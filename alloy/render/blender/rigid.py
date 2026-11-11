# SPDX-License-Identifier: Apache-2.0
"""Rigid shapes as objects with shared meshes, plus one Geometry Nodes instancer per hero shape for replica worlds.

Shape world pose = body_q (Warp xyzw) * shape xform; static shapes (body -1) use the xform directly. The recorder has
already shifted every exported world by its world_offsets entry, so poses are used as recorded.
Shapes of a body-chain curve carry `curve` and are hidden (the tube draws them), as are shapes of bodies a skin
replaces. look.bevels adds a display-only geometric bevel to matching box / cylinder / cone / mesh shapes.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Callable

import bpy
import numpy as np
from mathutils import Matrix

import meshdata
import motion
import nodes as nd
import primitives
from io_bundle import Bundle
from look import Look, LookError
from materials import MaterialLibrary

log = logging.getLogger(__name__)
SMOOTH_ANGLE_DEG = 30.0
BEVEL_TYPES = ("box", "cylinder", "cone", "mesh")


class ShapePoser:
    """Vectorised world poses for a list of shapes at a fractional frame."""

    def __init__(self, bundle: Bundle, shapes: list[dict]):
        self.bundle = bundle
        xf = np.array([s.get("xform", [0, 0, 0, 0, 0, 0, 1]) for s in shapes], np.float64).reshape(-1, 7)
        self.xp, self.xq = xf[:, :3], motion.normalize_quat(xf[:, 3:7])
        self.body = np.array([int(s.get("body", -1)) for s in shapes], np.int64)
        self.dynamic = self.body >= 0

    def poses(self, fr: float) -> tuple[np.ndarray, np.ndarray]:
        """Positions [N,3] and xyzw quaternions [N,4] of all shapes at frame fr."""
        p, q = self.xp.copy(), self.xq.copy()
        if self.dynamic.any() and self.bundle.body_q is not None:
            bp, bq = motion.interp_poses(self.bundle.body_q, fr)
            idx = self.body[self.dynamic]
            dp, dq = motion.compose(bp[idx], bq[idx], self.xp[self.dynamic], self.xq[self.dynamic])
            p[self.dynamic] = dp
            q[self.dynamic] = dq
        return p, q


class RigidSet:
    """All non-replica shapes (one object each) and replica instancers; updates poses per (sub)frame."""

    def __init__(self, bundle: Bundle, look: Look, mats: MaterialLibrary, coll: bpy.types.Collection,
                 hidden_bodies: set[int] | None = None, skins=None):
        self.bundle, self.look, self.mats, self.coll = bundle, look, mats, coll
        self.hidden_bodies = hidden_bodies or set()
        # Bodies a skin hides in the hero world: keep their proxies as replica sources, or instance a skin mesh.
        self.proxy_bodies: set[int] = getattr(skins, "proxy_bodies", set())
        self.replica_sources: dict[int, bpy.types.Object] = getattr(skins, "replica_sources", {})
        self.bevels = [(re.compile(b["match"]), b) for b in (look.section("bevels") or [])]
        self.body_labels = {int(b["id"]): b.get("label", "") for b in bundle.scene.get("bodies", [])}
        self._meshes: dict[tuple, bpy.types.Mesh] = {}
        self.shapes: list[dict] = []
        self.objects: list[bpy.types.Object] = []
        self.replica_sets: list[tuple[bpy.types.Mesh, ShapePoser]] = []
        self._replica_objects: list[tuple[bpy.types.Object, list[str]]] = []
        self._skin_anchored: set[int] = set()
        self._last = None
        self._build()

    def labels(self, shape: dict) -> list[str]:
        return [shape.get("label", ""), self.body_labels.get(int(shape.get("body", -1)), "")]

    def _build(self) -> None:
        heroes: dict[int, bpy.types.Object] = {}
        replicas: dict[int, list[dict]] = {}
        for s in self.bundle.scene.get("shapes", []):
            if s.get("curve") or self.look.is_hidden(self.labels(s)):
                continue
            if s.get("replica_of") is not None:
                replicas.setdefault(int(s["replica_of"]), []).append(s)
                continue
            if int(s.get("body", -1)) in self.hidden_bodies:
                self._replica_source(s, heroes)
                continue
            obj = self._shape_object(s)
            heroes[int(s["id"])] = obj
            self.shapes.append(s)
            self.objects.append(obj)
        self.poser = ShapePoser(self.bundle, self.shapes)
        for hero_id, reps in replicas.items():
            self._replica_instancer(hero_id, heroes, reps)
        self._apply(0.0, everything=True)
        log.info("rigid: %d shape objects, %d meshes, %d replica sets (%d replicas)", len(self.objects),
                 len(self._meshes), len(self.replica_sets), sum(len(r) for r in replicas.values()))

    def _replica_source(self, s: dict, heroes: dict) -> None:
        """A hero shape whose body a skin hides: not rendered here, but still the source its replicas instance (its
        own proxy, or the skin's mesh moved into this shape's frame on the body's first shape only)."""
        body = int(s.get("body", -1))
        if body in self.proxy_bodies:
            obj = self._shape_object(s)
        elif body in self.replica_sources and body not in self._skin_anchored:
            self._skin_anchored.add(body)
            me = self.replica_sources[body].data.copy()
            me.transform(Matrix(np.linalg.inv(motion.pose_matrices(
                np.asarray(s["xform"][:3], float)[None], np.asarray(s["xform"][3:7], float)[None])[0]).tolist()))
            obj = meshdata.new_object(f"{s.get('label') or s['id']}_replica_skin", me, self.coll)
        else:
            return
        obj.hide_render = True
        heroes[int(s["id"])] = obj

    def _mesh_for(self, s: dict, mat: bpy.types.Material) -> bpy.types.Mesh:
        typ = s["type"]
        scale = tuple(round(float(x), 6) for x in s.get("scale", [1, 1, 1]))
        geo = ("mesh", str(s.get("mesh")), scale) if typ == "mesh" else (typ, tuple(round(float(x), 7)
                                                                                  for x in s.get("size", [])))
        key = geo + (mat.name,)
        if key in self._meshes:
            return self._meshes[key]
        if typ == "mesh":
            m = self.bundle.mesh(s["mesh"])
            verts, faces, uv = m["vertices"] * np.asarray(scale, np.float32), m["faces"], m.get("uv")
            if uv is None:
                verts, faces = primitives.weld_exact(verts, faces)
            _, corners, sizes = primitives.tris(faces)
            me = meshdata.make_mesh(f"mesh_{s.get('mesh')}", verts, corners, sizes, SMOOTH_ANGLE_DEG, uv)
        else:
            verts, corners, sizes = primitives.shape_mesh(typ, s.get("size", []))
            me = meshdata.make_mesh(f"{typ}_{s.get('label', '')}", verts, corners, sizes, SMOOTH_ANGLE_DEG)
        me.materials.append(mat)
        self._meshes[key] = me
        return me

    def _shape_object(self, s: dict) -> bpy.types.Object:
        mat = self.mats.for_labels(self.labels(s), s.get("color"), float(s.get("opacity", 1.0)))
        obj = meshdata.new_object(s.get("label") or f"shape_{s['id']}", self._mesh_for(s, mat), self.coll)
        obj.rotation_mode = "QUATERNION"
        self._bevel(obj, s)
        return obj

    def _bevel(self, obj: bpy.types.Object, s: dict) -> None:
        """Geometric bevel (angle-limited 30 deg, hardened normals) from the first matching look.bevels rule."""
        if s["type"] not in BEVEL_TYPES:
            return
        rule = next((b for rx, b in self.bevels if any(lb and rx.search(lb) for lb in self.labels(s))), None)
        if rule is None:
            return
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = float(rule.get("bevel_m", 0.0003))
        mod.segments = int(rule.get("segments", 3))
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(30.0)
        mod.harden_normals = True
        mod.use_clamp_overlap = True

    def _replica_instancer(self, hero_id: int, heroes: dict, reps: list[dict]) -> None:
        hero = heroes.get(hero_id)
        if hero is None:
            log.info("replica_of %d: no hero source (hidden, or covered by a skin mesh); %d replicas skipped",
                     hero_id, len(reps))
            return
        me = meshdata.point_mesh(f"replicas_{hero.name}", len(reps))
        obj = meshdata.new_object(f"replicas_{hero.name}", me, self.coll)
        g, gin, gout = nd.gn_group(f"replicas_{hero.name}")
        info = nd.node(g, "GeometryNodeObjectInfo", {"As Instance": True}, transform_space="ORIGINAL")
        info.inputs["Object"].default_value = hero
        iop = nd.node(g, "GeometryNodeInstanceOnPoints")
        nd.link(g, gin.outputs[0], iop.inputs["Points"])
        nd.link(g, info.outputs["Geometry"], iop.inputs["Instance"])
        nd.link(g, nd.named_attr(g, "rot", "QUATERNION"), iop.inputs["Rotation"])
        nd.link(g, iop.outputs[0], gout.inputs[0])
        nd.gn_modifier(obj, g)
        self.replica_sets.append((me, ShapePoser(self.bundle, reps)))
        self._replica_objects.append((obj, [lb for r in reps for lb in self.labels(r)]))

    def _apply(self, fr: float, everything: bool = False) -> None:
        p, q = self.poser.poses(fr)
        qw = motion.xyzw_to_wxyz(q)
        for i, obj in enumerate(self.objects):
            if everything or self.poser.dynamic[i]:
                obj.location = p[i]
                obj.rotation_quaternion = qw[i]
        for me, poser in self.replica_sets:
            if everything or poser.dynamic.any():
                rp, rq = poser.poses(fr)
                meshdata.set_attr(me, "rot", motion.xyzw_to_wxyz(rq), "QUATERNION")
                meshdata.set_positions(me, rp)

    def update(self, fr: float, center: int) -> None:
        if fr == self._last:
            return
        self._last = fr
        self._apply(fr)

    def objects_matching(self, rx: re.Pattern) -> list[bpy.types.Object]:
        """Shape objects (and replica instancers) with a shape or body label matching `rx`."""
        hits = [o for s, o in zip(self.shapes, self.objects) if any(lb and rx.search(lb) for lb in self.labels(s))]
        return hits + [o for o, labels in self._replica_objects if any(lb and rx.search(lb) for lb in labels)]

    def origin_fn(self, pattern: str) -> Callable[[float], np.ndarray]:
        """World position over time of the first shape, then body, whose label matches `pattern`."""
        rx = re.compile(pattern)
        for i, s in enumerate(self.shapes):
            if rx.search(s.get("label", "")):
                return lambda fr, i=i: self.poser.poses(fr)[0][i]
        for bid, label in self.body_labels.items():
            if rx.search(label) and self.bundle.body_q is not None:
                return lambda fr, b=bid: motion.interp_poses(self.bundle.body_q, fr)[0][b]
        raise LookError(f"focus.label {pattern!r} matches no visible shape or body label")
