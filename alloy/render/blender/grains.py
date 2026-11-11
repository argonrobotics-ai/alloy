# SPDX-License-Identifier: Apache-2.0
"""Grains and particles: body instancers (bundle `instancers`) and point sets (instanced grains or spheres).

Each set is one points carrier object with a Geometry Nodes Instance on Points, so 200k grains cost one object.
"""

from __future__ import annotations

import logging
import math
import zlib

import bpy
import numpy as np

import meshdata
import motion
import nodes as nd
import primitives
from io_bundle import Bundle
from look import Look, LookError
from materials import MaterialLibrary

log = logging.getLogger(__name__)


def prototype(name: str, arrays: primitives.MeshArrays, mat, coll, smooth_deg: float | None = 30.0):
    """A hidden source object for instancing (Object Info reads its geometry)."""
    verts, corners, sizes = arrays
    me = meshdata.make_mesh(f"proto_{name}", verts, corners, sizes, smooth_deg)
    me.materials.append(mat)
    obj = meshdata.new_object(f"proto_{name}", me, coll)
    obj.hide_render = True
    obj.hide_viewport = True
    return obj


def instance_group(name: str, proto: bpy.types.Object, rotation: str,
                   max_tilt_deg: float = 15.0) -> bpy.types.NodeTree:
    """Instance `proto` on every point; rotation from the `rot` attribute or a stable random per point index.

    ``rotation``: "attribute", "random" (any orientation) or "flat" (random yaw, tilt within ``max_tilt_deg``:
    grains lying in a bed; the grain's long axis is X, so small X/Y angles keep it horizontal).
    """
    g, gin, gout = nd.gn_group(f"instances_{name}")
    info = nd.node(g, "GeometryNodeObjectInfo", {"As Instance": True}, transform_space="ORIGINAL")
    info.inputs["Object"].default_value = proto
    iop = nd.node(g, "GeometryNodeInstanceOnPoints")
    nd.link(g, gin.outputs[0], iop.inputs["Points"])
    nd.link(g, info.outputs["Geometry"], iop.inputs["Instance"])
    if rotation == "attribute":
        nd.link(g, nd.named_attr(g, "rot", "QUATERNION"), iop.inputs["Rotation"])
    else:
        rnd = nd.node(g, "FunctionNodeRandomValue", data_type="FLOAT_VECTOR")
        tilt = math.radians(max_tilt_deg) / math.sqrt(2.0)   # per axis, so the combined tilt stays within the max
        lo, hi = ((-tilt, -tilt, 0.0), (tilt, tilt, 2 * math.pi)) if rotation == "flat" else \
            ((0.0, 0.0, 0.0), (2 * math.pi, 2 * math.pi, 2 * math.pi))
        rnd.inputs[0].default_value = lo
        rnd.inputs[1].default_value = hi
        rnd.inputs["Seed"].default_value = zlib.crc32(name.encode()) % 100000
        nd.link(g, nd.node(g, "GeometryNodeInputIndex").outputs[0], rnd.inputs["ID"])
        eul = nd.node(g, "FunctionNodeEulerToRotation")
        nd.link(g, rnd.outputs[0], eul.inputs[0])
        nd.link(g, eul.outputs[0], iop.inputs["Rotation"])
    nd.link(g, iop.outputs[0], gout.inputs[0])
    return g


def sphere_group(name: str, radius: float, mat) -> bpy.types.NodeTree:
    """Plain particles as a Cycles point cloud (true spheres, cheapest per point)."""
    g, gin, gout = nd.gn_group(f"points_{name}")
    m2p = nd.node(g, "GeometryNodeMeshToPoints", {"Radius": radius})
    nd.link(g, gin.outputs[0], m2p.inputs["Mesh"])
    nd.link(g, nd.set_material_node(g, m2p.outputs[0], mat), gout.inputs[0])
    return g


def _long_axis(bundle: Bundle, ent: dict) -> int:
    """Index of the source shape's longest local axis (capsule / cylinder / cone: Z)."""
    if ent.get("mesh") is not None:
        return int(np.argmax(np.ptp(bundle.mesh(ent["mesh"])["vertices"], axis=0)))
    if ent.get("shape_type") in ("capsule", "cylinder", "cone"):
        return 2
    size = list(ent.get("size", [1.0])) + [0.0, 0.0]
    return int(np.argmax(size[:3]))


def _display_grain(bundle: Bundle, ent: dict, cfg: dict) -> primitives.MeshArrays:
    """A preset grain (long axis X) turned onto the source shape's long axis: display-only geometry swap."""
    verts, corners, sizes = primitives.grain_mesh(cfg["instance"], float(ent.get("size", [0.001])[0]),
                                                  float(cfg.get("scale", 1.0)))
    perm = ([0, 1, 2], [2, 0, 1], [1, 2, 0])[_long_axis(bundle, ent)]  # cyclic, so faces keep their winding
    return verts[:, perm], corners, sizes


def _instancer_arrays(bundle: Bundle, ent: dict, cfg: dict) -> primitives.MeshArrays:
    """Source shape geometry with the instancer's shape-in-body xform baked in, so instances use body poses."""
    if cfg.get("instance"):
        verts, corners, sizes = _display_grain(bundle, ent, cfg)
    elif ent.get("mesh") is not None:
        m = bundle.mesh(ent["mesh"])
        verts = m["vertices"] * np.asarray(ent.get("scale", [1, 1, 1]), np.float32)
        verts, faces = primitives.weld_exact(verts, m["faces"])
        _, corners, sizes = primitives.tris(faces)
    else:
        verts, corners, sizes = primitives.shape_mesh(ent.get("shape_type", "sphere"), ent.get("size", [0.002]))
    xf = np.asarray(ent.get("xform", [0, 0, 0, 0, 0, 0, 1]), np.float64)
    verts = xf[:3] + motion.quat_rotate(np.broadcast_to(xf[3:7], (len(verts), 4)), verts.astype(np.float64))
    return verts.astype(np.float32), corners, sizes


class InstancerSet:
    """Bundle instancers: many one-shape bodies drawn as instances at their recorded poses."""

    def __init__(self, bundle: Bundle, look: Look, mats: MaterialLibrary, coll: bpy.types.Collection):
        self.items: list[tuple[bpy.types.Mesh, np.ndarray]] = []
        self._last = None
        cfgs = look.section("instances")
        for ent in bundle.entities("instancers"):
            name = ent["name"]
            if look.is_hidden([name]):
                continue
            cfg = cfgs.get(name, {})
            mat_name = cfg.get("material") or look.assigned([name])
            mat = mats.get(mat_name) if mat_name else mats.tinted(ent.get("color"))
            smooth = None if cfg.get("instance") else 30.0
            proto = prototype(name, _instancer_arrays(bundle, ent, cfg), mat, coll, smooth)
            arr = bundle.array("inst", name)
            me = meshdata.point_mesh(name, arr.shape[1])
            obj = meshdata.new_object(name, me, coll)
            nd.gn_modifier(obj, instance_group(name, proto, "attribute"))
            self.items.append((me, arr))
        log.info("instancers: %d", len(self.items))

    def update(self, fr: float, center: int) -> None:
        if fr == self._last:
            return
        self._last = fr
        for me, arr in self.items:
            p, q = motion.interp_poses(arr, fr)
            meshdata.set_attr(me, "rot", motion.xyzw_to_wxyz(q), "QUATERNION")
            meshdata.set_positions(me, p)


class PointSet:
    """Point sets not meshed as liquids: preset grains with stable random rotations, or spheres."""

    def __init__(self, bundle: Bundle, look: Look, mats: MaterialLibrary, coll: bpy.types.Collection):
        self.items: list[tuple[bpy.types.Mesh, np.ndarray]] = []
        self._last = None
        liquids = set(look.section("liquids"))
        cfgs = look.section("points")
        for ent in bundle.entities("points"):
            name = ent["name"]
            if name in liquids or look.is_hidden([name]):
                continue
            cfg = cfgs.get(name, {})
            mat_name = cfg.get("material") or look.assigned([name])
            mat = mats.get(mat_name) if mat_name else mats.tinted(None)
            radius = float(ent.get("radius", 0.001))
            me = meshdata.point_mesh(name, int(ent["count"]))
            obj = meshdata.new_object(name, me, coll)
            grain = cfg.get("instance")
            if grain:
                arrays = primitives.grain_mesh(grain, radius, float(cfg.get("scale", 1.0)))
                rotation = cfg.get("rotation", "random")
                if rotation not in ("random", "flat"):
                    raise LookError(f'points.{name}.rotation: expected "random" or "flat", got {rotation!r}')
                group = instance_group(name, prototype(name, arrays, mat, coll, None), rotation,
                                       float(cfg.get("max_tilt_deg", 15.0)))
            else:
                group = sphere_group(name, radius * float(cfg.get("scale", 1.0)), mat)
            nd.gn_modifier(obj, group)
            self.items.append((me, bundle.array("pts", name)))
        log.info("point sets: %d", len(self.items))

    def update(self, fr: float, center: int) -> None:
        if fr == self._last:
            return
        self._last = fr
        for me, arr in self.items:
            meshdata.set_positions(me, motion.lerp_frames(arr, fr))
