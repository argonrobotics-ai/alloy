# SPDX-License-Identifier: Apache-2.0
"""Skinned and attached display meshes (look.skins), deformed or carried by recorded body poses.

- **Mesh.** `.obj` (with `usemtl` groups and `vt` UVs) or `.glb`/`.gltf` (Blender's importer: the model's own PBR
  materials, textures and UVs, all primitives joined into one object). World coordinates of the exported world at
  `rest_frame` before the grid offset, or the body's local frame with `"local": true`.
- **Materials.** `material` replaces every material; `materials` maps OBJ `usemtl` names and `material_map` maps
  glTF material names to look materials (others keep the model's own, or the default plastic for OBJ).
- **Following bodies.** `"body": label` attaches rigidly (the object's matrix follows the body, with object motion
  blur). Otherwise a weights file blends several bodies per vertex: v = blend_i(w_i, T_i(t) T_i(rest)^-1) v0, in numpy
  every frame and motion-blur sub-frame. Dual-quaternion blending is the default: linear blend skinning collapses a
  90 degree living hinge to 71 % of its thickness, dual quaternions keep it. `"blend": "lbs"` selects LBS.
- **Grid replicas.** A skin covers its bodies in the hero world only. `"replicas"` says what the other worlds show
  for those bodies: `"proxy"` (default: their own proxy shapes), `"skin"` (this mesh, instanced per replica body),
  `{"mesh": "lod.glb"}` (a lighter mesh in the same frame as `mesh`) or `"none"`. Skin meshes need one `body`.
"""

from __future__ import annotations

import logging
import re

import bpy
import numpy as np
from mathutils import Matrix, kdtree

import imports
import meshdata
import nodes as nd
import primitives
from io_bundle import Bundle
from look import Look, LookError
from materials import MaterialLibrary
from skinning import (ObjMesh, blend_dual_quat, blend_linear, body_ids, load_weights, normalise_weights,
                      relative_poses, rigid_matrix)

log = logging.getLogger(__name__)


class SkinSet:
    """All look.skins entries; `hidden_bodies` lists bodies whose shapes the skins replace."""

    def __init__(self, bundle: Bundle, look: Look, mats: MaterialLibrary, coll: bpy.types.Collection):
        self.bundle = bundle
        self.items: list[dict] = []
        self.hidden_bodies: set[int] = set()
        self.proxy_bodies: set[int] = set()  # hidden in the hero world, proxies kept in replica worlds
        self.replica_sources: dict[int, bpy.types.Object] = {}  # body -> hidden mesh in the body frame
        self._named: list[tuple[bpy.types.Object, list[str]]] = []
        self._last = None
        for name, cfg in look.section("skins").items():
            if not look.is_hidden([name]):
                self._build(name, cfg, mats, coll)
        log.info("skins: %d", len(self.items))

    def _build(self, name: str, cfg: dict, mats: MaterialLibrary, coll) -> None:
        where = f"skins.{name}"
        if self.bundle.body_q is None:
            raise LookError(f"{where}: the bundle has no bodies to follow")
        path = self.bundle.root / str(cfg.get("mesh", ""))
        if not path.is_file() or path.suffix.lower() not in imports.IMPORTERS:
            raise LookError(f"{where}.mesh: no .obj/.glb/.gltf file {cfg.get('mesh')!r} under the bundle root")
        ids, idx, w, points = self._weights(cfg, where)
        local = bool(cfg.get("local", False))
        if local and len(ids) != 1:
            raise LookError(f"{where}.local needs a single body ('body'), got {len(ids)} bodies")
        world = int(next(b.get("world", -1) for b in self.bundle.scene["bodies"] if int(b["id"]) == ids[0]))
        offset = np.zeros(3) if local else self.bundle.offset(world)
        me = load_mesh(path, name, cfg, mats, offset, where)
        obj = meshdata.new_object(name, me, coll)
        if cfg.get("hide_bodies", True):
            self.hidden_bodies.update(int(i) for i in ids)
            self._replicas(name, cfg, mats, coll, me, ids, offset, where)
        rest = int(cfg.get("rest_frame", 0))
        if len(ids) == 1:
            self.items.append({"kind": "rigid", "obj": obj, "body": int(ids[0]), "rest": rest, "local": local})
        else:
            v0 = _positions(me)
            if points is not None:
                idx, w = _nearest_rows(v0, points + offset, idx, w)
            elif len(idx) != len(v0):
                raise LookError(f"{where}.weights: {len(idx)} rows for {len(v0)} mesh vertices (add 'points')")
            _rest_attributes(me, v0)
            blend = blend_linear if cfg.get("blend") in ("lbs", "linear") else blend_dual_quat
            self.items.append({"kind": "deform", "mesh": me, "v0": v0, "idx": idx, "ids": ids, "blend": blend,
                               "w": normalise_weights(idx, w, f"{where}.weights"), "rest": rest})
        _shell_modifiers(obj, cfg)
        labels = {int(b["id"]): b.get("label", "") for b in self.bundle.scene["bodies"]}
        self._named.append((obj, [name] + [labels.get(int(i), "") for i in ids]))

    def _replicas(self, name: str, cfg: dict, mats: MaterialLibrary, coll, me, ids, offset, where: str) -> None:
        """What replica worlds show for the bodies this skin hides (see the module docstring)."""
        mode = cfg.get("replicas", "proxy")
        if mode == "none":
            return
        if mode == "proxy":
            self.proxy_bodies.update(int(i) for i in ids)
            return
        if len(ids) != 1:
            raise LookError(f"{where}.replicas: a skin mesh on replicas needs a single 'body', got {len(ids)}")
        if mode == "skin":
            src = me.copy()
        elif isinstance(mode, dict) and mode.get("mesh"):
            lod = self.bundle.root / str(mode["mesh"])
            if not lod.is_file() or lod.suffix.lower() not in imports.IMPORTERS:
                raise LookError(f"{where}.replicas.mesh: no .obj/.glb/.gltf file {mode['mesh']!r} under the bundle")
            src = load_mesh(lod, f"{name}_lod", cfg, mats, offset, where)
        else:
            raise LookError(f'{where}.replicas is {mode!r}, expected "proxy", "skin", "none" or {{"mesh": file}}')
        body, rest = int(ids[0]), int(cfg.get("rest_frame", 0))
        if not cfg.get("local", False):  # world coordinates at rest_frame -> the body frame
            src.transform(Matrix(np.linalg.inv(rigid_matrix(self.bundle, body, rest, float(rest), True)).tolist()))
        obj = meshdata.new_object(f"{name}_replica_src", src, coll)
        obj.hide_render = True
        self.replica_sources[body] = obj

    def objects_matching(self, rx: re.Pattern) -> list[bpy.types.Object]:
        """Skin objects whose name or followed body labels match `rx`."""
        return [o for o, names in self._named if any(n and rx.search(n) for n in names)]

    def _weights(self, cfg: dict, where: str):
        if cfg.get("body"):
            return body_ids(self.bundle, [cfg["body"]], where), np.zeros((1, 1), np.int64), np.ones((1, 1)), None
        wpath = self.bundle.root / str(cfg.get("weights", ""))
        if not wpath.is_file():
            raise LookError(f"{where}: give 'body' or a 'weights' file under the bundle root, "
                            f"got {cfg.get('weights')!r}")
        labels, idx, w, points = load_weights(wpath)
        ids = body_ids(self.bundle, labels, where)
        if idx.max() >= len(ids):
            raise LookError(f"{where}.weights: body index {int(idx.max())} >= {len(ids)} listed bodies")
        used = np.unique(idx[w > 0])
        return (ids[used], np.zeros_like(idx), w, points) if len(used) == 1 else (ids, idx, w, points)

    def update(self, fr: float, center: int) -> None:
        if fr == self._last:
            return
        self._last = fr
        for it in self.items:
            if it["kind"] == "rigid":
                m = rigid_matrix(self.bundle, it["body"], it["rest"], fr, it["local"])
                it["obj"].matrix_world = Matrix(m.tolist())
                continue
            t, q = relative_poses(self.bundle, it["ids"], it["rest"], fr)
            meshdata.set_positions(it["mesh"], it["blend"](it["v0"], it["idx"], it["w"], t, q))


def load_mesh(path, name: str, cfg: dict, mats: MaterialLibrary, offset: np.ndarray, where: str):
    """The skin's mesh with its material slots resolved (see the module docstring)."""
    if path.suffix.lower() == ".obj":
        return _obj_mesh(path, name, cfg, mats, offset)
    try:
        me = imports.joined_mesh(path, name)
    except Exception as exc:  # noqa: BLE001 - re-raised with the look field
        raise LookError(f"{where}.mesh: cannot import {path.name} ({exc})") from exc
    if np.any(offset):
        me.transform(Matrix.Translation(tuple(float(x) for x in offset)))
    remap = cfg.get("material_map") or {}
    for i, slot in enumerate(me.materials):
        own = imports.base_name(slot.name) if slot else ""
        if cfg.get("material"):
            me.materials[i] = mats.get(cfg["material"])
        elif own in remap:
            me.materials[i] = mats.get(remap[own])
    return me


def _obj_mesh(path, name: str, cfg: dict, mats: MaterialLibrary, offset: np.ndarray):
    m = ObjMesh(path)
    me = meshdata.make_mesh(name, m.verts.astype(np.float64) + offset, m.corners, m.sizes,
                            smooth_angle_deg=float(cfg.get("smooth_angle_deg", 30.0)), corner_uv=m.corner_uv)
    groups = cfg.get("materials") or {}
    default = mats.get(cfg["material"]) if cfg.get("material") else mats.tinted(None)
    me.materials.append(default)
    for g in m.groups:
        me.materials.append(default if cfg.get("material") else mats.get(groups[g]) if g in groups else default)
    me.polygons.foreach_set("material_index", (m.face_group + 1).astype(np.int32))
    return me


def _positions(me: bpy.types.Mesh) -> np.ndarray:
    v = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", v)
    return v.reshape(-1, 3).astype(np.float64)


def _nearest_rows(v0: np.ndarray, points: np.ndarray, idx: np.ndarray, w: np.ndarray):
    """Per mesh vertex, the weight row of the nearest sample point."""
    tree = kdtree.KDTree(len(points))
    for i, p in enumerate(points):
        tree.insert(p, i)
    tree.balance()
    rows = np.array([tree.find(v)[1] for v in v0], np.int64)
    return idx[rows], w[rows]


def _rest_attributes(me: bpy.types.Mesh, v0: np.ndarray) -> None:
    """rest_p / rest_n so procedural and triplanar textures stick to the deforming surface."""
    corners = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", corners)
    sizes = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", sizes)
    meshdata.set_attr(me, "rest_p", v0)
    meshdata.set_attr(me, "rest_n", primitives.vertex_normals(v0, _fan_triangles(corners, sizes)))
    meshdata.set_attr(me, "has_rest", np.ones(len(v0), np.float32), "FLOAT")


def _fan_triangles(corners: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    tris, start = [], 0
    for n in sizes:
        tris += [[corners[start], corners[start + k], corners[start + k + 1]] for k in range(1, n - 1)]
        start += n
    return np.asarray(tris, np.int64).reshape(-1, 3)


def _shell_modifiers(obj: bpy.types.Object, cfg: dict) -> None:
    thick = float(cfg.get("thickness_m", 0.0) or 0.0)
    if thick > 0.0:
        nd.solidify(obj, thick, cfg, even_default=True)
    levels = int(cfg.get("subdivide", 0) or 0)
    if levels > 0:
        mod = obj.modifiers.new("subdivide", "SUBSURF")
        mod.levels = mod.render_levels = levels
