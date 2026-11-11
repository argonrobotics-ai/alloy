# SPDX-License-Identifier: Apache-2.0
"""Deforming surfaces (cloth, soft bodies, cut pieces) and centreline curves (cables, thread).

Both keep a fixed-topology carrier mesh whose vertex positions are written per (sub)frame with foreach_set, so
Cycles gets deformation motion blur. Display-only processing (twin weld, solidify, subdivision, tube sweep) runs in
the modifier stack.
"""

from __future__ import annotations

import logging

import bpy
import numpy as np

import meshdata
import motion
import nodes as nd
import primitives
from io_bundle import Bundle
from look import Look
from materials import MaterialLibrary
from weld import WELD_DISTANCE_M, first_break, sealed_flags

log = logging.getLogger(__name__)


def _materials_for(look: Look, mats: MaterialLibrary, name: str, cfg: dict) -> list:
    names = cfg.get("materials") or ([cfg["material"]] if cfg.get("material") else [])
    if not names:
        assigned = look.assigned([name])
        return [mats.get(assigned) if assigned else mats.tinted(None)]
    return [mats.get(n) for n in names]


class SurfaceSet:
    """One object per bundle surface, with weld / solidify / subdivision modifiers from the look.

    Modifier order: weld (delete cut-tagged faces whose corners are all sealed, i.e. the two coincident copies
    inside an intact seam, then merge sealed twins), then subdivide, then solidify. So an unbroken pre-cut shows no
    groove or inner sheet, and an opened cut shows its cut faces (see weld.py for when a seam counts as open).
    """

    def __init__(self, bundle: Bundle, look: Look, mats: MaterialLibrary, coll: bpy.types.Collection):
        self.items: list[tuple[bpy.types.Mesh, np.ndarray, np.ndarray, np.ndarray]] = []
        self._last = None
        cfgs = look.section("surfaces")
        for s in bundle.entities("surfaces"):
            if look.is_hidden([s["name"]]):
                continue
            self._build(bundle, s, cfgs.get(s["name"], {}), _materials_for(look, mats, s["name"],
                                                                         cfgs.get(s["name"], {})), coll)
        log.info("surfaces: %d", len(self.items))

    def _build(self, bundle: Bundle, s: dict, cfg: dict, mat_list: list, coll) -> None:
        topo = bundle.surface_topology(s)
        arr = bundle.array("surf", s["name"])
        rest = np.asarray(arr[0], np.float32)
        faces = topo["faces"]
        _, corners, sizes = primitives.tris(faces)
        me = meshdata.make_mesh(s["name"], rest, corners, sizes, smooth_angle_deg=None)
        tags = np.minimum(topo["face_tag"].astype(np.int32), len(mat_list) - 1)
        me.polygons.foreach_set("material_index", tags)
        meshdata.set_attr(me, "rest_p", rest)
        meshdata.set_attr(me, "rest_n", primitives.vertex_normals(rest, faces))
        meshdata.set_attr(me, "has_rest", np.ones(len(rest), np.float32), "FLOAT")
        for m in mat_list:
            me.materials.append(m)
        obj = meshdata.new_object(s["name"], me, coll)
        twins = topo["twins"]
        broken_at = first_break(arr, twins)
        if len(twins):
            # Any non-zero tag is a cut face (tags > 1 only pick other materials: rim, flesh, seeds, ...).
            meshdata.set_attr(me, "cut", topo["face_tag"] >= 1, "BOOLEAN", "FACE")
            meshdata.set_attr(me, "sealed", sealed_flags(rest, twins, broken_at, 0.0), "FLOAT")
            self._weld_modifier(obj, s["name"])
            log.info("surface %s: %d twin pairs, %d open during the recording", s["name"], len(twins),
                     int(np.isfinite(broken_at).sum()))
        self._display_modifiers(obj, cfg)
        self.items.append((me, arr, twins, broken_at))

    @staticmethod
    def _weld_modifier(obj: bpy.types.Object, name: str) -> None:
        g, gin, gout = nd.gn_group(f"weld_{name}")
        # On the face domain the point attribute `sealed` is averaged, so 1.0 means every corner is sealed.
        all_sealed = nd.node(g, "FunctionNodeCompare", {"B": 0.999}, data_type="FLOAT", operation="GREATER_THAN")
        nd.link(g, nd.named_attr(g, "sealed", "FLOAT"), all_sealed.inputs["A"])
        inner = nd.node(g, "FunctionNodeBooleanMath", operation="AND")
        nd.link(g, nd.named_attr(g, "cut", "BOOLEAN"), inner.inputs[0])
        nd.link(g, all_sealed.outputs[0], inner.inputs[1])
        delete = nd.node(g, "GeometryNodeDeleteGeometry", domain="FACE", mode="ONLY_FACE")
        nd.link(g, gin.outputs[0], delete.inputs["Geometry"])
        nd.link(g, inner.outputs[0], delete.inputs["Selection"])
        point_sealed = nd.node(g, "FunctionNodeCompare", {"B": 0.5}, data_type="FLOAT", operation="GREATER_THAN")
        nd.link(g, nd.named_attr(g, "sealed", "FLOAT"), point_sealed.inputs["A"])
        merge = nd.node(g, "GeometryNodeMergeByDistance", {"Distance": WELD_DISTANCE_M}, mode="ALL")
        nd.link(g, delete.outputs[0], merge.inputs["Geometry"])
        nd.link(g, point_sealed.outputs[0], merge.inputs["Selection"])
        nd.link(g, merge.outputs[0], gout.inputs[0])
        nd.gn_modifier(obj, g, "weld_twins")

    @staticmethod
    def _display_modifiers(obj: bpy.types.Object, cfg: dict) -> None:
        """Subdivision then solidify, both after the weld (modifier order is stack order)."""
        levels = int(cfg.get("subdivide", 0) or 0)
        if levels > 0:
            mod = obj.modifiers.new("subdivide", "SUBSURF")
            mod.levels = levels
            mod.render_levels = levels
        thick = float(cfg.get("thickness_m", 0.0) or 0.0)
        if thick > 0.0:
            nd.solidify(obj, thick, cfg, even_default=False)

    def update(self, fr: float, center: int) -> None:
        if fr == self._last:
            return
        self._last = fr
        for me, arr, twins, broken_at in self.items:
            pos = motion.lerp_frames(arr, fr)
            if len(twins):
                meshdata.set_attr(me, "sealed", sealed_flags(pos, twins, broken_at, fr), "FLOAT")
            meshdata.set_positions(me, pos)
