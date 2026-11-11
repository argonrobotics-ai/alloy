# SPDX-License-Identifier: Apache-2.0
"""Cables and thread: smooth tubes through recorded centrelines, with UVs that stick to the jacket.

Geometry Nodes: edge chain -> Catmull-Rom curve -> swept open circle (first point repeated, so the seam can carry
its own UVs) -> UVMap stored on face corners -> seam merged (normals smooth across it, corner UVs kept) -> end caps.
u = rest arc length / curves.<n>.uv_scale_m: the arc length is measured on the first recorded frame and carried by
each centreline point, so a print repeats along the cable and does not swim when the cable stretches or slides.
v = 0..1 around the circumference, plus uv_twist turns per metre.
"""

from __future__ import annotations

import logging
import math

import bpy
import numpy as np

import meshdata
import motion
import nodes as nd
from io_bundle import Bundle
from look import Look
from materials import MaterialLibrary

log = logging.getLogger(__name__)
TUBE_SEGMENTS = 16
UV_SCALE_M = 0.01


def _profile(g, radius: float):
    """Open circle whose last point repeats the first, with prof_v = 0..1 around it."""
    arc = nd.node(g, "GeometryNodeCurveArc", {"Resolution": TUBE_SEGMENTS + 1, "Radius": radius,
                                              "Sweep Angle": 2.0 * math.pi}, mode="RADIUS")
    store = nd.node(g, "GeometryNodeStoreNamedAttribute", {"Name": "prof_v"}, data_type="FLOAT", domain="POINT")
    nd.link(g, arc.outputs["Curve"], store.inputs["Geometry"])
    nd.link(g, nd.node(g, "GeometryNodeSplineParameter").outputs["Factor"], store.inputs["Value"])
    return store.outputs[0]


def _store_uv(g, mesh, uv_scale_m: float, twist: float):
    u = nd.math(g, "DIVIDE", nd.named_attr(g, "rest_s", "FLOAT"), max(uv_scale_m, 1e-9))
    v = nd.math(g, "MULTIPLY_ADD", nd.named_attr(g, "rest_s", "FLOAT"), twist, nd.named_attr(g, "prof_v", "FLOAT"))
    comb = nd.node(g, "ShaderNodeCombineXYZ")
    nd.link(g, u, comb.inputs[0])
    nd.link(g, v, comb.inputs[1])
    store = nd.node(g, "GeometryNodeStoreNamedAttribute", {"Name": "UVMap"}, data_type="FLOAT2", domain="CORNER")
    nd.link(g, mesh, store.inputs["Geometry"])
    nd.link(g, comb.outputs[0], store.inputs["Value"])
    return store.outputs[0]


def _cap(g, curve, radius: float, factor: float, outward: float):
    """A disc closing one tube end, facing along the curve tangent (outward = +1 at the end, -1 at the start)."""
    sample = nd.node(g, "GeometryNodeSampleCurve", {"Factor": factor}, mode="FACTOR")
    nd.link(g, curve, sample.inputs["Curves"])
    disc = nd.node(g, "GeometryNodeMeshCircle", {"Vertices": TUBE_SEGMENTS, "Radius": radius},
                   fill_type="TRIANGLE_FAN")
    align = nd.node(g, "FunctionNodeAlignRotationToVector", axis="Z")
    nd.link(g, nd.vmath(g, "SCALE", sample.outputs["Tangent"], scale=outward), align.inputs["Vector"])
    move = nd.node(g, "GeometryNodeTransform")
    nd.link(g, disc.outputs[0], move.inputs["Geometry"])
    nd.link(g, sample.outputs["Position"], move.inputs["Translation"])
    nd.link(g, align.outputs[0], move.inputs["Rotation"])
    return move.outputs[0]


def tube_group(name: str, radius: float, resolution: int, mat, uv_scale_m: float, twist: float):
    """GN group turning the centreline chain (with a rest_s point attribute) into a UV-mapped capped tube."""
    g, gin, gout = nd.gn_group(f"tube_{name}")
    m2c = nd.node(g, "GeometryNodeMeshToCurve")
    stype = nd.node(g, "GeometryNodeCurveSplineType", spline_type="CATMULL_ROM")
    sres = nd.node(g, "GeometryNodeSetSplineResolution", {"Resolution": max(resolution, 1)})
    c2m = nd.node(g, "GeometryNodeCurveToMesh", {"Fill Caps": False})
    nd.link(g, gin.outputs[0], m2c.inputs["Mesh"])
    nd.link(g, m2c.outputs[0], stype.inputs["Curve"])
    nd.link(g, stype.outputs[0], sres.inputs["Geometry"])
    curve = sres.outputs[0]
    nd.link(g, curve, c2m.inputs["Curve"])
    nd.link(g, _profile(g, radius), c2m.inputs["Profile Curve"])
    if "Scale" in c2m.inputs:
        c2m.inputs["Scale"].default_value = 1.0
    merge = nd.node(g, "GeometryNodeMergeByDistance", {"Distance": radius * 1e-3}, mode="ALL")
    nd.link(g, _store_uv(g, c2m.outputs[0], uv_scale_m, twist), merge.inputs["Geometry"])
    join = nd.node(g, "GeometryNodeJoinGeometry")
    caps = (_cap(g, curve, radius, 0.0, -1.0), _cap(g, curve, radius, 1.0, 1.0))
    for geo in (nd.smooth_node(g, merge.outputs[0]), *caps):
        nd.link(g, geo, join.inputs[0])
    nd.link(g, nd.set_material_node(g, join.outputs[0], mat), gout.inputs[0])
    return g


class CurveSet:
    """One tube per bundle curve; the centreline points are rewritten per (sub)frame."""

    def __init__(self, bundle: Bundle, look: Look, mats: MaterialLibrary, coll: bpy.types.Collection):
        from deform import _materials_for

        self.items: list[tuple[bpy.types.Mesh, np.ndarray]] = []
        self._last = None
        cfgs = look.section("curves")
        for c in bundle.entities("curves"):
            if look.is_hidden([c["name"]]):
                continue
            cfg = cfgs.get(c["name"], {})
            mat = _materials_for(look, mats, c["name"], cfg)[0]
            arr = bundle.array("curve", c["name"])
            me = meshdata.chain_mesh(c["name"], arr.shape[1])
            meshdata.set_positions(me, np.asarray(arr[0]))
            meshdata.set_attr(me, "rest_s", motion.rest_arc_length(arr[0]).astype(np.float32), "FLOAT")
            obj = meshdata.new_object(c["name"], me, coll)
            radius = float(cfg.get("radius_m", c.get("radius", 0.002)))
            group = tube_group(c["name"], radius, int(cfg.get("resolution", 3)), mat,
                               float(cfg.get("uv_scale_m", UV_SCALE_M)), float(cfg.get("uv_twist", 0.0)))
            nd.gn_modifier(obj, group)
            self.items.append((me, arr))
        log.info("curves: %d", len(self.items))

    def update(self, fr: float, center: int) -> None:
        if fr == self._last:
            return
        self._last = fr
        for me, arr in self.items:
            meshdata.set_positions(me, motion.lerp_frames(arr, fr))
