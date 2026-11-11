# SPDX-License-Identifier: Apache-2.0
"""Liquid surfaces from particles: points -> Points to Volume -> Volume to Mesh -> Taubin smoothing (all in GN).

Motion blur: the surface is meshed once per rendered frame from that frame's particles (the handler ignores
sub-frames), and each surface vertex carries the `velocity` of its nearest particle. Cycles turns a `velocity`
attribute into deformation blur over the shutter, so topology never has to match between motion steps.
Optional `contact` rules seat the surface on its container's walls (contact.py). A moving container makes Blender
re-evaluate the group at each motion step (same particles, so the same topology; it costs one meshing per step).
"""

from __future__ import annotations

import logging
from typing import Callable

import bpy
import numpy as np

import contact
import meshdata
import nodes as nd
from io_bundle import Bundle
from look import Look, LookError
from materials import MaterialLibrary

log = logging.getLogger(__name__)
TAUBIN_LAMBDA, TAUBIN_MU = 0.6, -0.62
ISO_THRESHOLD = 0.3
ERODE = 0.75  # level set built ERODE * particle radius fatter, then shrunk back: fills the crevices between beads
SMOOTH_REACH = 2.0  # blur iterations per pass = SMOOTH_REACH * particle radius / voxel


def _taubin(g, geo, passes: int, iterations: int):
    """`passes` shrink-free smoothing passes (a Laplacian step then an inflating step, each over `iterations`
    blur iterations, so one pass removes bumps about one particle across whatever the voxel size)."""
    for _ in range(passes):
        for factor in (TAUBIN_LAMBDA, TAUBIN_MU):
            pos = nd.node(g, "GeometryNodeInputPosition").outputs[0]
            blur = nd.node(g, "GeometryNodeBlurAttribute", {"Iterations": iterations, "Weight": 1.0},
                           data_type="FLOAT_VECTOR")
            nd.link(g, pos, blur.inputs["Value"])
            delta = nd.vmath(g, "SUBTRACT", blur.outputs["Value"], pos)
            setp = nd.node(g, "GeometryNodeSetPosition")
            nd.link(g, geo, setp.inputs["Geometry"])
            nd.link(g, nd.vmath(g, "SCALE", delta, scale=factor), setp.inputs["Offset"])
            geo = setp.outputs[0]
    return geo


def _velocity_transfer(g, particles, surface):
    """Store each surface vertex's nearest-particle velocity as the `velocity` attribute Cycles blurs with."""
    store = nd.node(g, "GeometryNodeStoreNamedAttribute", {"Name": "velocity"}, data_type="FLOAT_VECTOR",
                    domain="POINT")
    nd.link(g, surface, store.inputs["Geometry"])
    nd.link(g, nd.sample_nearest_point(g, particles, nd.named_attr(g, "vel")), store.inputs["Value"])
    return store.outputs[0]


def _erode(g, geo, distance: float, walls: contact.Snapper | None = None):
    """Move the smoothed surface inward along its normals by `distance` (undoes the fattened level set).

    With contact `walls` the erosion fades to 0 at the wall, so the surface rises into a small meniscus there."""
    if distance <= 0.0:
        return geo
    setp = nd.node(g, "GeometryNodeSetPosition")
    nd.link(g, geo, setp.inputs["Geometry"])
    normal = nd.node(g, "GeometryNodeInputNormal").outputs[0]
    scale = nd.math(g, "MULTIPLY", walls.erode_weight(), -distance) if walls else -distance
    nd.link(g, nd.vmath(g, "SCALE", normal, scale=scale), setp.inputs["Offset"])
    return setp.outputs[0]


def liquid_group(name: str, radius: float, voxel: float, smooth: int, mat, particle_r: float,
                 erode: float = ERODE, walls: list | None = None) -> bpy.types.NodeTree:
    """GN surface: level set of radius + erode*particle_r, Taubin smoothing, erode back, velocity attribute.

    With contact `walls`: snap to the walls -> smooth (relaxes wet vertices along the wall) -> erode, fading to 0 at
    the wall -> snap again, re-seating every wet vertex."""
    g, gin, gout = nd.gn_group(f"liquid_{name}")
    fat = radius + erode * particle_r
    m2p = nd.node(g, "GeometryNodeMeshToPoints", {"Radius": fat})
    p2v = nd.node(g, "GeometryNodePointsToVolume", {"Voxel Size": voxel, "Radius": fat, "Density": 1.0},
                  resolution_mode="VOXEL_SIZE")
    v2m = nd.node(g, "GeometryNodeVolumeToMesh", {"Threshold": ISO_THRESHOLD, "Adaptivity": 0.0},
                  resolution_mode="GRID")
    nd.link(g, gin.outputs[0], m2p.inputs["Mesh"])
    nd.link(g, m2p.outputs[0], p2v.inputs["Points"])
    nd.link(g, p2v.outputs[0], v2m.inputs["Volume"])
    snapper = contact.Snapper(g, gin.outputs[0], walls) if walls else None
    geo = snapper.snap(v2m.outputs[0], first=True) if snapper else v2m.outputs[0]
    geo = _taubin(g, geo, smooth, max(1, round(SMOOTH_REACH * particle_r / voxel)))
    geo = _erode(g, geo, erode * particle_r, snapper)
    if snapper:
        geo = snapper.snap(geo, first=False)
    geo = _velocity_transfer(g, gin.outputs[0], geo)
    geo = nd.set_material_node(g, nd.smooth_node(g, geo), mat)
    nd.link(g, geo, gout.inputs[0])
    return g


class LiquidSet:
    """One meshed liquid object per look.liquids entry."""

    def __init__(self, bundle: Bundle, look: Look, mats: MaterialLibrary, coll: bpy.types.Collection,
                 lookup: Callable | None = None):
        self.items: list[tuple[bpy.types.Mesh, np.ndarray, np.ndarray]] = []
        self._last = None
        points = {p["name"]: p for p in bundle.entities("points")}
        for name, cfg in look.section("liquids").items():
            if name not in points:
                raise LookError(f"liquids.{name}: no point set of that name in the bundle ({sorted(points)})")
            if look.is_hidden([name]):
                continue
            ent = points[name]
            mat = mats.get(cfg.get("material") or look.assigned([name]) or "water")
            radius = float(ent.get("radius", 0.001)) * float(cfg.get("radius_scale", 1.6))
            voxel = float(cfg.get("voxel_m", radius * 0.5))
            me = meshdata.point_mesh(name, int(ent["count"]))
            obj = meshdata.new_object(name, me, coll)
            particle_r = float(ent.get("radius", 0.001))
            group = liquid_group(name, radius, voxel, int(cfg.get("smooth", 2)), mat, particle_r,
                                 float(cfg.get("erode", ERODE)), self._walls(name, cfg, particle_r, lookup))
            nd.gn_modifier(obj, group)
            self.items.append((me, bundle.array("pts", name), bundle.array("vel", name)))
            log.info("liquid %s: %d particles, level-set radius %.2f mm, voxel %.2f mm", name, ent["count"],
                     radius * 1e3, voxel * 1e3)

    @staticmethod
    def _walls(name: str, cfg: dict, particle_r: float, lookup) -> list | None:
        rules = cfg.get("contact") or []
        if not rules:
            return None
        if lookup is None:
            raise LookError(f"liquids.{name}.contact needs rigid shapes or skins to match")
        out = contact.walls(name, rules, particle_r, lookup)
        for w in out:
            log.info("liquid %s: contact with %s, snap within %.2f mm, overlap %.2f mm", name,
                     [o.name for o in w.coll.objects], w.distance * 1e3, w.overlap * 1e3)
        return out

    def update(self, fr: float, center: int) -> None:
        """Re-mesh only when the rendered frame changes; sub-frames blur from the velocity attribute."""
        if center == self._last:
            return
        self._last = center
        for me, pts, vel in self.items:
            k = int(min(max(center, 0), pts.shape[0] - 1))
            meshdata.set_attr(me, "vel", np.asarray(vel[k]))
            meshdata.set_positions(me, np.asarray(pts[k]))
