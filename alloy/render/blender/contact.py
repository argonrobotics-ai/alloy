# SPDX-License-Identifier: Apache-2.0
"""Liquid-container contact (liquids.<name>.contact): close the air gap between a meshed liquid and its glass.

A particle liquid's level set stops about a particle radius short of the walls, with particle-scale bumps there.
Every liquid vertex within `distance_m` of a matched container surface moves to the nearest surface point and
`overlap_m` past it into the wall, so liquid and glass overlap slightly: the usual way to render a liquid-glass
interface (one refracting boundary, a clean meniscus line). Between 1 x and FALLOFF x `distance_m` the pull fades
smoothly to nothing, so the snapped band has no step at its edge. "Into the wall" is the side away from the vertex's
nearest particle, so flipped or one-sided container meshes and vertices that start inside the wall all work.

Snapped vertices are stored as `wet`. Smoothing then runs over every vertex (on the wall it relaxes the projected
marching-cubes triangles along the wall); the erosion of the fattened level set fades out between 2 x and 1 x
`distance_m` from a wall, so a small meniscus rises there instead of a crease; and the second snap re-seats every
wet vertex, so at the end none is off the wall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import bpy

import nodes as nd
from look import LookError

DISTANCE_RADII = 1.5  # default distance_m = 1.5 x particle radius
OVERLAP_M = 0.0002
FALLOFF = 2.0  # the pull fades from full at distance_m to none at FALLOFF x distance_m


@dataclass
class Wall:
    """One contact rule: the matched container objects (an off-scene collection) and its snap distances."""

    coll: bpy.types.Collection
    distance: float
    overlap: float


def walls(name: str, rules: list, particle_r: float, lookup: Callable[[re.Pattern], list]) -> list[Wall]:
    """Collect each rule's matched objects (rigid shapes, replicas or skins, possibly moving) into a collection."""
    out = []
    for i, rule in enumerate(rules):
        objs = lookup(re.compile(rule["match"]))
        if not objs:
            raise LookError(f"liquids.{name}.contact[{i}].match {rule['match']!r} matches no visible shape, "
                            "body or skin")
        coll = bpy.data.collections.new(f"contact_{name}_{i}")
        for obj in objs:
            coll.objects.link(obj)
        out.append(Wall(coll, float(rule.get("distance_m", DISTANCE_RADII * particle_r)),
                        float(rule.get("overlap_m", OVERLAP_M))))
    return out


def container_geometry(g, coll: bpy.types.Collection):
    """Realized world-space mesh of a container collection (the liquid object sits at the origin)."""
    info = nd.node(g, "GeometryNodeCollectionInfo", {"Separate Children": False, "Reset Children": False},
                   transform_space="RELATIVE")
    info.inputs["Collection"].default_value = coll
    real = nd.node(g, "GeometryNodeRealizeInstances")
    nd.link(g, info.outputs[0], real.inputs["Geometry"])
    return real.outputs[0]


class Snapper:
    """Builds the snap and erosion-weight nodes of one liquid group against its walls."""

    def __init__(self, g, particles, rules: list[Wall]):
        self.g, self.particles, self.rules = g, particles, rules
        self.boxes = [container_geometry(g, w.coll) for w in rules]

    def snap(self, geo, first: bool):
        """Seat vertices near each wall: fully within distance_m, fading smoothly to no pull at FALLOFF x distance_m
        (no step where the snapped band ends). The first pass stores `wet` and `wall_gap` (distance over distance_m,
        min over walls); the second also re-seats every wet vertex, wherever smoothing moved it."""
        for i, (wall, box) in enumerate(zip(self.rules, self.boxes)):
            prox = nd.node(self.g, "GeometryNodeProximity", target_element="FACES")
            nd.link(self.g, box, prox.inputs[0])
            nd.link(self.g, self._pos(), prox.inputs["Source Position"])
            dist = prox.outputs["Distance"]
            pull = self._falloff(dist, wall)
            pull = pull if first else nd.math(self.g, "MAXIMUM", pull, self._wet_near(dist, wall))
            move = nd.vmath(self.g, "SUBTRACT", self._target(box, prox.outputs["Position"], wall.overlap), self._pos())
            setp = nd.node(self.g, "GeometryNodeSetPosition")
            nd.link(self.g, geo, setp.inputs["Geometry"])
            nd.link(self.g, nd.vmath(self.g, "SCALE", move, scale=pull), setp.inputs["Offset"])
            geo = setp.outputs[0]
            if first:
                geo = self._record(geo, i, dist, wall)
        return geo

    def _falloff(self, dist, wall: Wall):
        """1 within distance_m, smoothstep down to 0 at FALLOFF x distance_m."""
        n = nd.node(self.g, "ShaderNodeMapRange", {"From Min": wall.distance, "From Max": FALLOFF * wall.distance,
                                                   "To Min": 1.0, "To Max": 0.0},
                    data_type="FLOAT", interpolation_type="SMOOTHSTEP", clamp=True)
        nd.link(self.g, dist, n.inputs["Value"])
        return n.outputs["Result"]

    def _record(self, geo, i: int, dist, wall: Wall):
        """Store `wet` (snapped by this or an earlier wall) and `wall_gap` (min over walls) after the first snap."""
        hit = nd.node(self.g, "FunctionNodeCompare", {"B": wall.distance}, data_type="FLOAT", operation="LESS_THAN")
        nd.link(self.g, dist, hit.inputs["A"])
        gap = nd.math(self.g, "DIVIDE", dist, wall.distance)
        wet = self._bool("OR", hit.outputs[0], nd.named_attr(self.g, "wet", "BOOLEAN")) if i else hit.outputs[0]
        geo = self._store(geo, "wet", "BOOLEAN", wet)
        return self._store(geo, "wall_gap", "FLOAT", nd.math(self.g, "MINIMUM", gap, self._gap()) if i else gap)

    def _wet_near(self, dist, wall: Wall):
        """Wet vertices this wall should re-seat: within FALLOFF x distance_m (with several walls, each keeps its
        own)."""
        near = nd.node(self.g, "FunctionNodeCompare", {"B": FALLOFF * wall.distance}, data_type="FLOAT",
                       operation="LESS_THAN")
        nd.link(self.g, dist, near.inputs["A"])
        return self._bool("AND", near.outputs[0], nd.named_attr(self.g, "wet", "BOOLEAN"))

    def erode_weight(self):
        """0 within distance_m of a wall, ramping to 1 at twice that."""
        n = nd.node(self.g, "ShaderNodeMath", operation="SUBTRACT", use_clamp=True)
        nd.link(self.g, self._gap(), n.inputs[0])
        n.inputs[1].default_value = 1.0
        return n.outputs[0]

    def _target(self, box, nearest, overlap: float):
        """Nearest wall point pushed `overlap` along the wall normal, away from the nearest particle's side."""
        sample = nd.node(self.g, "GeometryNodeSampleNearestSurface", data_type="FLOAT_VECTOR")
        nd.link(self.g, box, sample.inputs["Mesh"])
        nd.link(self.g, nd.node(self.g, "GeometryNodeInputNormal").outputs[0], sample.inputs["Value"])
        nd.link(self.g, self._pos(), sample.inputs["Sample Position"])
        normal = nd.vmath(self.g, "NORMALIZE", sample.outputs["Value"])
        particle = nd.sample_nearest_point(self.g, self.particles, self._pos())
        side = nd.math(self.g, "SIGN", nd.vmath(self.g, "DOT_PRODUCT", normal,
                                                nd.vmath(self.g, "SUBTRACT", particle, nearest)))
        push = nd.vmath(self.g, "SCALE", normal, scale=nd.math(self.g, "MULTIPLY", side, -overlap))
        return nd.vmath(self.g, "ADD", nearest, push)

    def _pos(self):
        return nd.node(self.g, "GeometryNodeInputPosition").outputs[0]

    def _gap(self):
        return nd.named_attr(self.g, "wall_gap", "FLOAT")

    def _bool(self, op: str, a, b):
        n = nd.node(self.g, "FunctionNodeBooleanMath", operation=op)
        nd.link(self.g, a, n.inputs[0])
        nd.link(self.g, b, n.inputs[1])
        return n.outputs[0]

    def _store(self, geo, name: str, data_type: str, value):
        store = nd.node(self.g, "GeometryNodeStoreNamedAttribute", {"Name": name}, data_type=data_type,
                        domain="POINT")
        nd.link(self.g, geo, store.inputs["Geometry"])
        nd.link(self.g, value, store.inputs["Value"])
        return store.outputs[0]
