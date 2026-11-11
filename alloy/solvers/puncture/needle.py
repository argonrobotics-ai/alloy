# SPDX-License-Identifier: Apache-2.0
"""Needle centreline geometry on device (config.py module doc for the parametrisation).

All functions take the needle arrays (``Needles``) and an index. Arc length ``s`` runs from the tip (0) to the swage
(``length``). ``project`` returns the unclamped closest-point parameter: < 0 is beyond the tip, > length beyond the
swage (arc: the angle is wrapped into the half-turn either side of the arc's middle).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .config import NeedleSpec

PI = wp.constant(3.14159265358979)


@wp.struct
class Needles:
    body: wp.array[wp.int32]
    world: wp.array[wp.int32]
    radius: wp.array[float]
    length: wp.array[float]
    wire: wp.array[float]
    origin: wp.array[wp.vec3]
    e1: wp.array[wp.vec3]
    e2: wp.array[wp.vec3]
    thread_start: wp.array[wp.int32]
    thread_count: wp.array[wp.int32]
    stopper: wp.array[wp.int32]
    thread_flen: wp.array[float]
    thread_eta: wp.array[float]


@dataclass
class NeedleArrays:
    """Host-built ``Needles`` plus the count (struct arrays keep their address for graph capture)."""

    count: int
    dev: Needles

    @classmethod
    def build(cls, specs: list[NeedleSpec], device) -> NeedleArrays:
        n = max(len(specs), 1)
        pad = list(specs) + [NeedleSpec(body=-1, length=1.0, wire_radius=1.0, origin=(0, 0, 0), e1=(1, 0, 0))] * (
            n - len(specs))

        def arr(vals, dtype):
            return wp.array(np.asarray(vals), dtype=dtype, device=device)

        d = Needles()
        d.body = arr([s.body for s in pad], wp.int32)
        d.world = arr([s.world for s in pad], wp.int32)
        d.radius = arr([s.radius for s in pad], float)
        d.length = arr([s.length for s in pad], float)
        d.wire = arr([s.wire_radius for s in pad], float)
        d.origin = arr([s.origin for s in pad], wp.vec3)
        d.e1 = arr([_unit(s.e1) for s in pad], wp.vec3)
        d.e2 = arr([_unit(s.e2) if s.radius > 0 else (0.0, 0.0, 0.0) for s in pad], wp.vec3)
        d.thread_start = arr([s.thread_start for s in pad], wp.int32)
        d.thread_count = arr([s.thread_count for s in pad], wp.int32)
        d.stopper = arr([int(s.stopper) for s in pad], wp.int32)
        d.thread_flen = arr([s.thread_flen for s in pad], float)
        d.thread_eta = arr([s.thread_eta for s in pad], float)
        return cls(count=len(specs), dev=d)


def _unit(v) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(a)
    if n <= 0.0:
        raise ValueError(f"needle axis {v} has zero length")
    return tuple(float(x) for x in a / n)


@wp.func
def pose(nd: Needles, i: int, body_q: wp.array[wp.transform]) -> wp.transform:
    b = nd.body[i]
    if b < 0:
        return wp.transform_identity()
    return body_q[b]


@wp.func
def point_local(nd: Needles, i: int, s: float) -> wp.vec3:
    r = nd.radius[i]
    if r > 0.0:
        a = s / r
        return nd.origin[i] + r * (wp.cos(a) * nd.e1[i] + wp.sin(a) * nd.e2[i])
    return nd.origin[i] + s * nd.e1[i]


@wp.func
def tangent_local(nd: Needles, i: int, s: float) -> wp.vec3:
    """Unit dP/ds (tip -> swage)."""
    r = nd.radius[i]
    if r > 0.0:
        a = s / r
        return -wp.sin(a) * nd.e1[i] + wp.cos(a) * nd.e2[i]
    return nd.e1[i]


@wp.func
def project_local(nd: Needles, i: int, q: wp.vec3) -> float:
    """Unclamped closest-point arc length of body-frame point q (module doc)."""
    r = nd.radius[i]
    v = q - nd.origin[i]
    if r > 0.0:
        phi = wp.atan2(wp.dot(v, nd.e2[i]), wp.dot(v, nd.e1[i]))
        mid = 0.5 * nd.length[i] / r
        while phi < mid - PI:
            phi = phi + 2.0 * PI
        while phi > mid + PI:
            phi = phi - 2.0 * PI
        return r * phi
    return wp.dot(v, nd.e1[i])


@wp.func
def point_world(nd: Needles, i: int, X: wp.transform, s: float) -> wp.vec3:
    return wp.transform_point(X, point_local(nd, i, s))


@wp.func
def tangent_world(nd: Needles, i: int, X: wp.transform, s: float) -> wp.vec3:
    return wp.transform_vector(X, tangent_local(nd, i, s))


@wp.func
def project_world(nd: Needles, i: int, X: wp.transform, p: wp.vec3) -> float:
    return project_local(nd, i, wp.transform_point(wp.transform_inverse(X), p))


@wp.func
def tip_axis_world(nd: Needles, i: int, X: wp.transform) -> wp.vec3:
    """Direction the tip advances (−dP/ds at s = 0)."""
    return -tangent_world(nd, i, X, 0.0)
