# SPDX-License-Identifier: Apache-2.0
"""Hole geometry shared by the drilling plug-in and the parts that pass through holes (drilling v2).

A hole lies in one workpiece layer. It has an entry point on the layer's face (face frame, z = 0), a unit axis into
the material (face frame) and up to S stages, one per drill diameter that cut it (a pilot, then the final drill).
Every drill has the same point half-angle kappa (T = tan kappa). A stage of radius R whose point tip is at axial
distance b from the entry has cut, at each radius r < R, the material from the face down to its point cone: axial
u in [0, b - r/T]. The hole is the union of its stages.

Volumes are integrated over the radius, not along the axis. At radius r the removed axial length is
clamp(B(r) - r/T, 0, L), with B(r) the deepest bottom among the stages that reach r and L the layer's span along the
axis. So the union needs no sorting and is exact in float64 (piecewise polynomial in r). Along an oblique axis the
layer is taken as the span [0, L], L = t / cos(tilt). That is exact once a stage is through (Cavalieri: a cylinder
between parallel planes holds pi R^2 L) and approximate while a point enters an inclined face.

Example::

    >>> import numpy as np
    >>> v = union_volume_np(np.array([0.003]), np.array([0.01]), 0.003, np.tan(np.radians(59.0)))
    >>> round(v / (np.pi * 0.003**2 * 0.003), 12)                    # a through hole: pi R^2 t
    1.0
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

PI64 = wp.constant(wp.float64(math.pi))


@wp.func
def ring_integral(r0: wp.float64, r1: wp.float64, b: wp.float64, span: wp.float64, t: wp.float64) -> wp.float64:
    """Integral of 2 pi r clamp(b - r/t, 0, span) dr over [r0, r1]: the removed volume of one radial band."""
    v = wp.float64(0.0)
    if r1 <= r0:
        return v
    ra = t * (b - span)                       # inside ra the band is cut through the whole span
    rb = t * b                                # beyond rb nothing is cut
    hi = wp.min(r1, ra)
    if hi > r0:
        v = v + PI64 * span * (hi * hi - r0 * r0)
    lo = wp.max(r0, ra)
    hi = wp.min(r1, rb)
    if hi > lo:
        v = v + PI64 * b * (hi * hi - lo * lo) - wp.float64(2.0) * PI64 * (hi * hi * hi - lo * lo * lo) / (
            wp.float64(3.0) * t)
    return v


@wp.func
def union_volume(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int,
                 span: wp.float64, t: wp.float64) -> wp.float64:
    """Volume of the union of a hole's ``n`` stages (``stage_d[w, base + k]``, ``stage_b[w, base + k]``) in a layer
    of axial span ``span``; ``t`` = tan of the point half-angle."""
    v = wp.float64(0.0)
    for j in range(n):
        rj = wp.float64(0.5) * wp.float64(stage_d[w, base + j])
        dup = int(0)
        prev = wp.float64(0.0)
        deep = wp.float64(-1.0e30)
        for k in range(n):
            rk = wp.float64(0.5) * wp.float64(stage_d[w, base + k])
            if rk == rj and k < j:
                dup = 1
            if rk < rj and rk > prev:
                prev = rk
            if rk >= rj:
                deep = wp.max(deep, stage_b[w, base + k])
        if dup == 0:
            v = v + ring_integral(prev, rj, deep, span, t)
    return v


@wp.func
def stage_start(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int,
                radius: wp.float64, t: wp.float64) -> wp.float64:
    """Where a new stage of ``radius`` starts: the tip distance at which its point first meets uncut material in the
    hole's other stages (0 in a fresh hole; r_p / t over a through pilot of radius r_p)."""
    best = wp.float64(1.0e30)
    for c in range(n + 1):
        rc = wp.float64(0.0)
        if c < n:
            rc = wp.float64(0.5) * wp.float64(stage_d[w, base + c])
        if rc < radius:
            deep = wp.float64(-1.0e30)
            for k in range(n):
                rk = wp.float64(0.5) * wp.float64(stage_d[w, base + k])
                if rk > rc:
                    deep = wp.max(deep, stage_b[w, base + k])
            best = wp.min(best, wp.max(deep, rc / t))
    return wp.max(best, wp.float64(0.0))


@wp.func
def void_radius(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int,
                cur: int, bottom: wp.float64, radius: wp.float32) -> wp.float32:
    """The radius inside which another stage is at least as deep as ``bottom``: the cutting edges there are in the
    void of a pilot (or of a wider hole) and cut nothing."""
    r = wp.float32(0.0)
    for k in range(n):
        if k != cur and stage_b[w, base + k] >= bottom:
            r = wp.max(r, 0.5 * stage_d[w, base + k])
    return wp.min(r, radius)


@wp.func
def overlap(a0: wp.float32, a1: wp.float32, b0: wp.float32, b1: wp.float32) -> wp.float32:
    return wp.max(wp.min(a1, b1) - wp.max(a0, b0), 0.0)


@wp.func
def engaged_edges(s: wp.float32, span: wp.float32, radius: wp.float32, rc: wp.float32, tan_half: wp.float32,
                  void: wp.float32) -> wp.vec3:
    """(lips, moment, chisel) fractions of the cutting edges in material, with the tip at axial distance ``s`` in a
    layer of span ``span``: an edge point at radius r lies r / tan_half behind the tip, so it is inside the layer for
    r in [(s - span) tan_half, s tan_half], and outside ``void`` (a pilot already cut it)."""
    r_hi = wp.clamp(s * tan_half, 0.0, radius)
    r_lo = wp.max(wp.clamp((s - span) * tan_half, 0.0, radius), void)
    lips = wp.float32(0.0)
    moment = wp.float32(0.0)
    if radius > rc:
        lips = overlap(r_lo, r_hi, rc, radius) / (radius - rc)
        r1 = wp.max(r_lo, rc)
        r2 = wp.min(r_hi, radius)
        if r2 > r1:
            moment = (r2 * r2 - r1 * r1) / (radius * radius - rc * rc)
    chisel = wp.float32(0.0)
    if rc > 0.0:
        chisel = overlap(r_lo, r_hi, 0.0, rc) / rc
    return wp.vec3(lips, moment, chisel)


@wp.func
def lateral(p: wp.vec3, origin: wp.vec3, axis: wp.vec3) -> wp.vec3:
    """Offset of ``p`` from the line through ``origin`` along unit ``axis``, across the axis."""
    d = p - origin
    return d - wp.dot(d, axis) * axis


# --- host twins (tests, laws, scoring) -------------------------------------------------------------------------

def _ring_np(r0: float, r1: float, b: float, span: float, t: float) -> float:
    if r1 <= r0:
        return 0.0
    v, ra, rb = 0.0, t * (b - span), t * b
    hi = min(r1, ra)
    if hi > r0:
        v += math.pi * span * (hi * hi - r0 * r0)
    lo, hi = max(r0, ra), min(r1, rb)
    if hi > lo:
        v += math.pi * b * (hi * hi - lo * lo) - 2.0 * math.pi * (hi**3 - lo**3) / (3.0 * t)
    return v


def union_volume_np(radii: np.ndarray, bottoms: np.ndarray, span: float, t: float) -> float:
    """Host twin of ``union_volume`` (radii, not diameters). Example: see the module docstring."""
    r, b = np.asarray(radii, np.float64), np.asarray(bottoms, np.float64)
    v = 0.0
    for j, rj in enumerate(r):
        if any(r[k] == rj for k in range(j)):
            continue
        smaller = r[r < rj]
        v += _ring_np(float(smaller.max()) if smaller.size else 0.0, float(rj), float(b[r >= rj].max()), span, t)
    return v


def stage_start_np(radii: np.ndarray, bottoms: np.ndarray, radius: float, t: float) -> float:
    """Host twin of ``stage_start``: where a new stage of ``radius`` first meets material."""
    r, b = np.asarray(radii, np.float64), np.asarray(bottoms, np.float64)
    best = math.inf
    for rc in [0.0, *r.tolist()]:
        if rc < radius:
            cover = b[r > rc]
            best = min(best, max(float(cover.max()) if cover.size else -math.inf, rc / t))
    return max(best, 0.0)


__all__ = ["engaged_edges", "lateral", "overlap", "ring_integral", "stage_start", "stage_start_np", "union_volume",
           "union_volume_np", "void_radius"]
