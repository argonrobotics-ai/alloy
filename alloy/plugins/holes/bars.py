# SPDX-License-Identifier: Apache-2.0
"""Bars a drill cannot cut (drilling v2, phase C, D-7): rebar in a concrete layer stops a carbide masonry bit.

A layer may hold straight bars (``Bar``: a line in the layer's face frame, z down into the material, and a radius).
The drill's point solid is the drill-point cone of half-angle kappa that reaches the full radius R at u = R / tan
kappa behind the tip, then the cylinder. Its clearance to a bar whose axis is perpendicular to the drill axis (bars
lie along a face) is exact at the bar's closest approach to the drill axis; a bar tilted out of the face plane (up to
10 deg; steeper ones are refused) is searched along its axis, the distance to the convex solid being convex there:

    clearance = min over u >= 0 of sqrt(max(rho - r(u), 0)^2 + (h0 + u)^2) - r_bar,   r(u) = min(u tan kappa, R)

with h0 the bar axis's distance ahead of the tip along the drill axis and rho its distance across. The minimum lies at
u = 0, the cone's stationary point, where r(u) = rho, at u = R / tan kappa, or where h0 + u = 0 (``bar_clearance``).
A stage's bottom never cuts past where the clearance reaches 0 (``bar_limit``, bisection): there the bit is jammed
and the front holds it along the hole axis. Pressed on, it turns into the bar at the Shaw-Oxford feed of the bar's
``work`` row under the jam thrust, its blows doing nothing on steel: an ESTIMATE (a carbide masonry bit cuts steel
no better than a sharp twist drill; no measured rate was found). With steel_200bhn the chisel edge alone takes
C_J d^2 (1.04 kN for a 12.7 mm bit) before any cut, so a light push leaves the bar intact and a hard one cuts it.
The deepest overlap is booked as the bar's cut depth (``bar_cut``). The rate is an upper bound, so ``bar_cut`` is
too: a check that fails at 0.5 mm of cut is conservative, and a pass is real. The whole chisel threshold applies
even when only a lip is on the bar (part of the estimate).

The hole's bottom is the bar's memory of its cut: the solid at bottom b is rho <= R, z + rho / t <= b, so a hole
whose bottom is past the largest z + rho / t over the bar's part within the drill's radius (``bar_through``) has cut
all of the bar it can reach, and that bar no longer stops it. A bit pressed on a bar therefore cuts through it at
the estimated rate and drills on, and a hole already past a bar (seeded, or cut through it) is re-entered freely.
The through bottom is exact for a bar across the axis (z0 + sqrt(r_bar^2 - max(rho0 - R, 0)^2) + R / t at its
closest approach) and an upper bound for a tilted one (the bit cuts a little longer). A bit that meets a bar before
its hole has a stage (an exposed bar) is held and reported jammed, but cuts nothing into the bar.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

JAM_PROBE = wp.constant(1.0e-6)     # m: a stage whose bottom cannot go this much deeper is jammed on a bar
MAX_TILT = math.sin(math.radians(10.0)) + 1e-9   # a bar's direction may leave the face plane by this much (sine)
SEARCH = wp.constant(0.1)           # m: a tilted bar's nearest point lies this close to its closest approach
SQUARE = wp.constant(1.0e-3)        # |cos| between bar and drill axis below which the closest approach is used
GOLDEN = wp.constant(0.618034)
GOLDEN_STEPS = wp.constant(48)
BISECT = wp.constant(40)


@dataclass
class Bar:
    """A straight bar in a layer, in the layer's face frame (z = 0 on the face, -z into the material).

    Example: a #4 rebar (12.7 mm) along y under 25 mm of cover at x = 0.1 m:
    ``Bar(point=(0.1, 0.0, -0.025 - 0.00635), direction=(0, 1, 0), radius=0.00635)``."""

    point: tuple[float, float, float]
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    radius: float = 0.00635
    name: str = "rebar"
    work: str = "steel_200bhn"                 # its Kd row: a bit pressed on it cuts at that Shaw feed (estimate)

    def __post_init__(self) -> None:
        from alloy.plugins.holes.work import PercussiveMaterial, work_material  # noqa: PLC0415

        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError(f"bar {self.name!r}: radius must be finite and > 0")
        if not np.isfinite(self.point).all() or not np.linalg.norm(self.direction) > 0.0:
            raise ValueError(f"bar {self.name!r}: point must be finite and direction non-zero")
        self.point = tuple(float(v) for v in self.point)
        self.direction = tuple(float(v) for v in np.asarray(self.direction) / np.linalg.norm(self.direction))
        if abs(self.direction[2]) > MAX_TILT:
            raise ValueError(f"bar {self.name!r}: more than 10 deg out of the face plane (the clearance is exact "
                             f"for bars along the face, within 1 um to 10 deg)")
        if isinstance(work_material(self.work), PercussiveMaterial):
            raise ValueError(f"bar {self.name!r}: work {self.work!r} is percussive; give a twist-drilling Kd row")


@wp.struct
class Bars:
    p: wp.array[wp.vec3]          # a point on each bar's axis, its layer's face frame
    u: wp.array[wp.vec3]          # its unit direction
    r: wp.array[wp.float32]       # its radius
    lips: wp.array[wp.float32]    # its Shaw-Oxford C_T (the drill's factors, its work row's Kd)
    chisel: wp.array[wp.float32]  # its C_J
    first: wp.array[wp.int32]     # per layer: its first bar
    n: wp.array[wp.int32]         # per layer: its bar count


@wp.func
def _gap2(rho: wp.float32, h0: wp.float32, u: wp.float32, radius: wp.float32, t: wp.float32) -> wp.float32:
    s = wp.max(u, 0.0)
    lateral = wp.max(rho - wp.min(s * t, radius), 0.0)
    ax = h0 + s
    return lateral * lateral + ax * ax


@wp.func
def point_gap(q: wp.vec3, a: wp.vec3, radius: wp.float32, t: wp.float32) -> wp.float32:
    """Distance [m] from point ``q`` (relative to the tip) to the drill-point solid along unit axis ``a``: the
    point-to-profile distance in the (radial, axial) half-plane, over the five stations that can be nearest."""
    h0 = wp.dot(q, a)
    rho = wp.length(q - h0 * a)
    full = radius / t
    best = _gap2(rho, h0, 0.0, radius, t)
    best = wp.min(best, _gap2(rho, h0, wp.clamp((t * rho - h0) / (1.0 + t * t), 0.0, full), radius, t))
    best = wp.min(best, _gap2(rho, h0, wp.min(rho / t, full), radius, t))
    best = wp.min(best, _gap2(rho, h0, full, radius, t))
    best = wp.min(best, _gap2(rho, h0, wp.max(-h0, 0.0), radius, t))
    return wp.sqrt(best)


@wp.func
def bar_clearance(tip: wp.vec3, a: wp.vec3, radius: wp.float32, t: wp.float32, p0: wp.vec3, ub: wp.vec3,
                  rb: wp.float32) -> wp.float32:
    """Clearance [m] between the drill-point solid (tip ``tip``, unit axis ``a`` into the material, radius ``radius``,
    tan of the half-angle ``t``) and a bar (axis point ``p0``, unit direction ``ub``, radius ``rb``); < 0: overlap.
    The solid is convex, so the distance from the bar's axis points to it is convex along the bar: at the closest
    approach to the drill axis for a bar across it, else a golden-section search around there."""
    c = wp.dot(ub, a)
    d0 = p0 - tip
    lam = wp.float32(0.0)
    if 1.0 - c * c > 1.0e-9:
        lam = (c * wp.dot(d0, a) - wp.dot(d0, ub)) / (1.0 - c * c)
    if wp.abs(c) < SQUARE:
        return point_gap(d0 + lam * ub, a, radius, t) - rb             # square to the axis: nanometres off
    lo = lam - SEARCH
    hi = lam + SEARCH
    for _ in range(GOLDEN_STEPS):
        m1 = hi - GOLDEN * (hi - lo)
        m2 = lo + GOLDEN * (hi - lo)
        if point_gap(d0 + m1 * ub, a, radius, t) <= point_gap(d0 + m2 * ub, a, radius, t):
            hi = m2
        else:
            lo = m1
    return point_gap(d0 + (0.5 * (lo + hi)) * ub, a, radius, t) - rb


@wp.func
def bar_through(entry: wp.vec3, a: wp.vec3, radius: wp.float32, t: wp.float32, p0: wp.vec3, ub: wp.vec3,
                rb: wp.float32) -> wp.float32:
    """The bottom (axial along ``a`` from ``entry``) past which the point solid has cut all of a bar within its radius:
    the largest z + rho / t over the bar's part inside the drill's cylinder. Across the axis: z0 + sqrt(rb^2 -
    max(rho0 - R, 0)^2) + R / t, z0 and rho0 at the closest approach; tilted: z0 + rb + R / t + |c| (R + rb) /
    sqrt(1 - c^2), an upper bound. -1e30 for a bar the bit cannot reach (rho0 >= R + rb)."""
    c = wp.dot(ub, a)
    d0 = p0 - entry
    lam = wp.float32(0.0)
    if 1.0 - c * c > 1.0e-9:
        lam = (c * wp.dot(d0, a) - wp.dot(d0, ub)) / (1.0 - c * c)
    q = d0 + lam * ub
    z0 = wp.dot(q, a)
    rho0 = wp.length(q - z0 * a)
    if rho0 >= radius + rb:
        return wp.float32(-1.0e30)
    if wp.abs(c) < SQUARE:
        off = wp.max(rho0 - radius, 0.0)
        return z0 + wp.sqrt(wp.max(rb * rb - off * off, 0.0)) + radius / t
    return z0 + rb + radius / t + wp.abs(c) * (radius + rb) / wp.sqrt(1.0 - c * c)


@wp.func
def layer_clearance(bars: Bars, i: int, entry: wp.vec3, a: wp.vec3, b: wp.float32, radius: wp.float32,
                    t: wp.float32, bottom: wp.float32) -> wp.float32:
    """The smallest clearance to layer ``i``'s bars with the tip at axial ``b`` along ``a`` from ``entry`` (face
    frame), leaving out the bars a hole whose bottom is at ``bottom`` has cut through; 1e30 without bars."""
    c = wp.float32(1.0e30)
    tip = entry + b * a
    for k in range(bars.n[i]):
        j = bars.first[i] + k
        if bottom < bar_through(entry, a, radius, t, bars.p[j], bars.u[j], bars.r[j]):
            c = wp.min(c, bar_clearance(tip, a, radius, t, bars.p[j], bars.u[j], bars.r[j]))
    return c


@wp.func
def blocking_bar(bars: Bars, i: int, entry: wp.vec3, a: wp.vec3, b: wp.float32, radius: wp.float32,
                 t: wp.float32, bottom: wp.float32) -> wp.int32:
    """The bar of layer ``i`` nearest the point solid with the tip at ``b``, of those a hole whose bottom is at
    ``bottom`` has not cut through (the layer's first without bars)."""
    best = bars.first[i]
    c = wp.float32(1.0e30)
    tip = entry + b * a
    for k in range(bars.n[i]):
        j = bars.first[i] + k
        if bottom < bar_through(entry, a, radius, t, bars.p[j], bars.u[j], bars.r[j]):
            cj = bar_clearance(tip, a, radius, t, bars.p[j], bars.u[j], bars.r[j])
            if cj < c:
                c = cj
                best = j
    return best


@wp.func
def bar_limit(bars: Bars, i: int, entry: wp.vec3, a: wp.vec3, old: wp.float64, new: wp.float64,
              radius: wp.float32, t: wp.float32) -> wp.float64:
    """The deepest bottom in [old, new] the point reaches without entering a bar the hole (bottom ``old``) has not
    cut through (``new`` if none is in the way)."""
    bottom = wp.float32(old)
    if bars.n[i] == 0 or layer_clearance(bars, i, entry, a, wp.float32(new), radius, t, bottom) >= 0.0:
        return new
    lo = old
    hi = new
    for _ in range(BISECT):
        mid = wp.float64(0.5) * (lo + hi)
        if layer_clearance(bars, i, entry, a, wp.float32(mid), radius, t, bottom) >= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def bars_struct(per_layer: list[tuple[Bar, ...]], device: str, shaw_of: Callable[[str], Any]) -> Bars:
    """The device bars of a stack (one dummy slot when there are none: arrays are never empty); ``shaw_of(work)``
    gives a work row's Shaw-Oxford constants at the drill's factors."""
    flat = [b for row in per_layer for b in row]
    counts = [len(row) for row in per_layer]
    shaws = [shaw_of(b.work) for b in flat]
    s = Bars()
    s.lips = wp.array([float(x.lips) for x in shaws] or [0.0], dtype=wp.float32, device=device)
    s.chisel = wp.array([float(x.chisel) for x in shaws] or [0.0], dtype=wp.float32, device=device)
    s.p = wp.array([b.point for b in flat] or [(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    s.u = wp.array([b.direction for b in flat] or [(0.0, 1.0, 0.0)], dtype=wp.vec3, device=device)
    s.r = wp.array([b.radius for b in flat] or [0.0], dtype=wp.float32, device=device)
    s.first = wp.array(np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32), dtype=wp.int32, device=device)
    s.n = wp.array(np.asarray(counts, np.int32), dtype=wp.int32, device=device)
    return s


def through_np(entry: np.ndarray, a: np.ndarray, radius: float, t: float, bar: Bar) -> float:
    """Host twin of ``bar_through`` (float64): the bottom past which a hole has cut through ``bar``."""
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    ub = np.asarray(bar.direction, np.float64)
    d0 = np.asarray(bar.point, np.float64) - np.asarray(entry, np.float64)
    c = float(ub @ a)
    lam = (c * float(d0 @ a) - float(d0 @ ub)) / (1.0 - c * c) if 1.0 - c * c > 1e-9 else 0.0
    q = d0 + lam * ub
    z0 = float(q @ a)
    rho0 = float(np.linalg.norm(q - z0 * a))
    if rho0 >= radius + bar.radius:
        return -math.inf
    if abs(c) < 1.0e-3:                                                     # SQUARE, as on the device
        return z0 + math.sqrt(max(bar.radius**2 - max(rho0 - radius, 0.0) ** 2, 0.0)) + radius / t
    return z0 + bar.radius + radius / t + abs(c) * (radius + bar.radius) / math.sqrt(1.0 - c * c)


def clearance_np(tip: np.ndarray, a: np.ndarray, radius: float, t: float, bar: Bar) -> float:
    """Host twin of ``bar_clearance`` (float64), for checks and laws."""
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    ub = np.asarray(bar.direction, np.float64)
    d0 = np.asarray(bar.point, np.float64) - np.asarray(tip, np.float64)
    c = float(ub @ a)
    lam = (c * float(d0 @ a) - float(d0 @ ub)) / (1.0 - c * c) if 1.0 - c * c > 1e-9 else 0.0
    full = radius / t

    def gap(q: np.ndarray) -> float:
        h0 = float(q @ a)
        rho = float(np.linalg.norm(q - h0 * a))

        def gap2(u: float) -> float:
            s = max(u, 0.0)
            return max(rho - min(s * t, radius), 0.0) ** 2 + (h0 + s) ** 2

        cands = (0.0, min(max((t * rho - h0) / (1 + t * t), 0.0), full), min(rho / t, full), full, max(-h0, 0.0))
        return math.sqrt(min(gap2(u) for u in cands))

    if abs(c) < 1.0e-3:                                                     # SQUARE, as on the device
        return gap(d0 + lam * ub) - bar.radius
    lo, hi = lam - 0.1, lam + 0.1
    for _ in range(80):
        m1, m2 = hi - 0.618034 * (hi - lo), lo + 0.618034 * (hi - lo)
        lo, hi = (lo, m2) if gap(d0 + m1 * ub) <= gap(d0 + m2 * ub) else (m1, hi)
    return gap(d0 + 0.5 * (lo + hi) * ub) - bar.radius


__all__ = ["JAM_PROBE", "Bar", "Bars", "bar_clearance", "bar_limit", "bar_through", "bars_struct", "blocking_bar",
           "clearance_np", "layer_clearance", "through_np"]
