# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 11: the overhang and the touchdown on the device (the numpy twin: ``grid_surface_arc``'s
``overhang_frame`` and ``pinned_for``; GA pool.py:176-240, 1134-1256).

A full slice whose toes cannot spread (the contact freezes) holds more past its cap: its face turns past the 88 deg
limit and overhangs, under volume control (GA: stable past the half circle, where a pinned drop's pressure falls as
it grows). A bulge past a toe is star-shaped about a point near the chord's middle, not about the root, so this
family is shot in the polar frame about C = M - d n (d = 0.25 mm below the chord's midpoint, GA's centre): the fan
from C holds the face, ``corner`` the joint's region between C and the rays, and the pressure stays the root's.
The family runs from the cap's start toward a face leaving toe 1 along the plate; it holds the slice ("OVER"), or
its face reaches the plate past toe 1 first ("TOUCH": GA's touchdown, ``touch`` where along the plate; the caller
moves toe 1 there without fusion, the metal between lying on unfused plate, GA's cold lap), or it leans on the
member past toe 2 first ("TOUCH2": its face there past 180 deg to it; GA touches down on either side).
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.thermal import grid_surface as S
from alloy.plugins.thermal.grid_surface_roots import FAIL, bracket, guess, narrow, settle

OVER, TOUCH, TOUCH2 = 5, 6, 7
DEPTH = wp.constant(2.5e-4)                 # m, C below the chord's midpoint (GA pool.py:1146-1152, at a convex face)
STEP = wp.constant(0.02)                    # rad, the walk along the family before refining


@wp.func
def cross2(a: wp.vec2, b: wp.vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def overhang_frame(fr: S.Frame, depth: float) -> S.Frame:
    """From slice frame ``fr`` (about the root): the frame about C = M - depth n, n the chord's normal off the root."""
    root = fr.root
    p1 = root + fr.t1 * fr.u
    p2 = root + fr.t2 * S.far_ray(fr)
    m = 0.5 * (p1 + p2)
    d = p2 - p1
    n = wp.normalize(wp.vec2(-d[1], d[0]))
    lean = wp.dot(n, m - root)
    if wp.abs(lean) < 1.0e-9:
        n = fr.up
    elif lean < 0.0:
        n = -n
    c = m - depth * n
    u = wp.normalize(p1 - c)
    v = wp.normalize(n - wp.dot(n, u) * u)
    d2 = p2 - c
    sweep = wp.atan2(wp.dot(d2, v), wp.dot(d2, u))
    if sweep < 0.0:
        sweep = sweep + 2.0 * wp.pi
    out = S.frame_of(c, u, d2, wp.length(p1 - c), wp.length(d2), fr.up, fr.g, fr.rho, fr.gamma, fr.peak, fr.at,
                     fr.sigma)
    out.u = u
    out.v = v
    out.sweep = sweep
    out.ref = root - c
    out.corner = 0.5 * (cross2(p2 - c, root - c) + cross2(root - c, p1 - c))
    out.guard = 1
    out.e1 = fr.u
    out.m1 = fr.v
    out.t1r = fr.t1
    return out


@wp.func
def member(of: S.Frame, psi0: float, out: wp.array2d(dtype=float), w: int, base: int) -> S.Sol:
    """The family's surface leaving toe 1 at psi0 (in the overhang frame), NONE if no pressure closes it."""
    r = S.p_at_start(of, psi0, out, w, base)
    if r[1] <= 0.0:
        none = S.Sol()
        none.status = S.NONE
        return none
    return S.solved(of, psi0, r[0], OVER, out, w, base)


@wp.func
def solid2(of: S.Frame, a2: float, s: S.Sol) -> float:
    """The face's angle at toe 2 to the member beyond it (past pi: it leans on that member)."""
    return s.th2 + S.chord_angle(of) - a2


@wp.func
def touch2(s: S.Sol) -> S.Sol:
    out = s
    out.status = TOUCH2
    return out


@wp.func
def pinned_for(fr: S.Frame, target: float, cap: S.Sol, out: wp.array2d(dtype=float), w: int, base: int) -> S.Sol:
    """The overhang holding ``target`` past ``cap`` (the root frame's top), or the touchdown edge (module doc)."""
    of = overhang_frame(fr, DEPTH)
    d = wp.cos(cap.psi0) * fr.u + wp.sin(cap.psi0) * fr.v           # the cap's start, in the world
    hi = wp.atan2(wp.dot(d, of.v), wp.dot(d, of.u))
    e2 = S.far_ray(fr)
    a2 = wp.atan2(wp.dot(e2, of.v), wp.dot(e2, of.u))
    if a2 < 0.0:
        a2 = a2 + 2.0 * wp.pi
    s_hi = member(of, hi, out, w, base)
    if s_hi.status != S.NONE and solid2(of, a2, s_hi) >= wp.pi:   # it already leans on the member past toe 2
        return touch2(s_hi)
    lo = hi
    s_lo = s_hi
    reached = int(0)
    broke = int(0)
    floor = wp.max(wp.atan2(wp.dot(fr.u, of.v), wp.dot(fr.u, of.u)), 0.0) + 1.0e-4   # a face along the plate
    while lo - STEP > floor:
        prev = s_lo
        lo = lo - STEP
        s_lo = member(of, lo, out, w, base)
        if s_lo.status == S.NONE or s_lo.touch > 0.0 or s_lo.area >= target or solid2(of, a2, s_lo) >= wp.pi:
            hi = lo + STEP
            s_hi = prev
            broke = 1
            if s_lo.status != S.NONE and s_lo.touch == 0.0:
                reached = 1
            break
    if broke == 1 and reached == 1 and solid2(of, a2, s_lo) >= wp.pi and s_lo.area < target:
        a = lo                                                        # past toe 2 first: its touchdown edge
        b = hi
        for _i in range(40):
            mid = 0.5 * (a + b)
            sm = member(of, mid, out, w, base)
            if sm.status == S.NONE or solid2(of, a2, sm) >= wp.pi:
                a = mid
            else:
                b = mid
        return touch2(member(of, b, out, w, base))
    if broke == 0:                                                    # the family's end: the face lies on the plate
        s_lo = member(of, floor, out, w, base)
        if s_lo.status != S.NONE and s_lo.area < target:
            s_lo.status = TOUCH
            s_lo.touch = fr.t1
            return s_lo
        lo = floor
        if s_lo.status != S.NONE and s_lo.touch == 0.0:
            reached = 1
    if reached == 1:
        br = bracket(lo, hi, target - s_lo.area, target - s_hi.area)
        for _i in range(60):
            if br.hi - br.lo <= 1.0e-6:
                break
            x = guess(br)
            s = member(of, x, out, w, base)
            fx = -FAIL
            if s.status != S.NONE:
                fx = target - s.area
            br = narrow(br, x, fx)
        found = member(of, settle(br), out, w, base)
        if found.status != S.NONE and solid2(of, a2, found) >= wp.pi:
            return touch2(found)
        return found
    a = lo
    b = hi
    for _i in range(40):                                              # the edge: the last start not touching
        mid = 0.5 * (a + b)
        s = member(of, mid, out, w, base)
        if s.status == S.NONE or s.touch > 0.0:
            a = mid
        else:
            b = mid
    t = member(of, a, out, w, base)
    res = member(of, b, out, w, base)
    res.status = TOUCH
    res.touch = t.touch
    return res


@wp.func
def in_tri(x: wp.vec2, a: wp.vec2, b: wp.vec2, c: wp.vec2) -> bool:
    d1 = cross2(b - a, x - a)
    d2 = cross2(c - b, x - b)
    d3 = cross2(a - c, x - c)
    return (d1 >= 0.0 and d2 >= 0.0 and d3 >= 0.0) or (d1 <= 0.0 and d2 <= 0.0 and d3 <= 0.0)


@wp.func
def in_region(x: wp.vec2, fr: S.Frame, out: wp.array2d(dtype=float), w: int, base: int) -> bool:
    """x under the traced surface: in the fan from the frame's centre, or (an overhang frame) in its corner."""
    if S.in_surface(x, fr, out, w, base):
        return True
    if fr.guard == 0:
        return False
    root = fr.root + fr.ref
    p1 = fr.root + fr.t1 * fr.u
    p2 = fr.root + fr.t2 * S.far_ray(fr)
    return in_tri(x, fr.root, p2, root) or in_tri(x, fr.root, root, p1)


__all__ = ["OVER", "TOUCH", "TOUCH2", "in_region", "overhang_frame", "pinned_for"]
