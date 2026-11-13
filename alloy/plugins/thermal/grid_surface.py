# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 11: each slice's free surface with the liquid's weight and the arc's pressure, on the device
(``bead_model="surface"``). The model and its numerics are those of ``grid_surface_arc``, the numpy reference the
laws compare with (its docstring has the equations): Young-Laplace integrated along the polar angle about the
joint's root, so the curve stays between the rays and its toe angles and area are exact.

- ``shoot``: RK4 in the polar angle from toe 1 to the far ray (``STEPS`` steps; 32 agree with 256 to 1e-6); a
  failed shot is classed by the side its tangent leaves (+1 out along the radius, -1 toward the root).
- ``top``: the fullest surface the toes hold, a toe at 88 deg to the chord (GA ``_caps``): at a fixed start angle the
  far-ray miss falls with the pressure, so the pressure is a bracketed root; the other toe is checked, else the
  mirrored shot binds toe 2.
- ``surface_at``: the surface at a pressure, the first root walking from the top's start angle toward the root
  (warm-started on last exchange's angle); a draining slice takes the edge surface meeting the wall tangentially
  short of toe 2 (GA's surface resting on the solid), its toe receded.
- ``area_for``: the surface holding an area (Illinois on the pressure, warm-started); "full" past the top, "thin"
  below the least any surface holds (the caller keeps the cap model's segment there, which holds any area).
- ``in_surface``: a point under the surface, from the traced radii (``STEPS`` + 1 per slice, linear in the angle).
Float32 throughout, as the grid: the miss resolves 1e-8 m, the area 1e-6 relative, against 0.25 mm cells.
"""

from __future__ import annotations

import math

import warp as wp

from alloy.plugins.thermal.grid_surface_roots import FAIL, bracket, finite, guess, narrow, settle

STEPS = 32                                  # RK4 steps over the sweep; samples per slice: STEPS + 1
SCAN = 24                                   # start angles tried from the top's toward the root before refining
PSI_MIN = wp.constant(1.0e-3)               # a tangent this close to the radius ends a shot
TOE_LIMIT = math.radians(88.0)              # GA POOL_TOE_SLOPE_MAX, to the chord
HELD, WET, FULL, THIN, NONE = 1, 2, 3, 4, 0


@wp.struct
class Frame:
    root: wp.vec2          # the joint's root (y, z)
    u: wp.vec2             # e1: polar angle 0
    v: wp.vec2             # its normal toward e2
    sweep: float           # the polar angle of e2
    t1: float              # toes along e1, e2 [m]
    t2: float
    g: wp.vec2             # gravity in the slice plane [m/s^2], world
    rho: float             # liquid density [kg/m^3]
    gamma: float           # surface tension [N/m]
    peak: float            # the arc's peak pressure in this slice [Pa] (0: no arc)
    at: wp.vec2            # the arc root (y, z), world
    sigma: float           # the arc pressure's sigma [m]
    up: wp.vec2            # a flat joint's liquid side
    ref: wp.vec2           # the joint's root less the polar centre (0 about the root): P is the root's pressure
    corner: float          # the region's area outside the fan from the centre (0 about the root)
    guard: int             # 1: flag a face entering the plate past toe 1 (``grid_surface_overhang``)
    e1: wp.vec2            # the plate's direction and its normal toward the liquid (the guard's)
    m1: wp.vec2
    t1r: float             # toe 1 along e1 from the root


@wp.struct
class Shot:
    status: int            # 0 reached the far ray; +1 ran out along a radius; -1 fell toward the root
    r: float
    phi: float
    area: float
    touch: float           # where along e1 the face first entered the plate past toe 1 (0: it did not)


@wp.struct
class Sol:
    status: int            # HELD, WET, FULL, THIN, NONE
    p: float
    psi0: float
    area: float
    toe2: float            # where the surface meets e2 (short of t2 when WET)
    th1: float             # toe angles to the chord [rad]
    th2: float
    touch: float           # as Shot.touch


@wp.func
def frame_of(root: wp.vec2, e1: wp.vec2, e2: wp.vec2, t1: float, t2: float, up: wp.vec2, g: wp.vec2, rho: float,
             gamma: float, peak: float, at: wp.vec2, sigma: float) -> Frame:
    """The polar frame turning from e1 to e2 through the liquid (``up``'s side for a flat joint)."""
    u = wp.normalize(e1)
    f = wp.normalize(e2)
    perp = wp.vec2(-u[1], u[0])
    cr = u[0] * f[1] - u[1] * f[0]
    side = float(1.0)
    sweep = float(wp.pi)
    if wp.abs(cr) < 1.0e-6 and wp.dot(u, f) < 0.0:
        if wp.dot(perp, up) < 0.0:
            side = -1.0
    else:
        if cr < 0.0:
            side = -1.0
        sweep = wp.atan2(wp.abs(cr), wp.dot(u, f))
    fr = Frame()
    fr.root = root
    fr.u = u
    fr.v = side * perp
    fr.sweep = sweep
    fr.t1 = t1
    fr.t2 = t2
    fr.g = g
    fr.rho = rho
    fr.gamma = gamma
    fr.peak = peak
    fr.at = at
    fr.sigma = sigma
    fr.up = up
    fr.ref = wp.vec2(0.0, 0.0)
    fr.corner = 0.0
    fr.guard = 0
    fr.e1 = u
    fr.m1 = side * perp
    fr.t1r = t1
    return fr


@wp.func
def far_ray(fr: Frame) -> wp.vec2:
    return wp.cos(fr.sweep) * fr.u + wp.sin(fr.sweep) * fr.v


@wp.func
def mirror(fr: Frame) -> Frame:
    """The same slice shot from toe 2."""
    return frame_of(fr.root, far_ray(fr), fr.u, fr.t2, fr.t1, fr.up, fr.g, fr.rho, fr.gamma, fr.peak, fr.at, fr.sigma)


@wp.func
def chord_angle(fr: Frame) -> float:
    """The chord's direction from toe 1 to toe 2 in the frame, in [0, pi] (sin of a float32 pi is -8.7e-8)."""
    return wp.atan2(fr.t2 * wp.max(wp.sin(fr.sweep), 0.0), fr.t2 * wp.cos(fr.sweep) - fr.t1)


@wp.func
def chord(fr: Frame) -> float:
    return wp.length(wp.vec2(fr.t2 * wp.cos(fr.sweep) - fr.t1, fr.t2 * wp.sin(fr.sweep)))


@wp.func
def scale(fr: Frame) -> float:
    """A pressure scale [Pa]: tension over the chord, the weight over it, the arc."""
    c = wp.max(chord(fr), 1.0e-9)
    return fr.gamma / c + fr.rho * wp.length(fr.g) * c + fr.peak


@wp.func
def deriv(fr: Frame, p: float, a: float, y: wp.vec3) -> wp.vec4:
    """(dr, dphi, dA)/da and the shot's status at polar angle a, state (r, phi, A)."""
    r = y[0]
    psi = y[1] - a
    if r <= 0.0 or psi > wp.pi - PSI_MIN:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if psi < PSI_MIN:
        return wp.vec4(0.0, 0.0, 0.0, 1.0)
    x = r * (wp.cos(a) * fr.u + wp.sin(a) * fr.v)
    pa = float(0.0)
    if fr.peak > 0.0:
        d = fr.root + x - fr.at
        pa = fr.peak * wp.exp(-0.5 * wp.dot(d, d) / (fr.sigma * fr.sigma))
    k = (p + fr.rho * wp.dot(fr.g, x - fr.ref) - pa) / fr.gamma
    s = wp.sin(psi)
    return wp.vec4(r * wp.cos(psi) / s, k * r / s, 0.5 * r * r, 0.0)


@wp.func
def part(k: wp.vec4) -> wp.vec3:
    return wp.vec3(k[0], k[1], k[2])


@wp.func
def shoot(fr: Frame, psi0: float, p: float, keep: int, out: wp.array2d(dtype=float), w: int, base: int) -> Shot:
    """From toe 1 leaving at psi0 to the far ray; ``keep``: the radius at each step into out[w, base + i]."""
    h = fr.sweep / float(STEPS)
    y = wp.vec3(fr.t1, psi0, 0.0)
    r_max = 10.0 * wp.max(fr.t1, fr.t2)
    res = Shot()
    res.status = 0
    res.touch = 0.0
    if keep == 1:
        out[w, base] = y[0]
    for i in range(STEPS):
        a = float(i) * h
        k1 = deriv(fr, p, a, y)
        st = k1[3]
        k2 = k1
        k3 = k1
        k4 = k1
        if st == 0.0:
            k2 = deriv(fr, p, a + 0.5 * h, y + (0.5 * h) * part(k1))
            st = k2[3]
        if st == 0.0:
            k3 = deriv(fr, p, a + 0.5 * h, y + (0.5 * h) * part(k2))
            st = k3[3]
        if st == 0.0:
            k4 = deriv(fr, p, a + h, y + h * part(k3))
            st = k4[3]
        if st != 0.0:
            res.status = int(st)
            return res
        y = y + (h / 6.0) * (part(k1) + 2.0 * part(k2) + 2.0 * part(k3) + part(k4))
        psi = y[1] - float(i + 1) * h
        if y[0] > r_max or psi < PSI_MIN:
            res.status = 1
            return res
        if y[0] <= 0.0 or psi > wp.pi - PSI_MIN:
            res.status = -1
            return res
        if fr.guard == 1 and res.touch == 0.0:
            a1 = float(i + 1) * h
            d = y[0] * (wp.cos(a1) * fr.u + wp.sin(a1) * fr.v) - fr.ref
            if wp.dot(d, fr.e1) > fr.t1r and wp.dot(d, fr.m1) < 0.0:
                res.touch = wp.dot(d, fr.e1)
        if keep == 1:
            out[w, base + i + 1] = y[0]
    res.r = y[0]
    res.phi = y[1]
    res.area = y[2] + fr.corner
    return res


@wp.func
def miss(s: Shot, t2: float) -> float:
    if s.status != 0:
        return float(s.status) * FAIL
    return s.r - t2


@wp.func
def p_at_start(fr: Frame, psi0: float, out: wp.array2d(dtype=float), w: int, base: int) -> wp.vec2:
    """(the pressure whose curve leaving toe 1 at psi0 meets toe 2, 1), or (0, 0) if none: the miss falls with P."""
    c = wp.max(chord(fr), 1.0e-9)
    mid = 0.5 * (fr.t1 * fr.u + fr.t2 * far_ray(fr)) - fr.ref
    p0 = 2.0 * fr.gamma * wp.sin(chord_angle(fr) - psi0) / c - fr.rho * wp.dot(fr.g, mid)
    sc = scale(fr)
    d = 0.01 * sc
    found = int(0)
    b = bracket(0.0, 0.0, 0.0, 0.0)
    for _k in range(48):
        flo = miss(shoot(fr, psi0, p0 - d, 0, out, w, base), fr.t2)
        fhi = miss(shoot(fr, psi0, p0 + d, 0, out, w, base), fr.t2)
        if flo > 0.0 and fhi < 0.0:
            b = bracket(p0 - d, p0 + d, flo, fhi)
            found = 1
            break
        d = 2.0 * d
    if found == 0:
        return wp.vec2(0.0, 0.0)
    for _i in range(80):
        if b.hi - b.lo <= 1.0e-7 * sc:               # float32 resolves ~1e-8 of the scale; a wide puddle needs it
            break
        x = guess(b)
        fx = miss(shoot(fr, psi0, x, 0, out, w, base), fr.t2)
        if fx == 0.0:
            return wp.vec2(x, 1.0)
        b = narrow(b, x, fx)
    return wp.vec2(settle(b), 1.0)


@wp.func
def solved(fr: Frame, psi0: float, p: float, status: int, out: wp.array2d(dtype=float), w: int, base: int) -> Sol:
    """The surface from toe 1 at psi0 and pressure p, its radii traced into out (NONE if the shot fails)."""
    s = shoot(fr, psi0, p, 1, out, w, base)
    res = Sol()
    res.status = NONE
    if s.status != 0:
        return res
    beta = chord_angle(fr)
    res.status = status
    res.p = p
    res.psi0 = psi0
    res.area = s.area
    res.toe2 = s.r
    res.th1 = beta - psi0
    res.th2 = s.phi - beta
    res.touch = s.touch
    return res


@wp.func
def top(fr: Frame, limit: float, out: wp.array2d(dtype=float), w: int, base: int) -> Sol:
    """The fullest surface: toe 1 at ``limit`` to the chord with toe 2 below it, else toe 2 at it (FULL; NONE if
    neither toe can reach the limit)."""
    beta = chord_angle(fr)
    r1 = p_at_start(fr, beta - limit, out, w, base)
    if r1[1] > 0.0:
        s = solved(fr, beta - limit, r1[0], FULL, out, w, base)
        if s.status == FULL and s.th2 <= limit + 1.0e-5:
            return s
    mf = mirror(fr)
    bm = chord_angle(mf)
    r2 = p_at_start(mf, bm - limit, out, w, base)
    none = Sol()
    none.status = NONE
    if r2[1] <= 0.0:
        return none
    sm = shoot(mf, bm - limit, r2[0], 0, out, w, base)
    if sm.status != 0:
        return none
    psi0 = beta - (sm.phi - bm)                       # toe 1's angle to the chord, seen from toe 2
    r3 = p_at_start(fr, psi0, out, w, base)
    if r3[1] <= 0.0:
        return none
    return solved(fr, psi0, r3[0], FULL, out, w, base)


@wp.func
def surface_at(fr: Frame, p: float, cap: Sol, warm: float, out: wp.array2d(dtype=float), w: int, base: int) -> Sol:
    """The surface at pressure p: the cap at or above its pressure; else the first root of the far-ray miss from the
    cap's start angle toward the root (first tried around ``warm``, last exchange's angle; 0: none), or the edge
    surface of a draining slice (WET); NONE if no surface between the toes has this pressure."""
    if p >= cap.p:
        return cap
    lo = cap.psi0
    flo = miss(shoot(fr, lo, p, 0, out, w, base), fr.t2)
    none = Sol()
    none.status = NONE
    if not flo > 0.0:
        return none
    hi = float(wp.pi - 1.0e-5)
    fhi = float(0.0)
    found = int(0)
    if warm > lo and warm < hi:                       # the continuation: a narrow bracket around last exchange's
        a = wp.max(lo, warm - 0.03)
        b = wp.min(hi, warm + 0.03)
        fa = miss(shoot(fr, a, p, 0, out, w, base), fr.t2)
        fb = miss(shoot(fr, b, p, 0, out, w, base), fr.t2)
        if fa > 0.0 and fb < 0.0 and finite(fa) and finite(fb):
            lo = a
            flo = fa
            hi = b
            fhi = fb
            found = 1
    if found == 0:
        step = (hi - lo) / float(SCAN)
        x0 = lo
        for i in range(SCAN):
            x = lo + float(i + 1) * step
            fx = miss(shoot(fr, x, p, 0, out, w, base), fr.t2)
            if fx < 0.0:
                lo = x0
                hi = x
                fhi = fx
                found = 1
                break
            x0 = x
            flo = fx
    if found == 0 or (not finite(flo) and not finite(fhi)):
        return none
    if not finite(flo):                               # turns out along the wall, then meets it short of toe 2
        a = lo
        b = hi
        for _i in range(40):
            m = 0.5 * (a + b)
            if finite(miss(shoot(fr, m, p, 0, out, w, base), fr.t2)):
                b = m
            else:
                a = m
        fb = miss(shoot(fr, b, p, 0, out, w, base), fr.t2)
        if fb < 0.0:
            return solved(fr, b, p, WET, out, w, base)
        lo = b
        flo = fb
    br = bracket(lo, hi, flo, fhi)
    for _i in range(60):
        if br.hi - br.lo <= 1.0e-6:
            break
        x = guess(br)
        fx = miss(shoot(fr, x, p, 0, out, w, base), fr.t2)
        if fx == 0.0:
            return solved(fr, x, p, HELD, out, w, base)
        br = narrow(br, x, fx)
    return solved(fr, settle(br), p, HELD, out, w, base)


@wp.func
def area_for(fr: Frame, target: float, cap: Sol, warm_p: float, warm_psi: float, out: wp.array2d(dtype=float),
             w: int, base: int) -> Sol:
    """The surface holding ``target`` [m^2]: the cap (FULL) if it holds less; else Illinois on the pressure (the
    held area grows with it, draining included), started below min(warm_p, the cap's); THIN if no surface holds as
    little."""
    if target >= cap.area:
        return cap
    sc = scale(fr)
    hi = cap.p
    fhi = target - cap.area
    lo = wp.min(warm_p, cap.p) - 0.02 * sc
    flo = float(0.0)
    psi = warm_psi
    found = int(0)
    d = 0.02 * sc
    for _k in range(40):
        s = surface_at(fr, lo, cap, psi, out, w, base)
        flo = FAIL
        if s.status != NONE:
            flo = target - s.area
            psi = s.psi0
        if flo > 0.0:
            found = 1
            break
        hi = lo
        fhi = flo
        d = 2.0 * d
        lo = lo - d
    thin = Sol()
    thin.status = THIN
    if found == 0:
        return thin
    br = bracket(lo, hi, flo, fhi)
    best = Sol()
    best.status = NONE
    for _i in range(60):
        if br.hi - br.lo <= 1.0e-6 * sc:
            break
        x = guess(br)
        s = surface_at(fr, x, cap, psi, out, w, base)
        fx = FAIL
        if s.status != NONE:
            fx = target - s.area
            psi = s.psi0
            best = s
        if fx == 0.0:
            break
        br = narrow(br, x, fx)
    best = surface_at(fr, settle(br), cap, psi, out, w, base)
    if best.status == NONE or wp.abs(best.area - target) > 1.0e-4 * target:
        return thin
    return best


@wp.func
def in_surface(x: wp.vec2, fr: Frame, out: wp.array2d(dtype=float), w: int, base: int) -> bool:
    """x (y, z) lies between the rays and under the traced surface."""
    d = x - fr.root
    lu = wp.dot(d, fr.u)
    lv = wp.dot(d, fr.v)
    a = wp.atan2(lv, lu)
    if a < 0.0 or a > fr.sweep:
        return False
    k = a / fr.sweep * float(STEPS)
    i = wp.min(int(k), STEPS - 1)
    f = k - float(i)
    r = (1.0 - f) * out[w, base + i] + f * out[w, base + i + 1]
    return wp.length(d) <= r


__all__ = ["FULL", "Frame", "HELD", "NONE", "STEPS", "Sol", "THIN", "TOE_LIMIT", "WET", "area_for", "frame_of",
           "in_surface", "shoot", "surface_at", "top"]
