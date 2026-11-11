# SPDX-License-Identifier: Apache-2.0
"""The peel front, one thread per world (``plugin.py`` docstring, "Peel"). No atomics: each world's thread owns its
vertices and its outputs.

- The front sits at material distance ``a`` from the grip: bonded vertices with d_i < a are released.
- The arm is a straight line from each bonded vertex's nearest grip vertex g(i) to its front point
  f_i = x_i - t_b·(d_i - a), with t_b the bonded direction in the surface. Its stretch is s_i(a) = |f_i - x_g(i)| - a.
- It carries T = min(k·s + c_d·ds/dt, T_c) for s > 0. Here k = E·t·b/a, c_d is critical damping for the grip mass,
  and T_c = b·E·t·(sqrt(c² + 2G(c)/(E·t)) - c) is Kendall with the elastic term, with c = 1 - cos(theta).
- The front advances to the largest a that brings every s_i to T_c/k: a closed form, because s_i falls
  monotonically with a.
"""

from __future__ import annotations

import warp as wp

from .kernels import FAR, FREE, GRIP, SUBSTRATE, dir_to_world
from .wetout import W_STD


ARM_CAP = wp.constant(0.02)
"""k·dt_x²/m cap of the arm on the grip body (dt_x: the exchange, not a film substep). The pull reaches the body 1-3
exchanges late; a lagged explicit spring
with critical damping is stable (spectral radius <= 0.95) at 0.02 for lags 1 to 3, but not at 0.1 with lag 2
(1.09) nor at 0.25 with lag 1 (1.14). In steady peeling the front lags the inextensible arm by T_c/k: 2.4 mm at
9.5 N on a 0.2 kg grip at a 1 ms step."""


@wp.struct
class Film:
    start: wp.array[wp.int32]          # [W + 1] each world's particle range
    q: wp.array[wp.vec3]
    state: wp.array[wp.int32]
    shape: wp.array[wp.int32]          # world-local; + base[w] = the model's shape id
    base: wp.array[wp.int32]           # [W] each world's first shape
    area: wp.array[float]
    h: wp.array[float]                 # mean rest edge length at the vertex
    d: wp.array[float]                 # rest distance to the nearest grip vertex (FAR: none)
    gnear: wp.array[wp.int32]          # that grip vertex
    normal: wp.array[wp.vec3]          # bonded: surface normal, body frame
    bond_dir: wp.array[wp.vec3]        # bonded: rest direction away from the grip, body frame
    wet: wp.array[float]               # wetted fraction w (wetout.py)


@wp.struct
class Peel:
    g90: wp.array[float]               # per shape: effective G at 90 and 180 deg [J/m²], lap-shear cap [Pa]
    g180: wp.array[float]
    tau: wp.array[float]


@wp.struct
class Front:
    a: wp.array[float]                 # [W] front distance from the grip [m] (0: not set)
    xg_prev: wp.array[wp.vec3]
    stretch: wp.array[float]           # [W] the arm's stretch at the last step (its rate damps the pull)
    force: wp.array[wp.vec3]           # pull on the grip body [N], world
    point: wp.array[wp.vec3]           # its point (grip vertex mean)
    front_point: wp.array[wp.vec3]     # the reaction's point on the substrate
    grip_shape: wp.array[wp.int32]
    sub_shape: wp.array[wp.int32]
    peel_force: wp.array[wp.float64]   # readings [W]
    angle: wp.array[wp.float64]
    width: wp.array[wp.float64]
    bonded: wp.array[wp.float64]
    released: wp.array[wp.float64]
    dissipated: wp.array[wp.float64]
    work: wp.array[wp.float64]
    impulse: wp.array[wp.float64]      # ∫ T dt on the grip [N·s]: its rate is the time-averaged pull
    front: wp.array[wp.float64]


@wp.func
def g_of_c(g90: float, g180: float, c: float) -> float:
    if c <= 1.0:
        return g90
    return g90 + (g180 - g90) * (wp.min(c, 2.0) - 1.0)


@wp.func
def kendall(g: float, c: float, et: float) -> float:
    """F/b from G = (F/b)·c + (F/b)²/(2·E·t)."""
    return et * (wp.sqrt(c * c + 2.0 * g / et) - c)


@wp.func
def stretch(fm: Film, i: int, tb: wp.vec3, a: float) -> float:
    return wp.length(fm.q[i] - tb * (fm.d[i] - a) - fm.q[fm.gnear[i]]) - a


@wp.func
def advance_to(fm: Film, i: int, tb: wp.vec3, s_star: float) -> float:
    """The a that brings vertex i's line to stretch ``s_star``; FAR if no advance can (a pull along the bond)."""
    v = fm.q[i] - tb * fm.d[i] - fm.q[fm.gnear[i]]
    den = 2.0 * (wp.dot(v, tb) - s_star)
    if den >= 0.0:
        return FAR
    return (s_star * s_star - wp.dot(v, v)) / den


@wp.func
def peelable(fm: Film, i: int) -> bool:
    return fm.state[i] == SUBSTRATE and fm.d[i] < FAR and fm.gnear[i] >= 0


@wp.struct
class Census:
    xg: wp.vec3                        # grip vertex mean
    ng: int
    gs: int                            # grip shape
    a0: float                          # nearest peelable bond
    bonded: float                      # bonded area [m²]
    peel_area: float                   # peelable bonded area


@wp.struct
class Band:
    tb: wp.vec3                        # bonded direction (unit, in the surface)
    xf: wp.vec3                        # front point
    b: float                           # front width
    s: int                             # substrate shape at the front
    wet: float                         # area-mean wet-out of the bonds at the front (G and tau scale by wet/W_STD)


@wp.func
def census(fm: Film, w: int, p0: int, p1: int) -> Census:
    r = Census()
    r.xg = wp.vec3(0.0)
    r.ng = 0
    r.gs = -1
    r.a0 = FAR
    r.bonded = 0.0
    r.peel_area = 0.0
    for i in range(p0, p1):
        if fm.state[i] == GRIP:
            r.xg = r.xg + fm.q[i]
            r.ng = r.ng + 1
            r.gs = fm.shape[i] + fm.base[w]
        if fm.state[i] == SUBSTRATE:
            r.bonded = r.bonded + fm.area[i]
        if peelable(fm, i):
            r.a0 = wp.min(r.a0, fm.d[i])
            r.peel_area = r.peel_area + fm.area[i]
    if r.ng > 0:
        r.xg = r.xg / float(r.ng)
    return r


@wp.func
def in_band(fm: Film, i: int, a: float) -> bool:
    return peelable(fm, i) and fm.d[i] < a + 1.5 * fm.h[i]


@wp.func
def band(fm: Film, w: int, p0: int, p1: int, a: float, body_q: wp.array[wp.transform],
         shape_body: wp.array[wp.int32]) -> Band:
    """Bonded vertices within 1.5 rows of the front: direction, normal, front point, width, substrate."""
    r = Band()
    tb = wp.vec3(0.0)
    n = wp.vec3(0.0)
    xf = wp.vec3(0.0)
    wsum = float(0.0)
    hsum = float(0.0)
    dmin = float(FAR)
    wet = float(0.0)
    r.s = -1
    for i in range(p0, p1):
        if in_band(fm, i, a):
            s = fm.shape[i] + fm.base[w]
            tb = tb + fm.area[i] * dir_to_world(body_q, shape_body, s, fm.bond_dir[i])
            n = n + fm.area[i] * dir_to_world(body_q, shape_body, s, fm.normal[i])
            wsum = wsum + fm.area[i]
            hsum = hsum + fm.area[i] * fm.h[i]
            wet = wet + fm.area[i] * fm.wet[i]
            if fm.d[i] < dmin:
                dmin = fm.d[i]
                r.s = fm.shape[i] + fm.base[w]
    if wsum == 0.0:                                   # no bond within 1.5 rows of the front: nothing pulls
        r.s = -1
        return r
    n = wp.normalize(n)
    r.tb = wp.normalize(tb - wp.dot(tb, n) * n)
    side = wp.normalize(wp.cross(n, r.tb))
    lo = float(FAR)
    hi = float(-FAR)
    for i in range(p0, p1):
        if in_band(fm, i, a):
            xf = xf + fm.area[i] * (fm.q[i] - r.tb * (fm.d[i] - a))
            lo = wp.min(lo, wp.dot(fm.q[i], side))
            hi = wp.max(hi, wp.dot(fm.q[i], side))
    r.xf = xf / wsum
    r.b = wp.max(hi - lo, 0.5 * hsum / wsum)
    r.wet = wet / wsum
    return r


@wp.func
def advance(fm: Film, p0: int, p1: int, tb: wp.vec3, a: float, s_star: float) -> float:
    """The front distance that brings every arm line to stretch ``s_star`` (never backwards)."""
    a_new = a
    for i in range(p0, p1):
        if peelable(fm, i):
            if stretch(fm, i, tb, a) > s_star:
                a_new = wp.max(a_new, advance_to(fm, i, tb, s_star))
    return a_new


@wp.func
def release(fm: Film, p0: int, p1: int, tb: wp.vec3, a: float) -> wp.vec2:
    """Frees the bonds the front passed; returns (released area, largest stretch of the rest)."""
    area = float(0.0)
    s_max = float(-FAR)
    for i in range(p0, p1):
        if peelable(fm, i):
            if fm.d[i] < a:
                fm.state[i] = FREE
                area = area + fm.area[i]
            else:
                s_max = wp.max(s_max, stretch(fm, i, tb, a))
    return wp.vec2(area, s_max)


@wp.func
def arm_stiffness(et: float, b: float, a: float, m: float, dt: float) -> float:
    """E·t·b/a, capped at ARM_CAP·m/dt² for a grip body of mass m (the pull is explicit and lagged)."""
    k = et * b / a
    if m > 0.0:
        k = wp.min(k, ARM_CAP * m / (dt * dt))
    return k


@wp.func
def arm_pull(k: float, s: float, s_prev: float, m: float, dt: float, t_c: float) -> float:
    """k·s plus critical damping 2·sqrt(k·m) on the stretch rate (the backing's and adhesive's viscoelasticity, and
    the only damping a force-limited robot gets), clamped to [0, T_c]; 0 when slack. Steady peeling holds s at
    T_c/k, so the damping acts only while the arm stretches or relaxes."""
    if s <= 0.0:
        return 0.0
    c = float(0.0)
    if m > 0.0:
        c = 2.0 * wp.sqrt(k * m)
    return wp.clamp(k * s + c * (s - wp.max(s_prev, 0.0)) / dt, 0.0, t_c)


@wp.func
def record(fr: Front, fm: Film, w: int, t: float, u: wp.vec3, c: float, bd: Band, cs: Census, g: float,
           rel: float, a: float, dt: float):
    t_prev = float(fr.peel_force[w])
    fr.impulse[w] = fr.impulse[w] + wp.float64(t * dt)            # applied over the next exchange
    fr.work[w] = fr.work[w] + wp.float64(0.5 * (t_prev + t) * wp.dot(u, cs.xg - fr.xg_prev[w]))
    fr.xg_prev[w] = cs.xg
    fr.force[w] = -t * u
    fr.point[w] = cs.xg
    fr.front_point[w] = bd.xf
    fr.sub_shape[w] = bd.s - fm.base[w]                  # state is world-local (the harness copies worlds)
    fr.peel_force[w] = wp.float64(t)
    fr.angle[w] = wp.float64(wp.degrees(wp.acos(wp.clamp(1.0 - c, -1.0, 1.0))))
    fr.width[w] = wp.float64(bd.b)
    fr.front[w] = wp.float64(a)
    fr.released[w] = fr.released[w] + wp.float64(rel)
    fr.dissipated[w] = fr.dissipated[w] + wp.float64(g * rel)
    fr.bonded[w] = wp.float64(cs.bonded - rel)


@wp.kernel
def peel_front(fm: Film, pl: Peel, fr: Front, body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32],
               et: float, grip_mass: wp.array[float], dt: float, dt_x: float):
    """``dt``: this film step (a substep of the exchange); ``dt_x``: the exchange, the lag the explicit pull has."""
    w = wp.tid()
    p0 = fm.start[w]
    p1 = fm.start[w + 1]
    cs = census(fm, w, p0, p1)
    fr.bonded[w] = wp.float64(cs.bonded)
    fr.grip_shape[w] = -1
    if cs.gs >= 0:
        fr.grip_shape[w] = cs.gs - fm.base[w]
    if cs.ng == 0 or cs.peel_area == 0.0:
        fr.force[w] = wp.vec3(0.0)
        fr.peel_force[w] = wp.float64(0.0)
        fr.sub_shape[w] = -1
        return
    a = fr.a[w]
    if a <= 0.0:                                   # first step with a grip and bonds: the arm starts here
        a = cs.a0
        fr.xg_prev[w] = cs.xg
    bd = band(fm, w, p0, p1, a, body_q, shape_body)
    if bd.s < 0:
        fr.force[w] = wp.vec3(0.0)
        fr.peel_force[w] = wp.float64(0.0)
        fr.sub_shape[w] = -1
        return
    u = wp.normalize(cs.xg - bd.xf)
    c = wp.clamp(1.0 + wp.dot(u, bd.tb), 1.0e-6, 2.0)
    scale = bd.wet / W_STD                              # the cards' rows are a standard application's
    g = g_of_c(pl.g90[bd.s], pl.g180[bd.s], c) * scale
    t_c = wp.min(bd.b * kendall(g, c, et), pl.tau[bd.s] * scale * cs.peel_area)
    k = arm_stiffness(et, bd.b, a, grip_mass[w], dt_x)
    a_new = advance(fm, p0, p1, bd.tb, a, t_c / k)
    rs = release(fm, p0, p1, bd.tb, a_new)
    t = arm_pull(k, rs[1], fr.stretch[w], grip_mass[w], dt, t_c)
    fr.stretch[w] = wp.max(rs[1], 0.0)
    fr.a[w] = wp.min(a_new, FAR)
    record(fr, fm, w, t, u, c, bd, cs, g, rs[0], a_new, dt)
