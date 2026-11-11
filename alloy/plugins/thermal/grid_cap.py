# SPDX-License-Identifier: Apache-2.0
"""Heat v3: the bead's cap model for ``GridThermal``'s deposition (``bead_model="cap"``; docs/notes/heat-v3-design.md
§Slices 3-6). Shared helpers (``Bead``, ``locate``, ``centre``) live here too; ``grid_deposit`` lays the metal.

Each slice's liquid is held as surface tension holds it between its contact points (GA report_welding §6, without
gravity or arc pressure, so the free surface is a circular arc):
- toes along the joint's two rays (``cap_of``): the outermost fused point (peak liquid fraction >= 0.5), and further
  where the liquid at the toe wets: onto parent metal hot enough that the effusivity-weighted contact temperature
  (e_l T_edge + e_s T) / (e_l + e_s) with the liquid at the toe's edge reaches the solidus, at most 1 mm per
  exchange (``wet_toe``); toes only grow;
- the bead: the wedge between the rays up to the toe chord plus the circular segment on that chord holding the rest
  of the slice's area (concave when the area is less than the wedge, a bulge past a half disc on a short chord); the
  area is the slice's weld metal plus the wire it is still owed (``held``), so metal fed over a tack adds to it;
- ``reshape``: every exchange, each slice's weld metal that is still liquid (H > H_l / 2) and lies outside its cap
  moves into the cap's air cells nearest the root, carrying its enthalpy and peak (mass and energy exact). Frozen
  metal stays where it froze.
- ``pool_time``: when each fed slice last held liquid metal near the root (GA's slice in the pool; its freeze time once
  it stays solid), on the pool's clock (``tick``: seconds since the start, and since the arc struck).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from alloy.plugins.thermal.grid_kernels import BIG, Grid, width

BEAD_MODELS = ("nearest", "cap")
WET_LIQUID = (31.0, 6950.0, 808.0)  # liquid steel's molecular k, ρ, c for the contact temperature (GA §6)
VOLUME_TOL = 1.0e-4  # cells of one uniform band count as equal: their float32 edge widths differ by up to ~4e-6


@wp.func
def locate(g: Grid, axis: int, v: float) -> int:
    """Cell index along ``axis`` holding coordinate v (clamped to the grid)."""
    n = g.nx
    if axis == 1:
        n = g.ny
    elif axis == 2:
        n = g.nz
    lo = int(0)
    hi = n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        e = g.ex[mid]
        if axis == 1:
            e = g.ey[mid]
        elif axis == 2:
            e = g.ez[mid]
        if e <= v:
            lo = mid
        else:
            hi = mid
    return lo


@wp.struct
class Bead:
    area: float  # m², ε · π d_w² / 4: deposited wire cross-section
    reach: float  # m, the farthest a deposited cell's centre may lie from the anchor's
    flatten: float  # κ on Δz
    norm: float  # p
    weld: int  # the cell code of weld metal (part + 1)
    h_drop: float  # J/kg, H at the droplets' temperature
    t_drop: float  # K
    max_fill: int  # cells per world per exchange at most
    model: int  # 0: nearest-first; 1: a capillary cap between the fused toes (``cap_region``)
    root: wp.vec2  # (y, z) of the joint's root in the slice plane (cap model)
    root_at_anchor: int  # 1: the root is the anchor's foot on the parent surface (a bead on plate)
    e1: wp.vec2  # the joint's two surfaces as rays from the root, sweeping counter-clockwise through the bead
    e2: wp.vec2
    t_half: float  # K, fused: peak >= T_s + (T_l - T_s) / 2 (GA: peak liquid fraction >= 0.5)
    t_solidus: float  # K
    t_liquidus: float  # K
    wet: float  # e_l / e_s: the liquid's effusivity over the solid's (GA pool.Pool._contact_freezes)
    advance: float  # m, the most a toe moves onto wettable metal per exchange (GA: 1 mm per update)
    h_liquidus: float  # J/kg
    spread: float  # m, σ of the Gaussian that shares each exchange's wire among slices along x (GA: a 1 mm kernel)
    drop_radius: float  # m, the droplets' heat cylinder below the arc root (GA §5, spray)
    drop_depth: float  # m
    lag: float  # s, the wire's metal lands where the arc root was this long ago (its pressure pushes the pool back)
    arc_force: float  # N, the arc's force (GA ARC_FORCE); > 0: the pool flows along the weld (``grid_flow``)
    arc_sigma: float  # m, σ of the arc's pressure along the weld (GA ARC_PRESSURE_SIGMA)
    gamma: float  # N/m, the liquid's surface tension


def bead_struct(c, m, cell: np.ndarray) -> tuple[Bead, float]:
    """The deposition constants of config ``c`` on material ``m``, and the droplets' power per m/s of wire feed
    [W·s/m]."""
    hot = np.asarray(0.5 * (m.t_solidus + 273.15 + 1200.0))           # the solid's e at ~1300 °C (ESTIMATE)
    k_hot, c_hot = float(m.k_solid(hot)), float(m.c_solid(hot))
    b = Bead()
    b.area = c.deposit_efficiency * math.pi * c.wire_diameter ** 2 / 4.0
    b.reach, b.flatten, b.norm, b.max_fill = c.deposit_reach, c.bead_flatten, c.bead_norm, c.max_fill
    b.weld = (c.weld_part if c.weld_part >= 0 else int(cell.max())) + 1
    h_drop = float(m.enthalpy(c.drop_temperature))
    b.h_drop, b.t_drop = h_drop, c.drop_temperature
    b.model = BEAD_MODELS.index(c.bead_model)
    b.root = wp.vec2(*(c.joint_root or (0.0, 0.0)))
    b.root_at_anchor = int(c.joint_root is None)
    r = np.asarray(c.joint_rays, np.float64)
    r = r / np.linalg.norm(r, axis=1, keepdims=True)
    b.e1, b.e2 = wp.vec2(*r[0]), wp.vec2(*r[1])
    b.t_half = 0.5 * (m.t_solidus + m.t_liquidus)
    b.t_solidus, b.t_liquidus, b.advance = m.t_solidus, m.t_liquidus, 1.0e-3
    b.h_liquidus, b.spread = m.h_liquidus, c.deposit_spread
    b.drop_radius, b.drop_depth = c.drop_cylinder
    b.lag, b.gamma = c.deposit_lag, c.surface_tension
    b.arc_force, b.arc_sigma = c.arc_pressure
    b.wet = math.sqrt(WET_LIQUID[0] * WET_LIQUID[1] * WET_LIQUID[2]) / math.sqrt(k_hot * m.density * c_hot)
    return b, m.density * b.area * (h_drop - float(m.enthalpy(c.ambient)))


@wp.func
def centre(g: Grid, x: int, y: int, z: int) -> wp.vec3:
    return wp.vec3(g.ex[x] + 0.5 * width(g, 0, x), g.ey[y] + 0.5 * width(g, 1, y), g.ez[z] + 0.5 * width(g, 2, z))


@wp.func
def cross2(a: wp.vec2, b: wp.vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def toe(g: Grid, cell: wp.array2d(dtype=wp.int32), peak: wp.array2d(dtype=float), w: int, ax: int, root: wp.vec2,
        e: wp.vec2, n: wp.vec2, step: float, reach: float, t_half: float) -> float:
    """The outermost fused point along ray ``e`` from ``root`` in slice ``ax`` (GA: the contact line lies at the
    outermost fused points around the root): the farthest parent cell just inside the surface (offset half a step
    along the metal-side normal ``n``) within ``reach`` whose peak >= t_half; 0 if none."""
    s = float(0.0)
    k = int(0)
    while (float(k) + 1.0) * step <= reach:
        q = root + (float(k) + 0.5) * step * e + 0.5 * step * n
        c = (ax * g.ny + locate(g, 1, q[0])) * g.nz + locate(g, 2, q[1])
        if cell[w, c] != 0 and peak[w, c] >= t_half:
            s = (float(k) + 1.0) * step
        k = k + 1
    return s


@wp.func
def wet_toe(g: Grid, bead: Bead, cell: wp.array2d(dtype=wp.int32), temp: wp.array2d(dtype=float), w: int, ax: int,
            root: wp.vec2, e: wp.vec2, n: wp.vec2, step: float, start: float) -> float:
    """From ``start``, the toe advances along ray ``e`` while the liquid at its edge (the weld cell just outside the
    surface at the toe) would not freeze on the parent cell ahead: the effusivity-weighted contact temperature
    (e_l T_edge + e_s T) / (e_l + e_s) reaches the solidus (GA §6); at most ``advance`` per exchange, to ``reach``."""
    s = start
    while s + step <= wp.min(bead.reach, start + bead.advance):
        edge = root + (s - 0.5 * step) * e - 0.5 * step * n
        ce = (ax * g.ny + locate(g, 1, edge[0])) * g.nz + locate(g, 2, edge[1])
        if cell[w, ce] != bead.weld or temp[w, ce] < bead.t_liquidus:
            return s
        t_wet = bead.t_solidus + bead.wet * (bead.t_solidus - temp[w, ce])
        q = root + (s + 0.5 * step) * e + 0.5 * step * n
        c = (ax * g.ny + locate(g, 1, q[0])) * g.nz + locate(g, 2, q[1])
        if cell[w, c] == 0 or temp[w, c] < t_wet:
            return s
        s = s + step
    return s


@wp.func
def cap_angle(chord: float, area: float) -> float:
    """θ of the circular segment on ``chord`` holding ``area`` (θ in (0, 2π); bisection on the monotone
    L² (θ − sin θ) / (8 sin²(θ/2)))."""
    lo = float(1.0e-4)
    hi = float(2.0 * wp.pi - 1.0e-4)
    for _i in range(60):
        mid = 0.5 * (lo + hi)
        sh = wp.sin(0.5 * mid)
        if chord * chord * (mid - wp.sin(mid)) / (8.0 * sh * sh) < area:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@wp.struct
class Cap:
    root: wp.vec2
    mid: wp.vec2  # the toe chord's midpoint
    m: wp.vec2  # the chord's unit normal, away from the root (the air side)
    centre: wp.vec2
    radius: float
    convex: int


@wp.func
def held(g: Grid, bead: Bead, cell: wp.array2d(dtype=wp.int32), budget: wp.array2d(dtype=wp.float64), w: int,
         ax: int, root: wp.vec2) -> float:
    """Slice ax's bead area [m²]: its weld metal within ``reach`` of the root (a tack's, the wire's) plus the wire it
    is still owed. Without metal from elsewhere it is the slice's fed volume per length."""
    v = budget[w, ax]
    for y in range(locate(g, 1, root[0] - bead.reach), locate(g, 1, root[0] + bead.reach) + 1):
        for z in range(locate(g, 2, root[1] - bead.reach), locate(g, 2, root[1] + bead.reach) + 1):
            if cell[w, (ax * g.ny + y) * g.nz + z] == bead.weld:
                v = v + wp.float64(width(g, 0, ax) * width(g, 1, y) * width(g, 2, z))
    return float(v / wp.float64(width(g, 0, ax)))


@wp.func
def cap_of(g: Grid, bead: Bead, cell: wp.array2d(dtype=wp.int32), peak: wp.array2d(dtype=float),
           temp: wp.array2d(dtype=float), toes: wp.array2d(dtype=wp.vec2), w: int, ax: int, root: wp.vec2,
           step: float, target: float) -> Cap:
    """Slice ax's bead for the area ``target`` [m²]: the toes, the wedge between the rays and the chord, and the
    circular segment on that chord holding the rest. A toe (stored in ``toes``, only growing) lies at the outermost
    fused point, or further where the slice's liquid wets: onto parent metal hot enough that the effusivity-weighted
    contact temperature (e_l T_pool + e_s T) / (e_l + e_s) reaches the solidus (GA §6), at most ``advance`` per
    exchange."""
    n1 = wp.vec2(bead.e1[1], -bead.e1[0])
    n2 = wp.vec2(-bead.e2[1], bead.e2[0])
    old = toes[w, ax]
    t1 = wp.max(old[0], wp.max(2.0 * step, toe(g, cell, peak, w, ax, root, bead.e1, n1, step, bead.reach,
                                                bead.t_half)))
    t2 = wp.max(old[1], wp.max(2.0 * step, toe(g, cell, peak, w, ax, root, bead.e2, n2, step, bead.reach,
                                                bead.t_half)))
    t1 = wet_toe(g, bead, cell, temp, w, ax, root, bead.e1, n1, step, t1)
    t2 = wet_toe(g, bead, cell, temp, w, ax, root, bead.e2, n2, step, t2)
    toes[w, ax] = wp.vec2(t1, t2)
    p1 = root + t1 * bead.e1
    p2 = root + t2 * bead.e2
    out = Cap()
    out.root = root
    out.mid = 0.5 * (p1 + p2)
    chord = wp.length(p2 - p1)
    m = wp.normalize(wp.vec2(-(p2 - p1)[1], (p2 - p1)[0]))
    lean = wp.dot(m, out.mid - root)
    if wp.abs(lean) < 1.0e-9:
        m = -n1                                   # a flat joint: the cap stands off the metal
    elif lean < 0.0:
        m = -m
    out.m = m
    rest = target - 0.5 * wp.abs(cross2(t1 * bead.e1, t2 * bead.e2))
    out.convex = 1
    if rest >= 0.0:
        th = cap_angle(chord, rest)
        out.radius = chord / (2.0 * wp.sin(0.5 * th))
        out.centre = out.mid - (out.radius * wp.cos(0.5 * th)) * m
    else:
        out.convex = 0
        th = wp.min(cap_angle(chord, -rest), wp.pi)
        out.radius = chord / (2.0 * wp.sin(0.5 * th))
        out.centre = out.mid + (out.radius * wp.cos(0.5 * th)) * m
    return out


@wp.func
def in_cap(x: wp.vec2, cap: Cap, bead: Bead) -> bool:
    """Inside the bead: the wedge between the rays on the root's side of the chord, plus (convex) the cap beyond the
    chord or minus (concave) the circle bulging into the wedge."""
    r = x - cap.root
    side = wp.dot(x - cap.mid, cap.m)
    wedge = side <= 0.0 and cross2(bead.e1, r) >= 0.0 and cross2(r, bead.e2) >= 0.0 and \
        cross2(bead.e1, bead.e2) > 1.0e-6
    inside = wp.length(x - cap.centre) <= cap.radius
    if cap.convex == 1:
        return wedge or (side > 0.0 and inside)
    return wedge and not inside


@wp.kernel
def pool_time(g: Grid, bead: Bead, cell: wp.array2d(dtype=wp.int32), temp: wp.array2d(dtype=float),
              slice_fed: wp.array2d(dtype=wp.float64), clock: wp.array(dtype=wp.float64), dt: float,
              liquid_s: wp.array2d(dtype=wp.float64), live_s: wp.array2d(dtype=wp.float64)):
    """Slice ax's last liquid time [s]: the end of this exchange if metal (weld or parent) within ``reach`` of the
    root is at or above T_half (GA: liquid fraction >= 0.5). ``live_s``: every slice of a fixed-root cap (the pool's
    run, ``grid_flow``: a slice whose steel is fused is in the pool before it holds weld metal, and after it has
    drained); ``liquid_s``: the slices holding fed metal only (-1 until then; the freeze times the measures read)."""
    w, ax = wp.tid()
    if bead.model != 1 or bead.root_at_anchor == 1:
        return
    for y in range(locate(g, 1, bead.root[0] - bead.reach), locate(g, 1, bead.root[0] + bead.reach) + 1):
        for z in range(locate(g, 2, bead.root[1] - bead.reach), locate(g, 2, bead.root[1] + bead.reach) + 1):
            c = (ax * g.ny + y) * g.nz + z
            if cell[w, c] != 0 and temp[w, c] >= bead.t_half:
                live_s[w, ax] = clock[w] + wp.float64(dt)
                if slice_fed[w, ax] > wp.float64(0.0):
                    liquid_s[w, ax] = clock[w] + wp.float64(dt)
                return


@wp.kernel
def tick(power: wp.array(dtype=float), dt: float, clock: wp.array(dtype=wp.float64),
         arc_on: wp.array(dtype=wp.float64)):
    """The pool's clock: seconds since the start, and since the arc struck (0 while it is off)."""
    w = wp.tid()
    clock[w] = clock[w] + wp.float64(dt)
    if power[w] > 0.0:
        arc_on[w] = arc_on[w] + wp.float64(dt)
    else:
        arc_on[w] = wp.float64(0.0)


@wp.func
def carry(cell: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=float), lo: wp.array2d(dtype=float),
          peak: wp.array2d(dtype=float), mass: wp.array(dtype=float), w: int, src: int, dst: int, code: int):
    """Weld cell ``src``'s metal moves into air cell ``dst`` (volumes equal within ``VOLUME_TOL``): its peak, and its
    enthalpy scaled by the ledger's own cell masses, m_src / m_dst, so the energy m H is the same in the new cell.
    The metal's volume changes by the two cells' difference (measured at most 4e-6 of a cell per move); the fed and
    deposited books do not count it."""
    hh = (wp.float64(h[w, src]) + wp.float64(lo[w, src])) * (wp.float64(mass[src]) / wp.float64(mass[dst]))
    hi = float(hh)
    h[w, dst] = hi
    lo[w, dst] = float(hh - wp.float64(hi))
    peak[w, dst] = peak[w, src]
    cell[w, dst] = code
    cell[w, src] = 0


@wp.kernel
def reshape(g: Grid, bead: Bead, h_flow: float, moves: int, cell: wp.array2d(dtype=wp.int32),
            h: wp.array2d(dtype=float), lo: wp.array2d(dtype=float), peak: wp.array2d(dtype=float),
            budget: wp.array2d(dtype=wp.float64), slice_fed: wp.array2d(dtype=wp.float64),
            toes: wp.array2d(dtype=wp.vec2), temp: wp.array2d(dtype=float), mass: wp.array(dtype=float)):
    """The cap model's liquid follows its slice's cap (GA: each liquid slice's pool, every update): with the toes
    re-read as fusion spreads, up to ``moves`` weld cells still liquid (H > h_flow) outside the cap move, farthest from
    the root first, into the cap's air cells nearest the root, each carrying its own enthalpy and peak (``carry``:
    energy exact; only between cells of one uniform band, equal within ``VOLUME_TOL``: an exact float32 test stopped
    most moves, since such cells' widths differ by a few 1e-6). Fixed roots only (``root_at_anchor`` 0)."""
    w, ax = wp.tid()
    if bead.model != 1 or bead.root_at_anchor == 1 or slice_fed[w, ax] <= wp.float64(0.0):
        return
    root = bead.root
    ry = locate(g, 1, root[0])
    rz = locate(g, 2, root[1])
    cap = cap_of(g, bead, cell, peak, temp, toes, w, ax, root, 0.5 * wp.min(width(g, 1, ry), width(g, 2, rz)),
                 held(g, bead, cell, budget, w, ax, root))
    y0 = locate(g, 1, root[0] - bead.reach)
    y1 = locate(g, 1, root[0] + bead.reach) + 1
    z0 = locate(g, 2, root[1] - bead.reach)
    z1 = locate(g, 2, root[1] + bead.reach) + 1
    for _k in range(moves):
        src = int(-1)
        dst = int(-1)
        far = float(-1.0)
        near = float(BIG)
        for y in range(y0, y1):
            for z in range(z0, z1):
                c = (ax * g.ny + y) * g.nz + z
                x3 = centre(g, ax, y, z)
                x2 = wp.vec2(x3[1], x3[2])
                d = wp.length(x2 - root)
                code = cell[w, c]
                if code == bead.weld:
                    if h[w, c] + lo[w, c] > h_flow and d > far and not in_cap(x2, cap, bead):
                        far = d
                        src = c
                elif code == 0:
                    if d < near and in_cap(x2, cap, bead):
                        near = d
                        dst = c
        if src < 0 or dst < 0:
            return
        vs = width(g, 1, (src // g.nz) % g.ny) * width(g, 2, src % g.nz)
        vd = width(g, 1, (dst // g.nz) % g.ny) * width(g, 2, dst % g.nz)
        if wp.abs(vs - vd) > VOLUME_TOL * vs:
            return
        carry(cell, h, lo, peak, mass, w, src, dst, bead.weld)
