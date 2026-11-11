# SPDX-License-Identifier: Apache-2.0
"""Heat v3: the liquid pool flows along the weld toward one pressure (docs/notes/heat-v3-design.md §Slice 8; GA
report_welding §6, pool.py: the pool's metal is shared among its contiguous liquid slices at one pressure, the arc's
pressure holding the surface down under the arc). On when ``arc_pressure``'s force is > 0 (cap model, fixed root).

Every exchange, one thread per world:
1. the pool: the contiguous run of live slices (``grid_cap.pool_time``'s ``live_s``: weld or parent metal liquid at
   the root, so steel fused ahead of the arc and a slice that has drained are in it) holding the arc root's slice
   or one within two of it (GA: the liquid run at the root; the torch's axis finds the root also with the arc off);
2. each slice's area A (its weld metal and the wire still owed), its liquid weld metal L (H > H_l / 2) and its toes
   (the stored ones, or the outermost fused points if farther);
3. the pool's pressure P: at P a slice holds the wedge between its toes plus the circular segment on the toe chord of
   curvature κ = (P - p_arc) / γ (Young-Laplace without gravity; concave where the arc presses harder than P; at most
   a half disc, GA's 88° toe limit), never less than its frozen metal A - L; P is set (bisection) so the pool holds
   what it has. p_arc = F / (2π σ²) exp(-r² / 2σ²), r from the arc root (the first metal on the torch's axis) to
   the slice's toe chord, averaged over the chord, while the arc burns (GA ARC_FORCE, ARC_PRESSURE_SIGMA; GA applies
   it node by node on the chord, pool.py:847-853, this is its mean), 0 when it is out, so the dip under a parked
   arc fills back. A full slice's chord is 4-5 mm from a root in the corner, so the arc dents the pool only where
   its surface is near the root. The dent is about σ wide: cells across the joint must be finer than σ / 3
   (``dent_cell``; ``grid.Config`` warns);
4. up to ``moves`` liquid weld cells move, each from the slice holding most over its share (its cell farthest from
   the root) into the slice holding least under it (the air cell of its grown cap nearest the root), carrying their
   enthalpy and peak (``grid_cap.carry``: energy exact; only between cells of one uniform band, equal within
   ``VOLUME_TOL``). ``reshape`` then shapes each slice.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy.plugins.thermal.grid_cap import VOLUME_TOL, Bead, cap_of, carry, centre, held, in_cap, locate, toe
from alloy.plugins.thermal.grid_deposit import anchor
from alloy.plugins.thermal.grid_kernels import BIG, Arc, Grid, width

RUN_MAX = 80                                       # slices a pool reaches each way from where it is found
P_SPAN = 1.0e6                                     # Pa: the bisection's bounds on the pool's pressure


@wp.func
def liquid_now(liquid_s: wp.array2d(dtype=wp.float64), clock: wp.array(dtype=wp.float64), w: int, i: int) -> bool:
    """Slice i held liquid metal at the end of the last exchange."""
    return liquid_s[w, i] >= clock[w] - wp.float64(1.0e-9)


@wp.func
def pool_run(g: Grid, liquid_s: wp.array2d(dtype=wp.float64), clock: wp.array(dtype=wp.float64), w: int,
             ir: int) -> wp.vec2i:
    """The contiguous liquid slices holding slice ir or one within two of it; (-1, -1) if none."""
    start = int(-1)
    for d in range(5):
        i = ir + ((d + 1) // 2) * (1 - 2 * (d % 2))     # ir, ir - 1, ir + 1, ir - 2, ir + 2
        if start < 0 and i >= 0 and i < g.nx:
            if liquid_now(liquid_s, clock, w, i):
                start = i
    if start < 0:
        return wp.vec2i(-1, -1)
    lo = int(start)
    while lo > 0 and start - lo < RUN_MAX and liquid_now(liquid_s, clock, w, lo - 1):
        lo = lo - 1
    hi = int(start)
    while hi < g.nx - 1 and hi - start < RUN_MAX and liquid_now(liquid_s, clock, w, hi + 1):
        hi = hi + 1
    return wp.vec2i(lo, hi)


@wp.func
def holds(t: wp.vec2, kappa: float) -> float:
    """The area [m²] a slice with toes t holds at free-surface curvature kappa [1/m]: the wedge plus the circular
    segment of that curvature on the toe chord (taken out of the wedge when concave), at most a half disc."""
    wedge = 0.5 * t[0] * t[1]
    c = wp.sqrt(t[0] * t[0] + t[1] * t[1])
    if c <= 0.0:
        return 0.0
    k = wp.min(wp.abs(kappa), 2.0 / c)
    if k < 1.0e-9:
        return wedge
    th = 2.0 * wp.asin(wp.min(1.0, 0.5 * c * k))
    seg = 0.5 * (th - wp.sin(th)) / (k * k)
    if kappa < 0.0:
        return wp.max(0.0, wedge - seg)
    return wedge + seg


QUAD = 9                                           # points across a toe chord for the arc's mean pressure


@wp.func
def arc_pressure(bead: Bead, t: wp.vec2, x: float, at: wp.vec3, force: float) -> float:
    """The arc's mean pressure [Pa] on the toe chord of a slice at x with toes t: GA's F / (2π σ²) exp(-r² / 2σ²)
    with r from the arc root ``at`` to the chord's points (pool.py:847-853), averaged over the chord. GA keeps the
    pressure at each of its 25 chord nodes and solves the surface under that profile; here the slice's cap is one
    circular arc, so it answers to the chord's mean. A full slice's chord lies far from an arc root in the joint's
    corner, so the arc presses hard only where the surface is near its root."""
    p1 = bead.root + t[0] * bead.e1
    d = bead.root + t[1] * bead.e2 - p1
    if force <= 0.0 or wp.length(d) <= 0.0:
        return 0.0
    q = wp.vec2(at[1], at[2]) - p1
    dx = x - at[0]
    total = float(0.0)
    for k in range(QUAD):
        r = q - (float(k) + 0.5) / float(QUAD) * d
        total = total + wp.exp(-0.5 * (dx * dx + wp.dot(r, r)) / (bead.arc_sigma * bead.arc_sigma))
    return force / (2.0 * wp.pi * bead.arc_sigma * bead.arc_sigma) * total / float(QUAD)


@wp.func
def share(g: Grid, bead: Bead, areas: wp.array2d(dtype=float), liq: wp.array2d(dtype=float),
          ptoes: wp.array2d(dtype=wp.vec2), w: int, i: int, at: wp.vec3, force: float, p: float) -> float:
    """Slice i's area [m²] at pool pressure p: what its toes hold, never less than its frozen metal."""
    p_arc = arc_pressure(bead, ptoes[w, i], g.ex[i] + 0.5 * width(g, 0, i), at, force)
    return wp.max(areas[w, i] - liq[w, i], holds(ptoes[w, i], (p - p_arc) / bead.gamma))


@wp.func
def pool_volume(g: Grid, bead: Bead, areas: wp.array2d(dtype=float), liq: wp.array2d(dtype=float),
                ptoes: wp.array2d(dtype=wp.vec2), w: int, run: wp.vec2i, at: wp.vec3, force: float, p: float) -> float:
    total = float(0.0)
    for i in range(run[0], run[1] + 1):
        total = total + share(g, bead, areas, liq, ptoes, w, i, at, force, p) * width(g, 0, i)
    return total


@wp.func
def survey(g: Grid, bead: Bead, h_flow: float, cell: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=float),
           lo: wp.array2d(dtype=float), peak: wp.array2d(dtype=float), budget: wp.array2d(dtype=wp.float64),
           toes: wp.array2d(dtype=wp.vec2), areas: wp.array2d(dtype=float), liq: wp.array2d(dtype=float),
           ptoes: wp.array2d(dtype=wp.vec2), w: int, i: int, step: float):
    """Slice i's area, liquid weld metal [m²] and toes into the pool's scratch."""
    areas[w, i] = held(g, bead, cell, budget, w, i, bead.root)
    v = float(0.0)
    for y in range(locate(g, 1, bead.root[0] - bead.reach), locate(g, 1, bead.root[0] + bead.reach) + 1):
        for z in range(locate(g, 2, bead.root[1] - bead.reach), locate(g, 2, bead.root[1] + bead.reach) + 1):
            c = (i * g.ny + y) * g.nz + z
            if cell[w, c] == bead.weld and h[w, c] + lo[w, c] > h_flow:
                v = v + width(g, 1, y) * width(g, 2, z)
    liq[w, i] = v
    n1 = wp.vec2(bead.e1[1], -bead.e1[0])
    n2 = wp.vec2(-bead.e2[1], bead.e2[0])
    old = toes[w, i]
    t1 = toe(g, cell, peak, w, i, bead.root, bead.e1, n1, step, bead.reach, bead.t_half)
    t2 = toe(g, cell, peak, w, i, bead.root, bead.e2, n2, step, bead.reach, bead.t_half)
    ptoes[w, i] = wp.vec2(wp.max(old[0], t1), wp.max(old[1], t2))


@wp.func
def farthest_liquid(g: Grid, bead: Bead, h_flow: float, cell: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=float),
                    lo: wp.array2d(dtype=float), w: int, i: int) -> int:
    """Slice i's liquid weld cell farthest from the root, or -1."""
    best = int(-1)
    far = float(-1.0)
    for y in range(locate(g, 1, bead.root[0] - bead.reach), locate(g, 1, bead.root[0] + bead.reach) + 1):
        for z in range(locate(g, 2, bead.root[1] - bead.reach), locate(g, 2, bead.root[1] + bead.reach) + 1):
            c = (i * g.ny + y) * g.nz + z
            if cell[w, c] == bead.weld and h[w, c] + lo[w, c] > h_flow:
                x3 = centre(g, i, y, z)
                d = wp.length(wp.vec2(x3[1], x3[2]) - bead.root)
                if d > far:
                    far = d
                    best = c
    return best


@wp.func
def nearest_room(g: Grid, bead: Bead, cell: wp.array2d(dtype=wp.int32), peak: wp.array2d(dtype=float),
                 temp: wp.array2d(dtype=float), toes: wp.array2d(dtype=wp.vec2), w: int, i: int, step: float,
                 target: float) -> int:
    """The air cell of slice i's cap for area ``target`` nearest the root, or -1."""
    cap = cap_of(g, bead, cell, peak, temp, toes, w, i, bead.root, step, target)
    best = int(-1)
    near = float(BIG)
    for y in range(locate(g, 1, bead.root[0] - bead.reach), locate(g, 1, bead.root[0] + bead.reach) + 1):
        for z in range(locate(g, 2, bead.root[1] - bead.reach), locate(g, 2, bead.root[1] + bead.reach) + 1):
            c = (i * g.ny + y) * g.nz + z
            if cell[w, c] == 0:
                x3 = centre(g, i, y, z)
                x2 = wp.vec2(x3[1], x3[2])
                d = wp.length(x2 - bead.root)
                if d < near and in_cap(x2, cap, bead):
                    near = d
                    best = c
    return best


@wp.kernel
def flow(g: Grid, bead: Bead, h_flow: float, moves: int, arcs: wp.array(dtype=Arc),
         cell: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=float), lo: wp.array2d(dtype=float),
         peak: wp.array2d(dtype=float), budget: wp.array2d(dtype=wp.float64), slice_fed: wp.array2d(dtype=wp.float64),
         toes: wp.array2d(dtype=wp.vec2), temp: wp.array2d(dtype=float), liquid_s: wp.array2d(dtype=wp.float64),
         clock: wp.array(dtype=wp.float64), areas: wp.array2d(dtype=float), liq: wp.array2d(dtype=float),
         ptoes: wp.array2d(dtype=wp.vec2), want: wp.array2d(dtype=float), mass: wp.array(dtype=float)):
    """One exchange of the pool's flow along the weld for world w (module doc). ``slice_fed`` moves with each cell by
    the cell's own volume, so its total drifts by the cells' volume difference (at most VOLUME_TOL per move)."""
    w = wp.tid()
    if bead.arc_force <= 0.0 or bead.model != 1 or bead.root_at_anchor == 1:
        return
    a = arcs[w]
    cr = anchor(g, cell, w, a.p, a.d, -1, 1)               # the arc root: the first metal on the axis
    if cr < 0:
        return
    ir = cr // (g.nz * g.ny)
    at = centre(g, ir, (cr // g.nz) % g.ny, cr % g.nz)
    run = pool_run(g, liquid_s, clock, w, ir)
    if run[0] < 0 or run[1] <= run[0]:
        return
    force = float(0.0)
    if a.power > 0.0:
        force = bead.arc_force
    step = 0.5 * wp.min(width(g, 1, locate(g, 1, bead.root[0])), width(g, 2, locate(g, 2, bead.root[1])))
    have = float(0.0)
    for i in range(run[0], run[1] + 1):
        survey(g, bead, h_flow, cell, h, lo, peak, budget, toes, areas, liq, ptoes, w, i, step)
        have = have + areas[w, i] * width(g, 0, i)
    p_lo = float(-P_SPAN)
    p_hi = float(P_SPAN)
    for _k in range(60):
        mid = 0.5 * (p_lo + p_hi)
        if pool_volume(g, bead, areas, liq, ptoes, w, run, at, force, mid) < have:
            p_lo = mid
        else:
            p_hi = mid
    for i in range(run[0], run[1] + 1):
        want[w, i] = share(g, bead, areas, liq, ptoes, w, i, at, force, p_hi) - areas[w, i]
    unit = width(g, 1, locate(g, 1, bead.root[0])) * width(g, 2, locate(g, 2, bead.root[1]))
    for _m in range(moves):
        ia = int(-1)
        ib = int(-1)
        most = float(0.0)
        least = float(0.0)
        for i in range(run[0], run[1] + 1):
            if want[w, i] < -0.5 * unit and want[w, i] < most:
                most = want[w, i]
                ia = i
            if want[w, i] > 0.5 * unit and want[w, i] > least:
                least = want[w, i]
                ib = i
        if ia < 0 or ib < 0:
            return
        src = farthest_liquid(g, bead, h_flow, cell, h, lo, w, ia)
        dst = nearest_room(g, bead, cell, peak, temp, toes, w, ib, step, areas[w, ib] + want[w, ib])
        if src < 0:
            want[w, ia] = 0.0                     # nothing left to give
        elif dst < 0:
            want[w, ib] = 0.0                     # no room in its cap
        else:
            vs = width(g, 0, ia) * width(g, 1, (src // g.nz) % g.ny) * width(g, 2, src % g.nz)
            vd = width(g, 0, ib) * width(g, 1, (dst // g.nz) % g.ny) * width(g, 2, dst % g.nz)
            if wp.abs(vs - vd) > VOLUME_TOL * vs:
                want[w, ib] = 0.0                 # its room is in cells of another size: this slice takes no more
                continue
            carry(cell, h, lo, peak, mass, w, src, dst, bead.weld)
            want[w, ia] = want[w, ia] + vs / width(g, 0, ia)
            want[w, ib] = want[w, ib] - vd / width(g, 0, ib)
            areas[w, ia] = areas[w, ia] - vs / width(g, 0, ia)
            areas[w, ib] = areas[w, ib] + vd / width(g, 0, ib)
            slice_fed[w, ia] = slice_fed[w, ia] - wp.float64(vs)
            slice_fed[w, ib] = slice_fed[w, ib] + wp.float64(vd)


def dent_cell(y, z, root) -> float:
    """The coarser of the two cell widths [m] at the joint's root (y, z: cell edges; root None: the finest cells)."""
    widths = []
    for edges, at in zip((y, z), root or (None, None), strict=True):
        d = np.diff(np.asarray(edges, np.float64))
        widths.append(float(d.min()) if at is None else float(d[min(max(int(np.searchsorted(edges, at)) - 1, 0),
                                                                    len(d) - 1)]))
    return max(widths)


__all__ = ["dent_cell", "flow", "holds", "liquid_now", "pool_run"]
