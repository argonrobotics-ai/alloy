# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 11: the pool's flow along the weld with the surface model (``bead_model="surface"`` with an arc
force; ``grid_flow`` is the cap model's). The same exchange as ``grid_flow`` (the pool's run of live slices, one
pressure P shared along it, liquid weld cells moved from slices over their share to slices under it), with each
slice's share the area its own surface holds at P (``grid_surface``: the liquid's weight and the arc's pressure,
node by node as GA, pool.py:846-853), and with GA's lag on the arc (process.py:206-209: the root and the force reach
the surface through a first-order lag of POOL_TAU = sqrt(rho L^3 / gamma) = 27 ms; without it GA's arc / surface loop
oscillates with period two updates).

Per exchange: ``arc_state`` (per world: the arc root, the first metal on the torch's axis, and its lagged copy; the
lagged force; the pool's run); ``table`` (per slice of the run: its toes and metal as ``grid_flow.survey``, its top,
and the area at ``K`` pressures from the top's down, geometric in the pressure scale); ``flow_surface`` (per world:
P by bisection on the run's volume, each slice's share max(its frozen metal, the table's area at P, linear between
rows), shared along x with GA's 1 mm kernel (``smooth``), then the moves, into the air cell nearest the root of
the circle holding the share, ``reshape_surface`` then giving each slice its own surface).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from alloy.plugins.thermal import grid_surface as S
from alloy.plugins.thermal.grid_bead_surface import arc_frame, reshape_surface
from alloy.plugins.thermal.grid_cap import VOLUME_TOL, Bead, carry, centre, circle_of, in_cap, locate
from alloy.plugins.thermal.grid_deposit import anchor
from alloy.plugins.thermal.grid_flow import P_SPAN, farthest_liquid, pool_run, survey
from alloy.plugins.thermal.grid_kernels import BIG, Arc, Grid, width

K = 9                                              # table rows per slice
ROWS = wp.constant(wp.types.vector(length=K, dtype=float)(0.0, 0.01, 0.03, 0.07, 0.15, 0.3, 0.6, 1.2, 2.4))
POOL_TAU = math.sqrt(6950.0 * 5e-3 ** 3 / 1.2)     # s, GA params.py POOL_TAU (27 ms)
LONG_SIGMA = wp.constant(1.0e-3)                   # m, GA POOL_LONG_SIGMA: the pool's metal shared along x
TOE_LIMIT = wp.constant(S.TOE_LIMIT)


@wp.kernel
def arc_state(g: Grid, bead: Bead, arcs: wp.array(dtype=Arc), cell: wp.array2d(dtype=wp.int32),
              liquid_s: wp.array2d(dtype=wp.float64), clock: wp.array(dtype=wp.float64), lag: float,
              arc_at: wp.array(dtype=wp.vec3), arc_force: wp.array(dtype=float), arc_run: wp.array(dtype=wp.vec2i)):
    """World w's arc root (lagged by ``lag`` = 1 - exp(-dt / POOL_TAU); the first root is taken whole), its lagged
    force, and the pool's run (-1, -1: none)."""
    w = wp.tid()
    a = arcs[w]
    arc_run[w] = wp.vec2i(-1, -1)
    force = float(0.0)
    if a.power > 0.0:
        force = bead.arc_force
    arc_force[w] = arc_force[w] + (force - arc_force[w]) * lag
    cr = anchor(g, cell, w, a.p, a.d, -1, 1)
    if cr < 0:
        return
    ir = cr // (g.nz * g.ny)
    now = centre(g, ir, (cr // g.nz) % g.ny, cr % g.nz)
    if arc_at[w][0] >= BIG:
        arc_at[w] = now
    else:
        arc_at[w] = arc_at[w] + (now - arc_at[w]) * lag
    run = pool_run(g, liquid_s, clock, w, ir)
    if run[0] >= 0 and run[1] > run[0]:
        arc_run[w] = run


@wp.kernel
def table(g: Grid, bead: Bead, h_flow: float, cell: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=float),
          lo: wp.array2d(dtype=float), peak: wp.array2d(dtype=float), budget: wp.array2d(dtype=wp.float64),
          toes: wp.array2d(dtype=wp.vec2), areas: wp.array2d(dtype=float), liq: wp.array2d(dtype=float),
          ptoes: wp.array2d(dtype=wp.vec2), arc_at: wp.array(dtype=wp.vec3), arc_force: wp.array(dtype=float),
          arc_run: wp.array(dtype=wp.vec2i), surf_r: wp.array2d(dtype=float), tab_p: wp.array2d(dtype=float),
          tab_a: wp.array2d(dtype=float)):
    """Slice ax's metal and toes, and the area its surface holds at K pressures (row 0 the top's; NONE: 0)."""
    w, ax = wp.tid()
    run = arc_run[w]
    if run[0] < 0 or ax < run[0] or ax > run[1]:
        return
    step = 0.5 * wp.min(width(g, 1, locate(g, 1, bead.root[0])), width(g, 2, locate(g, 2, bead.root[1])))
    survey(g, bead, h_flow, cell, h, lo, peak, budget, toes, areas, liq, ptoes, w, ax, step)
    t = wp.vec2(wp.max(ptoes[w, ax][0], 2.0 * step), wp.max(ptoes[w, ax][1], 2.0 * step))
    fr = arc_frame(g, bead, t, ax, arc_at[w], arc_force[w])
    base = ax * (S.STEPS + 1)
    cap = S.top(fr, TOE_LIMIT, surf_r, w, base)
    sc = S.scale(fr)
    psi = cap.psi0
    for j in range(K):
        p = cap.p - ROWS[j] * sc
        a = float(0.0)
        if cap.status != S.NONE:
            s = S.surface_at(fr, p, cap, psi, surf_r, w, base)
            if s.status != S.NONE:
                a = s.area
                psi = s.psi0
        tab_p[w, ax * K + j] = p
        tab_a[w, ax * K + j] = a
    if cap.status == S.NONE:
        tab_p[w, ax * K] = BIG                     # no surface: the slice keeps what it has


@wp.func
def held_at(areas: wp.array2d(dtype=float), liq: wp.array2d(dtype=float), tab_p: wp.array2d(dtype=float),
            tab_a: wp.array2d(dtype=float), w: int, i: int, p: float) -> float:
    """Slice i's share [m²] at pool pressure p: the table's area (linear between rows; the top's above row 0;
    nothing below the last row), never less than its frozen metal."""
    frozen = areas[w, i] - liq[w, i]
    if tab_p[w, i * K] >= BIG:
        return areas[w, i]
    a = float(0.0)
    if p >= tab_p[w, i * K]:
        a = tab_a[w, i * K]
    else:
        for j in range(1, K):
            pj = tab_p[w, i * K + j]
            if a == 0.0 and p >= pj:
                pk = tab_p[w, i * K + j - 1]
                f = (p - pj) / wp.max(pk - pj, 1.0e-12)
                a = tab_a[w, i * K + j] + f * (tab_a[w, i * K + j - 1] - tab_a[w, i * K + j])
                a = wp.max(a, 1.0e-30)
    return wp.max(frozen, a)


@wp.func
def room(g: Grid, bead: Bead, cell: wp.array2d(dtype=wp.int32), ptoes: wp.array2d(dtype=wp.vec2), w: int, i: int,
         step: float, target: float) -> int:
    """The air cell nearest the root of the circle holding ``target`` on slice i's toes, or -1 (the slice's own
    surface takes its cells in ``reshape_surface``)."""
    t = wp.vec2(wp.max(ptoes[w, i][0], 2.0 * step), wp.max(ptoes[w, i][1], 2.0 * step))
    cap = circle_of(bead, bead.root, t[0], t[1], target)
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


@wp.func
def smooth(g: Grid, areas: wp.array2d(dtype=float), liq: wp.array2d(dtype=float), tab_p: wp.array2d(dtype=float),
           tab_a: wp.array2d(dtype=float), share: wp.array2d(dtype=float), sm: wp.array2d(dtype=float), w: int,
           run: wp.vec2i):
    """GA's surface tension along the weld (pool.py:678-710, 986-992): the slices' shares at P shared along x with a
    Gaussian of LONG_SIGMA (1 mm, cut at 3 sigma; shorter than the arc's 1.8 mm spot, so the crater under the arc
    stays), the run's metal conserved and each slice held to its top (four rounds, as GA) and to its frozen metal.
    A single slice holding little (a starved fin under the arc) costs far more surface than the bead around it."""
    tot = float(0.0)
    for i in range(run[0], run[1] + 1):
        tot = tot + share[w, i] * width(g, 0, i)
    for i in range(run[0], run[1] + 1):
        xi = g.ex[i] + 0.5 * width(g, 0, i)
        num = float(0.0)
        den = float(0.0)
        for j in range(wp.max(run[0], i - 40), wp.min(run[1], i + 40) + 1):
            dx = g.ex[j] + 0.5 * width(g, 0, j) - xi
            if wp.abs(dx) <= 3.0 * LONG_SIGMA:
                k = wp.exp(-0.5 * dx * dx / (LONG_SIGMA * LONG_SIGMA)) * width(g, 0, j)
                num = num + k * share[w, j]
                den = den + k
        sm[w, i] = num / wp.max(den, 1.0e-30)
    for _r in range(4):
        have = float(0.0)
        for i in range(run[0], run[1] + 1):
            have = have + sm[w, i] * width(g, 0, i)
        f = float(1.0)
        if tot > 0.0 and have > 0.0:
            f = tot / have
        over = int(0)
        for i in range(run[0], run[1] + 1):
            top = areas[w, i]
            if tab_p[w, i * K] < BIG:
                top = wp.max(tab_a[w, i * K], areas[w, i] - liq[w, i])
            v = sm[w, i] * f
            if v > top:
                v = top
                over = 1
            sm[w, i] = wp.max(v, areas[w, i] - liq[w, i])
        if over == 0:
            return


@wp.kernel
def flow_surface(g: Grid, bead: Bead, h_flow: float, moves: int, cell: wp.array2d(dtype=wp.int32),
                 h: wp.array2d(dtype=float), lo: wp.array2d(dtype=float), peak: wp.array2d(dtype=float),
                 slice_fed: wp.array2d(dtype=wp.float64), areas: wp.array2d(dtype=float),
                 liq: wp.array2d(dtype=float), ptoes: wp.array2d(dtype=wp.vec2), want: wp.array2d(dtype=float),
                 mass: wp.array(dtype=float), arc_run: wp.array(dtype=wp.vec2i), tab_p: wp.array2d(dtype=float),
                 tab_a: wp.array2d(dtype=float), sm: wp.array2d(dtype=float)):
    """One exchange of the pool's flow for world w (``grid_flow.flow`` with the surfaces' shares)."""
    w = wp.tid()
    run = arc_run[w]
    if run[0] < 0:
        return
    step = 0.5 * wp.min(width(g, 1, locate(g, 1, bead.root[0])), width(g, 2, locate(g, 2, bead.root[1])))
    have = float(0.0)
    for i in range(run[0], run[1] + 1):
        have = have + areas[w, i] * width(g, 0, i)
    p_lo = float(-P_SPAN)
    p_hi = float(P_SPAN)
    for _k in range(60):
        mid = 0.5 * (p_lo + p_hi)
        total = float(0.0)
        for i in range(run[0], run[1] + 1):
            total = total + held_at(areas, liq, tab_p, tab_a, w, i, mid) * width(g, 0, i)
        if total < have:
            p_lo = mid
        else:
            p_hi = mid
    for i in range(run[0], run[1] + 1):
        want[w, i] = held_at(areas, liq, tab_p, tab_a, w, i, p_hi)
    smooth(g, areas, liq, tab_p, tab_a, want, sm, w, run)
    for i in range(run[0], run[1] + 1):
        want[w, i] = sm[w, i] - areas[w, i]
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
        dst = room(g, bead, cell, ptoes, w, ib, step, areas[w, ib] + want[w, ib])
        if src < 0:
            want[w, ia] = 0.0
        elif dst < 0:
            want[w, ib] = 0.0
        else:
            vs = width(g, 0, ia) * width(g, 1, (src // g.nz) % g.ny) * width(g, 2, src % g.nz)
            vd = width(g, 0, ib) * width(g, 1, (dst // g.nz) % g.ny) * width(g, 2, dst % g.nz)
            if wp.abs(vs - vd) > VOLUME_TOL * vs:
                want[w, ib] = 0.0
                continue
            carry(cell, h, lo, peak, mass, w, src, dst, bead.weld)
            want[w, ia] = want[w, ia] + vs / width(g, 0, ia)
            want[w, ib] = want[w, ib] - vd / width(g, 0, ib)
            areas[w, ia] = areas[w, ia] - vs / width(g, 0, ia)
            areas[w, ib] = areas[w, ib] + vd / width(g, 0, ib)
            slice_fed[w, ia] = slice_fed[w, ia] - wp.float64(vs)
            slice_fed[w, ib] = slice_fed[w, ib] + wp.float64(vd)


def allocate(grid, ctx, worlds: int, slices: int) -> None:
    """The surface model's arrays on ``grid``: warm starts (state), the arc's lagged root and force (state), the
    traced radii, the pool's run and the tables (scratch)."""
    grid.surf_state = grid.add_state("surf_state", ctx.zeros((worlds, slices), wp.vec4))
    grid.arc_at = grid.add_state("arc_at", ctx.array(np.full((worlds, 3), BIG, np.float32), wp.vec3))
    grid.arc_force = grid.add_state("arc_force", ctx.zeros(worlds, wp.float32))
    grid._surf_r = ctx.zeros((worlds, slices * (S.STEPS + 1)), wp.float32)
    grid._arc_run = ctx.zeros(worlds, wp.vec2i)
    grid._tab = (ctx.zeros((worlds, slices * K), wp.float32), ctx.zeros((worlds, slices * K), wp.float32))
    grid._smooth = ctx.zeros((worlds, slices), wp.float32)


def exchange(grid, dev: str, dt: float, h_flow: float) -> None:
    """The surface model's exchange after deposition: the arc, the pool's flow (with an arc force), the reshape."""
    c, W, nx = grid.config, grid._W, grid._shape[0]
    areas, liq, ptoes, want = grid._pool
    if c.arc_pressure[0] > 0.0:
        wp.launch(arc_state, dim=W, device=dev, inputs=[grid._ga, grid._bead, grid._arcs, grid.cell, grid.liquid_s,
                  grid.clock_s, 1.0 - math.exp(-dt / POOL_TAU)], outputs=[grid.arc_at, grid.arc_force, grid._arc_run])
        wp.launch(table, dim=(W, nx), device=dev, inputs=[grid._ga, grid._bead, h_flow, grid.cell, grid.enthalpy,
                  grid.enthalpy_lo, grid.peak, grid.deposit_budget, grid.toes, areas, liq, ptoes, grid.arc_at,
                  grid.arc_force, grid._arc_run], outputs=[grid._surf_r, *grid._tab])
        wp.launch(flow_surface, dim=W, device=dev, inputs=[grid._ga, grid._bead, h_flow, c.flow_moves],
                  outputs=[grid.cell, grid.enthalpy, grid.enthalpy_lo, grid.peak, grid.slice_fed_m3, areas, liq,
                           ptoes, want, grid._mass_d, grid._arc_run, *grid._tab, grid._smooth])
    if c.reshape_moves > 0:
        wp.launch(reshape_surface, dim=(W, nx), device=dev, inputs=[grid._ga, grid._bead, h_flow, c.reshape_moves],
                  outputs=[grid.cell, grid.enthalpy, grid.enthalpy_lo, grid.peak, grid.deposit_budget,
                           grid.slice_fed_m3, grid.toes, grid._t, grid._mass_d, grid._surf_r, grid.surf_state,
                           grid.arc_at, grid.arc_force])


__all__ = ["K", "POOL_TAU", "allocate", "exchange"]
