# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 11: the surface model's cells (``bead_model="surface"``; docs/notes/heat-v3-design.md §Slice 11).
Each fed slice's liquid follows its free surface with the liquid's weight (``grid_surface``), as the cap model's
follows its circle (``grid_cap.reshape``):

1. the toes as the cap model's (``fused_toes``: fusion, then wetting, which moves a toe a step past the liquid);
2. the top: the fullest surface the toes hold, a toe at 88 deg to the chord (GA ``_caps``); a slice holding more
   spreads (``joint_down`` given: GA's overflow, L (sqrt(A / A_top) - 1) / 2 per side, one cell to ``advance``, not
   up a surface climbing against gravity, only where the contact does not freeze), measured from where its liquid
   met the ray last exchange (the spill test needs the liquid at the toe; wetting's step leaves air there), and its
   top is solved again;
3. the surface holding the slice's area (``area_for``, warm-started on last exchange's pressure and angle, kept in
   ``surf_state``: p, start angle, status, where it meets toe 2's ray); a slice holding less than any surface
   between its toes ("thin": both toes would drain) keeps the cap model's circle, which holds any area;
4. a full slice whose toes could not take its metal (``joint_down`` given) overhangs (``grid_surface_overhang``,
   GA's ``_overhang``): its face turns past the 88 deg limit; when the bulge lies on the plate at toe 1 it touches
   down, and toe 1 moves out (GA's overflow step from the touch, no fusion: the metal there lies on unfused plate,
   GA's cold lap) for the next exchange; when it leans on the member past toe 2 instead (its face there past 180
   deg), toe 2 moves up that member by the same step (GA touches down on either side; no climb rule there);
5. up to ``moves`` liquid weld cells outside the surface move, farthest from the root first, into its air cells
   nearest the root (``grid_cap.carry``: energy exact; cells of one uniform band). What a touching slice cannot
   hold yet stays where it is.
With an arc force, the surface also carries GA's arc pressure about the lagged arc root (``grid_surface_flow``).
``surf_r`` holds each slice's traced radii (``STEPS`` + 1 per slice).
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.thermal import grid_surface as S
from alloy.plugins.thermal import grid_surface_overhang as O
from alloy.plugins.thermal.grid_cap import (VOLUME_TOL, Bead, carry, centre, circle_of, fused_toes, held, in_cap,
                                            locate, spread)
from alloy.plugins.thermal.grid_kernels import BIG, Grid, width

TOE_LIMIT = wp.constant(S.TOE_LIMIT)


@wp.func
def arc_frame(g: Grid, bead: Bead, t: wp.vec2, ax: int, at: wp.vec3, force: float) -> S.Frame:
    """Slice ax's polar frame on toes t: the liquid's weight and GA's arc pressure F / (2 pi s^2) exp(-r^2 / 2 s^2)
    about the lagged arc root ``at`` (the slice's distance along x in the peak, the rest node by node)."""
    up = wp.vec2(-bead.e1[1], bead.e1[0])
    peak = float(0.0)
    if force > 0.0:
        dx = g.ex[ax] + 0.5 * width(g, 0, ax) - at[0]
        s2 = bead.arc_sigma * bead.arc_sigma
        peak = force / (2.0 * wp.pi * s2) * wp.exp(-0.5 * dx * dx / s2)
    return S.frame_of(bead.root, bead.e1, bead.e2, t[0], t[1], up, bead.g * bead.down, bead.rho_liquid, bead.gamma,
                      peak, wp.vec2(at[1], at[2]), bead.arc_sigma)


@wp.func
def overflow(bead: Bead, fr: S.Frame, target: float, full: float, step: float) -> wp.vec2:
    """How far each toe of an overfull slice moves out (GA pool.py:1029-1056), on the top's area ``full``."""
    if target <= full or full <= 0.0:
        return wp.vec2(0.0, 0.0)
    move = wp.clamp(0.5 * S.chord(fr) * (wp.sqrt(target / full) - 1.0), step, bead.advance)
    m1 = float(0.0)
    m2 = float(0.0)
    if wp.dot(bead.e1, bead.down) > -0.3:
        m1 = move
    if wp.dot(bead.e2, bead.down) > -0.3:
        m2 = move
    return wp.vec2(m1, m2)


@wp.func
def liquid_weld(g: Grid, bead: Bead, h_flow: float, cell: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=float),
                lo: wp.array2d(dtype=float), w: int, ax: int, y0: int, y1: int, z0: int, z1: int) -> bool:
    """Slice ax holds a liquid weld cell (H > h_flow) within the box."""
    for y in range(y0, y1):
        for z in range(z0, z1):
            c = (ax * g.ny + y) * g.nz + z
            if cell[w, c] == bead.weld and h[w, c] + lo[w, c] > h_flow:
                return True
    return False


@wp.kernel
def reshape_surface(g: Grid, bead: Bead, h_flow: float, moves: int, cell: wp.array2d(dtype=wp.int32),
                    h: wp.array2d(dtype=float), lo: wp.array2d(dtype=float), peak: wp.array2d(dtype=float),
                    budget: wp.array2d(dtype=wp.float64), slice_fed: wp.array2d(dtype=wp.float64),
                    toes: wp.array2d(dtype=wp.vec2), temp: wp.array2d(dtype=float), mass: wp.array(dtype=float),
                    surf_r: wp.array2d(dtype=float), surf_state: wp.array2d(dtype=wp.vec4),
                    arc_at: wp.array(dtype=wp.vec3), arc_force: wp.array(dtype=float)):
    """Slice ax of world w follows its surface (module doc). Fixed roots only (``root_at_anchor`` 0)."""
    w, ax = wp.tid()
    if bead.model != 2 or bead.root_at_anchor == 1 or slice_fed[w, ax] <= wp.float64(0.0):
        return
    root = bead.root
    y0 = locate(g, 1, root[0] - bead.reach)
    y1 = locate(g, 1, root[0] + bead.reach) + 1
    z0 = locate(g, 2, root[1] - bead.reach)
    z1 = locate(g, 2, root[1] + bead.reach) + 1
    if not liquid_weld(g, bead, h_flow, cell, h, lo, w, ax, y0, y1, z0, z1):
        return                                          # nothing can move: no solve (most of a long weld, frozen)
    step = 0.5 * wp.min(width(g, 1, locate(g, 1, root[0])), width(g, 2, locate(g, 2, root[1])))
    target = held(g, bead, cell, budget, w, ax, root)
    base = ax * (S.STEPS + 1)
    edge = toes[w, ax]                                  # where last exchange's surface met the rays: the liquid
    t = fused_toes(g, bead, cell, peak, temp, toes, w, ax, root, step)
    fr = arc_frame(g, bead, t, ax, arc_at[w], arc_force[w])
    cap = S.top(fr, TOE_LIMIT, surf_r, w, base)
    if bead.sag == 1 and cap.status == S.FULL and target > cap.area and edge[0] > 0.0 and edge[1] > 0.0:
        inner = wp.vec2(edge[0] - 0.5 * step, edge[1] - 0.5 * step)     # the spill test's edge cell lies inside
        moved = spread(g, bead, cell, temp, w, ax, root, step, inner, overflow(bead, fr, target, cap.area, step))
        moved = wp.vec2(wp.max(moved[0] + 0.5 * step, edge[0]), wp.max(moved[1] + 0.5 * step, edge[1]))
        moved = wp.vec2(wp.max(moved[0], t[0]), wp.max(moved[1], t[1]))
        if moved[0] != t[0] or moved[1] != t[1]:
            t = moved
            fr = arc_frame(g, bead, t, ax, arc_at[w], arc_force[w])
            cap = S.top(fr, TOE_LIMIT, surf_r, w, base)
    toes[w, ax] = t
    st = surf_state[w, ax]
    warm_p = cap.p
    warm_psi = cap.psi0
    if st[2] > 0.0 and st[2] < 4.5:                     # an overhang's angle is in its own frame
        warm_p = st[0]
        warm_psi = st[1]
    sol = cap
    if cap.status != S.NONE:
        sol = S.area_for(fr, target, cap, warm_p, warm_psi, surf_r, w, base)
    cells = fr
    if bead.sag == 1 and sol.status == S.FULL and target > cap.area * 1.000001:
        ov = O.pinned_for(fr, target, cap, surf_r, w, base)      # the toes could not take it: it overhangs
        if ov.status == O.OVER or ov.status == O.TOUCH or ov.status == O.TOUCH2:
            sol = ov
            cells = O.overhang_frame(fr, O.DEPTH)
            if ov.status == O.TOUCH and wp.dot(bead.e1, bead.down) > -0.3:   # it touches down: toe 1 moves out
                toes[w, ax] = wp.vec2(wp.max(t[0], ov.touch) + overflow(bead, fr, target, ov.area, step)[0], t[1])
            if ov.status == O.TOUCH2:                     # it leans on the member past toe 2: toe 2 moves up it
                move = wp.clamp(0.5 * S.chord(fr) * (wp.sqrt(target / wp.max(ov.area, 1.0e-12)) - 1.0), step,
                                bead.advance)
                toes[w, ax] = wp.vec2(t[0], t[1] + move)
    use = sol.status == S.HELD or sol.status == S.WET or sol.status == S.FULL or sol.status >= O.OVER
    circle = circle_of(bead, root, t[0], t[1], target)
    if use:
        S.shoot(cells, sol.psi0, sol.p, 1, surf_r, w, base)       # the radii of the surface the cells take
        surf_state[w, ax] = wp.vec4(sol.p, sol.psi0, float(sol.status), sol.toe2)
    else:
        surf_state[w, ax] = wp.vec4(0.0, 0.0, float(sol.status), t[1])
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
                inside = in_cap(x2, circle, bead)
                if use:
                    inside = O.in_region(x2, cells, surf_r, w, base)
                code = cell[w, c]
                if code == bead.weld:
                    if h[w, c] + lo[w, c] > h_flow and d > far and not inside:
                        far = d
                        src = c
                elif code == 0:
                    if d < near and inside:
                        near = d
                        dst = c
        if src < 0 or dst < 0:
            return
        vs = width(g, 1, (src // g.nz) % g.ny) * width(g, 2, src % g.nz)
        vd = width(g, 1, (dst // g.nz) % g.ny) * width(g, 2, dst % g.nz)
        if wp.abs(vs - vd) > VOLUME_TOL * vs:
            return
        carry(cell, h, lo, peak, mass, w, src, dst, bead.weld)


__all__ = ["reshape_surface"]
