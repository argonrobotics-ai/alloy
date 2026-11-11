# SPDX-License-Identifier: Apache-2.0
"""Heat v3 slice 2: weld-metal deposition on ``GridThermal``'s grid (docs/notes/heat-v3-design.md §Deposition).

The wire feeds ``wire_feed`` [W] (m/s) of wire of ``wire_diameter``; ``deposit_efficiency`` of it reaches the work
(GA report_welding §5: spray 98 %, short circuit 93 %, globular 89 %; the rest is spatter, booked as fed but not
deposited). Each exchange, one thread per world:

1. finds the beam's **anchor**: the last cell before the first parent-metal cell along the torch axis (a voxel walk
   from where the axis enters the grid, through weld metal, so the anchor stays on the parent surface as the bead
   grows); if the axis misses the parent metal, the exchange's wire is booked as ``missed_m3``;
2. adds the exchange's volume to the anchor's **slice** budget (cells with the anchor's x index: the grid's x is the
   joint, the travel axis, as in GA's weld frame; z is up);
3. fills air cells of that slice that touch metal, nearest the anchor first by the score
   (|Δy|^p + |κ Δz|^p)^(1/p) (``bead_norm`` p, ``bead_flatten`` κ), while the slice's budget holds at least half the
   next cell's volume. The budget keeps the remainder (it may go half a cell negative), so the deposited volume of
   every slice tracks its fed volume within half a cell for good.

A new cell is weld metal (``weld_part``) at the droplets' enthalpy, H(``drop_temperature``) (GA: 2673 K), with that
peak temperature. Its m·H is booked as ``filler_J``, an input to the ledger. The beam's power is reduced by what the
droplets carry, ρ ε A_w w_f (H_drop − H_wire) with H_wire = H(ambient at setup), so η·V·I is not counted twice.

Shapes (``bead_model``): "nearest" fills nearest the anchor first (a round stand-in); "cap" holds each slice's liquid
between its contact points (``grid_cap``). Each exchange's wire is shared among the slices around the anchor by a
Gaussian of σ ``spread`` along x (GA spreads the pool's metal with a 1 mm kernel), so a parked arc fills a crater, not
one slice. New metal takes the enthalpy of the droplets' cylinder below the arc root (GA §5: spray droplets enter a
1 x 1.5 mm cylinder; new filler at the droplets' cell enthalpy), and the rest of the droplets' heat goes into that
cylinder as power (``droplets``), so the droplets heat the root, not the bead's rim.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.thermal.grid_cap import Bead, Cap, cap_of, centre, held, in_cap, locate
from alloy.plugins.thermal.grid_kernels import BIG, Arc, Grid, Table, lookup, width


@wp.func
def slab(lo: float, hi: float, p: float, d: float) -> wp.vec2:
    """Ray parameter interval inside [lo, hi] along one axis."""
    if d == 0.0:
        if p < lo or p > hi:
            return wp.vec2(BIG, -BIG)
        return wp.vec2(-BIG, BIG)
    a = (lo - p) / d
    b = (hi - p) / d
    return wp.vec2(wp.min(a, b), wp.max(a, b))


@wp.func
def anchor(g: Grid, cell: wp.array2d(dtype=wp.int32), w: int, p: wp.vec3, d: wp.vec3, weld: int, first: int) -> int:
    """The last cell before the first parent-metal cell along p + t d (t >= 0 from where the ray enters the grid; weld
    metal is passed through, so the anchor stays on the parent surface as the bead grows), or -1 when the ray misses
    the grid or meets no parent metal."""
    sx = slab(g.ex[0], g.ex[g.nx], p[0], d[0])
    sy = slab(g.ey[0], g.ey[g.ny], p[1], d[1])
    sz = slab(g.ez[0], g.ez[g.nz], p[2], d[2])
    t0 = wp.max(wp.max(sx[0], sy[0]), wp.max(sz[0], 0.0))
    t1 = wp.min(wp.min(sx[1], sy[1]), sz[1])
    if t1 <= t0:
        return -1
    q = p + (0.5 * (t0 + wp.min(t1, t0 + 1.0e-6))) * d     # just inside the entry face
    x = locate(g, 0, q[0])
    y = locate(g, 1, q[1])
    z = locate(g, 2, q[2])
    prev = int(-1)
    for _step in range(g.nx + g.ny + g.nz + 3):
        if x < 0 or x >= g.nx or y < 0 or y >= g.ny or z < 0 or z >= g.nz:
            return -1
        c = (x * g.ny + y) * g.nz + z
        if cell[w, c] != 0 and cell[w, c] != weld:
            if first == 1:
                return c                                # the arc root: the first metal (weld = -1: any)
            return prev
        prev = c
        tx = float(BIG)
        ty = float(BIG)
        tz = float(BIG)
        if d[0] > 0.0:
            tx = (g.ex[x + 1] - p[0]) / d[0]
        elif d[0] < 0.0:
            tx = (g.ex[x] - p[0]) / d[0]
        if d[1] > 0.0:
            ty = (g.ey[y + 1] - p[1]) / d[1]
        elif d[1] < 0.0:
            ty = (g.ey[y] - p[1]) / d[1]
        if d[2] > 0.0:
            tz = (g.ez[z + 1] - p[2]) / d[2]
        elif d[2] < 0.0:
            tz = (g.ez[z] - p[2]) / d[2]
        if tx <= ty and tx <= tz:
            if d[0] > 0.0:
                x = x + 1
            else:
                x = x - 1
        elif ty <= tz:
            if d[1] > 0.0:
                y = y + 1
            else:
                y = y - 1
        else:
            if d[2] > 0.0:
                z = z + 1
            else:
                z = z - 1
    return -1


@wp.func
def touches_metal(g: Grid, cell: wp.array2d(dtype=wp.int32), w: int, x: int, y: int, z: int) -> bool:
    c = (x * g.ny + y) * g.nz + z
    if x > 0 and cell[w, c - g.ny * g.nz] != 0:
        return True
    if x + 1 < g.nx and cell[w, c + g.ny * g.nz] != 0:
        return True
    if y > 0 and cell[w, c - g.nz] != 0:
        return True
    if y + 1 < g.ny and cell[w, c + g.nz] != 0:
        return True
    if z > 0 and cell[w, c - 1] != 0:
        return True
    if z + 1 < g.nz and cell[w, c + 1] != 0:
        return True
    return False


@wp.func
def fill_slice(g: Grid, bead: Bead, mass: wp.array(dtype=float), cell: wp.array2d(dtype=wp.int32),
               h: wp.array2d(dtype=float), lo: wp.array2d(dtype=float), peak: wp.array2d(dtype=float),
               budget: wp.array2d(dtype=wp.float64), deposited: wp.array(dtype=wp.float64),
               toes: wp.array2d(dtype=wp.vec2), temp: wp.array2d(dtype=float), w: int, ix: int, at: wp.vec3,
               ay: int, az: int, h_new: float, t_new: float) -> wp.float64:
    """Fill slice ix's air cells while its budget holds half the next cell: nearest the anchor (model 0) or inside
    the slice's cap nearest the root (model 1). New cells at ``h_new`` / ``t_new``; returns the mass filled [kg]."""
    root = bead.root
    cap = Cap()
    cap.root = root
    if bead.model == 1:
        if bead.root_at_anchor == 1:
            root = wp.vec2(at[1], g.ez[az])
        step = 0.5 * wp.min(width(g, 1, ay), width(g, 2, az))
        cap = cap_of(g, bead, cell, peak, temp, toes, w, ix, root, step, held(g, bead, cell, budget, w, ix, root))
    root = cap.root
    box = wp.vec2(at[1], at[2])                       # the scan box: around the anchor, or the cap's root
    if bead.model == 1:
        box = root
    y0 = locate(g, 1, box[0] - bead.reach)
    y1 = locate(g, 1, box[0] + bead.reach) + 1
    z0 = locate(g, 2, box[1] - bead.reach)
    z1 = locate(g, 2, box[1] + bead.reach) + 1
    p = bead.norm
    filled = wp.float64(0.0)
    for _k in range(bead.max_fill):
        best = int(-1)
        best_s = float(BIG)
        for y in range(y0, y1):
            for z in range(z0, z1):
                c = (ix * g.ny + y) * g.nz + z
                if cell[w, c] == 0:
                    x3 = centre(g, ix, y, z)
                    v = wp.vec2(x3[1] - at[1], x3[2] - at[2])
                    if bead.model == 1:
                        x2 = wp.vec2(x3[1], x3[2])
                        s = wp.length(x2 - root)
                        if s < best_s and in_cap(x2, cap, bead):
                            best_s = s
                            best = c
                    elif wp.length(v) <= bead.reach:
                        s = wp.pow(wp.pow(wp.abs(v[0]), p) + wp.pow(wp.abs(bead.flatten * v[1]), p), 1.0 / p)
                        if s < best_s and touches_metal(g, cell, w, ix, y, z):
                            best_s = s
                            best = c
        if best < 0:
            return filled
        vol = width(g, 0, ix) * width(g, 1, (best // g.nz) % g.ny) * width(g, 2, best % g.nz)
        if budget[w, ix] < wp.float64(0.5 * vol):
            return filled
        cell[w, best] = bead.weld
        h[w, best] = h_new
        lo[w, best] = 0.0
        peak[w, best] = t_new
        budget[w, ix] = budget[w, ix] - wp.float64(vol)
        deposited[w] = deposited[w] + wp.float64(vol)
        filled = filled + wp.float64(mass[best])
    return filled


@wp.kernel
def deposit(g: Grid, bead: Bead, tb: Table, arcs: wp.array(dtype=Arc), feed: wp.array(dtype=float), dt: float,
            mass: wp.array(dtype=float), cell: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=float),
            lo: wp.array2d(dtype=float), peak: wp.array2d(dtype=float), budget: wp.array2d(dtype=wp.float64),
            fed: wp.array(dtype=wp.float64), deposited: wp.array(dtype=wp.float64),
            missed: wp.array(dtype=wp.float64), filler: wp.array(dtype=wp.float64),
            slice_fed: wp.array2d(dtype=wp.float64), toes: wp.array2d(dtype=wp.vec2), temp: wp.array2d(dtype=float),
            drop_idx: wp.array2d(dtype=wp.int32), drop_pow: wp.array2d(dtype=float), drop_n: wp.array(dtype=wp.int32),
            arc_on: wp.array(dtype=wp.float64)):
    """One exchange of deposition for world w (one thread per world: the cells it fills are its own). The wire is
    shared among the slices around the anchor (a Gaussian of σ ``spread`` along x), each slice fills; the new metal
    takes the enthalpy of the droplets' cylinder below the arc root (GA: new filler at the droplets' cell enthalpy),
    and the rest of the droplets' heat, m (H_drop - H_new), goes into that cylinder as power for this exchange
    (``drop_*``, added to the arc's power per cell by ``droplets``)."""
    w = wp.tid()
    drop_n[w] = 0
    dv = wp.float64(feed[w]) * wp.float64(bead.area) * wp.float64(dt)
    if dv <= wp.float64(0.0):
        return
    fed[w] = fed[w] + dv
    a = arcs[w]
    ca = anchor(g, cell, w, a.p, a.d, bead.weld, 0)
    if ca < 0:
        missed[w] = missed[w] + dv
        return
    az = ca % g.nz
    ay = (ca // g.nz) % g.ny
    ax = ca // (g.nz * g.ny)
    at = centre(g, ax, ay, az)
    # the droplets' cylinder: metal along the beam below the arc root (the first metal, weld or parent)
    m_cyl = float(0.0)
    h_cyl = float(0.0)
    n_drop = int(0)
    cr = anchor(g, cell, w, a.p, a.d, -1, 1)
    if cr >= 0:
        p0 = centre(g, cr // (g.nz * g.ny), (cr // g.nz) % g.ny, cr % g.nz)
        span = bead.drop_depth + bead.drop_radius
        for x in range(locate(g, 0, p0[0] - span), locate(g, 0, p0[0] + span) + 1):
            for y in range(locate(g, 1, p0[1] - span), locate(g, 1, p0[1] + span) + 1):
                for z in range(locate(g, 2, p0[2] - span), locate(g, 2, p0[2] + span) + 1):
                    c = (x * g.ny + y) * g.nz + z
                    v = centre(g, x, y, z) - p0
                    along = wp.dot(v, a.d)
                    if cell[w, c] != 0 and n_drop < drop_idx.shape[1] and along >= 0.0 and \
                            along <= bead.drop_depth and wp.length(v - along * a.d) <= bead.drop_radius:
                        drop_idx[w, n_drop] = c
                        drop_pow[w, n_drop] = mass[c]
                        m_cyl = m_cyl + mass[c]
                        h_cyl = h_cyl + mass[c] * (h[w, c] + lo[w, c])
                        n_drop = n_drop + 1
        if n_drop == 0:
            drop_idx[w, 0] = cr
            drop_pow[w, 0] = mass[cr]
            m_cyl = mass[cr]
            h_cyl = mass[cr] * (h[w, cr] + lo[w, cr])
            n_drop = 1
    h_new = bead.h_drop
    if m_cyl > 0.0:
        h_new = wp.clamp(h_cyl / m_cyl, bead.h_liquidus, bead.h_drop)
    t_new = lookup(tb, h_new)[0]
    # the wire shared among the slices around the anchor, where the arc was ``lag`` seconds ago, or at its strike if
    # that is later (the arc's pressure pushes the pool back as it travels; parked, the metal lands under it)
    xc = at[0] - wp.min(bead.lag, float(arc_on[w])) * a.v[0]
    x_lo = locate(g, 0, xc - 3.0 * bead.spread)
    x_hi = locate(g, 0, xc + 3.0 * bead.spread)
    wsum = float(0.0)
    for ix in range(x_lo, x_hi + 1):
        u = (g.ex[ix] + 0.5 * width(g, 0, ix) - xc) / bead.spread
        wsum = wsum + wp.exp(-0.5 * u * u) * width(g, 0, ix)
    one = int(0)
    if wsum <= 1.0e-30:                               # a kernel narrower than a slice: all into xc's slice
        x_lo = locate(g, 0, xc)
        x_hi = x_lo
        one = 1
    m_new = wp.float64(0.0)
    given = wp.float64(0.0)
    for ix in range(x_lo, x_hi + 1):
        u = (g.ex[ix] + 0.5 * width(g, 0, ix) - xc) / bead.spread
        share = dv - given                            # the last slice takes the rest: the shares sum to dv exactly
        if one == 0 and ix < x_hi:
            share = dv * wp.float64(wp.exp(-0.5 * u * u) * width(g, 0, ix) / wsum)
        given = given + share
        budget[w, ix] = budget[w, ix] + share
        slice_fed[w, ix] = slice_fed[w, ix] + share
        m_new = m_new + fill_slice(g, bead, mass, cell, h, lo, peak, budget, deposited, toes, temp, w, ix,
                                   at, ay, az, h_new, t_new)
    filler[w] = filler[w] + m_new * wp.float64(h_new)
    power = float(m_new) * (bead.h_drop - h_new) / dt
    for i in range(n_drop):
        drop_pow[w, i] = power * drop_pow[w, i] / m_cyl
    drop_n[w] = n_drop


@wp.kernel
def droplets(drop_idx: wp.array2d(dtype=wp.int32), drop_pow: wp.array2d(dtype=float), drop_n: wp.array(dtype=wp.int32),
             arc_w: wp.array2d(dtype=float)):
    """Add this exchange's droplet heat to the arc's power of the cylinder's cells (one thread per world)."""
    w = wp.tid()
    for i in range(drop_n[w]):
        arc_w[w, drop_idx[w, i]] = arc_w[w, drop_idx[w, i]] + drop_pow[w, i]
