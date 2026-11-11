# SPDX-License-Identifier: Apache-2.0
"""Kernels of the drop stage (``drops.py``; design §4.4 and §4.7).

Shedding, per exchange step after the film's substeps:
1. ``collect_kernel`` (world, edge cell): the film's open-edge outflux (``shed``, ``shed_mom``) moves into the cell's
   pending volume and momentum, and the species it carried (``Species.shed``) into the cell's pending species.
2. ``drip_kernel`` (world, tile): only active tiles. A hanging cell (``g·n > 0``) deeper than its retention depth
   ``h_ret`` loses ``(h - h_ret)·(1 - exp(-dt/tau))``. That volume is booked in ``dripped`` and becomes pending,
   with the same share of the cell's species (booked in the film's ``Species.out``), so the film keeps its
   concentration. This is the only drop launch that lowers ``h``.
3. ``count_kernel`` (world, tile), ``offset_kernel`` (world) and ``fill_kernel`` (world, tile): each cell's whole
   drops (``floor(pending / V)``) get request slots ``[W, R]`` in tile then cell order (a prefix over tiles, so
   deterministic). Positions and velocities are in world. A tile keeps ``tile_pend`` while it still holds a whole
   drop, so a full request buffer defers drops and loses none.
4. After the SPH port or the ballistic pool takes ``granted[w]`` of them, ``commit_kernel`` (world) removes their
   volume, momentum and share of the pending species from the pending cells, in slot order. A ballistic drop carries
   its species (``drops_flight.Pool.sp``); SPH particles carry none, so a drop's species going to SPH are booked
   lost (``sph_sp``).

Absorption, at the start of the update: ``mark_kernel`` (SPH particle: thin liquid, i.e. fewer than ``column``
liquid particles stacked above it, so a resting pool keeps its liquid) and ``deposit_kernel`` (world, particles in
index order) for SPH liquid, ``drops_flight.advance_kernel`` (world) for ballistic drops. Each arriving volume ``V``
adds ``V/A`` to ``src_acc`` and sets ``cwet``. Its normal momentum goes to ``body_forces`` (``impulse``, ``moment``).
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import point_velocity
from alloy.plugins.surface_flow.cell_kernels import Species
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.geometry import chart_xform, point_cell
from alloy.plugins.surface_flow.kernels import Cells, Frame
from alloy.plugins.surface_flow.tiles import Tiles
from alloy.solvers.sph.pressure import Particles, is_fluid, nbr_base, nbr_j



@wp.struct
class Edges:
    cell: wp.array[wp.int32]        # [E] global cell with an open face (no drain)
    pos: wp.array[wp.vec3]          # [E] the open face's centre, chart frame
    out: wp.array[wp.vec3]          # [E] unit direction out across it, chart frame
    along: wp.array[wp.vec3]        # [E] unit direction along it
    length: wp.array[wp.float32]    # [E]
    of: wp.array[wp.int32]          # [C] a cell's edge index, -1 for none


@wp.struct
class Pending:
    vol: wp.array2d[wp.float64]     # [W, C] m³ waiting to leave as drops
    mom: wp.array2d[wp.vec3]        # [W, C] its momentum per density, m³·m/s, chart frame
    tile: wp.array2d[wp.int32]      # [W, T] 1: the tile still holds a whole drop
    sp: wp.array3d[wp.float64]      # [W, C, K] kg of species waiting with the volume
    sp_in: wp.array3d[wp.float64]   # [W, C, K] kg the cell's shed film and drips brought into it (book)


@wp.struct
class Requests:
    pos: wp.array2d[wp.vec3]        # [W, R] world
    vel: wp.array2d[wp.vec3]        # [W, R] world
    cell: wp.array2d[wp.int32]      # [W, R] the pending cell each request drains
    n: wp.array[wp.int32]           # [W] requests filled
    tile_n: wp.array2d[wp.int32]    # [W, T] whole drops per tile
    tile_off: wp.array2d[wp.int32]  # [W, T] first slot of each tile
    slot: wp.array2d[wp.int32]      # [W, R] the ballistic pool slot each granted request took (drops_flight.spawn)


@wp.kernel
def collect_kernel(edges: Edges, shed: wp.array2d[wp.float64], shed_mom: wp.array2d[wp.vec3],
                   ctile: wp.array[wp.int32], sp: Species, pend: Pending):
    """Open-edge outflux into pending drops, its species with it; the cell's tile is marked so it is counted even
    after it dried."""
    w, e = wp.tid()
    c = edges.cell[e]
    for k in range(sp.count):
        m = sp.shed[w, c, k]
        pend.sp[w, c, k] = pend.sp[w, c, k] + m
        pend.sp_in[w, c, k] = pend.sp_in[w, c, k] + m
        sp.shed[w, c, k] = wp.float64(0.0)
    v = shed[w, c]
    if v != 0.0:
        pend.vol[w, c] = pend.vol[w, c] + v
        pend.mom[w, c] = pend.mom[w, c] + shed_mom[w, c]
        pend.tile[w, ctile[c]] = 1
        shed[w, c] = wp.float64(0.0)
        shed_mom[w, c] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def drip_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], cells: Cells, frame: Frame, ret: wp.array[wp.float32],
                h_floor: float, decay: float, flooded: wp.array2d[wp.int32], h: wp.array2d[wp.float64],
                dripped: wp.array2d[wp.float64], sp: Species, pend: Pending):
    """``ret[k]``: ``2 sin(θ_r/2) sqrt(σ/ρ)`` of chart k, so ``h_ret = ret / sqrt(g_n)``; ``decay = 1 - exp(-dt/tau)``.
    The drip takes the share ``dv/h`` of the cell's species (booked in ``sp.out``)."""
    w, t = wp.tid()
    if tile_on[w, t] == 0:
        return
    first, pitch = tl.first[t], tl.pitch[t]
    k = cells.chart[first]
    g = frame.g[w, k]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            if flooded[w, c] >= 0:
                continue                                    # under a supply pool: its film joins the pool (DC5)
            gn = wp.dot(g, cells.nrm[c])
            if gn <= 0.0:
                continue                                    # pressed onto its surface: no drip
            h_ret = wp.float64(wp.max(ret[k] / wp.sqrt(gn), h_floor))
            hc = h[w, c]
            if hc <= h_ret:
                continue
            dv = (hc - h_ret) * wp.float64(decay)
            h[w, c] = hc - dv
            vol = dv * cells.area[c]
            dripped[w, c] = dripped[w, c] + vol
            pend.vol[w, c] = pend.vol[w, c] + vol           # it falls from rest in the chart frame (no momentum)
            for s in range(sp.count):
                dm = sp.mass[w, c, s] * (dv / hc)            # kg/m²: the film keeps its concentration
                sp.mass[w, c, s] = sp.mass[w, c, s] - dm
                m = dm * cells.area[c]
                sp.out[w, c, s] = sp.out[w, c, s] + m
                pend.sp[w, c, s] = pend.sp[w, c, s] + m
                pend.sp_in[w, c, s] = pend.sp_in[w, c, s] + m


@wp.func
def _whole(pend: Pending, w: int, c: int, vol: wp.float64) -> int:
    return int(wp.floor(pend.vol[w, c] / vol))


@wp.kernel
def count_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], pend: Pending, vol: wp.float64, req: Requests):
    w, t = wp.tid()
    n = int(0)
    if tile_on[w, t] != 0 or pend.tile[w, t] != 0:
        first, pitch = tl.first[t], tl.pitch[t]
        for r in range(tl.rows[t]):
            for j in range(tl.cols[t]):
                n += _whole(pend, w, first + r * pitch + j, vol)
    req.tile_n[w, t] = n


@wp.kernel
def offset_kernel(tiles: int, cap: int, req: Requests):
    w = wp.tid()
    run = int(0)
    for t in range(tiles):
        req.tile_off[w, t] = run
        run += req.tile_n[w, t]
    req.n[w] = wp.min(run, cap)


@wp.func
def _place(edges: Edges, cells: Cells, h: wp.array2d[wp.float64], w: int, c: int, q: int, n: int,
           r: float) -> wp.vec3:
    """Chart-frame start of the q-th of n drops leaving cell c: off an open face, or under a hanging cell. Drops sit
    2r apart in rows across the face (or the cell), and further rows stand 2r further out, so SPH particles never
    overlap."""
    e = edges.of[c]
    nrm = cells.nrm[c]
    width = wp.sqrt(wp.float32(cells.area[c]))
    if e >= 0:
        width = edges.length[e]
    per = wp.max(int(wp.floor(width / (2.0 * r))), 1)
    row = q // per
    col = q - row * per
    in_row = wp.min(per, n - row * per)
    s = (float(col) - 0.5 * float(in_row - 1)) * 2.0 * r
    out = r + 2.0 * r * float(row)
    if e >= 0:
        return edges.pos[e] + out * edges.out[e] + r * nrm + s * edges.along[e]
    side = wp.normalize(wp.cross(nrm, wp.vec3(0.577, 0.577, 0.577)))
    return cells.pos[c] + (wp.float32(h[w, c]) + out) * nrm + s * side


@wp.kernel
def fill_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], edges: Edges, cells: Cells, charts: wp.array[Chart],
                cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], h: wp.array2d[wp.float64],
                pend: Pending, vol: wp.float64, radius: float, cap: int, req: Requests):
    w, t = wp.tid()
    if req.tile_n[w, t] == 0:
        pend.tile[w, t] = 0
        return
    first, pitch = tl.first[t], tl.pitch[t]
    k = cells.chart[first]
    cb = cbody[w, k]
    X = chart_xform(body_q, cb, clocal[w, k])
    com = wp.vec3(0.0, 0.0, 0.0)
    if cb >= 0:
        com = wp.transform_point(body_q[cb], body_com[cb])
    slot = req.tile_off[w, t]
    left = int(0)
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            n = _whole(pend, w, c, vol)
            if n == 0:
                continue
            v_c = pend.mom[w, c] / wp.float32(pend.vol[w, c])
            for q in range(n):
                if slot >= cap:
                    left = 1
                    break
                x = wp.transform_point(X, _place(edges, cells, h, w, c, q, n, radius))
                v = wp.transform_vector(X, v_c)
                if cb >= 0:
                    v = v + point_velocity(body_qd, com, cb, x)
                req.pos[w, slot] = x
                req.vel[w, slot] = v
                req.cell[w, slot] = c
                slot += 1
    pend.tile[w, t] = left


@wp.func
def _carry(drop_sp: wp.array3d[wp.float64], sph_sp: wp.array2d[wp.float64], carry: int, w: int, slot: int, k: int,
           m: wp.float64):
    """A leaving drop's species: into its ballistic pool slot, or booked lost with an SPH particle."""
    if carry != 0:
        drop_sp[w, slot, k] = m
    else:
        sph_sp[w, k] = sph_sp[w, k] + m


@wp.kernel
def commit_kernel(req: Requests, granted: wp.array[wp.int32], vol: wp.float64, cells: Cells, pend: Pending,
                  lump_pend: wp.array2d[wp.float64], lump_sp: wp.array3d[wp.float64],
                  lump_drop_sp: wp.array3d[wp.float64], n_sp: int, carry: int, drop_sp: wp.array3d[wp.float64],
                  sph_sp: wp.array2d[wp.float64], out_total: wp.array[wp.float64], lump_out: wp.array[wp.float64]):
    """Per world: the first granted[w] requests left; their volume and momentum leave their cells, each with its
    share of the cell's waiting species (or their lumps, with their share of the lump's, booked in ``lump_drop_sp``).
    A ballistic drop, ``carry``, takes it along in its pool slot's ``drop_sp``; an SPH particle cannot, so it is
    booked lost in ``sph_sp`` [W, K]. A refused request's tile keeps ``pend.tile``, so it is counted again while
    dry."""
    w = wp.tid()
    n = req.n[w]
    g = wp.min(granted[w], n)
    for s in range(g):
        c = req.cell[w, s]
        out_total[w] = out_total[w] + vol
        if c < 0:                                       # a lump's drop: its share of the waiting species leaves
            b = -c - 1
            share = vol / lump_pend[w, b]
            for k in range(n_sp):
                gone = lump_sp[w, b, k] * share
                lump_sp[w, b, k] = lump_sp[w, b, k] - gone
                lump_drop_sp[w, b, k] = lump_drop_sp[w, b, k] + gone
                _carry(drop_sp, sph_sp, carry, w, req.slot[w, s], k, gone)
            lump_pend[w, b] = lump_pend[w, b] - vol
            lump_out[w] = lump_out[w] + vol
            continue
        v0 = pend.vol[w, c]
        keep = wp.float32((v0 - vol) / v0)
        for k in range(n_sp):
            gone = pend.sp[w, c, k] * (vol / v0)
            pend.sp[w, c, k] = pend.sp[w, c, k] - gone
            _carry(drop_sp, sph_sp, carry, w, req.slot[w, s], k, gone)
        pend.vol[w, c] = v0 - vol
        pend.mom[w, c] = keep * pend.mom[w, c]
    for s in range(g, n):
        if req.cell[w, s] >= 0:
            pend.tile[w, cells.tile[req.cell[w, s]]] = 1


@wp.struct
class Arrive:
    """Where arriving liquid goes: the film's inflow, pending momentum and wet flags, the body-force impulse store,
    the ledger."""
    src_acc: wp.array2d[wp.float64]
    src_mom: wp.array2d[wp.vec3]        # [W, C] pending momentum per area [N·s/m²], chart frame (the film applies it)
    cwet: wp.array2d[wp.int32]
    h: wp.array2d[wp.float64]
    area: wp.array[wp.float64]
    impulse: wp.array2d[wp.vec3]
    moment: wp.array2d[wp.vec3]
    total: wp.array[wp.float64]
    max_depth: float


@wp.func
def arrive(a: Arrive, w: int, k: int, c: int, vol: wp.float64, m: float, v_rel: wp.vec3, n: wp.vec3,
           x: wp.vec3, X: wp.transform) -> bool:
    """``vol`` [m³] of liquid (mass m, velocity v_rel relative to the chart, at world x, outward normal n) joins
    cell c of chart k (frame X), unless the cell would exceed ``max_depth``. Its normal momentum goes to the chart's
    body (the impulse store); its tangential momentum joins the film (``src_mom``, chart frame)."""
    add = vol / a.area[c]
    if a.h[w, c] + a.src_acc[w, c] + add > wp.float64(a.max_depth):
        return False
    a.src_acc[w, c] = a.src_acc[w, c] + add
    a.cwet[w, c] = 1
    J = (m * wp.dot(v_rel, n)) * n
    J_t = wp.transform_vector(wp.transform_inverse(X), m * v_rel - J)
    a.src_mom[w, c] = a.src_mom[w, c] + J_t / wp.float32(a.area[c])
    a.impulse[w, k] = a.impulse[w, k] + J
    a.moment[w, k] = a.moment[w, k] + wp.cross(x, J)
    a.total[w] = a.total[w] + vol
    return True


@wp.func
def column_above(p: Particles, i: int, x: wp.vec3, n: wp.vec3, spacing: float) -> int:
    """SPH liquid neighbours of i stacked above it along n (within one spacing of the normal line, at least half a
    spacing up): 0 for a sheet one particle deep, 1 or more under a pool (rows hold neighbours closer than h = 2
    spacings, so the second particle up is counted only when nearer than h)."""
    b = nbr_base(p, i)
    n_up = int(0)
    for k in range(p.count[i]):
        j = nbr_j(p, b, i, k)
        if not is_fluid(p, j):
            continue
        d = p.q[j] - x
        up = wp.dot(d, n)
        if up > 0.5 * spacing and wp.length(d - up * n) < spacing:
            n_up += 1
    return n_up


@wp.kernel
def mark_kernel(p: Particles, column: int, spacing: float, charts: wp.array[Chart], prof: wp.array[wp.vec3],
                cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], h: wp.array2d[wp.float64],
                g_chart: wp.array2d[wp.vec3], tick: wp.array[wp.int32], gravity: wp.vec3, reach: float,
                u_away: float, mask: wp.array[wp.int32], at: wp.array[wp.vec2i]):
    """Thin SPH liquid (fewer than ``column`` liquid particles stacked above it) within ``reach`` of a chart's film top
    that the chart holds: not moving away from it faster than ``u_away``, and either pressed onto it by the chart
    frame's gravity (``g_chart``, chart frame; before the film's first exchange, ``gravity`` turned into the chart
    frame) or arriving at ``u_away`` or faster. Mask 1, at = (chart, cell).
    A pool resting on a chart keeps its liquid; a drop leaving a hanging film is not taken back (WP15 review)."""
    i = wp.tid()
    mask[i] = 0
    w = p.world[i]
    if w < 0 or not is_fluid(p, i):
        return
    x = p.q[i]
    for k in range(charts.shape[0]):
        cb = cbody[w, k]
        X = chart_xform(body_q, cb, clocal[w, k])
        near = point_cell(charts[k], prof, wp.transform_point(wp.transform_inverse(X), x))
        if near.cell < 0:
            continue
        gap = near.dist - wp.float32(h[w, near.cell])
        if gap > reach or gap < -reach:
            continue
        n = wp.transform_vector(X, near.normal)
        v = p.qd[i]
        if cb >= 0:
            v = v - point_velocity(body_qd, wp.transform_point(body_q[cb], body_com[cb]), cb, x)
        v_n = wp.dot(v, n)                                  # > 0: moving away from the chart
        if v_n > u_away:
            continue
        g = g_chart[w, k]
        if tick[w] == 0:
            g = wp.transform_vector(wp.transform_inverse(X), gravity)
        if wp.dot(g, near.normal) >= 0.0 and -v_n < u_away:
            continue                                        # gravity does not hold it on and it is not arriving
        if column_above(p, i, x, n, spacing) >= column:
            return
        mask[i] = 1
        at[i] = wp.vec2i(k, near.cell)
        return


@wp.kernel
def deposit_kernel(ws: wp.array[wp.int32], q: wp.array[wp.vec3], qd: wp.array[wp.vec3],
                   mass: wp.array[wp.float32], vol: wp.float64, charts: wp.array[Chart], prof: wp.array[wp.vec3],
                   cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                   body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], at: wp.array[wp.vec2i],
                   a: Arrive, mask: wp.array[wp.int32]):
    """Per world, in particle order: each marked particle's volume joins its cell (or it stays liquid: mask 0)."""
    w = wp.tid()
    for i in range(ws[w], ws[w + 1]):
        if mask[i] == 0:
            continue
        k = at[i][0]
        c = at[i][1]
        cb = cbody[w, k]
        X = chart_xform(body_q, cb, clocal[w, k])
        x = q[i]
        near = point_cell(charts[k], prof, wp.transform_point(wp.transform_inverse(X), x))
        v = qd[i]
        if cb >= 0:
            v = v - point_velocity(body_qd, wp.transform_point(body_q[cb], body_com[cb]), cb, x)
        if not arrive(a, w, k, c, vol, mass[i], v, wp.transform_vector(X, near.normal), x, X):
            mask[i] = 0


__all__ = ["Arrive", "Edges", "Pending", "Requests", "collect_kernel", "commit_kernel", "count_kernel",
           "deposit_kernel", "drip_kernel", "fill_kernel", "mark_kernel", "offset_kernel"]
