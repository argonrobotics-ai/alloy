# SPDX-License-Identifier: Apache-2.0
"""The jump deposit of solid jets, parallel over the rows of each jet's window (``supply_deposit.py`` has the model).

One thread per world walking a jet's whole window (thousands of cells, four passes) left the GPU latency-bound: a
faucet over a 100 x 100 film cost 8.3 ms per exchange at 64 worlds. Here each jet runs as a fixed sequence of
launches, jets one after another in a fixed order:

1. ``jet_setup`` [W]: the jet's flow, hit, R_J (capillary-viscous), window and fold span;
2. ``jet_bands`` [W, rows]: per row, the film's depth by distance band (before the sweep);
3. ``jet_radius`` [W]: the bands summed in row order; where the jump stands (drowning);
4. ``jet_ring`` [W, rows]: per row, the ring weights and the farthest in-reach cell;
5. ``jet_fallback`` [W]: ring off the chart or past a fold?;
6. ``jet_sweep`` [W, rows]: sweeps the film under the sheet, per-row swept volume and ring sums;
7. ``jet_finish`` [W]: the volume to enter, u_post, impact pressure, the ledger, the on-chart flag;
8. ``jet_enter`` [W, rows]: the ring's sources and traction.

Every element is written by one thread, and every sum is taken in row order, so runs stay deterministic and
graph-capturable, with no atomics. The physics is unchanged.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.geometry import cell_window, chart_xform
from alloy.plugins.surface_flow.supply_deposit import (N_BANDS, bands, cell_of, cover_of, drowned_at, jump_radius,
                                                       oblique_split, ring_step, ring_weight, row_span, surf_dist,
                                                       union_box)
from alloy.plugins.surface_flow.supply_kernels import CHART, SWEPT, NozzleSpec, weight_sum
from alloy.plugins.surface_flow.supply_species import NetSpecies, nozzle_conc


@wp.struct
class Jet:
    on: wp.int32              # 1: a jet with flow on a chart this exchange; 2: finished at its hit cell; 0: none
    chart: wp.int32
    cell: wp.int32
    i_hit: wp.int32
    win: wp.vec4i             # rows i0..i1 (fold-limited), columns j0..j1 (unwrapped)
    p: wp.vec3                # hit, chart frame
    wl: wp.vec3               # impact velocity, chart frame
    n_hit: wp.vec3
    t_hat: wp.vec3
    q: wp.float64
    a: wp.float32
    speed: wp.float32
    r_b: wp.float32
    r_j: wp.float32
    wr: wp.float32
    g_n: wp.float32
    g: wp.float32
    cos_b: wp.float32
    fallback: wp.int32
    d_far: wp.float32
    sum_w: wp.float64
    vol: wp.float64
    sm: wp.vec4d              # species mass swept from under the sheet [kg], re-entered at the ring
    u_post: wp.float32
    r_fp: wp.float32          # the footprint's equal-area radius a·sqrt(speed/w_n) (π·a²/cos(incidence))
    p_fp: wp.float32          # its impact pressure on the cells within r_fp (0: all on the hit cell)
    s_fp: wp.float32          # their impact share: the footprint over those cells' area


@wp.kernel
def jet_setup(r: int, body_q: wp.array[wp.transform], charts: wp.array[Chart],
              prof: wp.array[wp.vec3], cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
              cnrm: wp.array[wp.vec3], nozzles: wp.array[NozzleSpec], ray_noz: wp.array[wp.int32],
              ray_wt: wp.array[wp.float32], discharge: wp.array2d[wp.float64], hit_kind: wp.array2d[wp.int32],
              hit_idx: wp.array2d[wp.int32], hit_cell: wp.array2d[wp.int32], hit_p: wp.array2d[wp.vec3],
              hit_w: wp.array2d[wp.vec3], gravity: wp.vec3, jet: wp.array[Jet], written: wp.array2d[wp.vec4i]):
    """The jet's flow, hit and jump radius; its window (fold-limited) joins the box of cells written this exchange."""
    w = wp.tid()
    s = Jet()
    s.on = 0
    noz = nozzles[ray_noz[r]]
    q_all = -discharge[w, noz.edge]
    if q_all <= wp.float64(0.0) or hit_kind[w, r] != CHART:
        jet[w] = s
        return
    k = hit_idx[w, r]
    chart = charts[k]
    X = chart_xform(body_q, cbody[w, k], clocal[w, k])
    s.chart = k
    s.cell = hit_cell[w, r]
    s.p = hit_p[w, r]
    s.wl = wp.transform_vector(wp.transform_inverse(X), hit_w[w, r])
    s.q = q_all * wp.float64(noz.efficiency) * wp.float64(ray_wt[r]) / weight_sum(noz, ray_wt)
    s.speed = wp.max(wp.length(s.wl), 1.0e-6)
    s.a = wp.sqrt(float(s.q) / (3.14159265 * s.speed))
    s.r_b = jump_radius(noz, float(s.q), s.a)
    s.wr = ring_step(chart)
    s.n_hit = cnrm[s.cell]
    s.r_fp = s.a * wp.sqrt(s.speed / wp.max(-wp.dot(s.wl, s.n_hit), 1.0e-6))
    w_t = s.wl - wp.dot(s.wl, s.n_hit) * s.n_hit
    s.cos_b = wp.min(wp.length(w_t) / s.speed, 0.995)
    s.t_hat = wp.vec3(1.0, 0.0, 0.0)
    if wp.length(w_t) > 1.0e-9:
        s.t_hat = wp.normalize(w_t)
    s.g_n = wp.abs(wp.dot(gravity, wp.transform_vector(X, s.n_hit)))
    s.g = wp.length(gravity)
    loc = s.cell - chart.cell_start
    s.i_hit = loc // chart.ny
    span = row_span(chart, prof, s.i_hit)
    win = cell_window(chart, prof, s.i_hit, loc % chart.ny, s.r_b + 2.0 * s.wr)
    s.win = wp.vec4i(wp.max(win[0], span[0]), wp.min(win[1], span[1]), win[2], win[3])
    s.on = 1
    jet[w] = s
    written[w, k] = union_box(chart, written[w, k], s.win)


@wp.kernel
def jet_bands(charts: wp.array[Chart], prof: wp.array[wp.vec3], cpos: wp.array[wp.vec3],
              area: wp.array[wp.float64], h: wp.array2d[wp.float64], jet_cover: wp.array2d[wp.float32],
              jet: wp.array[Jet], part_h: wp.array3d[wp.float64], part_a: wp.array3d[wp.float64]):
    """Row ``ii`` of the window: the film's depth·area and area by distance band (cells under another sheet out)."""
    w, ii = wp.tid()
    s = jet[w]
    i = s.win[0] + ii
    if s.on != 1 or i > s.win[1]:
        return
    chart = charts[s.chart]
    bw = wp.max(s.wr, (s.r_b + s.wr) / float(N_BANDS))
    bh = bands()
    ba = bands()
    for j in range(s.win[2], s.win[3] + 1):
        c = cell_of(chart, i, j)
        d = surf_dist(chart, prof, cpos, c, i, j, s.p, s.i_hit)
        if d <= s.r_b + s.wr and jet_cover[w, c] <= 0.0:
            kb = wp.min(int(d / bw), N_BANDS - 1)
            bh[kb] = bh[kb] + h[w, c] * area[c]
            ba[kb] = ba[kb] + area[c]
    for kb in range(N_BANDS):
        part_h[w, ii, kb] = bh[kb]
        part_a[w, ii, kb] = ba[kb]


@wp.kernel
def jet_radius(jet: wp.array[Jet], part_h: wp.array3d[wp.float64], part_a: wp.array3d[wp.float64]):
    """The bands summed over the rows in order, and where the jump stands."""
    w = wp.tid()
    s = jet[w]
    if s.on != 1:
        return
    bh = bands()
    ba = bands()
    for ii in range(s.win[1] - s.win[0] + 1):
        for kb in range(N_BANDS):
            bh[kb] = bh[kb] + part_h[w, ii, kb]
            ba[kb] = ba[kb] + part_a[w, ii, kb]
    bw = wp.max(s.wr, (s.r_b + s.wr) / float(N_BANDS))
    s.r_j = wp.max(drowned_at(bh, ba, bw, s.r_b, float(s.q) / (3.14159265 * s.a), s.g_n), s.a)
    jet[w] = s


@wp.kernel
def jet_ring(charts: wp.array[Chart], prof: wp.array[wp.vec3], cpos: wp.array[wp.vec3], area: wp.array[wp.float64],
             jet: wp.array[Jet], part_far: wp.array2d[wp.float32], part_w: wp.array2d[wp.float64]):
    """Row ``ii``: its ring weight and its farthest in-reach cell."""
    w, ii = wp.tid()
    s = jet[w]
    i = s.win[0] + ii
    if s.on != 1 or i > s.win[1]:
        return
    chart = charts[s.chart]
    far = float(0.0)
    sw = wp.float64(0.0)
    for j in range(s.win[2], s.win[3] + 1):
        c = cell_of(chart, i, j)
        d = surf_dist(chart, prof, cpos, c, i, j, s.p, s.i_hit)
        if d <= s.r_j + s.wr:
            far = wp.max(far, d)
        sw = sw + wp.float64(ring_weight(d, s.r_j, s.wr, 0, 0.0) *
                             oblique_split(cpos[c] - s.p, s.n_hit, s.t_hat, s.cos_b)) * area[c]
    part_far[w, ii] = far
    part_w[w, ii] = sw


@wp.kernel
def jet_fallback(jet: wp.array[Jet], part_far: wp.array2d[wp.float32], part_w: wp.array2d[wp.float64]):
    w = wp.tid()
    s = jet[w]
    if s.on != 1:
        return
    far = float(0.0)
    sw = wp.float64(0.0)
    for ii in range(s.win[1] - s.win[0] + 1):
        far = wp.max(far, part_far[w, ii])
        sw = sw + part_w[w, ii]
    s.d_far = far
    s.sum_w = sw
    s.fallback = 0
    if sw <= wp.float64(0.0):                          # the ring is off the chart or past a fold
        s.fallback = 1
        s.sum_w = wp.float64(0.0)
    jet[w] = s


@wp.kernel
def jet_sweep(charts: wp.array[Chart], prof: wp.array[wp.vec3], cpos: wp.array[wp.vec3], area: wp.array[wp.float64],
              nozzles: wp.array[NozzleSpec], ray_noz: wp.array[wp.int32], r: int, jet: wp.array[Jet],
              h: wp.array2d[wp.float64], jet_cover: wp.array2d[wp.float32], jet_tau: wp.array2d[wp.float32],
              cwet: wp.array2d[wp.int32], n_sp: int, mass: wp.array3d[wp.float64], part_swept: wp.array2d[wp.float64],
              part_w: wp.array2d[wp.float64], part_wh: wp.array2d[wp.float64], part_sm: wp.array2d[wp.vec4d],
              part_fa: wp.array2d[wp.float64]):
    """Row ``ii``: sweep the film under the sheet, its species with it (m·cover); the row's swept volume and species
    mass, ring weight (fallback), weight·depth and the area of its cells in the jet's footprint."""
    w, ii = wp.tid()
    s = jet[w]
    i = s.win[0] + ii
    if s.on != 1 or i > s.win[1]:
        return
    chart = charts[s.chart]
    noz = nozzles[ray_noz[r]]
    swept = wp.float64(0.0)
    sw = wp.float64(0.0)
    swh = wp.float64(0.0)
    sm = wp.vec4d()
    fa = wp.float64(0.0)
    for j in range(s.win[2], s.win[3] + 1):
        c = cell_of(chart, i, j)
        d = surf_dist(chart, prof, cpos, c, i, j, s.p, s.i_hit)
        if d <= s.r_fp:
            fa = fa + area[c]
        cov = cover_of(d, s.r_j, s.wr, s.fallback, s.d_far)
        if cov > 0.0:
            v = h[w, c] * wp.float64(cov)
            h[w, c] = h[w, c] - v
            if v > wp.float64(0.0):                                # a dry cell stays skipped (and keeps its residue)
                cwet[w, c] = 1
                for k in range(n_sp):                              # the species go with the film they are in
                    dm = mass[w, c, k] * wp.float64(cov)
                    mass[w, c, k] = mass[w, c, k] - dm
                    sm[k] = sm[k] + dm * area[c]
            swept = swept + v * area[c]
            jet_cover[w, c] = wp.max(jet_cover[w, c], cov)
            re = s.speed * wp.max(d, s.a) / noz.nu                 # Blasius on the radial sheet (estimate)
            jet_tau[w, c] = wp.max(jet_tau[w, c], 0.332 * noz.rho * s.speed * s.speed / wp.sqrt(re))
        wt = wp.float64(ring_weight(d, s.r_j, s.wr, s.fallback, s.d_far) *
                        oblique_split(cpos[c] - s.p, s.n_hit, s.t_hat, s.cos_b)) * area[c]
        sw = sw + wt
        swh = swh + wt * h[w, c]
    part_swept[w, ii] = swept
    part_w[w, ii] = sw
    part_wh[w, ii] = swh
    part_sm[w, ii] = sm
    part_fa[w, ii] = fa


@wp.kernel
def jet_finish(r: int, dt: wp.float64, nozzles: wp.array[NozzleSpec], ray_noz: wp.array[wp.int32],
               area: wp.array[wp.float64], jet: wp.array[Jet], part_swept: wp.array2d[wp.float64],
               part_w: wp.array2d[wp.float64], part_wh: wp.array2d[wp.float64], part_sm: wp.array2d[wp.vec4d],
               part_fa: wp.array2d[wp.float64], n_sp: int, src_acc: wp.array2d[wp.float64],
               src_sp: wp.array3d[wp.float64], impact_p: wp.array2d[wp.float32],
               impact_share: wp.array2d[wp.float32], cwet: wp.array2d[wp.int32], ledger: wp.array2d[wp.float64],
               swept_sp: wp.array2d[wp.float64], hit_on: wp.array2d[wp.float32], ns: NetSpecies,
               nconc: wp.array[wp.vec4d], nnode: wp.array[wp.int32]):
    """The volume to enter (the jet's flow and the swept film) and the swept species, u_post (at least the Froude-0.3
    depth), the impact pressure and share (impact_p is footprint-weighted: above p_stag on a cell whose share exceeds
    1, so Σ impact_p·A is exact), the ledgers and the on-chart flag of the jet's push."""
    w = wp.tid()
    s = jet[w]
    if s.on != 1:
        return
    noz = nozzles[ray_noz[r]]
    swept = wp.float64(0.0)
    sw = wp.float64(0.0)
    swh = wp.float64(0.0)
    sm = wp.vec4d()
    fa = wp.float64(0.0)
    for ii in range(s.win[1] - s.win[0] + 1):
        swept = swept + part_swept[w, ii]
        sw = sw + part_w[w, ii]
        swh = swh + part_wh[w, ii]
        sm = sm + part_sm[w, ii]
        fa = fa + part_fa[w, ii]
    ledger[w, SWEPT] = ledger[w, SWEPT] + swept
    for k in range(n_sp):
        swept_sp[w, k] = swept_sp[w, k] + sm[k]
        sm[k] = sm[k] + s.q * dt * nozzle_conc(ns, w, nconc[ray_noz[r]], nnode[ray_noz[r]], k)   # the jet's own
    s.sm = sm
    s.vol = s.q * dt + swept
    hit_on[w, r] = 1.0 - float(s.fallback)
    wn = wp.max(-wp.dot(s.wl, s.n_hit), 0.0)
    fp = 3.14159265 * s.r_fp * s.r_fp                  # the footprint π·a²/cos(incidence)
    force = 0.5 * noz.rho * wn * wn * fp               # ½ρw_n² over it
    s.p_fp = 0.0                                       # every jet and spray adds to impact_p (sprays run first)
    s.s_fp = 0.0
    if wn <= 1.0e-3 * s.speed:                         # grazing: no impact (its footprint runs off to infinity)
        fa = wp.float64(0.0)
        force = 0.0
        fp = 0.0
    if fa > wp.float64(0.0) and sw > wp.float64(0.0):  # spread over the cells within r_fp (jet_enter), ∫p dA kept
        s.p_fp = float(wp.float64(force) / fa)
        s.s_fp = float(wp.float64(fp) / fa)
    else:                                              # a footprint smaller than a cell: on the hit cell
        impact_p[w, s.cell] = impact_p[w, s.cell] + float(wp.float64(force) / area[s.cell])
        impact_share[w, s.cell] = impact_share[w, s.cell] + float(wp.float64(fp) / area[s.cell])
    if sw <= wp.float64(0.0):                          # nothing in reach (cannot happen with the hit cell)
        src_acc[w, s.cell] = src_acc[w, s.cell] + s.vol / area[s.cell]
        for k in range(n_sp):
            src_sp[w, s.cell, k] = src_sp[w, s.cell, k] + sm[k] / area[s.cell]
        cwet[w, s.cell] = 1
        hit_on[w, r] = 0.0
        s.on = 2
        jet[w] = s
        return
    s.sum_w = sw
    h_fr = wp.pow(float(s.q) / (6.28318531 * s.r_j * noz.froude * wp.sqrt(s.g)), 2.0 / 3.0)
    h_ring = wp.max(wp.max(float(swh / sw), h_fr), noz.h_min)
    s.u_post = wp.min(float(s.q) / (6.28318531 * s.r_j * h_ring), noz.u_max)
    jet[w] = s


@wp.kernel
def jet_enter(charts: wp.array[Chart], prof: wp.array[wp.vec3], cpos: wp.array[wp.vec3], cnrm: wp.array[wp.vec3],
              area: wp.array[wp.float64], nozzles: wp.array[NozzleSpec], ray_noz: wp.array[wp.int32], r: int,
              jet: wp.array[Jet], n_sp: int, src_acc: wp.array2d[wp.float64],
              src_sp: wp.array3d[wp.float64], src_mom: wp.array2d[wp.vec3], cwet: wp.array2d[wp.int32],
              impact_p: wp.array2d[wp.float32], impact_share: wp.array2d[wp.float32]):
    """Row ``ii``: the ring's share of the volume and of the swept species into the pending volume and species
    (cells marked wet), with its outward post-jump momentum ρ·dv·u_post into the pending momentum ``src_mom``
    [N·s/m²] (the film applies it over its step: exact on strided films)."""
    w, ii = wp.tid()
    s = jet[w]
    i = s.win[0] + ii
    if s.on != 1 or i > s.win[1]:
        return
    chart = charts[s.chart]
    noz = nozzles[ray_noz[r]]
    for j in range(s.win[2], s.win[3] + 1):
        c = cell_of(chart, i, j)
        rel = cpos[c] - s.p
        d = surf_dist(chart, prof, cpos, c, i, j, s.p, s.i_hit)
        if s.p_fp > 0.0 and d <= s.r_fp:                   # the impact pressure over the footprint, and its share
            impact_p[w, c] = impact_p[w, c] + s.p_fp
            impact_share[w, c] = impact_share[w, c] + s.s_fp
        wt = wp.float64(ring_weight(d, s.r_j, s.wr, s.fallback, s.d_far) *
                        oblique_split(rel, s.n_hit, s.t_hat, s.cos_b)) * area[c]
        if wt > wp.float64(0.0):
            dv = s.vol * wt / (s.sum_w * area[c])
            src_acc[w, c] = src_acc[w, c] + dv
            for k in range(n_sp):
                src_sp[w, c, k] = src_sp[w, c, k] + s.sm[k] * wt / (s.sum_w * area[c])
            cwet[w, c] = 1
            n = cnrm[c]
            tan = rel - wp.dot(rel, n) * n
            if wp.length(tan) > 1.0e-9:
                src_mom[w, c] = src_mom[w, c] + noz.rho * float(dv) * s.u_post * wp.normalize(tan)


__all__ = ["Jet", "jet_bands", "jet_enter", "jet_fallback", "jet_finish", "jet_radius", "jet_ring", "jet_setup",
           "jet_sweep"]
