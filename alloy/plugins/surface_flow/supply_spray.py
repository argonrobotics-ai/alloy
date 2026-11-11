# SPDX-License-Identifier: Apache-2.0
"""Spray deposits (fan, cone, aerosol), parallel: each ray's footprint and normaliser in its own thread, then a
gather over the union of the footprints, one thread per row, each cell summing its share from every ray in ray order.

A spray ray's flow q lands over a cone kernel of radius ``s·footprint`` around its hit. The serial splat this
replaces ran one thread per world over every ray's window: a 64-ray aerosol cost 3.3 ms per exchange at 64 worlds.
Each cell is written by one thread and its rays are summed in a fixed order, so runs stay deterministic and
graph-capturable, with no atomics; the result is bit-identical to the serial splat. Rows past a fold are out of a
ray's footprint, as for jets.

A stage's drip (supply_stage) of volume V = q·dt lands as GA's gob (concrete_finishing/coupling.py:978-986, WP6's
trowel): a cone of R = (3V/π)^(1/3) rounded up to whole cells (at least 2), each cell weighed max(1 - r/R, 0) by its
centre's distance r from the hit. On a plane chart it is normalised over the chart's lattice continued past its edges,
so a cone the edge cuts lays no thicker than a whole one, and the share past the edge (``q_off``) is booked ``lost``
(``spray_off``); on a revolution chart over the cells it reaches (rows within its fold span), nothing lost. A drip with
R at most one cell (under π·cell³/3: 1.05 mL on 10 mm cells; paint, rinse water) lands whole in the cell it strikes.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.charts import PLANE, Chart
from alloy.plugins.surface_flow.geometry import cell_window, chart_xform
from alloy.plugins.surface_flow.supply_deposit import cell_of, ring_step, row_span, surf_dist, union_box
from alloy.plugins.surface_flow.supply_kernels import CHART, JET, LOST, NozzleSpec, weight_sum
from alloy.plugins.surface_flow.supply_species import SP_LOST, SP_SENT, NetSpecies, nozzle_conc


@wp.struct
class Spray:
    on: wp.int32              # 1: spreads over its footprint; 2: all at its hit cell; 0: none
    chart: wp.int32
    cell: wp.int32
    i_hit: wp.int32
    win: wp.vec4i             # rows (fold-limited), columns (unwrapped)
    p: wp.vec3                # hit, chart frame
    radius: wp.float32
    q: wp.float64
    wn: wp.float32            # normal impact speed
    rho: wp.float32
    sum_w: wp.float64
    conc: wp.vec4d            # film species in the spray [kg/m³]
    noz: wp.int32             # its nozzle (rays of one nozzle are consecutive)
    q_shadow: wp.float64      # its flow hidden from its nozzle by an occluder (supply_shadow), and that occluder slot
    blocker: wp.int32
    bp: wp.vec3               # where the segment to its most-weighted hidden cell meets that occluder (world)
    q_off: wp.float64         # a stage drip's cone past its plane chart's edge (booked lost by spray_off)


@wp.func
def in_columns(chart: Chart, j: int, j0: int, j1: int) -> int:
    """1 when column ``j`` lies in the window ``j0..j1`` (unwrapped, at most one turn on a revolution chart)."""
    if chart.wrap != 0:
        dj = ((j - j0) % chart.ny + chart.ny) % chart.ny
        return int(dj <= j1 - j0)
    return int(j >= j0 and j <= j1)


@wp.func
def cone_cells(d: float, rc: float, a: wp.float64) -> wp.float64:
    """A cone kernel's weight·area at distance d from its centre (radius rc)."""
    return wp.float64(wp.max(1.0 - d / rc, 0.0)) * a


@wp.func
def drip_radius(step: float, v: wp.float64) -> float:
    """GA's gob radius for a drop of volume v: (3v/π)^(1/3) rounded up to whole cells (so at least 2: GA's floor); 0
    when that is at most one cell (the drop lands whole in its cell)."""
    r = float(wp.pow(wp.float64(3.0) * v / wp.float64(3.14159265358979), wp.float64(1.0) / wp.float64(3.0)))
    if r <= step:
        return 0.0
    return wp.ceil(r / step) * step


@wp.func
def off_chart(chart: Chart, p: wp.vec3, i_hit: int, j_hit: int, rc: float, a: wp.float64) -> wp.float64:
    """Σ weight·area of a cone about ``p`` over a plane chart's lattice continued past its edges (the cells off it)."""
    ri = int(wp.ceil(rc / chart.step[0])) + 1
    rj = int(wp.ceil(rc / chart.step[1])) + 1
    off = wp.float64(0.0)
    for i in range(i_hit - ri, i_hit + ri + 1):
        for j in range(j_hit - rj, j_hit + rj + 1):
            if i >= 0 and i < chart.nx and j >= 0 and j < chart.ny:
                continue
            x = wp.vec3(-chart.half[0] + (float(i) + 0.5) * chart.step[0],
                        -chart.half[1] + (float(j) + 0.5) * chart.step[1], 0.0)
            off = off + cone_cells(wp.length(x - p), rc, a)
    return off


@wp.kernel
def spray_rays(dt: wp.float64, body_q: wp.array[wp.transform], charts: wp.array[Chart], prof: wp.array[wp.vec3],
               cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], cpos: wp.array[wp.vec3],
               cnrm: wp.array[wp.vec3], area: wp.array[wp.float64], nozzles: wp.array[NozzleSpec],
               ray_noz: wp.array[wp.int32], ray_wt: wp.array[wp.float32], ray_fp: wp.array[wp.float32],
               discharge: wp.array2d[wp.float64], hit_kind: wp.array2d[wp.int32], hit_idx: wp.array2d[wp.int32],
               hit_cell: wp.array2d[wp.int32], hit_p: wp.array2d[wp.vec3], hit_w: wp.array2d[wp.vec3],
               hit_s: wp.array2d[wp.float32], ns: NetSpecies, nconc: wp.array[wp.vec4d], nnode: wp.array[wp.int32],
               spray: wp.array2d[Spray]):
    """One spray ray of one world: its flow, footprint radius (its angular share at its flight distance, at least a
    cell), fold-limited window and Σ weight·area over it. A stage's drip (supply_stage) lands as GA's cone, or whole in
    the cell it strikes (``on`` 2; module docstring): no jump, no sweep, whatever the drop's rate over the exchange."""
    w, r = wp.tid()
    s = Spray()
    s.on = 0
    noz = nozzles[ray_noz[r]]
    q_all = -discharge[w, noz.edge]
    if (noz.pattern == JET and noz.drip != 2) or q_all <= wp.float64(0.0) or hit_kind[w, r] != CHART:
        spray[w, r] = s
        return
    k = hit_idx[w, r]
    chart = charts[k]
    X = chart_xform(body_q, cbody[w, k], clocal[w, k])
    wl = wp.transform_vector(wp.transform_inverse(X), hit_w[w, r])
    s.chart = k
    s.cell = hit_cell[w, r]
    s.p = hit_p[w, r]
    s.noz = ray_noz[r]
    s.q = q_all * wp.float64(noz.efficiency) * wp.float64(ray_wt[r]) / weight_sum(noz, ray_wt)
    for kk in range(ns.n_film):
        s.conc[kk] = nozzle_conc(ns, w, nconc[ray_noz[r]], nnode[ray_noz[r]], kk)
    s.wn = wp.max(-wp.dot(wl, cnrm[s.cell]), 0.0)
    s.rho = noz.rho
    step = wp.min(chart.step[0], chart.step[1])
    if chart.kind != PLANE:
        step = chart.step[0]
    s.radius = wp.max(hit_s[w, r] * ray_fp[r], step)
    loc = s.cell - chart.cell_start
    s.i_hit = loc // chart.ny
    if noz.drip == 2:                                     # a stage's drip: GA's cone, or all into the cell it strikes
        s.radius = drip_radius(step, s.q * dt)
        if s.radius <= 0.0:
            s.win = wp.vec4i(s.i_hit, s.i_hit, loc % chart.ny, loc % chart.ny)
            s.sum_w = wp.float64(1.0)
            s.on = 2
            spray[w, r] = s
            return
    span = row_span(chart, prof, s.i_hit)
    win = cell_window(chart, prof, s.i_hit, loc % chart.ny, s.radius + ring_step(chart))
    s.win = wp.vec4i(wp.max(win[0], span[0]), wp.min(win[1], span[1]), win[2], win[3])
    sw = wp.float64(0.0)
    for i in range(s.win[0], s.win[1] + 1):
        for j in range(s.win[2], s.win[3] + 1):
            c = cell_of(chart, i, j)
            d = surf_dist(chart, prof, cpos, c, i, j, s.p, s.i_hit)
            sw = sw + wp.float64(wp.max(1.0 - d / s.radius, 0.0)) * area[c]
    s.sum_w = sw
    s.on = 1
    if sw <= wp.float64(0.0):
        s.on = 2
    elif noz.drip == 2 and chart.kind == PLANE:          # the cone past the chart's edge: lost, not laid inside
        off = off_chart(chart, s.p, s.i_hit, loc % chart.ny, s.radius, area[s.cell])
        s.sum_w = sw + off
        s.q_off = s.q * off / s.sum_w
    spray[w, r] = s


@wp.kernel
def spray_boxes(charts: wp.array[Chart], spray: wp.array2d[Spray], box: wp.array2d[wp.vec4i]):
    """Per world and chart, the union of its rays' windows (``union_box``; i0 > i1 when no ray hit it); before_film
    adds it to the box of cells the supply wrote this exchange (``supply_shadow.boxes_written``)."""
    w, k = wp.tid()
    chart = charts[k]
    b = wp.vec4i(1 << 30, -1, 0, -1)
    for r in range(spray.shape[1]):
        s = spray[w, r]
        if s.on != 0 and s.chart == k:
            b = union_box(chart, b, s.win)
    box[w, k] = b


@wp.kernel
def spray_off(dt: wp.float64, spray: wp.array2d[Spray], ns: NetSpecies, ledger: wp.array2d[wp.float64]):
    """Per world, in ray order: the stage drips' cones past their charts' edges into ``lost`` and their species from
    ``sent`` to ``lost`` (the gather lays the rest)."""
    w = wp.tid()
    for r in range(spray.shape[1]):
        s = spray[w, r]
        if s.on != 1 or s.q_off <= wp.float64(0.0):
            continue
        v = s.q_off * dt
        ledger[w, LOST] = ledger[w, LOST] + v
        for k in range(ns.n_film):
            m = v * s.conc[k]
            ns.led[w, SP_SENT, k] = ns.led[w, SP_SENT, k] - m
            ns.led[w, SP_LOST, k] = ns.led[w, SP_LOST, k] + m


@wp.kernel
def spray_gather(dt: wp.float64, charts: wp.array[Chart], prof: wp.array[wp.vec3], cpos: wp.array[wp.vec3],
                 area: wp.array[wp.float64], spray: wp.array2d[Spray], box: wp.array2d[wp.vec4i],
                 sbit: wp.array[wp.int32], src_acc: wp.array2d[wp.float64], impact_p: wp.array2d[wp.float32],
                 impact_share: wp.array2d[wp.float32], cwet: wp.array2d[wp.int32], ns: NetSpecies,
                 shade: wp.array2d[wp.int32]):
    """Row ``ii`` of chart k's box: each cell sums, over the rays in order, its share q·w/(Σw·area) of each ray whose
    footprint covers it (cells of the ray's own window, rows within its fold span), added to the pending volume as
    rate·dt (the cell marked wet), and the droplets' momentum flux ρ·rate·w_n with its impact share, the
    droplet-struck fraction 2·rate/w_n (WP15), so impact_p/share is the droplets' ½·ρ·w_n² as for a jet."""
    w, k, ii = wp.tid()
    b = box[w, k]
    i = b[0] + ii
    if i > b[1]:
        return
    chart = charts[k]
    for j in range(b[2], b[3] + 1):
        c = cell_of(chart, i, j)
        rate = wp.float64(0.0)
        imp = float(0.0)
        share = float(0.0)
        msp = wp.vec4d()
        for r in range(spray.shape[1]):
            s = spray[w, r]
            if s.on == 0 or s.chart != k or i < s.win[0] or i > s.win[1]:
                continue
            if in_columns(chart, j, s.win[2], s.win[3]) == 0:       # Σw ran over the ray's own window only
                continue
            if sbit[s.noz] >= 0 and ((shade[w, c] >> sbit[s.noz]) & 1) == 1:   # hidden from its nozzle (shadow)
                continue
            rr = wp.float64(0.0)
            if s.on == 2:
                if c == s.cell:
                    rr = s.q / area[c]
            else:
                d = surf_dist(chart, prof, cpos, c, i, j, s.p, s.i_hit)
                wt = wp.float64(wp.max(1.0 - d / s.radius, 0.0)) * area[c]
                if wt > wp.float64(0.0):
                    rr = s.q * wt / (s.sum_w * area[c])
            rate = rate + rr
            imp = imp + s.rho * float(rr) * s.wn
            if rr > wp.float64(0.0) and s.wn > 1.0e-6:
                share = share + 2.0 * float(rr) / s.wn
            msp = msp + s.conc * (rr * dt)
        if rate > wp.float64(0.0):
            src_acc[w, c] = src_acc[w, c] + rate * dt
            impact_p[w, c] = impact_p[w, c] + imp
            impact_share[w, c] = impact_share[w, c] + share
            for kk in range(ns.n_film):
                ns.src[w, c, kk] = ns.src[w, c, kk] + msp[kk]
            cwet[w, c] = 1


__all__ = ["Spray", "cone_cells", "drip_radius", "in_columns", "off_chart", "spray_boxes", "spray_gather", "spray_off",
           "spray_rays"]
