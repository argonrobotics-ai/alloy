# SPDX-License-Identifier: Apache-2.0
"""Deposit model functions of the surface-flow supply (interface note §4, ``before_film``) and the per-world
deposit of each nozzle's discharge into pools, lost, overspray or mist.

A solid jet (Watson 1964; the radius from Bhagat, Jha, Linden & Wilson 2018) enters at its hydraulic jump: the thin
supercritical sheet inside the jump is not resolved (its 25 m/s would set the film's step), so the film under it is
swept to the ring at R_J, where the jet's flow and the swept film enter at the post-jump speed u_post =
Q/(2π·R_J·h), h at least the depth of the post-jump Froude number ``froude`` (capped by ``u_max``, below the
film's ``u_cap``). A deep film downstream drowns the jump inward (``drowned_at``), so a jet into a cup fills it.
Distances run along the surface (``surf_dist``), and the jump region stops at folds (``row_span``). The jump's
kernels run row-parallel in ``supply_jump.py``; sprays (fan, cone, aerosol) spread each ray's share over a cone
kernel as wide as the ray's angular share at its flight distance, in parallel in ``supply_spray.py``.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.fluid.kernels import GAS, node_amount
from alloy.plugins.surface_flow.charts import PLANE, Chart
from alloy.plugins.surface_flow.supply_kernels import (CAUGHT, CHART, LOST, LUMP, LUMPED, MIST, OVERSPRAY, POOL,
                                                       SENT, SHAPE, SPRAY, TO_TANK, NozzleSpec, PoolSpec, weight_sum)

EMPTY_BOX = wp.constant(wp.vec4i(1 << 30, -1, 0, -1))     # rows i0 > i1: nothing written


@wp.func
def cell_of(chart: Chart, i: int, j: int) -> int:
    """Global cell (i, j); a revolution chart's j wraps."""
    jj = j
    if chart.wrap != 0:
        jj = ((j % chart.ny) + chart.ny) % chart.ny
    return chart.cell_start + i * chart.ny + jj


@wp.func
def ring_step(chart: Chart) -> float:
    """The half width of a jump ring: one cell (a plane's larger step, a revolution chart's ds)."""
    if chart.kind == PLANE:
        return wp.max(chart.step[0], chart.step[1])
    return chart.step[0]


@wp.func
def profile_s(chart: Chart, prof: wp.array[wp.vec3], i: int) -> float:
    """Arc length of a revolution chart's row i centre along its profile."""
    return 0.5 * (prof[chart.prof_start + i][2] + prof[chart.prof_start + i + 1][2])


@wp.func
def surf_dist(chart: Chart, prof: wp.array[wp.vec3], cpos: wp.array[wp.vec3], c: int, i: int, j: int, p: wp.vec3,
              i_hit: int) -> float:
    """Distance from the hit ``p`` (chart frame, in row ``i_hit``) to cell (i, j) along the surface: in-plane on a
    plane chart; on a revolution chart sqrt(Δs² + (r̄·Δθ)²), so a jet inside a glass never reaches its outside."""
    if chart.kind == PLANE:
        return wp.length(cpos[c] - p)
    ds = profile_s(chart, prof, i) - profile_s(chart, prof, i_hit)
    jj = ((j % chart.ny) + chart.ny) % chart.ny
    th = (float(jj) + 0.5) * chart.step[1] - wp.atan2(p[1], p[0])
    th = th - 6.28318531 * wp.floor((th + 3.14159265) / 6.28318531)       # to [-π, π)
    r_c = wp.sqrt(cpos[c][0] * cpos[c][0] + cpos[c][1] * cpos[c][1])
    r_m = 0.5 * (r_c + wp.sqrt(p[0] * p[0] + p[1] * p[1]))
    return wp.sqrt(ds * ds + r_m * r_m * th * th)


@wp.func
def jump_radius(noz: NozzleSpec, q: float, a: float) -> float:
    """R_J = c·Q^(3/4)·ρ^(1/4)/(ν·σ)^(1/4) (Bhagat, Jha, Linden & Wilson 2018, c = 0.289), at least 2 jet radii."""
    r = noz.jump_coeff * wp.pow(q, 0.75) * wp.pow(noz.rho / (noz.nu * noz.sigma), 0.25)
    return wp.max(r, 2.0 * a)


@wp.func
def oblique_split(rel: wp.vec3, n: wp.vec3, t_hat: wp.vec3, cos_b: float) -> float:
    """The share of an oblique jet's sheet going towards ``rel`` (tangential direction φ from the jet's own tangential
    direction ``t_hat``), per unit angle: sin³β/(1 - cos β·cos φ)² (Hasson & Peck 1964; inviscid, it carries the jet's
    tangential momentum). 1 for a normal jet; β is the angle between the jet and the surface."""
    if cos_b <= 1.0e-6:
        return 1.0
    tan = rel - wp.dot(rel, n) * n
    if wp.length(tan) <= 1.0e-12:
        return 1.0
    cos_phi = wp.dot(wp.normalize(tan), t_hat)
    sin_b = wp.sqrt(wp.max(1.0 - cos_b * cos_b, 0.0))
    den = 1.0 - cos_b * cos_phi
    return sin_b * sin_b * sin_b / (den * den)


FOLD_COS = wp.constant(0.7071)       # a fold: the profile turns by more than 45° (treated like a chart edge)
N_BANDS = 48
bands = wp.types.vector(length=N_BANDS, dtype=wp.float64)


@wp.func
def row_tangent(chart: Chart, prof: wp.array[wp.vec3], i: int) -> wp.vec2:
    a = prof[chart.prof_start + i]
    b = prof[chart.prof_start + i + 1]
    return wp.normalize(wp.vec2(b[0] - a[0], b[1] - a[1]))


@wp.func
def row_span(chart: Chart, prof: wp.array[wp.vec3], i_hit: int) -> wp.vec2i:
    """The rows a jet's sheet can reach from row ``i_hit`` without crossing a fold (a revolution chart's profile
    turning by more than 45°: a cup's floor-wall corner, its rim). A plane chart has no folds."""
    if chart.kind == PLANE:
        return wp.vec2i(0, chart.nx - 1)
    t0 = row_tangent(chart, prof, i_hit)
    lo = i_hit
    while lo > 0 and wp.dot(row_tangent(chart, prof, lo - 1), t0) >= FOLD_COS:
        lo = lo - 1
    hi = i_hit
    while hi < chart.nx - 1 and wp.dot(row_tangent(chart, prof, hi + 1), t0) >= FOLD_COS:
        hi = hi + 1
    return wp.vec2i(lo, hi)


@wp.func
def drowned_at(bh: bands, ba: bands, bw: float, r_b: float, aw: float, g_n: float) -> float:
    """Where the jump stands: the first band whose film is as deep as the sheet's conjugate depth there. At large
    Froude number the inviscid sheet (h ≈ a²/(2r), speed w) jumps to a·w/sqrt(g·r) (Watson 1964's inviscid limit;
    beyond ~0.3·a·Re^(1/3) viscosity thins the sheet, so this errs towards the larger radius). None within R_J:
    R_J. ``bh``/``ba``: depth·area and area per band of width ``bw``; aw = Q/(π·a) = a·w."""
    if g_n <= 0.0:
        return r_b
    for k in range(N_BANDS):
        r = (float(k) + 0.5) * bw
        if r >= r_b:
            return r_b
        if ba[k] > wp.float64(0.0):
            if float(bh[k] / ba[k]) >= aw / wp.sqrt(g_n * r):
                return r
    return r_b


@wp.func
def ring_weight(d: float, r_j: float, wr: float, fallback: int, d_far: float) -> float:
    """Where the jet's water enters: a ring of half width ``wr`` at R_J, or (the ring off the chart) the band of
    in-reach cells farthest from the hit."""
    if fallback != 0:
        if d <= r_j + wr and d >= d_far - wr:
            return 1.0
        return 0.0
    return wp.max(1.0 - wp.abs(d - r_j) / wr, 0.0)


@wp.func
def cover_of(d: float, r_j: float, wr: float, fallback: int, d_far: float) -> float:
    """1 for a cell under the unresolved supercritical sheet (inside the jump, short of the ring), else 0."""
    if fallback != 0:
        if d < d_far - wr:
            return 1.0
        return 0.0
    if d <= r_j - wr:
        return 1.0
    return 0.0


@wp.func
def union_box(chart: Chart, a: wp.vec4i, b: wp.vec4i) -> wp.vec4i:
    """The box (rows i0..i1, unwrapped columns j0..j1) covering boxes ``a`` and ``b`` (i0 > i1: empty); on a
    revolution chart a span of a turn or more becomes the whole turn, so no cell is visited twice."""
    if a[0] > a[1]:
        return b
    if b[0] > b[1]:
        return a
    j0 = wp.min(a[2], b[2])
    j1 = wp.max(a[3], b[3])
    if chart.wrap != 0 and j1 - j0 + 1 >= chart.ny:
        j0 = 0
        j1 = chart.ny - 1
    return wp.vec4i(wp.min(a[0], b[0]), wp.max(a[1], b[1]), j0, j1)


@wp.kernel
def clear_written(charts: wp.array[Chart], written: wp.array2d[wp.vec4i], has_pool: int,
                  impact_p: wp.array2d[wp.float32], impact_share: wp.array2d[wp.float32],
                  jet_cover: wp.array2d[wp.float32], jet_tau: wp.array2d[wp.float32], pool_tau: wp.array2d[wp.float32]):
    """Row ``ii`` of the box the supply wrote last exchange (per world and chart): zero the per-exchange fields there
    (interface note, the clear rule), so the cost scales with the cells written."""
    w, k, ii = wp.tid()
    b = written[w, k]
    i = b[0] + ii
    if i > b[1]:
        return
    chart = charts[k]
    for j in range(b[2], b[3] + 1):
        c = cell_of(chart, i, j)
        impact_p[w, c] = 0.0
        impact_share[w, c] = 0.0
        jet_cover[w, c] = 0.0
        jet_tau[w, c] = 0.0
        if has_pool != 0:                                   # the film's pool_jet_tau (supply_jetting)
            pool_tau[w, c] = 0.0


@wp.kernel
def reset_written(written: wp.array2d[wp.vec4i]):
    """Empty boxes before this exchange's deposits grow them."""
    w, k = wp.tid()
    written[w, k] = EMPTY_BOX


@wp.func
def node_conc(w: int, i: int, k: int, medium: wp.array[wp.int32], bulk: wp.array[wp.float64],
              p_ref: wp.array[wp.float64], poly: wp.array[wp.float64], pressure: wp.array2d[wp.float64],
              volume: wp.array2d[wp.float64], species: wp.array3d[wp.float64]) -> wp.float64:
    """Species k's concentration in network node i [kg/m³] (0 in gas and empty nodes)."""
    if medium[i] == GAS:
        return wp.float64(0.0)
    a = node_amount(medium[i], pressure[w, i], volume[w, i], bulk[i], p_ref[i], poly[i])
    if a <= wp.float64(0.0):
        return wp.float64(0.0)
    return species[w, i, k] / a


@wp.kernel
def deposit(dt: wp.float64, nozzles: wp.array[NozzleSpec], ray_wt: wp.array[wp.float32], pools: wp.array[PoolSpec],
            discharge: wp.array2d[wp.float64], hit_kind: wp.array2d[wp.int32], hit_idx: wp.array2d[wp.int32],
            inflow: wp.array2d[wp.float64], ledger: wp.array2d[wp.float64], caught_by: wp.array2d[wp.float64]):
    """Each nozzle's last-step discharge, by its stored hits: into pools (the tank's inflow), caught by a shape (per
    occluder slot in ``caught_by``), or lost (a jet that hit nothing) / overspray (a spray's rays that hit nothing) /
    mist (its 1 - efficiency); books the ledger per world. Chart hits are entered in parallel after this: sprays by
    ``supply_spray``, solid jets by ``supply_jump``; the species by ``supply_species.species_sent``."""
    w = wp.tid()
    for n in range(nozzles.shape[0]):
        noz = nozzles[n]
        q_all = -discharge[w, noz.edge]
        if q_all <= wp.float64(0.0):
            continue
        ledger[w, SENT] = ledger[w, SENT] + q_all * dt
        eff = wp.float64(noz.efficiency)
        ledger[w, MIST] = ledger[w, MIST] + q_all * (wp.float64(1.0) - eff) * dt
        norm = weight_sum(noz, ray_wt)
        for r in range(noz.ray_start, noz.ray_start + noz.ray_count):
            q = q_all * eff * wp.float64(ray_wt[r]) / norm
            kind = hit_kind[w, r]
            if kind == POOL:
                tank = pools[hit_idx[w, r]].tank
                inflow[w, tank] = inflow[w, tank] + q
                ledger[w, TO_TANK] = ledger[w, TO_TANK] + q * dt
            elif kind == CHART:
                continue
            elif kind == LUMP:
                ledger[w, LUMPED] = ledger[w, LUMPED] + q * dt
            elif kind == SHAPE:
                ledger[w, CAUGHT] = ledger[w, CAUGHT] + q * dt
                caught_by[w, hit_idx[w, r]] = caught_by[w, hit_idx[w, r]] + q * dt
            elif noz.pattern == SPRAY:
                ledger[w, OVERSPRAY] = ledger[w, OVERSPRAY] + q * dt
            else:
                ledger[w, LOST] = ledger[w, LOST] + q * dt
