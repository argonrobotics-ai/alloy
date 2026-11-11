# SPDX-License-Identifier: Apache-2.0
"""Kernels of the process stage (``process.py``; design §4.8, §5). One thread per (world, tile), over the tile's cells
in order, gated on the film's tick (``tick % stride == 0``: a film step this exchange) and, for films without a
forming bulk, on ``tile_on``. No atomics and no host work, so every pass is deterministic and capture-safe.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.kernels import Cells, Faces, Fluid, ToolField
from alloy.plugins.surface_flow.links import cell_dir
from alloy.plugins.surface_flow.tiles import Tiles


@wp.struct
class SetLaw:
    """The setting law of a paste (design §3.2, C:424-436): ``τ(a) = τ_0 + A_thix·a/s_T + τ_IS·(a/(t_IS·s_T))^m``."""

    tau_0: wp.float64
    a_thix: wp.float64
    tau_is: wp.float64
    t_is: wp.float64
    m: wp.float64
    s_t: wp.float64
    film_factor: wp.float64


@wp.struct
class SoilLaw:
    """A soil's strength and removal constants (``process_specs.Soil``)."""

    sigma_dry: wp.float64
    sigma_wet: wp.float64
    t_soak: wp.float64                # at the water's temperature
    k_e: wp.float64
    k_i: wp.float64
    k_front: wp.float64
    m_clean: wp.float64
    beta: wp.float64
    k_wipe: wp.float64
    k_abr: wp.float64
    sigma_ref: wp.float64
    rho: wp.float64


@wp.struct
class DryLaw:
    """Solvent evaporation [m/s] while the solids are below ``phi_m``; the wet edge ``phi_we``."""

    rate: wp.float64
    phi_m: wp.float64
    phi_we: wp.float64
    lap_visible: wp.float64           # [m] solids the old film needs for a lap to show


@wp.func
def set_yield(law: SetLaw, age: wp.float64) -> wp.float64:
    """At-rest yield stress [Pa] of the bulk at ``age`` [s] (an age < 0, not placed yet, counts as 0)."""
    a = wp.max(age, wp.float64(0.0))
    s = law.s_t
    return law.tau_0 + law.a_thix * a / s + law.tau_is * wp.pow(a / (law.t_is * s), law.m)


@wp.func
def due(tick: wp.array[wp.int32], stride: int, tile_on: wp.array2d[wp.int32], every: int, w: int, t: int) -> bool:
    """A film step this exchange, on a tile that steps (or on every tile when ``every``)."""
    if tick[w] % stride != 0:
        return False
    return every != 0 or tile_on[w, t] != 0


@wp.kernel
def age_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int, every: int,
               h: wp.array2d[wp.float64], h_wet: wp.float64, film_dt: wp.float64, age: wp.array2d[wp.float64]):
    """``age`` [s] advances by the film step: on every cell when ``every`` (a forming bulk, placed at the start), else
    from a cell's first wetting (``age < 0``: not placed yet)."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, every, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            a = age[w, c]
            if a < wp.float64(0.0):
                if every != 0 or h[w, c] <= h_wet:
                    continue
                a = wp.float64(0.0)
            age[w, c] = a + film_dt


@wp.kernel
def yield_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int, every: int,
                 law: SetLaw, age: wp.array2d[wp.float64], yield_cell: wp.array2d[wp.float32]):
    """The film's at-rest yield for this film step: ``film_factor·τ(age)``."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, every, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            yield_cell[w, c] = wp.float32(law.film_factor * set_yield(law, age[w, c]))


@wp.kernel
def bulk_yield_kernel(law: SetLaw, age: wp.array2d[wp.float64], bulk_yield: wp.array2d[wp.float32]):
    """A forming bulk's yield ``τ(age)`` every exchange (forming reads it each exchange): derived from ``age`` alone,
    so a reset, a rollback or a set age never leaves a stale yield."""
    w, c = wp.tid()
    bulk_yield[w, c] = wp.float32(set_yield(law, age[w, c]))


@wp.func
def cell_mu(fl: Fluid, w: int, c: int) -> wp.float64:
    """The film's viscosity [Pa·s] at a cell: the per-cell input where set (never below the material's), else the
    material's."""
    mu = fl.rho * fl.nu
    if fl.per_cell != 0:
        mu = wp.max(fl.visc_cell[w, c], mu)
    return wp.float64(mu)


@wp.func
def wall_shear(fl: Fluid, cells: Cells, faces: Faces, q: wp.array2d[wp.float32], w: int, c: int,
               hc: wp.float64) -> wp.float64:
    """The film's shear on its surface [Pa]: ``3μ|q̄|/max(h, h_wet)²``, with ``q̄`` the mean of the cell's face
    discharges (each face's ``q`` along its direction; two faces per axis), plus the yield stress while it flows."""
    fs = cells.face[c]
    qv = wp.vec3(0.0)
    for s in range(4):
        f = fs[s]
        if f >= 0:
            qv += q[w, f] * cell_dir(faces, f, c)
    qm = wp.float64(0.5 * wp.length(qv))
    hh = wp.max(hc, fl.h_wet)
    tau = wp.float64(3.0) * cell_mu(fl, w, c) * qm / (hh * hh)
    if qm > wp.float64(0.0):
        ty = fl.tau_y
        if fl.per_cell != 0:
            ty = fl.yield_cell[w, c]
        tau += wp.float64(ty)
    return tau


@wp.func
def front_rate(soil_prev: wp.array2d[wp.float64], h: wp.array2d[wp.float64], u: wp.array2d[wp.float32],
               cells: Cells, faces: Faces, law: SoilLaw, h_wet: wp.float64, w: int, c: int,
               hyd: wp.float64) -> wp.float64:
    """Wilson's cleaning front (WP8's ``rinse_soil.front_rate``): over the cell's faces whose neighbour is clean
    (``< m_clean``) and whose film flows in at ``u_n``, ``k_front·hyd·beta·ρ·u_n²·h_n·L/A`` [1/s], the rate at which
    the cell's start load goes. The front crosses a cell of width Δx in Δx/v, ``v = k'·hyd·β·ρ·u_n²·h``."""
    fc = cells.face[c]
    rate = wp.float64(0.0)
    for k in range(4):
        f = fc[k]
        if f < 0:
            continue
        n = faces.b[f]
        sgn = wp.float64(-1.0)                        # c is the face's a side: inflow from b is u < 0
        if faces.a[f] != c:
            n = faces.a[f]
            sgn = wp.float64(1.0)
        if n >= 0 and soil_prev[w, n] < law.m_clean:
            un = sgn * wp.float64(u[w, f])
            hn = h[w, n]
            if un > wp.float64(0.0) and hn > h_wet:
                m_flux = law.beta * law.rho * un * un * hn
                rate += law.k_front * hyd * m_flux * wp.float64(faces.length[f]) / cells.area[c]
    return rate


@wp.kernel
def soil_copy_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int,
                     soil: wp.array2d[wp.float64], soil_prev: wp.array2d[wp.float64]):
    """The soil at the step's start, which the front reads in the neighbours (one writer per cell, no races)."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, 0, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            soil_prev[w, c] = soil[w, c]


@wp.struct
class SoilOut:
    """Where a film step's soil goes: the surface load, its soaking, the removal by mechanism (impact, shear, front,
    wipe; kg/m²), the film's soil species and wet flags, and on flooded cells the soil bound for the pool's tank
    (``pool_soil`` kg/m², summed into the tank by ``pool_soil_kernel``)."""

    soil: wp.array2d[wp.float64]
    hyd: wp.array2d[wp.float64]
    by: wp.array3d[wp.float64]
    sp_src: wp.array3d[wp.float64]
    cwet: wp.array2d[wp.int32]
    pool_soil: wp.array2d[wp.float64]


@wp.func
def tool_loss(tf: ToolField, fl: Fluid, law: SoilLaw, w: int, c: int, hc: wp.float64, wet: bool) -> wp.float64:
    """Soil loss under a ceiling tool times σ_c [kg·Pa/(m²·s)]: Archard abrasion ``k_abr·cover·press·|v|·σ_ref``, wet
    or dry, and on a wet cell the viscous wipe ``k_wipe·cover·μ|v|/max(gap, h, h_wet)``. The gap floors at ``h_wet``
    as the film's wall shear does: a cell under a pool's water or a jet's sheet is wet with no film (h = 0), and a
    pressed pad's gap is <= 0 (WP8's pan: a pad on a flooded floor divided by zero)."""
    rate = wp.float64(0.0)
    if tf.on != 0 and tf.cover[w, c] > 0.0:
        cov, v = wp.float64(tf.cover[w, c]), wp.float64(wp.length(tf.vel[w, c]))
        rate = law.k_abr * cov * wp.float64(tf.press[w, c]) * v * law.sigma_ref
        if wet:
            gap = wp.max(wp.max(wp.float64(tf.ceil[w, c]), hc), fl.h_wet)
            rate = rate + law.k_wipe * cov * cell_mu(fl, w, c) * v / gap
    return rate


@wp.struct
class Writers:
    """The boxes of chart cells written this exchange by the jets (the supply's ``written`` [W, K]), the tools
    (``wins`` [W, n] on ``tool_chart``) and the pools (the supply's ``flood_cur`` [W, K], the cells its pools flood):
    only tiles meeting one are scanned for soil under a sheet, a pad or a pool's water."""
    boxes: wp.array2d[wp.vec4i]
    n_boxes: int
    flood: wp.array2d[wp.vec4i]
    n_flood: int
    wins: wp.array2d[wp.vec4i]
    tool_chart: wp.array[wp.int32]
    n_tools: int


@wp.func
def meets(box: wp.vec4i, i0: int, i1: int, j0: int, j1: int, ny: int, wrap: int) -> bool:
    """Box (i0, i1, j0, j1 of rows and columns; columns may run past ny on a wrapping chart) meets the tile's."""
    if box[1] < i0 or box[0] > i1:
        return False
    hit = False
    for s in range(-1, 2):
        if s == 0 or wrap != 0:
            if box[3] + s * ny >= j0 and box[2] + s * ny <= j1:
                hit = True
    return hit


@wp.func
def written_near(wr: Writers, charts: wp.array[Chart], k: int, w: int, i0: int, i1: int, j0: int, j1: int) -> bool:
    ch = charts[k]
    hit = False
    if wr.n_boxes > 0:
        hit = meets(wr.boxes[w, k], i0, i1, j0, j1, ch.ny, ch.wrap)
    if wr.n_flood > 0 and meets(wr.flood[w, k], i0, i1, j0, j1, ch.ny, ch.wrap):
        hit = True
    for n in range(wr.n_tools):
        if wr.tool_chart[n] == k and meets(wr.wins[w, n], i0, i1, j0, j1, ch.ny, ch.wrap):
            hit = True
    return hit


@wp.kernel
def soil_gate_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int, cells: Cells,
                     charts: wp.array[Chart], wr: Writers, soil: wp.array2d[wp.float64],
                     jet_cover: wp.array2d[wp.float32], tf: ToolField, flooded: wp.array2d[wp.int32],
                     gate: wp.array2d[wp.int32]):
    """The tiles the soil steps this film step: the film's, and any with soil under a jet's sheet (swept dry), under
    a pressing tool (abrasion acts dry) or under a pool's water (its film went to the pool), from this exchange's
    fields. Only an off tile that meets a box written this exchange (a jet's, a tool's, the pools') is scanned."""
    w, t = wp.tid()
    if tick[w] % stride != 0:
        return
    gate[w, t] = tile_on[w, t]
    first, pitch = tl.first[t], tl.pitch[t]
    k = cells.chart[first]
    li = first - charts[k].cell_start
    i0, j0 = li // pitch, li % pitch
    if tile_on[w, t] != 0 or not written_near(wr, charts, k, w, i0, i0 + tl.rows[t] - 1, j0, j0 + tl.cols[t] - 1):
        return
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            if soil[w, c] > wp.float64(0.0):
                if jet_cover[w, c] > 0.0 or flooded[w, c] >= 0 or (tf.on != 0 and tf.cover[w, c] > 0.0
                                                                   and tf.press[w, c] > 0.0):
                    gate[w, t] = 1


@wp.kernel
def soil_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int, cells: Cells,
                faces: Faces, fl: Fluid, tf: ToolField, law: SoilLaw, h: wp.array2d[wp.float64],
                q: wp.array2d[wp.float32], u: wp.array2d[wp.float32], impact_p: wp.array2d[wp.float32],
                impact_share: wp.array2d[wp.float32], jet_tau: wp.array2d[wp.float32],
                jet_cover: wp.array2d[wp.float32], pool_tau: wp.array2d[wp.float32], flooded: wp.array2d[wp.int32],
                soil_prev: wp.array2d[wp.float64], soil0: wp.array2d[wp.float64], film_dt: wp.float64, k_soil: int,
                out: SoilOut):
    """One film step of the soil on wet cells (film deeper than ``h_wet``, under a jet's sheet, which sweeps the
    film from under it, or under a pool's water, which took its film: shear from submerged jets ``pool_tau``) and
    under ceiling tools: the losses at the step's starting strength (scaled together to the
    soil left), then on wet cells soaking ``hyd ← 1 - (1 - hyd)·exp(-Δt/t_soak)`` (exact). A dry cell under a tool
    loses only abrasion. The loss is booked by mechanism (the tool's: wipe) and, when the film carries the soil
    species (``k_soil >= 0``), joins it as a pending source."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, 0, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    soak = wp.exp(-film_dt / law.t_soak)
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            m = soil_prev[w, c]
            hc = h[w, c]
            under = flooded[w, c] >= 0                    # under a pool's water: its film is the pool's
            wet = hc > fl.h_wet or jet_cover[w, c] > 0.0 or under
            if m <= zero:
                continue
            hy = out.hyd[w, c]
            sig = law.sigma_dry * (one - hy) * (one - hy) + law.sigma_wet
            t_w = tool_loss(tf, fl, law, w, c, hc, wet) / sig * film_dt
            t_i, t_e, t_f = zero, zero, zero
            if wet:
                frac = wp.float64(impact_share[w, c])     # impact_p = frac·p_stag (the supply's footprint share)
                t_i = law.k_i * wp.max(wp.float64(impact_p[w, c]) / sig - frac, zero) * film_dt
                tau = wp.float64(jet_tau[w, c])        # under a sheet the sheet's shear: the film there is the sweep's
                if jet_cover[w, c] <= 0.0:             # backflow (3μq̄/h² of a 10 µm refill reads kPa at its rim)
                    tau = wp.max(tau, wall_shear(fl, cells, faces, q, w, c, hc))
                if under:                              # a submerged jet's wall shear (the supply's pool_jet_tau)
                    tau = wp.max(tau, wp.float64(pool_tau[w, c]))
                t_e = law.k_e * wp.max(tau - sig, zero) * film_dt
                t_f = front_rate(soil_prev, h, u, cells, faces, law, fl.h_wet, w, c, hy) * soil0[w, c] * film_dt
                out.hyd[w, c] = one - (one - hy) * soak
            total = t_i + t_e + t_f + t_w
            if total <= zero:
                continue
            scale = wp.min(total, m) / total
            t_i, t_e, t_f, t_w = t_i * scale, t_e * scale, t_f * scale, t_w * scale
            dm = t_i + t_e + t_f + t_w
            out.soil[w, c] = m - dm
            out.by[w, c, 0] = out.by[w, c, 0] + t_i
            out.by[w, c, 1] = out.by[w, c, 1] + t_e
            out.by[w, c, 2] = out.by[w, c, 2] + t_f
            out.by[w, c, 3] = out.by[w, c, 3] + t_w
            if k_soil >= 0 and under:                 # its film is the pool's: the tank takes it (pool_soil_kernel)
                out.pool_soil[w, c] = out.pool_soil[w, c] + dm
            elif k_soil >= 0:
                out.sp_src[w, c, k_soil] = out.sp_src[w, c, k_soil] + dm
                out.cwet[w, c] = 1


@wp.kernel
def pool_soil_kernel(charts: wp.array[Chart], flood: wp.array2d[wp.vec4i], flooded: wp.array2d[wp.int32],
                     area: wp.array[wp.float64], k_soil: int, pool_soil: wp.array2d[wp.float64],
                     add_sp: wp.array3d[wp.float64], to_pools: wp.array[wp.float64]):
    """Per world, after the soil step: the soil eroded on flooded cells joins their pool's tank (DC5's
    ``pool_add_sp``, booked in the tank's species ledger), summed over each chart's flood box in row order
    (deterministic), and ``to_pools`` [kg] books it."""
    w = wp.tid()
    for k in range(charts.shape[0]):
        ch = charts[k]
        b = flood[w, k]
        span = wp.min(b[3] - b[2] + 1, ch.ny)
        for i in range(wp.max(b[0], 0), wp.min(b[1], ch.nx - 1) + 1):
            for jj in range(span):
                j = b[2] + jj
                if ch.wrap != 0:
                    j = (j % ch.ny + ch.ny) % ch.ny
                elif j < 0 or j >= ch.ny:
                    continue
                c = ch.cell_start + i * ch.ny + j
                m = pool_soil[w, c]
                p = flooded[w, c]
                if m > wp.float64(0.0) and p >= 0:
                    add_sp[w, p, k_soil] = add_sp[w, p, k_soil] + m * area[c]
                    to_pools[w] = to_pools[w] + m * area[c]
                    pool_soil[w, c] = wp.float64(0.0)


@wp.kernel
def dry_rate_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int, law: DryLaw,
                    h: wp.array2d[wp.float64], mass: wp.array3d[wp.float64], k_s: int, film_dt: wp.float64,
                    evap: wp.array2d[wp.float32]):
    """This film step's solvent evaporation [m/s]: the law's rate, but never taking the film below ``solids/phi_m``."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, 0, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            room = h[w, c] - mass[w, c, k_s] / law.phi_m
            e = wp.min(law.rate, room / film_dt)
            evap[w, c] = wp.float32(wp.max(e, wp.float64(0.0)))


@wp.kernel
def absorb_rate_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int,
                       sorp: wp.array[wp.float64], cap: wp.array[wp.float64], area: wp.array[wp.float64],
                       absorbed: wp.array2d[wp.float64], film_dt: wp.float64, dry: int, law: DryLaw,
                       h: wp.array2d[wp.float64], mass: wp.array3d[wp.float64], k_s: int,
                       evap: wp.array2d[wp.float32], rate: wp.array2d[wp.float32]):
    """This film step's uptake [m/s] into a porous substrate: the exact Lucas-Washburn increment from the depth taken
    so far, ``(sqrt(a² + S²·Δt) - a)/Δt``, never past the capacity. The film's cell kernel takes it as solvent, so with
    drying on it also never takes the film below ``solids/phi_m`` after this step's evaporation (WP15 review)."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, 0, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            s = sorp[c]
            if s <= wp.float64(0.0):
                continue
            a = absorbed[w, c] / area[c]
            inc = wp.sqrt(a * a + s * s * film_dt) - a
            inc = wp.min(inc, cap[c] - a)
            if dry != 0:
                room = h[w, c] - mass[w, c, k_s] / law.phi_m - wp.float64(evap[w, c]) * film_dt
                inc = wp.min(inc, room)
            rate[w, c] = wp.float32(wp.max(inc, wp.float64(0.0)) / film_dt)


@wp.kernel
def dry_state_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int, law: DryLaw,
                     h: wp.array2d[wp.float64], h_wet: wp.float64, mass: wp.array3d[wp.float64],
                     supplied: wp.array3d[wp.float64], k_s: int, laps: wp.array2d[wp.int32],
                     lap_open: wp.array2d[wp.int32], phi_prev: wp.array2d[wp.float64],
                     sup_prev: wp.array2d[wp.float64], seed: wp.array2d[wp.int32]):
    """Laps: a pass (film steps in a row that bring new paint, a rise in the solids' ``supplied``) that lands on film
    already past the wet edge, with at least ``lap_visible`` solids, counts once. A cell's solids fraction is kept for
    the next step (a cell dried down to its solids counts as ``phi_m``; bare cells, fringes thinner than
    ``lap_visible``, and solids that never held solvent (mist dust, pure solids: ``seed`` 0, set once a cell's
    solids are under ``phi_m`` of its depth, cleared when it is bare) as 0, so dust joins the next coat's solids
    without being an earlier wet edge)."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, 0, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            sup = supplied[w, c, k_s]
            if sup > sup_prev[w, c]:
                if lap_open[w, c] == 0 and phi_prev[w, c] >= law.phi_we:
                    laps[w, c] = laps[w, c] + 1
                lap_open[w, c] = 1
            else:
                lap_open[w, c] = 0
            sup_prev[w, c] = sup
            sol = mass[w, c, k_s]
            hc = h[w, c]
            phi = wp.float64(0.0)
            if hc > h_wet:
                phi = wp.min(sol / hc, law.phi_m)
                if sol < law.phi_m * hc * wp.float64(1.0 - 1.0e-9):
                    seed[w, c] = 1                          # solvent in it (a coat, wet or past its wet edge)
            elif sol > wp.float64(0.0):
                phi = law.phi_m
            if sol <= wp.float64(0.0):
                seed[w, c] = 0                              # bare again
            if sol < law.lap_visible or seed[w, c] == 0:
                phi = wp.float64(0.0)                       # an invisible fringe, or dust, cannot show a lap
            phi_prev[w, c] = phi


@wp.kernel
def sanitizer_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int,
                     h: wp.array2d[wp.float64], h_wet: wp.float64, mass: wp.array3d[wp.float64], k_s: int,
                     c_min: wp.float64, film_dt: wp.float64, expo: wp.array2d[wp.float64],
                     best: wp.array2d[wp.float64]):
    """The sanitizer clock: unbroken contact time [s] at ``c >= c_min`` (c = mass/h on a wet cell), and its best."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, 0, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            hc = h[w, c]
            if hc > h_wet and mass[w, c, k_s] >= c_min * hc:
                e = expo[w, c] + film_dt
                expo[w, c] = e
                best[w, c] = wp.max(best[w, c], e)
            else:
                expo[w, c] = wp.float64(0.0)


__all__ = ["DryLaw", "SetLaw", "SoilLaw", "SoilOut", "front_rate", "soil_copy_kernel", "absorb_rate_kernel",
           "age_kernel", "bulk_yield_kernel", "cell_mu", "dry_rate_kernel", "dry_state_kernel", "due",
           "pool_soil_kernel", "sanitizer_kernel", "set_yield", "soil_kernel", "Writers", "soil_gate_kernel",
           "tool_loss", "wall_shear", "yield_kernel"]
