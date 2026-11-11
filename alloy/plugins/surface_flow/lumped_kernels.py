# SPDX-License-Identifier: Apache-2.0
"""Kernels of the lumped object films (lumped.py): one thread per world steps every lumped body, so a sponge squeezing
onto chart cells has one writer per cell (no atomics, capture-safe)."""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.kernels import ToolField
from alloy.plugins.surface_flow.process_kernels import SoilLaw
from alloy.plugins.surface_flow.tool_geometry import cell_at

IN = wp.constant(0)             # ledger columns [W, B, 5] m³: from jets, shed (dripped), squeezed onto charts, pooled,
SHED = wp.constant(1)           # taken up from the film under a sponge's pad
SQUEEZED = wp.constant(2)
POOLED = wp.constant(3)
UPTAKE = wp.constant(4)


@wp.struct
class LumpSpec:
    area: wp.array[wp.float64]        # [B] wetted area [m²]
    hold: wp.array[wp.float64]        # [B] clinging film [m] (capacity hold·area without pores)
    law: wp.array[SoilLaw]            # [B] (unused where soil_on is 0)
    soil_on: wp.array[wp.int32]       # [B]
    pore: wp.array[wp.vec4d]          # [B] (free volume [m³], modulus [Pa], strain cap, tau_cap [s]); free 0: none
    tool: wp.array[wp.int32]          # [B] the Applicator a sponge is (-1: none)
    wet: wp.array[wp.float64]         # [B] m³: wet above H_WET·area (the charts' h > H_WET), or under a jet
    k_soil: wp.array[wp.int32]        # [B] the film species each body's soil joins (-1: not carried)
    k_san: int                        # the sanitizer's species (-1: no clock)
    c_min: wp.float64
    drip_on: int                      # 1: shed water waits in ``drip`` for the drop stage (else it is only booked)
    built: wp.array[wp.float64]       # [B] m³ of water in the body's built mass (its weight is the body's own)


@wp.struct
class LumpState:
    water: wp.array2d[wp.float64]     # [W, B] m³
    sp: wp.array3d[wp.float64]        # [W, B, K] kg, well mixed in the water
    soil: wp.array2d[wp.float64]      # [W, B] kg/m² over the body's area
    hyd: wp.array2d[wp.float64]
    by: wp.array3d[wp.float64]        # [W, B, 4] kg/m² removed: impact, shear, (front), (tool)
    ledger: wp.array3d[wp.float64]    # [W, B, 5] m³ (columns above)
    sp_in: wp.array3d[wp.float64]     # [W, B, K] kg species books: from jets, taken up, shed, squeezed onto charts
    sp_up: wp.array3d[wp.float64]
    sp_shed: wp.array3d[wp.float64]
    sp_sq: wp.array3d[wp.float64]
    soil_sp: wp.array2d[wp.float64]   # [W, B] kg of eroded soil that joined the water (its soil species)
    expo: wp.array2d[wp.float64]      # [W, B] s at c >= c_min, unbroken
    expo_best: wp.array2d[wp.float64]
    eps: wp.array2d[wp.float64]       # [W, B] a sponge's strain
    drip: wp.array2d[wp.float64]      # [W, B] m³ shed and waiting to drip (the drop stage's LumpDrip.pending)
    drip_sp: wp.array3d[wp.float64]   # [W, B, K] kg of species with it
    sp_wet: wp.array3d[wp.float64]    # [W, B, K] kg of it a resting body gave the cells under it (lumped_drip.py)
    sp_drop: wp.array3d[wp.float64]   # [W, B, K] kg of it that left with drops (drops_kernels.commit_kernel)
    tank: wp.array2d[wp.float64]      # [W, B] m³ of it whose drips started in a pool's water (into its tank)
    sp_tank: wp.array3d[wp.float64]   # [W, B, K] kg with it
    pool_of: wp.array2d[wp.int32]     # [W, B] the pool a body is in this exchange (-1: none; lumped_pool.pool_book)
    vessel_of: wp.array2d[wp.int32]   # [W, B] the cavity pool whose vessel holds its lowest point (-1: none)


@wp.struct
class LumpIn:
    vol: wp.array2d[wp.float64]       # [W, B] m³ the jets brought this exchange (DC5's supply; cleared by its rule)
    sp: wp.array3d[wp.float64]        # [W, B, K] kg
    p: wp.array2d[wp.float32]         # [W, B] the largest stagnation pressure ½ρw_n² [Pa]
    share: wp.array2d[wp.float32]     # [W, B] the jets' footprints over the area (<= 1)
    tau: wp.array2d[wp.float32]       # [W, B] the jet sheet's shear [Pa]


@wp.struct
class FilmOut:
    """The chart cells a squeezed sponge writes (the writer rule: ``src_acc``, ``sp.src``, ``cwet``)."""
    charts: wp.array[Chart]
    area: wp.array[wp.float64]
    tool_chart: wp.array[wp.int32]   # [n] the chart each tool works
    wins: wp.array2d[wp.vec4i]       # [W, n] the cells each tool wrote last
    src_acc: wp.array2d[wp.float64]
    src_sp: wp.array3d[wp.float64]
    cwet: wp.array2d[wp.int32]
    h: wp.array2d[wp.float64]        # the film a sponge takes up (after the film step; as the tools' stores do)
    base: wp.array2d[wp.float32]     # the substrate's height (the pad touches the film where h + base >= ceil)
    sp_mass: wp.array3d[wp.float64]
    taken: wp.array[wp.float64]      # [W] m³ the film lost to lumps (the film's ledger adds it)
    taken_sp: wp.array2d[wp.float64]  # [W, K] kg
    absorb: wp.array2d[wp.float64]   # [W, C] film porous pads absorbed [m] (WP15's tool stage), until taken here
    absorb_sp: wp.array3d[wp.float64]  # [W, C, K] its species [kg/m²]
    room: wp.array2d[wp.float64]     # [W, tools] the flow's tool_room: what each porous pad's sponge can take [m³]


@wp.func
def cell_take(tf: ToolField, fo: FilmOut, w: int, c: int, rate: wp.float64) -> wp.float64:
    """The depth [m] a sponge's pad takes from cell c where it presses its substrate (the film below its underside):
    ``min(1/2, 1 - exp(-rate·cover))`` of the cell's film; else 0. What a touching pad would plough, its pores absorb
    in the tool stage (WP15's porous pads; ``absorb_kernel`` takes it)."""
    hc = fo.h[w, c]
    if hc <= wp.float64(0.0) or tf.press[w, c] <= 0.0:
        return wp.float64(0.0)
    cov = wp.float64(tf.cover[w, c])
    return hc * wp.min(wp.float64(0.5), wp.float64(1.0) - wp.exp(-rate * cov))


@wp.func
def uptake(tf: ToolField, fo: FilmOut, w: int, k: int, room: wp.float64, rate: wp.float64, st: LumpState, b: int,
           n_sp: int) -> wp.float64:
    """A sponge under capacity takes each cell's ``cell_take`` (``rate = dt/τ_cap`` of the film step) and the cell's
    species by the same fraction (concentration unchanged; h stays >= 0), all scaled to fit ``room`` [m³] (GA's
    capillary uptake). On a frozen film k steps of dt take what one of k·dt does. Returns the volume."""
    win = fo.wins[w, k]
    ch = fo.charts[fo.tool_chart[k]]
    want = wp.float64(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            c = cell_at(ch, i, j)
            want = want + cell_take(tf, fo, w, c, rate) * fo.area[c]
    if want <= wp.float64(0.0):
        return wp.float64(0.0)
    scale = wp.min(wp.float64(1.0), room / want)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            c = cell_at(ch, i, j)
            d = cell_take(tf, fo, w, c, rate) * scale
            if d > wp.float64(0.0):
                f = d / fo.h[w, c]
                fo.h[w, c] = fo.h[w, c] - d
                for s in range(n_sp):
                    dm = fo.sp_mass[w, c, s] * f
                    fo.sp_mass[w, c, s] = fo.sp_mass[w, c, s] - dm
                    st.sp[w, b, s] = st.sp[w, b, s] + dm * fo.area[c]
                    st.sp_up[w, b, s] = st.sp_up[w, b, s] + dm * fo.area[c]
                    fo.taken_sp[w, s] = fo.taken_sp[w, s] + dm * fo.area[c]
    taken = want * scale
    fo.taken[w] = fo.taken[w] + taken
    return taken


@wp.func
def capacity_of(spec: LumpSpec, st: LumpState, w: int, b: int) -> wp.float64:
    pore = spec.pore[b]
    if pore[0] > wp.float64(0.0):
        return pore[0] * (wp.float64(1.0) - st.eps[w, b])
    return spec.hold[b] * spec.area[b]


@wp.kernel
def room_kernel(spec: LumpSpec, st: LumpState, fo: FilmOut):
    """Before the film's step: each sponge's room (capacity at its last strain - water) for its porous pad
    (``tool_room``: with room the pad absorbs the film it would plough)."""
    w = wp.tid()
    for b in range(spec.area.shape[0]):
        k = spec.tool[b]
        if k >= 0 and spec.pore[b][0] > wp.float64(0.0):
            fo.room[w, k] = wp.max(capacity_of(spec, st, w, b) - st.water[w, b], wp.float64(0.0))


@wp.kernel
def absorb_kernel(spec: LumpSpec, st: LumpState, fo: FilmOut, n_sp: int):
    """After the film's step, before the lump step: each sponge takes what its porous pad absorbed over the exchange
    (``absorb``, ``absorb_sp`` over its tool's cells: uptake, the film's ``to_lumps``) and zeroes it; what is over its
    capacity goes back to those cells pro rata with the sponge's mixed species (``src_acc``, ``sp.src``), under the
    pad, booked squeezed (the pad ploughs it out by the ceiling path)."""
    w = wp.tid()
    zero = wp.float64(0.0)
    for b in range(spec.area.shape[0]):
        k = spec.tool[b]
        if k < 0 or spec.pore[b][0] <= zero:
            continue
        win = fo.wins[w, k]
        ch = fo.charts[fo.tool_chart[k]]
        got = zero
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = cell_at(ch, i, j)
                got = got + fo.absorb[w, c] * fo.area[c]
        if got <= zero:
            continue
        st.water[w, b] = st.water[w, b] + got
        st.ledger[w, b, UPTAKE] = st.ledger[w, b, UPTAKE] + got
        fo.taken[w] = fo.taken[w] + got
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = cell_at(ch, i, j)
                for s in range(n_sp):
                    dm = fo.absorb_sp[w, c, s] * fo.area[c]
                    st.sp[w, b, s] = st.sp[w, b, s] + dm
                    st.sp_up[w, b, s] = st.sp_up[w, b, s] + dm
                    fo.taken_sp[w, s] = fo.taken_sp[w, s] + dm
        out = st.water[w, b] - capacity_of(spec, st, w, b)
        frac = zero
        if out > zero:
            frac = out / st.water[w, b]
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = cell_at(ch, i, j)
                share = fo.absorb[w, c] * fo.area[c] / got
                if out > zero and share > zero:                    # back to the cells it came from, pro rata
                    fo.src_acc[w, c] = fo.src_acc[w, c] + out * share / fo.area[c]
                    fo.cwet[w, c] = 1
                    for s in range(n_sp):
                        dm = st.sp[w, b, s] * frac * share
                        fo.src_sp[w, c, s] = fo.src_sp[w, c, s] + dm / fo.area[c]
                        st.sp_sq[w, b, s] = st.sp_sq[w, b, s] + dm
                fo.absorb[w, c] = zero
                for s in range(n_sp):
                    fo.absorb_sp[w, c, s] = zero
        if out > zero:
            st.water[w, b] = st.water[w, b] - out
            st.ledger[w, b, SQUEEZED] = st.ledger[w, b, SQUEEZED] + out
            for s in range(n_sp):
                st.sp[w, b, s] = st.sp[w, b, s] * (wp.float64(1.0) - frac)


@wp.func
def foam_strain(p: wp.float64, pore: wp.vec4d) -> wp.float64:
    """ε with ``p/E = ε + 3ε³`` (Newton from min(x, 1); the cubic is monotone), capped at the stop; 0 for no (or a
    non-finite) pressure."""
    if not (p > wp.float64(0.0)):
        return wp.float64(0.0)
    x = p / pore[1]
    e = wp.min(x, wp.float64(1.0))
    for _ in range(30):
        e = e - (e + wp.float64(3.0) * e * e * e - x) / (wp.float64(1.0) + wp.float64(9.0) * e * e)
    return wp.max(wp.min(e, pore[2]), wp.float64(0.0))


@wp.func
def pad_pressure(tf: ToolField, fo: FilmOut, w: int, k: int) -> wp.vec2d:
    """(mean pad pressure over the tool's cover [Pa], covered area [m²]) of tool k."""
    win = fo.wins[w, k]
    ch = fo.charts[fo.tool_chart[k]]
    num = wp.float64(0.0)
    den = wp.float64(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            c = cell_at(ch, i, j)
            a = wp.float64(tf.cover[w, c]) * fo.area[c]
            num = num + wp.float64(tf.press[w, c]) * a
            den = den + a
    if den <= wp.float64(0.0):
        return wp.vec2d(0.0, 0.0)
    return wp.vec2d(num / den, den)


@wp.func
def squeeze_onto(tf: ToolField, fo: FilmOut, w: int, k: int, dv: wp.float64, frac: wp.float64, st: LumpState,
                 b: int, n_sp: int, cover_area: wp.float64):
    """``dv`` [m³] (and ``frac`` of the body's species) onto the cells under tool k, by cover·area."""
    win = fo.wins[w, k]
    ch = fo.charts[fo.tool_chart[k]]
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            c = cell_at(ch, i, j)
            wt = wp.float64(tf.cover[w, c]) * fo.area[c] / cover_area
            if wt > wp.float64(0.0):
                fo.src_acc[w, c] = fo.src_acc[w, c] + dv * wt / fo.area[c]
                for s in range(n_sp):
                    dm = st.sp[w, b, s] * frac * wt
                    fo.src_sp[w, c, s] = fo.src_sp[w, c, s] + dm / fo.area[c]
                    st.sp_sq[w, b, s] = st.sp_sq[w, b, s] + dm
                fo.cwet[w, c] = 1


@wp.func
def covered(tf: ToolField, ch: Chart, w: int, i: int, j: int) -> bool:
    """Cell (i, j) of chart ``ch`` (j wrapped on a revolution chart) lies under a tool (cover > 0)."""
    if i < 0 or i >= ch.nx:
        return False
    jj = j
    if ch.wrap != 0:
        jj = (j % ch.ny + ch.ny) % ch.ny
    elif j < 0 or j >= ch.ny:
        return False
    return tf.cover[w, ch.cell_start + i * ch.ny + jj] > 0.0


@wp.func
def ring_cell(tf: ToolField, ch: Chart, w: int, i: int, j: int) -> int:
    """The cell (i, j) if it is just outside the tools' cover (uncovered, a 4-neighbour covered), else -1."""
    if i < 0 or i >= ch.nx:
        return -1
    jj = j
    if ch.wrap != 0:
        jj = (j % ch.ny + ch.ny) % ch.ny
    elif j < 0 or j >= ch.ny:
        return -1
    if covered(tf, ch, w, i, jj):
        return -1
    if covered(tf, ch, w, i - 1, jj) or covered(tf, ch, w, i + 1, jj) or covered(tf, ch, w, i, jj - 1) \
            or covered(tf, ch, w, i, jj + 1):
        return ch.cell_start + i * ch.ny + jj
    return -1


@wp.func
def squeeze_ring(tf: ToolField, fo: FilmOut, w: int, k: int, dv: wp.float64, frac: wp.float64, st: LumpState,
                 b: int, n_sp: int) -> bool:
    """``dv`` [m³] (and ``frac`` of the body's species) onto the ring of cells just outside tool k's pad, pro rata by
    area (WP15's ruling on JOB3's wring: squeezed water does not pile up under a pad that bears on the substrate).
    False (nothing moved) when the pad has no such ring on its chart."""
    win = fo.wins[w, k]
    ch = fo.charts[fo.tool_chart[k]]
    n_j = win[3] - win[2] + 3
    if ch.wrap != 0:
        n_j = wp.min(n_j, ch.ny)
    ring = wp.float64(0.0)
    for i in range(win[0] - 1, win[1] + 2):
        for jj in range(n_j):
            c = ring_cell(tf, ch, w, i, win[2] - 1 + jj)
            if c >= 0:
                ring = ring + fo.area[c]
    if ring <= wp.float64(0.0):
        return False
    for i in range(win[0] - 1, win[1] + 2):
        for jj in range(n_j):
            c = ring_cell(tf, ch, w, i, win[2] - 1 + jj)
            if c >= 0:
                wt = fo.area[c] / ring
                fo.src_acc[w, c] = fo.src_acc[w, c] + dv * wt / fo.area[c]
                for s in range(n_sp):
                    dm = st.sp[w, b, s] * frac * wt
                    fo.src_sp[w, c, s] = fo.src_sp[w, c, s] + dm / fo.area[c]
                    st.sp_sq[w, b, s] = st.sp_sq[w, b, s] + dm
                fo.cwet[w, c] = 1
    return True


@wp.kernel
def weight_kernel(spec: LumpSpec, st: LumpState, bodies: wp.array2d[wp.int32], gravity: wp.array[wp.vec3],
                  rho: float, body_f: wp.array[wp.spatial_vector]):
    """Per world: each body carries the weight of its water beyond what it was built with, ``ρ·(water - built)·g``
    at its centre of mass (WP15's ruling: graph-safe, no live mass change; the water's inertia stays the built
    one)."""
    w = wp.tid()
    g = gravity[w]
    for b in range(spec.area.shape[0]):
        m = rho * float(st.water[w, b] - spec.built[b])
        add_wrench(body_f, bodies[w, b], wp.spatial_vector(m * g[0], m * g[1], m * g[2], 0.0, 0.0, 0.0))


@wp.func
def erode(spec: LumpSpec, st: LumpState, inp: LumpIn, w: int, b: int, dt: wp.float64) -> wp.float64:
    """One exchange of the §5 laws on a wet body: impact ``k_i·share·(p/σ_c - 1)+`` and the sheet's shear
    ``k_e·share·(τ - σ_c)+`` at the starting strength (scaled to the soil left), then soaking. Returns kg removed."""
    zero, one = wp.float64(0.0), wp.float64(1.0)
    m = st.soil[w, b]
    law = spec.law[b]
    hy = st.hyd[w, b]
    st.hyd[w, b] = one - (one - hy) * wp.exp(-dt / law.t_soak)
    if m <= zero:
        return zero
    sig = law.sigma_dry * (one - hy) * (one - hy) + law.sigma_wet
    share = wp.float64(inp.share[w, b])
    t_i = law.k_i * share * wp.max(wp.float64(inp.p[w, b]) / sig - one, zero) * dt
    t_e = law.k_e * share * wp.max(wp.float64(inp.tau[w, b]) - sig, zero) * dt
    total = t_i + t_e
    if total <= zero:
        return zero
    scale = wp.min(total, m) / total
    st.soil[w, b] = m - total * scale
    st.by[w, b, 0] = st.by[w, b, 0] + t_i * scale
    st.by[w, b, 1] = st.by[w, b, 1] + t_e * scale
    return total * scale * spec.area[b]


@wp.func
def clock(spec: LumpSpec, st: LumpState, inp: LumpIn, w: int, b: int, dt: wp.float64):
    """The sanitizer clock: unbroken time with the water's sanitizer at ``c >= c_min`` on a wet body (as erosion's)."""
    if spec.k_san < 0:
        return
    v = st.water[w, b]
    wet = v > spec.wet[b] or (inp.share[w, b] > 0.0 and v > wp.float64(0.0))
    if wet and st.sp[w, b, spec.k_san] >= spec.c_min * v:
        st.expo[w, b] = st.expo[w, b] + dt
        st.expo_best[w, b] = wp.max(st.expo_best[w, b], st.expo[w, b])
    else:
        st.expo[w, b] = wp.float64(0.0)


@wp.kernel
def lump_kernel(spec: LumpSpec, st: LumpState, inp: LumpIn, tf: ToolField, fo: FilmOut, n_sp: int, dt: wp.float64,
                tick: wp.array[wp.int32], stride: int):
    """Per world, each body: the jets' water and species in; erosion (the soil joins the water); the sanitizer clock;
    a sponge's strain from its pad; water over capacity squeezed onto the pad's cells, else shed."""
    w = wp.tid()
    zero = wp.float64(0.0)
    for b in range(spec.area.shape[0]):
        dv = inp.vol[w, b]
        st.water[w, b] = st.water[w, b] + dv
        st.ledger[w, b, IN] = st.ledger[w, b, IN] + dv
        for s in range(n_sp):
            st.sp[w, b, s] = st.sp[w, b, s] + inp.sp[w, b, s]
            st.sp_in[w, b, s] = st.sp_in[w, b, s] + inp.sp[w, b, s]
        if spec.soil_on[b] != 0 and (st.water[w, b] > spec.wet[b] or inp.share[w, b] > 0.0):
            dm = erode(spec, st, inp, w, b, dt)
            ks = spec.k_soil[b]
            if ks >= 0:
                st.sp[w, b, ks] = st.sp[w, b, ks] + dm
                st.soil_sp[w, b] = st.soil_sp[w, b] + dm
        clock(spec, st, inp, w, b, dt)
        pore = spec.pore[b]
        cap = spec.hold[b] * spec.area[b]
        pad = wp.vec2d(0.0, 0.0)
        k = spec.tool[b]
        if pore[0] > zero:
            if k >= 0 and tf.on != 0:
                pad = pad_pressure(tf, fo, w, k)
            st.eps[w, b] = foam_strain(pad[0], pore)
            cap = pore[0] * (wp.float64(1.0) - st.eps[w, b])
            if pad[1] > zero and st.water[w, b] < cap and tick[w] % stride == 0:    # on film steps, at their dt
                f_dt = dt * wp.float64(stride)
                got = uptake(tf, fo, w, k, cap - st.water[w, b], f_dt / pore[3], st, b, n_sp)
                st.water[w, b] = st.water[w, b] + got
                st.ledger[w, b, UPTAKE] = st.ledger[w, b, UPTAKE] + got
        v = st.water[w, b]
        if v <= cap:
            continue
        out = v - cap
        frac = out / v
        if pad[0] > zero:                             # pressed: beside its pad (WP15: a porous pad bears on the
            if not squeeze_ring(tf, fo, w, k, out, frac, st, b, n_sp):    # substrate), else under it
                squeeze_onto(tf, fo, w, k, out, frac, st, b, n_sp, pad[1])
            st.ledger[w, b, SQUEEZED] = st.ledger[w, b, SQUEEZED] + out
            for s in range(n_sp):
                st.sp[w, b, s] = st.sp[w, b, s] * (wp.float64(1.0) - frac)
        elif st.pool_of[w, b] >= 0 or st.vessel_of[w, b] >= 0:   # immersed or in a vessel: lumped_pool.pool_ask
            continue
        else:
            st.ledger[w, b, SHED] = st.ledger[w, b, SHED] + out
            if spec.drip_on != 0:
                st.drip[w, b] = st.drip[w, b] + out
            for s in range(n_sp):
                gone = st.sp[w, b, s] * frac
                st.sp[w, b, s] = st.sp[w, b, s] - gone
                st.sp_shed[w, b, s] = st.sp_shed[w, b, s] + gone
                if spec.drip_on != 0:
                    st.drip_sp[w, b, s] = st.drip_sp[w, b, s] + gone
        st.water[w, b] = cap


__all__ = ["FilmOut", "IN", "LumpIn", "LumpSpec", "LumpState", "POOLED", "SHED", "SQUEEZED", "UPTAKE", "absorb_kernel",
           "capacity_of", "cell_take", "foam_strain", "lump_kernel", "room_kernel", "uptake"]
