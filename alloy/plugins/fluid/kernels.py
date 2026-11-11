# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the lumped-fluid core: one thread per world steps that world's whole network.

Nodes hold an absolute pressure P [Pa] and a volume V [m³] (written by the node's owner each step). A node's amount is
``a = κ(V)·P + λ(V)``: gas a = P·V/n [Pa·m³] (polytropic index n, 1 = isothermal); liquid a = fluid volume
V·(1 + (P - P_ref)/K) [m³]. A volume change from V_old to V_new is exact first: gas P·(V_old/V_new)^n (n = 1: the
amount is kept), liquid the amount is kept. Inflow into an n > 1 node arrives at the node's temperature (no
temperature state). An open tank (``TANK``) holds liquid under a free surface: P = P_ref + ρ·g·V/A, amount V
(κ = A/(ρ·g), kept in ``bulk``; ``volume`` is unused).
Edges carry Q = g·(P_b - P_a) into node ``a`` (b a node or a fixed-pressure reservoir), with g evaluated at the start
of the step (``edge_conductance``), plus flow sources. The step is the θ-method on the amounts (θ = ½, Crank-Nicolson;
θ = 1 when a node is stiff, dt·Σg/κ > 2, where CN's amplification tends to -1), a dense solve without pivoting (the
matrix is a diagonally dominant M-matrix). The volume change comes first, exactly, and the flow step starts from the
pressures after it (conductances and the explicit half included). A single node is stepped exactly instead. Every edge's transported amount accumulates in a float64 ledger, so amount changes balance exactly;
``discharge[w, e]`` is the step's mean flow into node a (the ledger increment over dt). ``inflow[w, i]`` is an
external inflow into node i over the step (a film's drains into a tank), booked in ``inflow_total`` and zeroed.

Energy ledgers (float64): ``node_work[w, i]`` = ∫ (P - P_ref)·dV of node i's volume changes (the fluid's work on its
owner, trapezoidal over the exact volume step); ``work[w, e]`` = liquid and tank edges only: a passive edge's throttling loss
∫ Q·(P_b - P_a) dt (>= 0), a source's pump work ∫ q·(P_a - P_res) dt (P_res: its tank). Exact in time for one node,
θ-pressures otherwise. Gas edges book 0 (an isothermal exergy ledger is v2).
"""

from __future__ import annotations

import warp as wp

MAX_NODES = 16        # a dense per-world solve, O(n³) per step (16: ~1.4k flops); arrays are sized per network
MAX_EDGES = 32

GAS = wp.constant(0)
LIQUID = wp.constant(1)
TANK = wp.constant(2)        # an open tank: P = P_ref + ρ·g·V/A, so κ = A/(ρ·g) (held in ``bulk``), amount = V

LINEAR = wp.constant(0)      # g = G·u                                    (leak, laminar line, valve)
ORIFICE = wp.constant(1)     # g = Cd·A·sqrt(2/ρ) / sqrt(max(|ΔP|, ΔP_lam))·u   (liquid)
CHECK = wp.constant(2)       # g = G·u while P_b - P_a > p_set (flow only into node a), else 0
RELIEF = wp.constant(3)      # vents G·u·(P_a - p_set) while P_a > p_set (p_res holds p_set), else 0
PUMP_GAS = wp.constant(4)    # vacuum pump to a reservoir at P_ult: g = S·u (throughput S·(P - P_ult))
SOURCE = wp.constant(5)      # a flow source q = flow·u into node a (liquid pump); no conductance
ORIFICE_GAS = wp.constant(6)  # ISO 6358: Q = C·p_up·ψ·ρ0·R·T0 (C in cd_area, b in p_set), g = Q / max(|ΔP|, ΔP_lam)·u
EXPEL = wp.constant(7)       # aerosol can: liquid out of gas node a's container, Q = Cd·A·sqrt(2·(P_a - P_res)/ρ)·u
                             # (liquid m³/s; capacity in ``flow``); the gas expands into the freed volume
RHO0_R_T0 = 1.185 * 287.05 * 293.15   # Pa: ISO 6358 ρ0·R_air·T0 (throughput = C·p_up·ψ·this); literal in the kernel

THETA_STIFF = 2.0                 # dt·Σg/κ above this: θ = 1 (literal in step_network)


@wp.func
def node_kappa(medium: wp.int32, v: wp.float64, bulk: wp.float64, poly: wp.float64) -> wp.float64:
    if medium == LIQUID:
        return v / bulk
    if medium == TANK:                                    # κ = A/(ρ·g), independent of the (unused) volume
        return bulk
    return v / poly


@wp.func
def node_lambda(medium: wp.int32, v: wp.float64, bulk: wp.float64, p_ref: wp.float64) -> wp.float64:
    if medium == LIQUID:
        return v * (wp.float64(1.0) - p_ref / bulk)
    if medium == TANK:                                    # amount κ·(P - P_ref) = the liquid volume in the tank
        return -bulk * p_ref
    return wp.float64(0.0)


@wp.func
def after_volume_change(medium: wp.int32, p: wp.float64, v_old: wp.float64, v_new: wp.float64, bulk: wp.float64,
                        p_ref: wp.float64, poly: wp.float64) -> wp.float64:
    """The pressure after the exact volume change V_old -> V_new (closed node): gas P·(V_old/V_new)^n, liquid the
    fluid volume kept. Unchanged (bit-exact) when V is."""
    if v_new == v_old:
        return p
    if medium == GAS and poly != wp.float64(1.0):
        return p * wp.pow(v_old / v_new, poly)
    k_old = node_kappa(medium, v_old, bulk, poly)
    return (k_old * p + node_lambda(medium, v_old, bulk, p_ref) - node_lambda(medium, v_new, bulk, p_ref)) / \
        node_kappa(medium, v_new, bulk, poly)


@wp.func
def iso6358_psi(r: wp.float64, b: wp.float64) -> wp.float64:
    """ISO 6358 flow function of the pressure ratio r = p_down/p_up: 1 when choked (r <= b), else the ellipse."""
    if r <= b:
        return wp.float64(1.0)
    x = (r - b) / (wp.float64(1.0) - b)
    return wp.sqrt(wp.max(wp.float64(1.0) - x * x, wp.float64(0.0)))


@wp.func
def edge_conductance(kind: wp.int32, g0: wp.float64, cd_area: wp.float64, rho: wp.float64, dp_lam: wp.float64,
                     p_set: wp.float64, u: wp.float64, pa: wp.float64, pb: wp.float64) -> wp.float64:
    """g of one edge from the pressures at the start of the step (pb: the far node or the reservoir)."""
    g = wp.float64(0.0)
    if kind == LINEAR or kind == PUMP_GAS:
        g = g0 * u
    elif kind == ORIFICE:
        dp = wp.max(wp.abs(pb - pa), dp_lam)
        g = cd_area * wp.sqrt(wp.float64(2.0) / rho) / wp.sqrt(dp) * u
    elif kind == CHECK:
        if pb - pa > p_set:
            g = g0 * u
    elif kind == RELIEF:                                  # pb is p_set for a relief edge
        if pa > p_set:
            g = g0 * u
    elif kind == ORIFICE_GAS:                             # cd_area = C [m³/(s·Pa)], p_set = b
        p_up = wp.max(pa, pb)
        dp = wp.max(wp.abs(pb - pa), dp_lam)
        if p_up > wp.float64(0.0):
            r = wp.max(p_up - dp, wp.float64(0.0)) / p_up
            g = cd_area * wp.float64(99716.2183875) * p_up * iso6358_psi(r, p_set) / dp * u
    return wp.max(g, wp.float64(0.0))


@wp.func
def _solve(w: wp.int32, n: wp.int32, mat: wp.array3d[wp.float64], rhs: wp.array2d[wp.float64]):
    """Gaussian elimination without pivoting, in place: rhs[w] becomes the solution."""
    for k in range(n):
        piv = mat[w, k, k]
        for i in range(k + 1, n):
            f = mat[w, i, k] / piv
            if f != wp.float64(0.0):
                for j in range(k, n):
                    mat[w, i, j] = mat[w, i, j] - f * mat[w, k, j]
                rhs[w, i] = rhs[w, i] - f * rhs[w, k]
    for kk in range(n):
        k = n - 1 - kk
        s = rhs[w, k]
        for j in range(k + 1, n):
            s = s - mat[w, k, j] * rhs[w, j]
        rhs[w, k] = s / mat[w, k, k]


@wp.func
def _book_inflow(w: wp.int32, n: wp.int32, dt: wp.float64, inflow: wp.array2d[wp.float64],
                 inflow_total: wp.array2d[wp.float64]):
    """The external inflow of this step goes into the ledger and is consumed (zeroed): nothing carries over."""
    for i in range(n):
        inflow_total[w, i] = inflow_total[w, i] + dt * inflow[w, i]
        inflow[w, i] = wp.float64(0.0)


@wp.func
def _source_flow(w: wp.int32, e: wp.int32, dt: wp.float64, a: wp.int32, flow: wp.float64,
                 opening: wp.array2d[wp.float32], medium: wp.array[wp.int32], bulk: wp.array[wp.float64],
                 p_ref: wp.array[wp.float64], pressure: wp.array2d[wp.float64]) -> wp.float64:
    """A source's flow this step, flow·opening; one drawing from an open tank takes at most what the tank holds."""
    q = flow * wp.float64(opening[w, e])
    if q < wp.float64(0.0) and medium[a] == TANK:
        q = wp.max(q, -bulk[a] * wp.max(pressure[w, a] - p_ref[a], wp.float64(0.0)) / dt)
    return q


@wp.func
def _refill_pumps(w: wp.int32, i: wp.int32, m: wp.int32, dt: wp.float64, kind: wp.array[wp.int32],
                  ea: wp.array[wp.int32], kappa: wp.float64, p_ref: wp.float64, p_new: wp.float64,
                  discharge: wp.array2d[wp.float64]) -> wp.float64:
    """After the step: if tank i ends below empty (an outlet took the rest while a pump drew its start volume), the
    pumps drawing from it give back the deficit (their discharge shrinks). Returns the tank's corrected pressure."""
    deficit = kappa * (p_ref - p_new)
    if deficit <= wp.float64(0.0):
        return p_new
    given = wp.float64(0.0)
    for e in range(m):
        if kind[e] == SOURCE and ea[e] == i and discharge[w, e] < wp.float64(0.0):
            give = wp.min(deficit - given, -discharge[w, e] * dt)
            discharge[w, e] = discharge[w, e] + give / dt
            given = given + give
    return p_new + given / kappa


@wp.func
def _expel(w: wp.int32, e: wp.int32, dt: wp.float64, a: wp.int32, p_out: wp.float64, cd_area: wp.float64,
           rho: wp.float64, dp_lam: wp.float64, capacity: wp.float64, opening: wp.array2d[wp.float32],
           pressure: wp.array2d[wp.float64], volume: wp.array2d[wp.float64], ledger: wp.array2d[wp.float64],
           discharge: wp.array2d[wp.float64]):
    """An aerosol can's dip-tube valve: liquid leaves at Cd·A·sqrt(2·ΔP/ρ) (start-of-step pressure, linear below
    ``dp_lam``) until ``capacity`` is spent; the gas node's volume grows by it (the exact polytropic change follows).
    The ledger books the liquid volume (negative: out); the discharge is its rate."""
    dp = pressure[w, a] - p_out
    dv = wp.float64(0.0)
    if dp > wp.float64(0.0):
        g = cd_area * wp.sqrt(wp.float64(2.0) / rho) / wp.sqrt(wp.max(dp, dp_lam)) * wp.float64(opening[w, e])
        dv = wp.min(g * dp * dt, wp.max(capacity + ledger[w, e], wp.float64(0.0)))
    volume[w, a] = volume[w, a] + dv
    ledger[w, e] = ledger[w, e] - dv
    discharge[w, e] = -dv / dt


@wp.func
def _step_one(w: wp.int32, m: wp.int32, dt: wp.float64, medium: wp.int32, bulk: wp.float64, p_ref: wp.float64,
              poly: wp.float64, p0: wp.float64, kind: wp.array[wp.int32], ea: wp.array[wp.int32],
              p_res: wp.array[wp.float64],
              flow: wp.array[wp.float64], opening: wp.array2d[wp.float32], volume: wp.array2d[wp.float64],
              inflow: wp.float64,
              g: wp.array2d[wp.float64], pressure: wp.array2d[wp.float64], volume_old: wp.array2d[wp.float64],
              ledger: wp.array2d[wp.float64], work: wp.array2d[wp.float64], discharge: wp.array2d[wp.float64]):
    """One node, all edges to reservoirs or sources: κ·dP/dt = Σ g_e·(P_e - P) + q from P0 (after the exact volume
    change), stepped exactly. ``inflow`` (the node's external inflow) adds to q."""
    k_new = node_kappa(medium, volume[w, 0], bulk, poly)
    gsum, gp, q = wp.float64(0.0), wp.float64(0.0), inflow
    for e in range(m):
        if kind[e] == SOURCE:
            q = q + discharge[w, e]
        elif kind[e] != EXPEL:
            gsum = gsum + g[w, e]
            gp = gp + g[w, e] * p_res[e]
    p1 = p0 + dt * q / k_new
    decay = wp.float64(0.0)                               # ∫ (P(t) - P_ss) dt over the step / (P0 - P_ss)
    decay2 = wp.float64(0.0)                              # ∫ (P(t) - P_ss)² dt / (P0 - P_ss)²
    p_ss = p0
    if gsum > wp.float64(0.0):
        b = gsum / k_new
        p_ss = (gp + q) / gsum
        e1 = wp.exp(-b * dt)
        p1 = p_ss + (p0 - p_ss) * e1
        decay = (wp.float64(1.0) - e1) / b
        decay2 = (wp.float64(1.0) - e1 * e1) / (wp.float64(2.0) * b)
    if medium == TANK:
        p1 = _refill_pumps(w, 0, m, dt, kind, ea, bulk, p_ref, p1, discharge)
    for e in range(m):
        if kind[e] == EXPEL:
            continue
        if kind[e] == SOURCE:
            qs = discharge[w, e]
            ledger[w, e] = ledger[w, e] + dt * qs
            if medium != GAS:                            # ∫ q·(P - P_res) dt: P exponential, or linear without g
                pint = (p_ss - p_res[e]) * dt + (p0 - p_ss) * decay
                if gsum == wp.float64(0.0):
                    pint = (p0 - p_res[e]) * dt + wp.float64(0.5) * q * dt * dt / k_new
                work[w, e] = work[w, e] + qs * pint
        elif gsum > wp.float64(0.0):
            inc = g[w, e] * ((p_res[e] - p_ss) * dt - (p0 - p_ss) * decay)
            ledger[w, e] = ledger[w, e] + inc
            discharge[w, e] = inc / dt
            if medium != GAS:                            # ∫ g·(P_res - P)² dt
                d = p_res[e] - p_ss
                d0 = p0 - p_ss
                work[w, e] = work[w, e] + g[w, e] * (d * d * dt - wp.float64(2.0) * d * d0 * decay + d0 * d0 * decay2)
        else:
            discharge[w, e] = wp.float64(0.0)
    pressure[w, 0] = p1
    volume_old[w, 0] = volume[w, 0]


@wp.kernel
def step_network(n: wp.int32, m: wp.int32, dt: wp.float64,
                 medium: wp.array[wp.int32], bulk: wp.array[wp.float64], p_ref: wp.array[wp.float64],
                 poly: wp.array[wp.float64],
                 kind: wp.array[wp.int32], ea: wp.array[wp.int32], eb: wp.array[wp.int32],
                 p_res: wp.array[wp.float64], g0: wp.array[wp.float64], cd_area: wp.array[wp.float64],
                 rho: wp.array[wp.float64], dp_lam: wp.array[wp.float64], p_set: wp.array[wp.float64],
                 flow: wp.array[wp.float64], opening: wp.array2d[wp.float32], volume: wp.array2d[wp.float64],
                 inflow: wp.array2d[wp.float64],
                 g: wp.array2d[wp.float64], mat: wp.array3d[wp.float64], rhs: wp.array2d[wp.float64],
                 pv: wp.array2d[wp.float64], pressure: wp.array2d[wp.float64], volume_old: wp.array2d[wp.float64],
                 ledger: wp.array2d[wp.float64], work: wp.array2d[wp.float64], node_work: wp.array2d[wp.float64],
                 discharge: wp.array2d[wp.float64], inflow_total: wp.array2d[wp.float64]):
    w = wp.tid()
    one, half = wp.float64(1.0), wp.float64(0.5)
    for e in range(m):                                   # 0. sources (into discharge) and aerosol cans
        if kind[e] == SOURCE:
            discharge[w, e] = _source_flow(w, e, dt, ea[e], flow[e], opening, medium, bulk, p_ref, pressure)
        if kind[e] == EXPEL:
            _expel(w, e, dt, ea[e], p_res[e], cd_area[e], rho[e], dp_lam[e], flow[e], opening, pressure, volume,
                   ledger, discharge)
    for i in range(n):                                   # 1. the exact volume change: amount -> rhs, pressure -> pv
        k_new = node_kappa(medium[i], volume[w, i], bulk[i], poly[i])
        if medium[i] == GAS and poly[i] != one:
            rhs[w, i] = k_new * after_volume_change(medium[i], pressure[w, i], volume_old[w, i], volume[w, i], bulk[i],
                                                    p_ref[i], poly[i])
        else:                                            # kept literally: n = 1 and liquid bit-identical
            k_old = node_kappa(medium[i], volume_old[w, i], bulk[i], poly[i])
            rhs[w, i] = (k_old * pressure[w, i] + node_lambda(medium[i], volume_old[w, i], bulk[i], p_ref[i])
                         - node_lambda(medium[i], volume[w, i], bulk[i], p_ref[i]))
        pv[w, i] = pressure[w, i]
        if volume[w, i] != volume_old[w, i]:
            pv[w, i] = rhs[w, i] / k_new
        node_work[w, i] = node_work[w, i] + (half * (pressure[w, i] + pv[w, i]) - p_ref[i]) * (volume[w, i] -
                                                                                             volume_old[w, i])
    for e in range(m):                                   # 2. conductances at the start of the flow step
        a0 = int(ea[e])
        b0 = int(eb[e])
        pb = wp.float64(p_res[e])
        if b0 >= 0:
            pb = wp.float64(pv[w, b0])
        g[w, e] = edge_conductance(kind[e], g0[e], cd_area[e], rho[e], dp_lam[e], p_set[e],
                                   wp.float64(opening[w, e]), wp.float64(pv[w, a0]), pb)
    stiff = int(0)
    for i in range(n):
        gsum = wp.float64(0.0)
        for e in range(m):
            if ea[e] == i or eb[e] == i:
                gsum = gsum + g[w, e]
        k_stiff = node_kappa(medium[i], volume[w, i], bulk[i], poly[i])
        if dt * gsum > wp.float64(2.0) * k_stiff:
            stiff = 1
    if n == 1:
        _step_one(w, m, dt, medium[0], bulk[0], p_ref[0], poly[0], pv[w, 0], kind, ea, p_res, flow, opening, volume,
                  inflow[w, 0], g, pressure, volume_old, ledger, work, discharge)
        _book_inflow(w, n, dt, inflow, inflow_total)
        return
    theta = half
    if stiff != 0:
        theta = one
    for i in range(n):                                   # 3. A = diag(κ'), rhs = the amount after the volume change
        for j in range(n):
            mat[w, i, j] = wp.float64(0.0)
        mat[w, i, i] = node_kappa(medium[i], volume[w, i], bulk[i], poly[i])
        if inflow[w, i] != wp.float64(0.0):              # external inflow (drains into a tank), constant over dt
            rhs[w, i] = rhs[w, i] + dt * inflow[w, i]
    for e in range(m):
        a = int(ea[e])
        b = int(eb[e])
        ge = wp.float64(g[w, e])
        if kind[e] == SOURCE:
            rhs[w, a] = rhs[w, a] + dt * discharge[w, e]
        elif b >= 0:
            d = pv[w, b] - pv[w, a]
            mat[w, a, a] = mat[w, a, a] + theta * dt * ge
            mat[w, a, b] = mat[w, a, b] - theta * dt * ge
            mat[w, b, b] = mat[w, b, b] + theta * dt * ge
            mat[w, b, a] = mat[w, b, a] - theta * dt * ge
            rhs[w, a] = rhs[w, a] + (one - theta) * dt * ge * d
            rhs[w, b] = rhs[w, b] - (one - theta) * dt * ge * d
        else:
            mat[w, a, a] = mat[w, a, a] + theta * dt * ge
            rhs[w, a] = rhs[w, a] + theta * dt * ge * p_res[e] + (one - theta) * dt * ge * (p_res[e] - pv[w, a])
    _solve(w, n, mat, rhs)
    for i in range(n):                                   # a pump never takes a tank below empty
        if medium[i] == TANK:
            rhs[w, i] = _refill_pumps(w, i, m, dt, kind, ea, bulk[i], p_ref[i], rhs[w, i], discharge)
    for e in range(m):                                   # ledger: amount carried into node a this step
        a2 = int(ea[e])
        b2 = int(eb[e])
        if kind[e] == EXPEL:
            continue
        if kind[e] == SOURCE:
            qs = discharge[w, e]
            ledger[w, e] = ledger[w, e] + dt * qs
            if medium[a2] != GAS:
                pa_th = theta * rhs[w, a2] + (one - theta) * pv[w, a2]
                work[w, e] = work[w, e] + dt * qs * (pa_th - p_res[e])
        else:
            pb_old = wp.float64(p_res[e])
            pb_new = wp.float64(p_res[e])
            if b2 >= 0:
                pb_old = wp.float64(pv[w, b2])
                pb_new = wp.float64(rhs[w, b2])
            dp_th = theta * (pb_new - rhs[w, a2]) + (one - theta) * (pb_old - pv[w, a2])
            q = g[w, e] * dp_th
            ledger[w, e] = ledger[w, e] + dt * q
            discharge[w, e] = q
            if medium[a2] != GAS:
                work[w, e] = work[w, e] + dt * q * dp_th
    for i in range(n):
        pressure[w, i] = rhs[w, i]
        volume_old[w, i] = volume[w, i]
    _book_inflow(w, n, dt, inflow, inflow_total)



@wp.kernel
def set_opening(col: wp.int32, values: wp.array[wp.float32], opening: wp.array2d[wp.float32]):
    """opening[w, col] = values[w] (an input array driving one edge)."""
    w = wp.tid()
    opening[w, col] = values[w]


@wp.func
def node_amount(medium: wp.int32, p: wp.float64, v: wp.float64, bulk: wp.float64, p_ref: wp.float64,
                poly: wp.float64) -> wp.float64:
    """The amount κ(V)·P + λ(V): a liquid's or tank's fluid volume [m³] (gas: P·V/n)."""
    return node_kappa(medium, v, bulk, poly) * p + node_lambda(medium, v, bulk, p_ref)


@wp.kernel
def step_species(n: wp.int32, m: wp.int32, n_species: wp.int32, dt: wp.float64, medium: wp.array[wp.int32],
                 bulk: wp.array[wp.float64], p_ref: wp.array[wp.float64], poly: wp.array[wp.float64],
                 ea: wp.array[wp.int32], eb: wp.array[wp.int32], conc_in: wp.array2d[wp.float64],
                 discharge: wp.array2d[wp.float64], pressure: wp.array2d[wp.float64],
                 volume: wp.array2d[wp.float64], mat: wp.array3d[wp.float64], rhs: wp.array2d[wp.float64],
                 species: wp.array3d[wp.float64], species_ledger: wp.array3d[wp.float64],
                 inflow_species: wp.array3d[wp.float64], species_in_total: wp.array3d[wp.float64],
                 out_conc: wp.array3d[wp.float64]):
    """Dissolved species (soap, sanitizer) carried by the step's edge flows, after ``step_network``: well-mixed
    liquid and tank nodes, upwind and implicit (backward Euler), so concentrations stay bounded at any dt.

    Per species and node: c'·a' + dt·Σ_out q·c' - dt·Σ_in q·c'_up = M + dt·(Σ_reservoir-in q·c_in + external), with a'
    the node's new amount; then M' = c'·a'. Gas nodes carry none. Each edge books the mass it carries into node a.
    ``out_conc`` keeps c' (gas: 0), what every outflow carried this step: a node the step emptied (a' = 0) still sent
    its last mass M at c' = M/(dt·Σ_out q), where M'/a' no longer reads it."""
    w = wp.tid()
    one = wp.float64(1.0)
    for k in range(n_species):
        for i in range(n):
            for j in range(n):
                mat[w, i, j] = wp.float64(0.0)
            mat[w, i, i] = node_amount(medium[i], pressure[w, i], volume[w, i], bulk[i], p_ref[i], poly[i])
            rhs[w, i] = species[w, i, k] + dt * inflow_species[w, i, k]
            if medium[i] == GAS:
                mat[w, i, i] = one
                rhs[w, i] = species[w, i, k]
        for e in range(m):
            a = int(ea[e])
            b = int(eb[e])
            q = discharge[w, e]                          # into a
            if medium[a] == GAS:
                continue
            if q > wp.float64(0.0):                      # b (or the reservoir / source) -> a
                if b >= 0:
                    mat[w, b, b] = mat[w, b, b] + dt * q
                    mat[w, a, b] = mat[w, a, b] - dt * q
                else:
                    rhs[w, a] = rhs[w, a] + dt * q * conc_in[e, k]
            elif q < wp.float64(0.0):                    # a -> b (or out to the reservoir)
                mat[w, a, a] = mat[w, a, a] - dt * q
                if b >= 0:
                    mat[w, b, a] = mat[w, b, a] + dt * q
        for i in range(n):                               # an empty node with no outflow keeps its (zero) mass
            if mat[w, i, i] <= wp.float64(1.0e-300):
                for j in range(n):
                    mat[w, i, j] = wp.float64(0.0)
                mat[w, i, i] = one
                rhs[w, i] = wp.float64(0.0)
        _solve(w, n, mat, rhs)                           # rhs = the new concentrations
        for e in range(m):
            a = int(ea[e])
            b = int(eb[e])
            q = discharge[w, e]
            if medium[a] == GAS:
                continue
            c_up = conc_in[e, k]
            if q < wp.float64(0.0):
                c_up = rhs[w, a]
            elif b >= 0:
                c_up = rhs[w, b]
            species_ledger[w, e, k] = species_ledger[w, e, k] + dt * q * c_up
        for i in range(n):
            out_conc[w, i, k] = wp.float64(0.0)
            if medium[i] != GAS:
                out_conc[w, i, k] = rhs[w, i]
                species[w, i, k] = rhs[w, i] * node_amount(medium[i], pressure[w, i], volume[w, i], bulk[i], p_ref[i],
                                                           poly[i])
            species_in_total[w, i, k] = species_in_total[w, i, k] + dt * inflow_species[w, i, k]
            inflow_species[w, i, k] = wp.float64(0.0)
