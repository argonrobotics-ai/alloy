# SPDX-License-Identifier: Apache-2.0
"""Two-phase chambers of the lumped-fluid core: liquid with a gas pocket at one pressure (SCR-WP8-11 Part A).

A ``mixed`` node (``Node(medium="mixed", volume=V, liquid=L, radius=r, pressure=P0)``) is a cylinder of bore radius
``r`` whose ports sit on its axis at a depth from its head (a syringe's nozzle end, a vial's stopper end). It holds a
liquid amount L [m³ at p_ref] and a gas amount a [Pa·m³ at T0 = 293.15 K] at one pressure P:

    V = V_l(P) + V_g(P),   V_l = L·(1 - (P - p_ref)/K),   V_g = a·(c/P)^(1/n)        (P = c·(a/V_g)^n)

``c`` is the pocket's isentropic constant (n = 1: c = 1, P·V_g = a). A volume change keeps a and c (P·V_g^n constant);
gas leaving keeps c (the rest expands on the same adiabat); gas arriving takes the pocket's c (no temperature state,
as the core's gas nodes). P never falls below the liquid's vapour pressure ``P_VAPOUR`` (a void opens instead).

Gas sits on top (instant stratification; no dispersed bubbles in v1). The head axis's cosine with "up" is owner-
written per world (``up [W, n]``, 1 = head up). A port passes gas while the pocket is larger than the chamber volume
above the port's level plane (``volume_above``: closed form for a cylinder), else liquid; what one step takes is capped
at what that phase holds on its side of the plane, and the rest of the step's volume follows in the other phase.

A mixed network holds only mixed nodes and ``port`` edges (between them, or to a gas or liquid reservoir), no species.
A port carries its upstream end's phase: liquid by its laminar ``conductance`` (+ orifice ``cd_area``), gas by its
``gas_conductance`` (+ the orifice at gas density P/(R·T0)), as volume at the upstream pressure. The θ-solve
(``kernels``' method) is in volume units, linearised at the step start: κ = L/K + V_g/(n·P), a gas port's volume
rescaled to the receiving node's pressure, then re-solved with κ the secant to its result (a small pocket's gas is far
from linear; θ is picked once, from the start κ); a void starts it below P_VAPOUR by void/κ (filled first, then
compressed). After it every port's volume is booked as amounts (``ledger``: liquid into node a; ``gas_ledger``: gas into
node a) and P is closed exactly from (V, L, a, c), so the books are exact and only P carries the linearisation error.

Ledgers per port (W worlds, m edges): ``ledger`` liquid m³ at p_ref and ``gas_ledger`` Pa·m³ at T0 into node a (gas
delivered to a reservoir in mL at P_atm: ``-gas_ledger/P_ATM·1e6``); ``work`` J of throttling (liquid ∫Q·ΔP dt, gas
a·ln(P_up/P_down), isothermal); ``phase`` the last step's carried phases (bit 1 liquid, bit 2 gas).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.fluid.kernels import _book_inflow, _solve

P_VAPOUR = 2339.0                 # Pa, water at 20 °C (CRC Handbook of Chemistry and Physics, vapour pressure table)
R_T0 = 287.05 * 293.15            # J/kg, air at T0: gas density = P / R_T0
MU_AIR = 1.81e-5                  # Pa·s, air at 20 °C
GAS_RES, LIQUID_RES = 0, 1        # a port reservoir's medium
PH_LIQUID, PH_GAS = 1, 2          # ``phase`` bits
MIXED_CODE = 3                    # the core's medium code of a mixed node (network.MEDIA["mixed"])
PI = math.pi
G_FAR = 2.0 / 3.0 - math.pi       # G(-r) / r³


@wp.func
def seg_integral(y: wp.float64, r: wp.float64) -> wp.float64:
    """G(Y) = ∫_0^Y A(y) dy, A(y) the area of a disk of radius r on the far side of a chord at offset y (πr² below
    -r, 0 above r)."""
    if y >= r:
        return wp.float64(2.0) / wp.float64(3.0) * r * r * r
    if y <= -r:
        return wp.float64(G_FAR) * r * r * r + wp.float64(PI) * r * r * (y + r)
    s = wp.sqrt(wp.max(r * r - y * y, wp.float64(0.0)))
    acos = wp.acos(wp.clamp(y / r, wp.float64(-1.0), wp.float64(1.0)))
    return r * r * (y * acos - s + r) + (s * s * s - r * r * r) / wp.float64(3.0)


@wp.func
def volume_above(v: wp.float64, r: wp.float64, depth: wp.float64, up: wp.float64) -> wp.float64:
    """The chamber volume above the level plane through a port on the axis ``depth`` from the head; ``up`` = cos of
    the head axis with up. ∫_0^ℓ A((z - d)·cot θ) dz = tan θ·(G((ℓ - d)·cot θ) - G(-d·cot θ)), ℓ = V/(π r²)."""
    area = wp.float64(PI) * r * r
    ell = v / area
    d = wp.clamp(depth, wp.float64(0.0), ell)
    c = wp.clamp(up, wp.float64(-1.0), wp.float64(1.0))
    if c >= wp.float64(1.0 - 1.0e-12):
        return area * d
    if c <= wp.float64(-1.0 + 1.0e-12):
        return area * (ell - d)
    if wp.abs(c) <= wp.float64(1.0e-12):
        return wp.float64(0.5) * v
    s = wp.sqrt(wp.float64(1.0) - c * c)
    cot = c / s
    out = (s / c) * (seg_integral((ell - d) * cot, r) - seg_integral(-d * cot, r))
    return wp.clamp(out, wp.float64(0.0), v)


@wp.func
def gas_volume(a: wp.float64, c: wp.float64, n: wp.float64, p: wp.float64) -> wp.float64:
    """V_g = a·(c/P)^(1/n)."""
    if a <= wp.float64(0.0):
        return wp.float64(0.0)
    if n == wp.float64(1.0):
        return a * c / p
    return a * wp.pow(c / p, wp.float64(1.0) / n)


@wp.func
def liquid_volume(liq: wp.float64, kb: wp.float64, p_ref: wp.float64, p: wp.float64) -> wp.float64:
    return liq * (wp.float64(1.0) - (p - p_ref) / kb)                  # V_l = L·(1 - (P - p_ref)/K)


@wp.func
def closure(v: wp.float64, liq: wp.float64, a: wp.float64, c: wp.float64, n: wp.float64, kb: wp.float64,
            p_ref: wp.float64) -> wp.float64:
    """The pressure at which the contents fill V: V_l(P) + V_g(P) = V (decreasing in P), at least P_VAPOUR.
    Safeguarded Newton (bisection inside the bracket), float64."""
    lo = wp.float64(P_VAPOUR)
    if liquid_volume(liq, kb, p_ref, lo) + gas_volume(a, c, n, lo) <= v:
        return lo                                        # the contents fit at the vapour pressure: a void opens
    p = p_ref + kb * (wp.float64(1.0) - v / wp.max(liq, wp.float64(1.0e-30)))
    if a > wp.float64(0.0) and v > liq:
        p = c * wp.pow(a / (v - liq), n)                 # the gas law with the liquid incompressible
    hi = wp.max(p, wp.float64(2.0) * lo)
    for _ in range(200):
        if liquid_volume(liq, kb, p_ref, hi) + gas_volume(a, c, n, hi) <= v:
            break
        lo = hi
        hi = hi * wp.float64(2.0)
    p = wp.clamp(p, lo, hi)
    for _ in range(80):
        f = liquid_volume(liq, kb, p_ref, p) + gas_volume(a, c, n, p) - v
        if f > wp.float64(0.0):
            lo = p
        else:
            hi = p
        df = -liq / kb - gas_volume(a, c, n, p) / (n * p)
        step = p - f / df
        if not (step > lo and step < hi):
            step = wp.float64(0.5) * (lo + hi)
        converged = wp.abs(step - p) <= wp.float64(1.0e-15) * p
        p = step
        if converged:
            break
    return p


@wp.func
def capacitance(liq: wp.float64, a: wp.float64, c: wp.float64, n: wp.float64, kb: wp.float64, p0: wp.float64,
                p1: wp.float64) -> wp.float64:
    """κ [m³/Pa] of the contents over [p0, p1], L/K + (V_g(p0) - V_g(p1))/(p1 - p0): the tangent L/K + V_g/(n·P) at
    p1 = p0, else the secant the solve iterates on (a small pocket's gas is far from linear over a step). A void
    (``cavity``) is no compliance: it fills at P_VAPOUR, so the solve starts below the floor by cavity/κ."""
    k = liq / kb + gas_volume(a, c, n, p0) / (n * p0)
    if wp.abs(p1 - p0) > wp.float64(1.0e-9) * p0:
        k = liq / kb + (gas_volume(a, c, n, p0) - gas_volume(a, c, n, p1)) / (p1 - p0)
    return wp.max(k, wp.float64(1.0e-30))


@wp.func
def cavity(v: wp.float64, liq: wp.float64, a: wp.float64, c: wp.float64, n: wp.float64, kb: wp.float64,
           p_ref: wp.float64, p: wp.float64) -> wp.float64:
    """The void at the vapour floor, V - V_l(P) - V_g(P) (0 above it)."""
    return wp.max(v - liquid_volume(liq, kb, p_ref, p) - gas_volume(a, c, n, p), wp.float64(0.0))


@wp.func
def gas_per_volume(a: wp.float64, c: wp.float64, n: wp.float64, p: wp.float64) -> wp.float64:
    """Gas amount per m³ of pocket at P: (P/c)^(1/n) (n = 1: P)."""
    if n == wp.float64(1.0):
        return p / c
    return wp.pow(p / c, wp.float64(1.0) / n)


@wp.func
def port_gas(v: wp.float64, r: wp.float64, depth: wp.float64, up: wp.float64, vg: wp.float64) -> wp.float64:
    """Signed: > 0 the gas volume a port can pass before the level reaches it; <= 0 minus the liquid it can pass."""
    return vg - volume_above(v, r, depth, up)


@wp.func
def port_g(liquid: wp.int32, g_liq: wp.float64, g_gas: wp.float64, cd_area: wp.float64, rho: wp.float64,
           dp_lam: wp.float64, p_up: wp.float64, dp: wp.float64, u: wp.float64) -> wp.float64:
    """Volume flow per Pa at the upstream pressure: laminar + orifice (liquid ρ, gas P_up/R_T0)."""
    g, dens = g_gas, p_up / wp.float64(R_T0)
    if liquid != 0:
        g = g_liq
        dens = rho
    if cd_area > wp.float64(0.0):
        g = g + cd_area * wp.sqrt(wp.float64(2.0) / dens) / wp.sqrt(wp.max(wp.abs(dp), dp_lam))
    return g * u


@wp.struct
class MixedNodes:
    radius: wp.array[wp.float64]
    bulk: wp.array[wp.float64]
    p_ref: wp.array[wp.float64]
    poly: wp.array[wp.float64]
    liquid: wp.array2d[wp.float64]        # [W, n] L, m³ at p_ref
    gas: wp.array2d[wp.float64]           # [W, n] a, Pa·m³ at T0
    gas_c: wp.array2d[wp.float64]         # [W, n] c
    up: wp.array2d[wp.float64]            # [W, n] owner-written cos(head axis, up)
    kap: wp.array2d[wp.float64]           # [W, n] scratch: κ
    cav: wp.array2d[wp.float64]           # [W, n] scratch: the cavity at the step start


@wp.struct
class MixedPorts:
    ea: wp.array[wp.int32]
    eb: wp.array[wp.int32]
    p_res: wp.array[wp.float64]
    res_liquid: wp.array[wp.int32]
    g_liq: wp.array[wp.float64]
    g_gas: wp.array[wp.float64]
    cd_area: wp.array[wp.float64]
    rho: wp.array[wp.float64]
    dp_lam: wp.array[wp.float64]
    depth_a: wp.array2d[wp.float64]       # [W, m] owner-writable
    depth_b: wp.array2d[wp.float64]
    opening: wp.array2d[wp.float32]
    g: wp.array2d[wp.float64]             # [W, m] scratch: conductance; sa/sb the rows' volume scales
    sa: wp.array2d[wp.float64]
    sb: wp.array2d[wp.float64]
    ledger: wp.array2d[wp.float64]
    gas_ledger: wp.array2d[wp.float64]
    work: wp.array2d[wp.float64]
    discharge: wp.array2d[wp.float64]
    phase: wp.array2d[wp.int32]


@wp.func
def _start(w: wp.int32, i: wp.int32, nodes: MixedNodes, volume: wp.array2d[wp.float64], pv: wp.array2d[wp.float64]):
    """Node i after its owner's exact volume change: P, κ and the cavity at the step start."""
    n = nodes.poly[i]
    kb = nodes.bulk[i]
    a = nodes.gas[w, i]
    c = nodes.gas_c[w, i]
    liq = nodes.liquid[w, i]
    p = closure(volume[w, i], liq, a, c, n, kb, nodes.p_ref[i])
    pv[w, i] = p
    nodes.kap[w, i] = capacitance(liq, a, c, n, kb, p, p)
    nodes.cav[w, i] = cavity(volume[w, i], liq, a, c, n, kb, nodes.p_ref[i], p)


@wp.func
def _port_signed(w: wp.int32, i: wp.int32, depth: wp.float64, nodes: MixedNodes, volume: wp.array2d[wp.float64],
                 pv: wp.array2d[wp.float64]) -> wp.float64:
    vg = gas_volume(nodes.gas[w, i], nodes.gas_c[w, i], nodes.poly[i], pv[w, i])   # live contents, start pressure
    return port_gas(volume[w, i], nodes.radius[i], depth, nodes.up[w, i], vg)


@wp.func
def _setup_port(w: wp.int32, e: wp.int32, nodes: MixedNodes, ports: MixedPorts, volume: wp.array2d[wp.float64],
                pv: wp.array2d[wp.float64]):
    """The phase the start-of-step upstream end passes, its conductance and the two rows' volume scales."""
    a = ports.ea[e]
    b = ports.eb[e]
    pa = pv[w, a]
    pb = ports.p_res[e]
    if b >= 0:
        pb = pv[w, b]
    liquid = int(0)
    if pb > pa and b < 0:
        liquid = ports.res_liquid[e]
    elif pb > pa and _port_signed(w, b, ports.depth_b[w, e], nodes, volume, pv) <= wp.float64(0.0):
        liquid = int(1)
    elif pb <= pa and _port_signed(w, a, ports.depth_a[w, e], nodes, volume, pv) <= wp.float64(0.0):
        liquid = int(1)
    ports.g[w, e] = port_g(liquid, ports.g_liq[e], ports.g_gas[e], ports.cd_area[e], ports.rho[e], ports.dp_lam[e],
                           wp.max(pa, pb), pb - pa, wp.float64(ports.opening[w, e]))
    ports.sa[w, e] = wp.where(liquid == 0 and pb > pa, pb / pa, wp.float64(1.0))     # gas at P_b expands into a
    ports.sb[w, e] = wp.where(liquid == 0 and pb <= pa and b >= 0, pa / pb, wp.float64(1.0))
    ports.phase[w, e] = wp.where(liquid != 0, PH_LIQUID, PH_GAS)      # the start phase, until _book's carried ones


@wp.func
def _take(w: wp.int32, up: wp.int32, depth: wp.float64, vol: wp.float64, nodes: MixedNodes,
          volume: wp.array2d[wp.float64], pv: wp.array2d[wp.float64], p_new: wp.array2d[wp.float64]) -> wp.vec2d:
    """(liquid, gas) amounts that ``vol`` (m³ at the start pressure) takes out of node ``up`` through a port at
    ``depth``: the phase at the port at the step start, up to what leaves the level at the port at the step's end
    (the solve's pressure, the owner's volume), then the other phase for the rest of ``vol``."""
    p0 = pv[w, up]
    p1 = wp.max(p_new[w, up], wp.float64(P_VAPOUR))
    v = volume[w, up]
    liq = nodes.liquid[w, up]
    a = nodes.gas[w, up]
    c = nodes.gas_c[w, up]
    n = nodes.poly[up]
    one = wp.float64(1.0)
    above = volume_above(v, nodes.radius[up], depth, nodes.up[w, up])
    lpv = one / (one - (p0 - nodes.p_ref[up]) / nodes.bulk[up])      # liquid amount per m³ at the start pressure
    gpv = gas_per_volume(a, c, n, p0)
    l_amt, g_amt = wp.float64(0.0), wp.float64(0.0)
    if port_gas(v, nodes.radius[up], depth, nodes.up[w, up], gas_volume(a, c, n, p0)) > wp.float64(0.0):
        g_amt = vol * gpv
        cap = wp.max(a - above * gas_per_volume(a, c, n, p1), wp.float64(0.0))
        if g_amt > cap:                                  # the level reaches the port this step: the rest is liquid
            g_amt = cap
            l_amt = wp.min((vol - cap / gpv) * lpv, liq)
    else:
        l_amt = vol * lpv
        cap = wp.max(liq - (v - above) / (one - (p1 - nodes.p_ref[up]) / nodes.bulk[up]), wp.float64(0.0))
        if l_amt > cap:
            l_amt = cap
            g_amt = wp.min((vol - cap / lpv) * gpv, a)
    return wp.vec2d(wp.max(l_amt, wp.float64(0.0)), wp.max(g_amt, wp.float64(0.0)))


@wp.func
def _book(w: wp.int32, e: wp.int32, theta: wp.float64, dt: wp.float64, nodes: MixedNodes, ports: MixedPorts,
          volume: wp.array2d[wp.float64], pv: wp.array2d[wp.float64], p_new: wp.array2d[wp.float64]):
    """Port e's volume this step as amounts: out of the upstream end, into the downstream one, into the ledgers."""
    one = wp.float64(1.0)
    a = ports.ea[e]
    b = ports.eb[e]
    pa0 = pv[w, a]
    pa1 = p_new[w, a]
    pb0 = ports.p_res[e]
    pb1 = ports.p_res[e]
    if b >= 0:
        pb0 = pv[w, b]
        pb1 = p_new[w, b]
    dp = theta * (pb1 - pa1) + (one - theta) * (pb0 - pa0)
    q = ports.g[w, e] * dp
    ports.discharge[w, e] = q
    into_a = q > wp.float64(0.0)
    up = a
    down = b
    depth = ports.depth_a[w, e]
    p_up = pa0
    if into_a:
        up = b
        down = a
        depth = ports.depth_b[w, e]
        p_up = pb0
    vol = wp.abs(q) * dt
    liq_amt, gas_amt, c_up = wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)
    if up >= 0:
        amts = _take(w, up, depth, vol, nodes, volume, pv, p_new)
        liq_amt = amts[0]
        gas_amt = amts[1]
        c_up = nodes.gas_c[w, up]
        nodes.liquid[w, up] = wp.max(nodes.liquid[w, up] - liq_amt, wp.float64(0.0))
        nodes.gas[w, up] = wp.max(nodes.gas[w, up] - gas_amt, wp.float64(0.0))
    elif ports.res_liquid[e] != 0:                      # a reservoir's liquid at P_res, its gas at P_res and T0
        liq_amt = vol / (one - (p_up - nodes.p_ref[down]) / nodes.bulk[down])
    else:
        gas_amt = vol * p_up
        c_up = wp.pow(p_up, one - nodes.poly[down])
    if down >= 0:
        nodes.liquid[w, down] = nodes.liquid[w, down] + liq_amt
        if gas_amt > wp.float64(0.0) and nodes.gas[w, down] <= wp.float64(0.0):
            nodes.gas_c[w, down] = c_up
        nodes.gas[w, down] = nodes.gas[w, down] + gas_amt
    sign = wp.float64(-1.0)
    if into_a:
        sign = one
    ports.ledger[w, e] = ports.ledger[w, e] + sign * liq_amt
    ports.gas_ledger[w, e] = ports.gas_ledger[w, e] + sign * gas_amt
    p_th_a = wp.max(theta * pa1 + (one - theta) * pa0, wp.float64(P_VAPOUR))
    p_th_b = wp.max(theta * pb1 + (one - theta) * pb0, wp.float64(P_VAPOUR))
    ratio = p_th_a / p_th_b                              # upstream over downstream
    if into_a:
        ratio = p_th_b / p_th_a
    v_liq = liq_amt
    if up >= 0:
        v_liq = liq_amt * (one - (p_up - nodes.p_ref[up]) / nodes.bulk[up])
    ports.work[w, e] = ports.work[w, e] + v_liq * wp.abs(dp) + gas_amt * wp.abs(wp.log(ratio))
    zero = wp.float64(0.0)
    ports.phase[w, e] = wp.where(liq_amt > zero, PH_LIQUID, 0) | wp.where(gas_amt > zero, PH_GAS, 0)


@wp.kernel
def step_mixed(n: wp.int32, m: wp.int32, dt: wp.float64, nodes: MixedNodes, ports: MixedPorts,
               volume: wp.array2d[wp.float64], volume_old: wp.array2d[wp.float64], pressure: wp.array2d[wp.float64],
               node_work: wp.array2d[wp.float64], mat: wp.array3d[wp.float64], rhs: wp.array2d[wp.float64],
               pv: wp.array2d[wp.float64], inflow: wp.array2d[wp.float64], inflow_total: wp.array2d[wp.float64]):
    """One world's mixed network (module docstring): start state, port phases, the θ-solve, booking, closure."""
    w = wp.tid()
    one, zero = wp.float64(1.0), wp.float64(0.0)
    for i in range(n):
        _start(w, i, nodes, volume, pv)
    for e in range(m):
        _setup_port(w, e, nodes, ports, volume, pv)
    theta = wp.float64(0.5)
    for i in range(n):
        gsum = zero
        for e in range(m):
            if ports.ea[e] == i:
                gsum = gsum + ports.sa[w, e] * ports.g[w, e]
            if ports.eb[e] == i:
                gsum = gsum + ports.sb[w, e] * ports.g[w, e]
        if dt * gsum > wp.float64(2.0) * nodes.kap[w, i]:
            theta = one
    for it in range(4):                                  # then κ as the secant over [start, solved] (capacitance)
        for i in range(n):
            gas = it == 0                                # (the secant keeps a: not where gas comes or goes)
            for e in range(m):
                at = ports.ea[e] == i or ports.eb[e] == i
                gas = gas or (at and ports.phase[w, e] == PH_GAS and ports.g[w, e] > zero)
            if not gas:
                nodes.kap[w, i] = capacitance(nodes.liquid[w, i], nodes.gas[w, i], nodes.gas_c[w, i], nodes.poly[i],
                                              nodes.bulk[i], pv[w, i], wp.max(rhs[w, i], wp.float64(P_VAPOUR)))
            for j in range(n):
                mat[w, i, j] = zero
            mat[w, i, i] = nodes.kap[w, i]
            rhs[w, i] = nodes.kap[w, i] * pv[w, i] - nodes.cav[w, i]     # a cavity fills first (capacitance)
        for e in range(m):
            a = ports.ea[e]
            b = ports.eb[e]
            ka = theta * dt * ports.sa[w, e] * ports.g[w, e]
            ex = (one - theta) * dt * ports.g[w, e]
            mat[w, a, a] = mat[w, a, a] + ka
            if b >= 0:
                kb = theta * dt * ports.sb[w, e] * ports.g[w, e]
                d = pv[w, b] - pv[w, a]
                mat[w, a, b] = mat[w, a, b] - ka
                mat[w, b, b] = mat[w, b, b] + kb
                mat[w, b, a] = mat[w, b, a] - kb
                rhs[w, a] = rhs[w, a] + ex * ports.sa[w, e] * d
                rhs[w, b] = rhs[w, b] - ex * ports.sb[w, e] * d
            else:
                rhs[w, a] = rhs[w, a] + ka * ports.p_res[e] + ex * ports.sa[w, e] * (ports.p_res[e] - pv[w, a])
        _solve(w, n, mat, rhs)
    for i in range(n):                                   # a cavity the step does not fill stays at the floor
        rhs[w, i] = wp.max(rhs[w, i], wp.float64(P_VAPOUR))
    for e in range(m):
        _book(w, e, theta, dt, nodes, ports, volume, pv, rhs)
    for i in range(n):                                   # ∫ (P - p_ref)·dV at the θ-pressure that drove the flows
        p_th = theta * rhs[w, i] + (one - theta) * pv[w, i]
        node_work[w, i] = node_work[w, i] + (p_th - nodes.p_ref[i]) * (volume[w, i] - volume_old[w, i])
        nodes.liquid[w, i] = nodes.liquid[w, i] + dt * inflow[w, i]       # an owner's liquid inflow (m³/s at p_ref)
        pressure[w, i] = closure(volume[w, i], nodes.liquid[w, i], nodes.gas[w, i], nodes.gas_c[w, i],
                                 nodes.poly[i], nodes.bulk[i], nodes.p_ref[i])
        volume_old[w, i] = volume[w, i]
    _book_inflow(w, n, dt, inflow, inflow_total)


def start_gas(node: Any) -> tuple[float, float]:
    """(a, c) of a mixed node's start pocket: V_g0 = V - V_l(P0) at P0 and T0 (a = P0·V_g0, c = P0^(1 - n))."""
    v_g = node.volume - node.liquid * (1.0 - (node.pressure - node.p_ref) / node.bulk_modulus)
    if v_g < -1e-12 * node.volume:
        raise ValueError(f"mixed node {node.name!r}: its liquid ({node.liquid} m³) does not fit in {node.volume} m³ "
                         f"at {node.pressure} Pa")
    return node.pressure * max(v_g, 0.0), node.pressure ** (1.0 - node.polytropic)


class MixedState:
    """The per-world arrays of a mixed network (built by ``FluidState`` when ``net.mixed``) and its step launch.
    Owner-written: ``FluidState.volume``, ``up [W, n]`` (cos of each head axis with up), ``depth_a``/``depth_b``
    ``[W, m]`` (port depths). State: ``liquid``, ``gas``, ``gas_c`` ``[W, n]``; ``gas_ledger``, ``phase`` ``[W, m]``;
    the core's ``ledger`` (liquid), ``work``, ``discharge``, ``node_work`` and ``pressure``."""

    def __init__(self, fs: Any) -> None:
        net, W, dev = fs.net, fs.world_count, fs.device
        self.fs, self._m = fs, fs.m

        def arr(values: object, dtype: object = wp.float64) -> wp.array:
            return wp.array(np.asarray(values), dtype=dtype, device=dev)

        def tile(values: list[float]) -> wp.array:
            return arr(np.tile(np.asarray(values, np.float64), (W, 1)))

        gas = [start_gas(nd) for nd in net.nodes]
        n, me, es = fs.n, max(fs.m, 1), list(net.edges) or [None]
        self.liquid, self.gas = tile([nd.liquid for nd in net.nodes]), tile([g[0] for g in gas])
        self.gas_c, self.up = tile([g[1] for g in gas]), tile([nd.up for nd in net.nodes])
        self.depth_a = tile([e.depth_a if e else 0.0 for e in es])
        self.depth_b = tile([e.depth_b if e else 0.0 for e in es])
        self.gas_ledger, self.phase = arr(np.zeros((W, me))), arr(np.zeros((W, me), np.int32), wp.int32)
        self.nodes = nodes = MixedNodes()
        nodes.radius = arr([nd.radius for nd in net.nodes])
        nodes.bulk, nodes.p_ref, nodes.poly = fs.bulk, fs.p_ref, fs.poly
        nodes.liquid, nodes.gas, nodes.gas_c, nodes.up = self.liquid, self.gas, self.gas_c, self.up
        nodes.kap, nodes.cav = arr(np.zeros((W, n))), arr(np.zeros((W, n)))
        self.ports = ports = MixedPorts()
        ports.ea, ports.eb, ports.p_res, ports.opening = fs.ea, fs.eb, fs.p_res, fs.opening
        ports.res_liquid = arr([int(bool(e) and e.reservoir_medium == "liquid") for e in es], wp.int32)
        ports.g_liq = arr([e.conductance if e else 0.0 for e in es])
        ports.g_gas = arr([e.gas_conductance if e else 0.0 for e in es])
        ports.cd_area, ports.rho, ports.dp_lam = fs.cd_area, fs.rho, fs.dp_lam
        ports.depth_a, ports.depth_b = self.depth_a, self.depth_b
        ports.g, ports.sa, ports.sb = fs._g, arr(np.ones((W, me))), arr(np.ones((W, me)))
        ports.ledger, ports.gas_ledger, ports.work, ports.discharge = fs.ledger, self.gas_ledger, fs.work, fs.discharge
        ports.phase = self.phase

    def state_arrays(self) -> dict[str, wp.array]:                    # for Plugin.add_state
        return {k: getattr(self, k) for k in ("liquid", "gas", "gas_c", "gas_ledger", "phase", "up", "depth_a",
                                              "depth_b")}

    def step(self, dt: float) -> None:
        fs = self.fs
        wp.launch(step_mixed, dim=fs.world_count,
                  inputs=[fs.n, self._m, wp.float64(dt), self.nodes, self.ports, fs.volume],
                  outputs=[fs.volume_old, fs.pressure, fs.node_work, fs._mat, fs._rhs, fs._pv, fs.inflow,
                           fs.inflow_total], device=fs.device)

    def gas_volume(self) -> np.ndarray:
        """Host: each pocket's volume [W, n] m³ at its pressure."""
        a, c, p = self.gas.numpy(), self.gas_c.numpy(), self.fs.pressure.numpy()
        n = np.asarray([nd.polytropic for nd in self.fs.net.nodes])
        return np.where(a > 0, a * (c / np.maximum(p, 1e-30)) ** (1.0 / n), 0.0)

    def air_out_ml(self, edge: str) -> np.ndarray:
        """Host: the gas a port has carried out of its node a, [W] mL at P_atm and T0 (GA's "air into the vein")."""
        from alloy.plugins.fluid.network import P_ATM  # noqa: PLC0415

        return -self.gas_ledger.numpy()[:, self.fs.net.edge_index(edge)] / P_ATM * 1e6


__all__ = ["MIXED_CODE", "MU_AIR", "MixedState", "P_VAPOUR", "PH_GAS", "PH_LIQUID", "R_T0", "capacitance", "cavity",
           "closure", "start_gas", "step_mixed", "volume_above"]
