# SPDX-License-Identifier: Apache-2.0
"""Lumped-fluid core (alloy/plugins/fluid): closed-form checks of the per-world network step (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.fluid import P_ATM, Edge, Network, Node

DEV = "cpu"
DT = 1.25e-3


def _run(net: Network, steps: int, worlds: int = 2, dt: float = DT, opening=None, every: int = 1):
    f = net.compile(DEV, worlds)
    if opening is not None:
        f.opening.assign(np.asarray(opening, np.float32))
    out = [f.pressure.numpy().copy()]
    for k in range(steps):
        f.step(dt)
        if (k + 1) % every == 0:
            out.append(f.pressure.numpy().copy())
    return f, np.asarray(out)                                # [K, W, n]


def test_one_gas_node_evacuates_on_the_exact_exponential():
    V, S, C, p_ult = 5e-6, 1e-4, 1e-6, 16e3
    net = Network([Node("cup", volume=V)], [Edge("pump", "pump_gas", "cup", reservoir=p_ult, conductance=S),
                                           Edge("leak", "linear", "cup", conductance=C)])
    _, p = _run(net, 200)
    t = DT * np.arange(201)
    p_ss = (S * p_ult + C * P_ATM) / (S + C)
    want = p_ss + (P_ATM - p_ss) * np.exp(-(S + C) * t / V)
    assert np.allclose(p[:, 0, 0], want, rtol=1e-12, atol=1e-6)


def test_two_tanks_equalise_through_a_conductance():
    v1, v2, g = 2e-5, 6e-5, 1e-4
    net = Network([Node("a", volume=v1, pressure=2 * P_ATM), Node("b", volume=v2)],
                  [Edge("line", "linear", "a", "b", conductance=g)])
    f, p = _run(net, 400)
    t = DT * np.arange(401)
    mean = (2 * P_ATM * v1 + P_ATM * v2) / (v1 + v2)
    dp = P_ATM * np.exp(-g * (1 / v1 + 1 / v2) * t)                      # P_a - P_b
    assert np.allclose(p[:, 0, 0] - p[:, 0, 1], dp, rtol=2e-4, atol=1e-3)   # Crank-Nicolson: O((dt/τ)²)
    amounts = p[:, 0, :] @ np.array([v1, v2])
    assert np.allclose(amounts, mean * (v1 + v2), rtol=1e-12)
    led = f.ledger.numpy()[0, 0]
    assert led == pytest.approx((p[-1, 0, 0] - 2 * P_ATM) * v1, rel=1e-9)   # into a: its amount change


def test_sixteen_nodes_equalise_and_conserve():
    """MAX_NODES (WP8's rinse-glassware: 3 bowls + a cavity tank per glass): a 16-node gas chain, the first at 2 atm,
    keeps its total amount exactly and equalises at the volume-weighted mean."""
    vols = 1e-5 * (1.0 + np.arange(16) % 3)
    nodes = [Node(f"n{i}", volume=float(v), pressure=2 * P_ATM if i == 0 else P_ATM) for i, v in enumerate(vols)]
    edges = [Edge(f"e{i}", "linear", f"n{i}", f"n{i + 1}", conductance=2e-3) for i in range(15)]
    _, p = _run(Network(nodes, edges), 4000, every=100)
    mean = (P_ATM * vols.sum() + P_ATM * vols[0]) / vols.sum()
    assert np.allclose(p[:, 0, :] @ vols, mean * vols.sum(), rtol=1e-12)
    assert np.allclose(p[-1, 0, :], mean, rtol=1e-6)


def test_liquid_node_follows_the_bulk_modulus_when_squeezed():
    K, v0 = 1.5e9, 1e-5
    net = Network([Node("oil", medium="liquid", volume=v0, bulk_modulus=K)], [])
    f = net.compile(DEV, 1)
    for v1 in (0.999 * v0, 0.998 * v0):
        f.volume.assign(np.array([[v1]]))
        f.step(DT)
        want = P_ATM + K * (v0 / v1 - 1.0)
        assert f.pressure.numpy()[0, 0] == pytest.approx(want, rel=1e-12)


def test_stiff_vent_does_not_ring():
    """dt·g/V ≫ 2 on a two-node network (the θ-step): θ = 1, the pressures approach P_atm monotonically."""
    net = Network([Node("a", volume=1e-6, pressure=20e3), Node("b", volume=1e-6, pressure=20e3)],
                  [Edge("vent", "linear", "a", conductance=1e-2), Edge("line", "linear", "a", "b", conductance=1e-6)])
    _, p = _run(net, 50, worlds=1)
    pa = p[:, 0, 0]
    assert np.all(np.diff(pa) >= -1e-9) and pa[-1] == pytest.approx(P_ATM, rel=1e-3)


def test_relief_caps_and_check_blocks_backflow():
    q, g, p_set = 1e-5, 1e-9, 5e5                       # liquid: a 10 mL/s pump into a relieved chamber
    net = Network([Node("c", medium="liquid", volume=1e-5, bulk_modulus=1e9), Node("d", medium="liquid", volume=1e-5,
                                                                                   bulk_modulus=1e9)],
                  [Edge("pump", "source", "c", flow=q), Edge("relief", "relief", "c", reservoir=P_ATM,
                                                                conductance=g, p_set=p_set),
                   Edge("check", "check", "d", "c", conductance=1e-9, p_set=0.0)])   # opens c -> d only
    f, p = _run(net, 4000, worlds=1)
    pc, pd = p[-1, 0, 0], p[-1, 0, 1]
    assert pc == pytest.approx(p_set + q / g, rel=0.02)     # relief: P settles at set + Q/G (minus the check's share)
    # filled through the check; the check's state is taken at the step start, so d may overshoot c by a little
    assert pd > P_ATM and pd <= pc * (1.0 + 1e-4)
    f.pressure.assign(np.array([[P_ATM, 10 * P_ATM]]))       # d above c: the check stays shut
    led0 = f.ledger.numpy()[0, 2]
    f.opening.assign(np.array([[0.0, 0.0, 1.0]], np.float32))
    f.step(DT)
    assert f.ledger.numpy()[0, 2] == led0


def test_ledger_balances_amounts_with_reservoirs_and_sources():
    net = Network([Node("a", volume=3e-6), Node("b", volume=7e-6, pressure=50e3)],
                  [Edge("pump", "pump_gas", "a", reservoir=10e3, conductance=5e-5),
                   Edge("leak", "linear", "b", conductance=2e-6), Edge("line", "linear", "a", "b", conductance=3e-5),
                   Edge("feed", "source", "b", flow=0.5)])
    f, p = _run(net, 300, worlds=1)
    amount = f.amount(p[:, 0, :], np.array([3e-6, 7e-6]))
    led = f.ledger.numpy()[0]
    boundary = led[0] + led[1] + led[3]                      # the line moves amount between nodes only
    assert amount[-1].sum() - amount[0].sum() == pytest.approx(boundary, rel=1e-10, abs=1e-12)


def test_worlds_are_isolated():
    net = Network([Node("cup", volume=5e-6)], [Edge("pump", "pump_gas", "cup", reservoir=16e3, conductance=1e-4)])
    _, p = _run(net, 40, worlds=3, opening=[[1.0], [0.0], [1.0]])
    assert np.array_equal(p[:, 0], p[:, 2]) and np.all(p[:, 1] == P_ATM)


def test_network_validation_names_the_problem():
    with pytest.raises(ValueError, match="different units"):
        Network([Node("g"), Node("l", medium="liquid")], [Edge("x", "linear", "g", "l", conductance=1.0)])
    with pytest.raises(ValueError, match="reservoir"):
        Edge("p", "pump_gas", "a", "b")
    with pytest.raises(ValueError, match="1-16"):
        Network([Node(f"n{i}") for i in range(17)], [])
    assert math.isfinite(P_ATM) and wp.float64 is not None


def test_liquid_energy_ledger_balances():
    """Pump work = throttling + stored compression energy (fixed volumes, Crank-Nicolson: exact); with a chamber
    growing, + its ∫(P - P_ref)·dV (first order in ΔP/K)."""
    from alloy.plugins.fluid import stored_energy  # noqa: PLC0415

    net = Network([Node("line", medium="liquid", volume=2e-5, bulk_modulus=1e9),
                   Node("cap", medium="liquid", volume=5.6e-5, bulk_modulus=1e9)],
                  [Edge("pump", "source", "line", flow=2e-6), Edge("valve", "linear", "line", "cap", conductance=1e-11)])
    for grow in (0.0, 1e-9):
        f = net.compile(DEV, 1)
        for k in range(400):
            v = f.volume.numpy()
            v[0, 1] = 5.6e-5 + grow * k
            f.volume.assign(v)
            f.step(DT)
        w, nw = f.work.numpy()[0], f.node_work.numpy()[0]
        e = stored_energy(net, f.pressure.numpy(), f.volume.numpy()).sum()
        assert w[1] > 0 and (nw.sum() > 0) == (grow > 0)
        assert w[0] - w[1] - nw.sum() - e == pytest.approx(0.0, abs=(1e-12 if grow == 0 else 1e-2) * w[0])
