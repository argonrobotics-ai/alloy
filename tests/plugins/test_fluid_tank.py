# SPDX-License-Identifier: Apache-2.0
"""Lumped-fluid core: the open tank node, the external inflow and the per-edge discharge (CPU, closed forms)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.fluid import G_TANK, P_ATM, Edge, Network, Node, stored_energy
from alloy.plugins.fluid.actuators import Cylinder, validate

DEV = "cpu"
DT = 1.25e-3
RHO = 998.0


def _tank(name: str = "sink", volume: float = 0.02, area: float = 0.1) -> Node:
    return Node(name, medium="tank", volume=volume, area=area, density=RHO)


def _drain(cd_area: float, a: str = "sink") -> Edge:
    """A sharp-edged waste hole to the drain (atmosphere), Torricelli's orifice (laminar floor 1 Pa = 0.1 mm head)."""
    return Edge("waste", "orifice", a, reservoir=P_ATM, cd_area=cd_area, density=RHO, dp_lam=1.0)


def test_a_tank_starts_at_its_head_and_its_amount_is_its_volume():
    f = Network([_tank(volume=0.02, area=0.1)], []).compile(DEV, 2)
    p = f.pressure.numpy()
    assert np.allclose(p, P_ATM + RHO * G_TANK * 0.02 / 0.1, rtol=0, atol=1e-9)
    assert np.allclose(f.amount(p, f.volume.numpy()), 0.02, rtol=1e-14)
    e = stored_energy(f.net, p, f.volume.numpy())
    assert np.allclose(e, RHO * G_TANK * 0.02 ** 2 / (2 * 0.1), rtol=1e-12)          # ρ·g·V²/(2A)


def test_tank_drains_by_torricelli():
    """F-14: dV/dt = -Cd·A·sqrt(2·g·V/A_t), so sqrt(V) falls linearly and the tank empties at T."""
    area, v0, cd_area = 0.1, 0.02, 0.61 * math.pi * 0.015 ** 2
    f = Network([_tank(volume=v0, area=area)], [_drain(cd_area)]).compile(DEV, 2)
    rate = 0.5 * cd_area * math.sqrt(2 * G_TANK / area)                           # d sqrt(V)/dt
    t_empty = math.sqrt(v0) / rate
    steps = int(0.9 * t_empty / DT)
    err = 0.0
    for k in range(steps):
        f.step(DT)
        if k % 500 == 499 or k == steps - 1:
            v = f.amount(f.pressure.numpy(), f.volume.numpy())[:, 0]
            want = (math.sqrt(v0) - rate * (k + 1) * DT) ** 2
            err = max(err, float(np.max(np.abs(v - want))) / v0)
    assert err < 0.02 * 0.01, err                              # F-14 bound 2 %; the exact per-step decay is far inside
    v = f.amount(f.pressure.numpy(), f.volume.numpy())[:, 0]
    out = -f.ledger.numpy()[:, 0]
    assert np.allclose(v0 - v, out, rtol=1e-12, atol=0)        # the drained amount is the edge's ledger


def test_tank_never_runs_below_empty():
    area, v0, cd_area = 0.01, 1e-4, 0.61 * math.pi * 0.01 ** 2
    f = Network([_tank(volume=v0, area=area)], [_drain(cd_area)]).compile(DEV, 1)
    for _ in range(4000):
        f.step(DT)
        assert f.pressure.numpy()[0, 0] >= P_ATM


def test_communicating_tanks_level_and_keep_their_volume():
    a, b = _tank("a", volume=0.02, area=0.1), _tank("b", volume=0.0, area=0.05)
    f = Network([a, b], [Edge("pipe", "linear", "a", "b", conductance=1e-6)]).compile(DEV, 2)
    for _ in range(4000):
        f.step(1e-2)
    v = f.amount(f.pressure.numpy(), f.volume.numpy())
    assert np.allclose(v.sum(axis=1), 0.02, rtol=1e-13)
    assert np.allclose(v[:, 0] / 0.1, v[:, 1] / 0.05, rtol=1e-4)                 # equal levels (12 τ)
    assert np.allclose(0.02 - v[:, 0], -f.ledger.numpy()[:, 0], rtol=1e-12)


def test_inflow_adds_exactly_is_booked_and_is_consumed():
    q, steps = 2e-4, 400
    for edges in ([], [Edge("pipe", "linear", "sink", "pan", conductance=0.0)]):   # one-node and network paths
        nodes = [_tank()] + ([_tank("pan", volume=0.0, area=0.02)] if edges else [])
        f = Network(nodes, edges).compile(DEV, 2)
        for _ in range(steps):
            inflow = f.inflow.numpy()
            assert not inflow.any()                            # consumed by every step
            inflow[:, 0] = q
            f.inflow.assign(inflow)
            f.step(DT)
        v = f.amount(f.pressure.numpy(), f.volume.numpy())[:, 0]
        assert np.allclose(v, 0.02 + q * steps * DT, rtol=1e-12)
        assert np.allclose(f.inflow_total.numpy()[:, 0], q * steps * DT, rtol=1e-12)


@pytest.mark.parametrize("two_nodes", [False, True])
def test_discharge_is_the_last_steps_mean_flow(two_nodes):
    nodes = [_tank()] + ([Node("hose", medium="liquid", volume=1e-4)] if two_nodes else [])
    edges = [_drain(1e-5), Edge("tap", "source", "sink", flow=3e-5)]
    if two_nodes:
        edges.append(Edge("hose_in", "linear", "hose", "sink", conductance=1e-9))
    f = Network(nodes, edges).compile(DEV, 2)
    for _ in range(20):
        before = f.ledger.numpy().copy()
        f.step(DT)
        inc = f.ledger.numpy() - before
        assert np.allclose(f.discharge.numpy() * DT, inc, rtol=1e-12, atol=1e-20)
    assert np.allclose(f.discharge.numpy()[:, 1], 3e-5)                           # a source's discharge is its flow
    assert np.all(f.discharge.numpy()[:, 0] < 0)                                   # the waste edge drains the tank


def test_tank_rules():
    with pytest.raises(ValueError, match="area > 0"):
        Node("t", medium="tank", volume=0.01)
    with pytest.raises(ValueError, match="different units"):
        Network([_tank(), Node("air", volume=1e-4)], [Edge("x", "linear", "sink", "air", conductance=1e-9)])
    net = Network([_tank()], [])
    with pytest.raises(ValueError, match="open tank"):
        validate((Cylinder("c", joint="j", a="sink"),), net)


def test_f16_a_rinsed_tank_dilutes_exponentially():
    """F-16 (network part): fresh water in and the mixture out at Q keep V; c = c0·exp(-Q·t/V)."""
    q, v0, c0 = 2e-4, 0.02, 5.0
    tank = Node("wash", medium="tank", volume=v0, area=0.1, species=(c0, 1.0))
    edges = [Edge("fresh", "source", "wash", flow=q), Edge("out", "source", "wash", flow=-q)]
    f = Network([tank], edges).compile(DEV, 2)
    steps = 8000
    for _ in range(steps):
        f.step(DT)
    t = steps * DT
    c = f.concentration()[:, 0, :]
    assert np.allclose(f.amount(f.pressure.numpy(), f.volume.numpy())[:, 0], v0, rtol=1e-12)
    assert np.allclose(c[:, 0], c0 * math.exp(-q * t / v0), rtol=0.03 * 0.01)       # F-16 bound 3 %
    assert np.allclose(c[:, 1], math.exp(-q * t / v0), rtol=3e-4)
    out = -f.species_ledger.numpy()[:, 1, 0]
    assert np.allclose(c0 * v0 - f.species.numpy()[:, 0, 0], out, rtol=1e-12)      # the mass that left


def test_species_ride_a_network_and_are_conserved():
    """A soap dispenser (source with 50 kg/m³) and mains water into a line, through an orifice into a tank that
    drains; plus an external species inflow. Mass = in - out exactly; concentrations bounded by the inlets."""
    nodes = [Node("line", medium="liquid", volume=1e-4), _tank()]
    edges = [Edge("mains", "orifice", "line", reservoir=P_ATM + 1e5, cd_area=3e-6, dp_lam=1.0, species=(0.0,)),
             Edge("soap", "source", "line", flow=2e-6, species=(50.0,)),
             Edge("tap", "orifice", "sink", "line", cd_area=3e-6, dp_lam=1.0),
             _drain(2e-6)]
    f = Network(nodes, edges).compile(DEV, 2)
    for k in range(3000):
        if k % 10 == 0:
            ext = f.inflow_species.numpy()
            ext[:, 1, 0] = 1e-5
            f.inflow_species.assign(ext)
        f.step(DT)
    led = f.species_ledger.numpy()[:, :, 0]
    boundary = led[:, 0] + led[:, 1] + led[:, 3]                                  # mains, soap, waste
    total = f.species.numpy()[:, :, 0].sum(axis=1)
    assert np.allclose(total, boundary + f.species_in_total.numpy()[:, 1, 0], rtol=1e-11)
    c = f.concentration()[:, :, 0]
    assert np.all(c >= 0) and np.all(c <= 50.0 + 1e-9) and np.all(c[:, 1] > 0)


def _can(capacity: float, p0: float = 4.0e5, headspace: float = 1.0e-4, cd_area: float = 8.2e-8):
    """An aerosol can: 100 mL of propellant gas at 4 bar abs over lacquer (ρ 1000), ~2 g/s at 25 m/s at the start."""
    return Network([Node("can", volume=headspace, pressure=p0)],
                   [Edge("valve", "expel", "can", cd_area=cd_area, density=1000.0, dp_lam=1.0, capacity=capacity)])


def test_an_aerosol_can_slows_as_it_empties():
    """dV/dt = Cd·A·sqrt(2·(P - P_atm)/ρ) with P = P0·V_g0/(V_g0 + V) (isothermal), against fine RK4."""
    f = _can(capacity=3e-4).compile(DEV, 2)
    q0 = -1.0
    steps = 48000                                                            # 60 s
    for k in range(steps):
        f.step(DT)
        if k == 0:
            q0 = -f.discharge.numpy()[0, 0]
    out = -f.ledger.numpy()[:, 0]

    def rate(v):
        p = 4.0e5 * 1e-4 / (1e-4 + v)
        return 8.2e-8 * math.sqrt(2.0 * max(p - P_ATM, 0.0) / 1000.0)

    v, h = 0.0, 1e-3
    for _ in range(60000):                                                   # RK4 to 60 s
        k1 = rate(v)
        k2 = rate(v + 0.5 * h * k1)
        k3 = rate(v + 0.5 * h * k2)
        k4 = rate(v + h * k3)
        v += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    assert q0 * 1000.0 == pytest.approx(2.0e-3, rel=0.02)                     # ~2 g/s at the start
    assert np.allclose(out, v, rtol=2e-3), (out, v)
    assert np.allclose(f.volume.numpy()[:, 0], 1e-4 + out, rtol=1e-12)        # the gas took the freed volume
    p = f.pressure.numpy()[:, 0]
    assert np.allclose(p * f.volume.numpy()[:, 0], 4.0e5 * 1e-4, rtol=1e-12)  # isothermal P·V kept


def test_an_aerosol_can_stops_when_its_liquid_is_gone():
    f = _can(capacity=2e-5).compile(DEV, 1)
    for _ in range(40000):
        f.step(DT)
    assert -f.ledger.numpy()[0, 0] == pytest.approx(2e-5, rel=1e-12)
    assert f.discharge.numpy()[0, 0] == 0.0


def test_a_pump_cannot_draw_a_tank_below_empty():
    f = Network([_tank(volume=1e-4, area=0.01)], [Edge("pump", "source", "sink", flow=-1e-3)]).compile(DEV, 1)
    for _ in range(200):
        f.step(DT)
        assert f.pressure.numpy()[0, 0] >= P_ATM
    v = f.amount(f.pressure.numpy(), f.volume.numpy())[0, 0]
    assert v == pytest.approx(0.0, abs=1e-15)
    assert -f.ledger.numpy()[0, 0] == pytest.approx(1e-4, rel=1e-12)            # it pumped exactly what was there


def test_a_tank_outlet_below_its_surface_pressure_is_refused():
    with pytest.raises(ValueError, match="below empty"):
        Network([_tank()], [Edge("suck", "orifice", "sink", reservoir=0.5 * P_ATM, cd_area=1e-5)])


def test_an_overflow_holds_the_level_at_the_rim():
    from alloy.plugins.fluid import tank_overflow

    tank = Node("bucket", medium="tank", volume=0.0, area=0.05, capacity=0.01)
    q, steps = 2e-3, 8000                                                      # 2 L/s for 10 s: 20 L into a 10 L bucket
    f = Network([tank], [Edge("hose", "source", "bucket", flow=q), tank_overflow(tank)]).compile(DEV, 1)
    for _ in range(steps):
        f.step(DT)
    v = f.amount(f.pressure.numpy(), f.volume.numpy())[0, 0]
    spilled = -f.ledger.numpy()[0, 1]
    over = 0.05 * q / (1.0e-5 * RHO * G_TANK)                                  # the head that vents q: G·ρ·g·Δh = q
    assert v == pytest.approx(0.01 + over, rel=1e-3)
    assert v + spilled == pytest.approx(q * steps * DT, rel=1e-12)
    moon = Network([tank], [Edge("hose", "source", "bucket", flow=q), tank_overflow(tank)]).compile(DEV, 1, gravity=1.62)
    for _ in range(3 * steps):                                                 # τ = A/(ρ·g·G) = 3.1 s at 1.62 m/s²
        moon.step(DT)
    v = moon.amount(moon.pressure.numpy(), moon.volume.numpy())[0, 0]
    assert v == pytest.approx(0.01 + 0.05 * q / (1.0e-5 * RHO * 1.62), rel=1e-3)   # the rim at the scene's gravity


def test_a_pump_and_an_outlet_together_still_stop_at_empty():
    """WP15's follow-up (2): a pump drawing the tank's start volume while its waste orifice drains in the same step;
    the pump gives back what would take the tank below empty."""
    tank = _tank(volume=2e-5, area=0.01)
    f = Network([tank], [Edge("pump", "source", "sink", flow=-5e-3), _drain(1e-4)]).compile(DEV, 1)
    for _ in range(50):
        f.step(DT)
        assert f.pressure.numpy()[0, 0] >= P_ATM
    v = f.amount(f.pressure.numpy(), f.volume.numpy())[0, 0]
    out = -f.ledger.numpy()[0].sum()
    assert v == pytest.approx(0.0, abs=1e-15) and out == pytest.approx(2e-5, rel=1e-12)


def test_tank_heads_use_the_scenes_gravity():
    f = Network([_tank(volume=0.02, area=0.1)], []).compile(DEV, 1, gravity=1.62)
    assert f.pressure.numpy()[0, 0] == pytest.approx(P_ATM + RHO * 1.62 * 0.2, rel=1e-12)
    assert f.amount(f.pressure.numpy(), f.volume.numpy())[0, 0] == pytest.approx(0.02, rel=1e-12)
