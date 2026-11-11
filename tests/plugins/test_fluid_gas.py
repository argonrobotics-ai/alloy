# SPDX-License-Identifier: Apache-2.0
"""Lumped-fluid core, pneumatics (alloy/plugins/fluid): the ISO 6358 gas orifice, the polytropic volume change and the
Poiseuille line conductance (CPU, closed forms)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.fluid import P_ATM, Edge, Network, Node, laminar_conductance
from alloy.plugins.fluid.kernels import RHO0_R_T0

C, B = 2.0e-9, 0.3                      # m³/(s·Pa), ISO 6358 sonic conductance and critical ratio of a small valve


def _first_step_flow(p_node: float, p_res: float, dt: float = 1.0e-6) -> float:
    """Throughput [Pa·m³/s] into a large node at p_node from a reservoir at p_res over one short step."""
    net = Network([Node("n", volume=1.0, pressure=p_node)],
                  [Edge("v", "orifice_gas", "n", reservoir=p_res, sonic_conductance=C, critical_ratio=B)])
    f = net.compile("cpu", 2)
    f.step(dt)
    return float(f.ledger.numpy()[0, 0]) / dt


def _iso6358(p_up: float, p_down: float) -> float:
    r = p_down / p_up
    psi = 1.0 if r <= B else math.sqrt(1.0 - ((r - B) / (1.0 - B)) ** 2)
    return C * p_up * psi * RHO0_R_T0


@pytest.mark.parametrize("p_node, p_res", [
    (P_ATM, 6.0e5),             # charging, choked (r = 0.17)
    (5.0e5, 6.0e5),             # charging, subsonic (r = 0.83)
    (6.0e5, P_ATM),             # exhausting, choked: flow out of the node
    (1.2e5, P_ATM),             # exhausting, subsonic
])
def test_gas_orifice_follows_iso_6358(p_node, p_res):
    want = _iso6358(max(p_node, p_res), min(p_node, p_res)) * (1.0 if p_res > p_node else -1.0)
    assert _first_step_flow(p_node, p_res) == pytest.approx(want, rel=1e-6)


def test_choked_flow_does_not_depend_on_the_downstream_pressure():
    assert _first_step_flow(P_ATM, 6.0e5) == pytest.approx(_first_step_flow(1.5e5, 6.0e5), rel=1e-6)


@pytest.mark.parametrize("n", [1.0, 1.4])
def test_closed_gas_volume_change_is_polytropic(n):
    net = Network([Node("c", volume=2.0e-5, pressure=P_ATM, polytropic=n)], [])
    f = net.compile("cpu", 2)
    vols = np.linspace(2.0e-5, 0.8e-5, 41)
    for v in vols[1:]:
        f.volume.assign(np.full((2, 1), v))
        f.step(1.0e-3)
    assert f.pressure.numpy()[0, 0] == pytest.approx(P_ATM * (vols[0] / vols[-1]) ** n, rel=1e-12)


def test_polytropic_node_charges_with_kappa_v_over_n():
    """dP/dt = n·Q/V for inflow into an n-node (inflow at the node's temperature): a linear edge, exact step."""
    v, g, n, dt = 1.0e-5, 1.0e-6, 1.4, 1.0e-3
    net = Network([Node("c", volume=v, pressure=P_ATM, polytropic=n)],
                  [Edge("in", "linear", "c", reservoir=3.0 * P_ATM, conductance=g)])
    f = net.compile("cpu", 2)
    f.step(dt)
    want = 3.0 * P_ATM + (P_ATM - 3.0 * P_ATM) * math.exp(-n * g * dt / v)
    assert f.pressure.numpy()[0, 0] == pytest.approx(want, rel=1e-12)


def test_laminar_conductance_is_poiseuille():
    d, length, mu = 4.0e-3, 2.0, 0.03
    assert laminar_conductance(d, length, mu) == pytest.approx(math.pi * d ** 4 / (128.0 * mu * length))
    with pytest.raises(ValueError):
        laminar_conductance(0.0, length, mu)


def test_gas_edges_and_indices_are_validated():
    with pytest.raises(ValueError, match="orifice_gas is the gas law"):
        Network([Node("l", medium="liquid")], [Edge("v", "orifice_gas", "l", sonic_conductance=C)])
    with pytest.raises(ValueError, match="polytropic"):
        Node("l", medium="liquid", polytropic=1.4)
    with pytest.raises(ValueError, match="critical_ratio"):
        Edge("v", "orifice_gas", "n", critical_ratio=1.0)
