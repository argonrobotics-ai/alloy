# SPDX-License-Identifier: Apache-2.0
"""Lumped hydraulics and pneumatics (alloy/plugins/library/fluid_network.py): the harness laws H-1..H-6, the vane stall,
balance and null, the declarative config, the inputs, the vane and the stability bound (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.fluid import P_ATM, Edge, Node
from alloy.plugins.fluid.actuators import Cylinder, Vane
from alloy.plugins.harness.scene import build_simulation
from alloy.plugins.library.fluid_network import FluidNetwork
from alloy.plugins.library.fluid_network_laws import VANE, stall_scene

pytestmark = pytest.mark.law


def test_fluid_network_quick_check_passes_every_law():
    report = ap.check("fluid_network", quick=True)
    assert report.passed, report.summary()
    laws = {c.name for c in report.checks if c.group == "law"}
    assert {"H-1_orifice_liquid", "H-1_orifice_gas", "H-2_laminar_line", "H-3_closed_isothermal",
            "H-3_closed_adiabatic", "H-4_stall_rises_below", "H-4_stall_sinks_above", "amount_balance",
            "null_closed_at_atmosphere", "H-5_step_response", "H-6_energy_ledger", "vane_stall_rises_below",
            "vane_stall_sinks_above", "F-14_tank_drain", "F-16_tank_dilution"} <= laws


@pytest.mark.slow
def test_fluid_network_full_check():
    report = ap.check("fluid_network")
    assert report.passed, report.summary()


def test_from_dict_matches_the_dataclass_config():
    spec = {"nodes": [{"name": "a", "pressure": 3e5}],
            "edges": [{"name": "v", "kind": "orifice_gas", "a": "a", "sonic_conductance": 1e-9}],
            "actuators": [{"name": "c", "joint": "cyl_slide", "a": "a", "bore": 0.02}],
            "openings": {"v": 0.5}}
    net = FluidNetwork.from_dict(spec)
    assert net.config.nodes == (Node("a", pressure=3e5),)
    assert net.config.edges == (Edge("v", "orifice_gas", "a", sonic_conductance=1e-9),)
    assert net.config.actuators == (Cylinder("c", joint="cyl_slide", a="a", bore=0.02),)
    assert net.config.openings == (("v", 0.5),)


def test_set_opening_writes_one_edge_column():
    plug = FluidNetwork()
    sim = build_simulation(stall_scene(plug, 0.5, periods=1), [plug], "cpu")
    plug.set_opening("exhaust_cap", [0.25, 0.5, 2.0])                    # clipped to [0, 1]
    col = plug.fluid.net.edge_index("exhaust_cap")
    assert np.allclose(plug.fluid.opening.numpy()[:, col], [0.25, 0.5, 1.0])
    assert np.allclose(plug.fluid.opening.numpy()[:, plug.fluid.net.edge_index("supply_cap")], 1.0)
    sim.step_period()


def test_stable_dt_is_two_over_the_chamber_frequency():
    plug = FluidNetwork()
    sim = build_simulation(stall_scene(plug, 0.5, periods=1), [plug], "cpu")
    cyl = plug.config.actuators[0]
    va, vb = cyl.volumes(cyl.x0)
    k = cyl.area_a ** 2 * 6.0e5 / va + cyl.area_b ** 2 * P_ATM / vb
    m = float(sim.model.body_mass.numpy()[int(sim.plugins.ctx.bodies("cyl_rod").numpy()[0])])
    assert plug.stable_dt(sim.model)["cyl"] == pytest.approx(2.0 / math.sqrt(k / m), rel=1e-6)


def test_vane_geometry_and_torque():
    v = Vane("v", joint="h", a="a", b="b", displacement=2e-6, travel=math.pi, dead_a=1e-7, dead_b=3e-7)
    assert v.volumes(0.5) == pytest.approx((1e-7 + 1e-6, 3e-7 + 2e-6 * (math.pi - 0.5)))
    assert v.force(4e5, 2e5) == pytest.approx(0.4)                      # (P_A - P_B)·D [N·m]
    assert FluidNetwork.from_dict({**{k: VANE[k] for k in ("nodes", "edges")},
                                   "actuators": [{"name": "v", "joint": "h", "a": "va", "b": "vb",
                                                  "displacement": 1e-6}]}).config.actuators[0] == \
        Vane("v", joint="h", a="va", b="vb", displacement=1e-6)


def test_actuators_need_the_right_joint_type():
    plug = FluidNetwork(**{**VANE, "actuators": (Vane("vane", joint="cyl_slide", a="va", b="vb"),)})
    with pytest.raises(Exception, match="revolute"):
        build_simulation(stall_scene(plug, 0.5, periods=1), [plug], "cpu")


def test_config_is_validated():
    with pytest.raises(ap.PluginError, match="chamber of both"):
        FluidNetwork(actuators=(Cylinder("a", "cyl_slide", "cap"), Cylinder("b", "cyl_slide", "cap")))
    with pytest.raises(ap.PluginError, match="unknown node"):
        FluidNetwork(actuators=(Cylinder("a", "cyl_slide", "nope"),))
    with pytest.raises(ap.PluginError, match="openings"):
        FluidNetwork(openings=(("supply_cap", 1.5),))
    with pytest.raises(ValueError, match="rod < bore"):
        Cylinder("a", "cyl_slide", "cap", bore=0.02, rod=0.03)
