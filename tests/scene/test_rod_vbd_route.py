# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD R2 / R5: the ROD_VBD route on CPU. A rod goes there wherever CHAIN cannot do the job: past CHAIN's
40-link cap (coordinator ruling 2026-11-08 11:53Z, option A; was a RouteError) or when a card's route override asks
(a clamp writes it). The `vbd` entry steps it with SolverVBDRod mixed in. Short elastic rods stay on CHAIN.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy._vendor import newton
from alloy.api import _rods
from alloy.solvers.rod import SolverVBDRod, has_rod_chains, with_rod_solve
from alloy.solvers.vbd import SolverVBDAlloy, SolverVBDMomentum, make_vbd


def _cable(length: float, **kw) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_rod("cable", length=length, radius=0.002, pos=(-0.5 * length, 0.0, 0.05), **kw)
    return sc


@pytest.mark.law
def test_a_soft_cable_past_chains_cap_routes_to_rod_vbd_and_says_why():
    """4 m of USB cable needs 47 links, more than CHAIN's 40: ROD_VBD with no override (was a RouteError)."""
    sim = alloy.build(_cable(4.0), device="cpu")
    r = sim.routes()["cable"]
    assert r["route"] == "rod_vbd" and r["solver"] == "vbd"
    assert any("links > CHAIN's 40" in why for why in r["reasons"])
    est = alloy.estimate(_cable(4.0), 1, 0.1, device="cpu")
    assert est["routes"]["cable"]["route"] == "rod_vbd" and est["routes"]["cable"]["reasons"] == r["reasons"]
    sim.close()


@pytest.mark.law
def test_the_override_routes_it_to_rod_vbd_with_segments_past_the_cap(monkeypatch):
    monkeypatch.setattr(_rods, "DEFAULT_ROUTE", "rod_vbd")
    sim = alloy.build(_cable(4.0), device="cpu")
    r = sim.routes()["cable"]
    assert r["route"] == "rod_vbd" and r["solver"] == "vbd"
    n = int(sim.built.report.decisions["cable"].numbers["links"])
    assert n > 40
    p0 = sim.state(["cable"], world=0)["cable.points"]
    assert p0.shape == (n + 1, 3)
    sim.step(30)
    p1 = sim.state(["cable"], world=0)["cable.points"]
    assert np.isfinite(p1).all() and p1[:, 2].min() > 0.0                      # falls onto the ground, not through
    assert p1[:, 2].max() < p0[:, 2].max()
    assert sim.check()["ok"]
    sim.close()


def test_segments_set_the_rod_vbd_segment_count(monkeypatch):
    monkeypatch.setattr(_rods, "DEFAULT_ROUTE", "rod_vbd")
    sim = alloy.build(_cable(0.3, segments=17), device="cpu")
    assert int(sim.built.report.decisions["cable"].numbers["links"]) == 17
    assert sim.state(["cable"], world=0)["cable.points"].shape == (18, 3)
    sim.close()


def test_make_vbd_mixes_the_rod_solve_in_only_for_rod_models():
    b = newton.ModelBuilder()
    b.add_ground_plane()
    pts = np.stack([np.linspace(0.0, 0.3, 6), np.zeros(6), np.full(6, 0.1)], axis=1)
    b.add_rod(rod=newton.Rod(pts, radius=0.003), label="r")
    b.color()
    rods = b.finalize(device="cpu")
    assert has_rod_chains(rods)
    for coarse, base in ((True, SolverVBDMomentum), (False, SolverVBDAlloy)):
        s = make_vbd(rods, {"iterations": 1, "coarse_step": coarse})
        assert isinstance(s, SolverVBDRod) and isinstance(s, base) and type(s) is with_rod_solve(base)
    b2 = newton.ModelBuilder()
    b2.add_ground_plane()
    b2.add_body(xform=None)
    b2.color()
    plain = b2.finalize(device="cpu")
    assert not has_rod_chains(plain)
    assert type(make_vbd(plain, {"iterations": 1})) is SolverVBDMomentum


WRIST = """<mujoco><compiler angle="radian"/><worldbody><body name="wrist" pos="0.388 0 0.05">
<joint name="roll" type="hinge" axis="1 0 0" damping="0.02"/><inertial pos="0 0 0" mass="0.3" diaginertia="2e-4 2e-4 2e-4"/>
<body name="left" pos="0 0.013 0"><joint name="lf" type="slide" axis="0 -1 0" range="0 0.02" damping="1"/>
<geom type="box" size="0.01 0.004 0.01" mass="0.2" friction="1.0"/></body>
<body name="right" pos="0 -0.013 0"><joint name="rf" type="slide" axis="0 1 0" range="0 0.02" damping="1"/>
<geom type="box" size="0.01 0.004 0.01" mass="0.2" friction="1.0"/></body></body></worldbody>
<actuator><position joint="roll" kp="20" kv="0.5"/><position joint="lf" kp="300" kv="5"/><position joint="rf" kp="300" kv="5"/></actuator></mujoco>"""
#: WP9's rows ([internal note] notes/wp9-runs/rod/r6/api_speed.py): wall s per simulated s on CPU, 1 world, at the rod-only
#: entry's 4 iterations (S4, 2026-11-09, notes/wp9-runs/s4/speed_*_iter4.log; at 10 the same host read 4.49 / 161.2)
MEASURED_ROD_S = {"free": 2.49, "robot": 94.9}


@pytest.mark.parametrize("kind", ["free", "robot"])
def test_rod_vbd_estimate_is_within_its_stated_3x(monkeypatch, kind):
    """A 0.4 m USB cable (20 links) on rod_vbd, dropped, or beside a wrist's two 0.2 kg pads (64 force exchanges per
    period, 2+ passes): the estimate charges the rod's links per iteration and the exchange clock (OSS-17: 16x low)."""
    monkeypatch.setattr(_rods, "DEFAULT_ROUTE", "rod_vbd")
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_rod("cable", length=0.4, radius=0.003, pos=(0.0, 0.0, 0.05 if kind == "robot" else 0.1), segments=20,
               material="cable_usb_composite")
    if kind == "robot":
        sc.add_robot("hand", mjcf=WRIST)
    est = alloy.estimate(sc, 1, 1.0, device="cpu")
    assert est["routes"]["cable"]["route"] == "rod_vbd" and est["vbd"]["links"] >= 20
    assert est["vbd"]["coupled"] is (kind == "robot")
    ratio = est["wall_time_s"] / MEASURED_ROD_S[kind]
    assert 1 / 3 < ratio < 3, est["wall_time_s"]
