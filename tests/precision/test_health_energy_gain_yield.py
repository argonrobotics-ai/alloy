# SPDX-License-Identifier: Apache-2.0
"""A joint spring that yields does not read as an energy gain (audit AGENTS-6, 2026-11-09; CPU).

Before: health summed the joint spring about its build reference, 0.5 k (q - s0)². Past the yield the reference
moves with the joint, so that sum grew with nothing behind it: a 1 N·m/rad hinge sagging under its link's weight past
a 0.02 N·m yield went INVALID with ENERGY_JUMP at 0.56 s and froze at q = 0.26 rad. The plug-in now books the
over-count (``alloy.plugins.joint_yield``, Energy), so health's net energy is the true one.

Laws: the clip sags through its yield with health OK and the booked work equal to the over-count
0.5 k (q - s0)² - 0.5 k (q - rest)²; the net energy only falls (plastic work leaves); an energy gain nothing
explains still trips ENERGY_JUMP beside it.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.precision.health_energy_gain import GAIN_REL

K, YIELD = 1.0, 0.02                      # N·m/rad, N·m; the link's weight gives 0.0245 N·m at q = 0
CLIP = """<mujoco><compiler angle="radian"/><worldbody><body name="link" pos="0 0 0.1">
  <joint name="arm" type="hinge" axis="0 1 0" stiffness="1" springref="0" damping="0.01"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.005" mass="0.05"/></body></worldbody></mujoco>"""


def _clip(monkeypatch, route: str, worlds: int = 1):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_robot("clip", mjcf=CLIP, pos=(0, 0, 0.3))
    scene.set_joint_yield("clip/arm", torque=YIELD)
    return alloy.build(scene, device="cpu")


@pytest.mark.parametrize("route, seconds", [("mjc", 2.0), ("warp", 0.8)])   # frozen at 0.56 s before, on both
def test_a_joint_that_yields_under_its_load_stays_ok_and_its_set_is_booked(route, seconds, monkeypatch) -> None:
    sim = _clip(monkeypatch, route)
    health = sim.engine.health
    assert health._energy_on                                       # passive: the energy check runs
    sim.run(seconds)
    st = sim.state(world=0)
    q, rest = float(st["clip/arm.q"]), float(st["clip/arm.rest"])
    assert q > 0.3 and q - rest == pytest.approx(YIELD / K, rel=0.02)          # it yielded and went on sagging
    h = sim.health()
    assert h["worlds"][0]["severity"] == "OK" and h["worlds"][0]["flags"] == [] and sim.check()["ok"]
    over = 0.5 * K * q * q - 0.5 * K * (q - rest) ** 2             # what the build-reference sum over-counts
    ref = health._ref
    assert float(ref.w_cum.numpy()[0]) == pytest.approx(over, rel=1e-4)
    assert float(ref.e_max.numpy()[0] - ref.e0.numpy()[0]) < 1e-3 * float(ref.e_scale.numpy()[0])   # only falls
    assert float(health.gain.state.peak.numpy()[0]) < 0.2 * GAIN_REL


def test_an_unexplained_gain_still_trips_energy_jump_beside_a_yielding_joint(monkeypatch) -> None:
    sim = _clip(monkeypatch, "mjc", worlds=2)
    sim.run(0.5)
    jqd = sim.engine.state.joint_qd.numpy()
    jqd[0] -= 40.0                                                 # world 0: a 40 rad/s kick nothing explains
    sim.engine.state.joint_qd.assign(jqd)
    sim.run(0.05)
    w0, w1 = sim.health()["worlds"]
    assert w0["severity"] == "INVALID" and "ENERGY_JUMP" in w0["flags"]
    assert w1["severity"] == "OK"


def test_a_reset_takes_the_booked_set_back(monkeypatch) -> None:
    sim = _clip(monkeypatch, "mjc")
    sim.run(1.0)
    assert float(sim.engine.health._ref.w_cum.numpy()[0]) > 0.05
    sim.reset()
    sim.run(1.0)
    q, rest = (float(sim.state(world=0)[k]) for k in ("clip/arm.q", "clip/arm.rest"))
    over = 0.5 * K * q * q - 0.5 * K * (q - rest) ** 2
    assert float(sim.engine.health._ref.w_cum.numpy()[0]) == pytest.approx(over, rel=1e-4)
    assert sim.health()["worlds"][0]["severity"] == "OK" and np.isfinite(q)
