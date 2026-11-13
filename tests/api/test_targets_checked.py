# SPDX-License-Identifier: Apache-2.0
"""Joint targets are checked before anything is written or stepped (Gemini audit S2-045).

A NaN target passed the range clamp (``np.clip`` keeps NaN), was reported as "held at the limit" and reached
MuJoCo's ctrl with health ok; a run shorter than half a control period stepped nothing and checked nothing (an
unknown joint passed); a ramp end that was not a number leaked a bare ValueError; a per-world list of the wrong
length raised UnknownNameError. Each now raises SimulationError (UnknownNameError for an unknown joint) naming the
joint and its fix, and every target that worked before still works the same."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.api as alloy
from alloy.cli.command import Session, respond

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _scene() -> alloy.Scene:
    sc = alloy.Scene(worlds=2, dt=0.02)
    sc.add_ground()
    sc.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    return sc


@pytest.fixture(scope="module")
def sim():
    s = alloy.build(_scene(), device="cpu")
    yield s
    s.close()


def _ctrl(sim) -> np.ndarray:
    return sim.engine.control.joint_target_q.numpy().copy()


@pytest.mark.parametrize("value", [math.nan, math.inf, [0.1, math.nan], [0.1, 0.2, 0.3], "bad", [None, 0.1]])
def test_a_bad_target_raises_and_writes_nothing(sim, value):
    before = _ctrl(sim)
    with pytest.raises(alloy.SimulationError) as err:
        sim.set_targets({"shoulder": value})
    assert err.value.field == "shoulder" and err.value.fix
    assert np.array_equal(_ctrl(sim), before)


def test_a_run_too_short_for_a_period_still_checks_its_targets(sim):
    t0 = sim.t
    with pytest.raises(alloy.UnknownNameError):
        sim.run(0.25 * sim.dt, targets={"arm/no_such_joint": 1.5})
    with pytest.raises(alloy.SimulationError, match="not finite"):
        sim.run(0.25 * sim.dt, targets={"shoulder": (0.0, math.nan)})
    with pytest.raises(alloy.SimulationError, match="not a number"):
        sim.run(0.1, targets={"shoulder": ("bad", "num")})
    assert sim.t == t0
    assert sim.run(0.25 * sim.dt, targets={"shoulder": 0.3}).meta["periods"] == 0     # a valid one: no periods


def test_targets_that_worked_still_work(sim):
    sim.set_targets({"shoulder": 0.4})
    sim.set_targets({"arm/shoulder": np.array([0.1, -0.1])})
    sim.set_targets({"shoulder": 3.0})                              # outside the range: still held at the limit
    assert float(_ctrl(sim).max()) == pytest.approx(2.0, rel=0.0, abs=0.0)
    traj = sim.run(0.2, record=["arm/shoulder"], targets={"shoulder": (0.0, [0.5, -0.5])})
    assert traj.meta["periods"] == 10 and sim.check()["ok"]
    assert np.allclose(_ctrl(sim)[:2], [0.5, -0.5], rtol=0.0, atol=1e-6)


def test_the_command_line_reports_a_bad_target_by_name():
    sess = Session()
    try:
        built = respond(sess, {"op": "build", "scene": _scene().to_dict(), "device": "cpu"})
        sid = built["result"]["sim"]
        for req in ({"op": "run", "seconds": 0.04, "targets": {"shoulder": {"from": "bad", "to": "num"}}},
                    {"op": "step", "steps": 1, "targets": {"shoulder": float("nan")}},
                    {"op": "targets", "targets": {"shoulder": [0.1, 0.2, 0.3]}}):
            out = respond(sess, {**req, "sim": sid})
            assert not out["ok"] and out["error"]["type"] == "SimulationError", out
            assert out["error"]["field"] == "shoulder"
    finally:
        sess.close_all()
