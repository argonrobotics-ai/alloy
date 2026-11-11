# SPDX-License-Identifier: Apache-2.0
"""Law: profile "precision" raises ``alloy.SimulationError`` from ``step`` and ``run`` in the control period a world
becomes INVALID (audit AGENTS-2 / API-12 / ERR-3 / WORLDS-4; CPU).

AGENTS.md section 8: "The precision profile checks every step and raises if a world becomes INVALID." Before this,
``step`` and ``run`` returned and only ``sim.check()`` raised, with an engine ``PrecisionError``.

One case per INVALID cause a user can reach: SPEED_FAIL (a huge push), FAR_FROM_ORIGIN (no ground), PEN_FAIL (a
start overlap), NONFINITE (a NaN pose), TUNNEL (a fast ball through a thin plate). Laws, each per case:
- stepping one period at a time, the first period after which ``sim.health()`` would say INVALID is the one that
  raises; ``step(n)`` and ``run(seconds)`` over the same span raise at that same time, not at their end;
- the error is an ``alloy.SimulationError`` naming the flag, with ``message``, ``fix`` and ``to_dict()``;
- after the raise the sim still answers: ``health()`` lists the world INVALID, ``check()`` returns its dict with
  ``ok`` False, ``report()`` returns; a later ``step`` does not raise for the same world again;
- ``sim.reset()`` re-arms it: the same failure raises again;
- profile "standard" never raises: the world freezes and health says so.
A DEGRADED flag is not a failure: a weld torn 7 mm (JOINT_FAIL) raises nothing under precision and is listed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

import alloy

DT = 0.01


@dataclass(frozen=True)
class Case:
    flag: str
    scene: Callable[[], alloy.Scene]
    arm: Callable[[alloy.Sim], None]          # what the user does before stepping (a push, a bad pose)
    periods: int                              # the failure comes within this many control periods


def _ball_pushed() -> alloy.Scene:
    scene = alloy.Scene(dt=DT)
    scene.add_sphere("ball", 0.05, pos=(0, 0, 1.0), mass=1.0)
    return scene


def _box_no_ground() -> alloy.Scene:
    scene = alloy.Scene(dt=DT)
    scene.add_box("box", size=(0.1, 0.1, 0.1), pos=(0, 0, 1.0), mass=1.0)
    return scene


def _block_in_table() -> alloy.Scene:
    scene = alloy.Scene(dt=DT)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.395), mass=0.2)      # 3 cm inside the top
    return scene


def _ball_over_plate() -> alloy.Scene:
    scene = alloy.Scene(dt=DT)
    scene.add_ground()
    scene.add_box("plate", size=(0.5, 0.5, 0.01), pos=(0, 0, 0.3), static=True)
    scene.add_sphere("ball", 0.01, pos=(0, 0, 0.315 + 8.0), density=7800.0)           # 12.5 m/s at the plate
    return scene


def _nothing(sim: alloy.Sim) -> None:
    return None


CASES = [
    Case("SPEED_FAIL", _ball_pushed, lambda sim: sim.apply_wrench("ball", force=(200.0, 0.0, 9.81)), 20),
    Case("FAR_FROM_ORIGIN", _box_no_ground, _nothing, 200),
    Case("PEN_FAIL", _block_in_table, _nothing, 5),
    Case("NONFINITE", _box_no_ground, lambda sim: sim.set_pose("box", pos=(float("nan"), 0.0, 1.0)), 5),
    Case("TUNNEL", _ball_over_plate, _nothing, 150),
]


def _armed(case: Case, profile: str = "precision") -> alloy.Sim:
    sim = alloy.build(case.scene(), device="cpu", profile=profile)
    case.arm(sim)
    return sim


def _periods_until_raise(sim: alloy.Sim, limit: int) -> tuple[int, alloy.SimulationError]:
    for k in range(1, limit + 1):
        try:
            sim.step(1)
        except alloy.SimulationError as err:
            return k, err
    pytest.fail(f"no raise within {limit} periods; health {sim.health()['worlds']}")


@pytest.fixture(scope="module", params=CASES, ids=[c.flag for c in CASES])
def failed(request) -> tuple[Case, alloy.Sim, int, alloy.SimulationError]:
    case = request.param
    sim = _armed(case)
    k, err = _periods_until_raise(sim, case.periods)
    return case, sim, k, err


@pytest.mark.law
def test_the_error_is_a_simulation_error_naming_the_flag(failed) -> None:
    case, _, _, err = failed
    assert isinstance(err, alloy.SimulationError)
    assert case.flag in err.message and "INVALID" in err.message
    assert err.fix and err.to_dict()["message"] == err.message


@pytest.mark.law
def test_the_sim_still_answers_and_does_not_raise_twice(failed) -> None:
    case, sim, _, _ = failed
    health = sim.health()
    assert health["invalid_worlds"] == [0] and any(case.flag in f for f in health["worlds"][0]["flags"])
    result = sim.check()
    assert result["ok"] is False and result["problems"]
    assert isinstance(sim.report(), dict)
    t = sim.t
    sim.step(3)                                           # the world is frozen; it raised once already
    assert sim.t == pytest.approx(t + 3 * DT)


@pytest.mark.law
def test_step_n_and_run_raise_in_that_same_period(failed) -> None:
    case, _, k, _ = failed
    for go in (lambda s: s.step(case.periods + 50), lambda s: s.run((case.periods + 50) * DT, record=[])):
        sim = _armed(case)
        with pytest.raises(alloy.SimulationError) as caught:
            go(sim)
        assert case.flag in caught.value.message
        assert sim.t == pytest.approx(k * DT, abs=1e-9), (sim.t, k)      # not at the end of the call


@pytest.mark.law
def test_reset_re_arms_the_raise(failed) -> None:
    case, sim, k, _ = failed
    sim.reset()
    case.arm(sim)
    again, err = _periods_until_raise(sim, case.periods)
    # the same period; the flags named can differ (a start overlap is judged once per build, so after a reset the
    # block in the table fails by ENERGY_JUMP | START_PEN, not PEN_FAIL)
    assert again == k and "INVALID" in err.message


@pytest.mark.law
@pytest.mark.parametrize("case", CASES, ids=[c.flag for c in CASES])
def test_standard_never_raises_and_freezes_the_world(case: Case) -> None:
    sim = _armed(case, profile="standard")
    sim.step(case.periods + 20)
    health = sim.health()                    # (a start overlap is INVALID there by other flags than PEN_FAIL)
    assert health["invalid_worlds"] == [0] and "LATCHED" in health["worlds"][0]["flags"]
    assert sim.check()["ok"] is False


WELD_RIG = """<mujoco><compiler angle="radian"/><worldbody>
<body name="a" pos="0 0 0.6"><joint name="ja" type="slide" axis="1 0 0" range="-1 1"/>
  <geom type="box" size="0.03 0.03 0.03" mass="5"/></body>
<body name="b" pos="0.1 0 0.6"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="20"/></body>
</worldbody><equality><weld body1="a" body2="b"/></equality>
<actuator><position joint="ja" kp="5000" kv="200"/></actuator></mujoco>"""


@pytest.mark.law
def test_a_degraded_flag_does_not_raise() -> None:
    """JOINT_FAIL is DEGRADED (the ladder): a weld pulled 7 mm apart at 3 kN is listed, not raised."""
    scene = alloy.Scene(dt=DT)
    scene.add_ground()
    scene.add_robot("rig", mjcf=WELD_RIG)
    sim = alloy.build(scene, device="cpu", profile="precision", welds="mujoco")   # MuJoCo's soft row, torn on purpose
    sim.step(5)
    sim.apply_wrench("rig/b", force=np.array([0.0, 3000.0, 0.0]))
    sim.run(0.5, record=[])
    row = sim.health()["worlds"][0]
    assert "JOINT_FAIL" in row["flags"] and row["severity"] == "DEGRADED"
