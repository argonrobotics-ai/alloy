# SPDX-License-Identifier: Apache-2.0
"""SPEED-NOW S1: IntegratedSimulation saves its period-start snapshots after health's check steps only when nothing
else restores them (alloy.integration.period_snapshots), on the CPU."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import alloy
from alloy.integration import period_snapshots as PS
from alloy.integration import profiles as P


@pytest.fixture(params=["on", "off"])
def graph(request, monkeypatch):
    for k, v in list(P.PROFILES.items()):
        monkeypatch.setitem(P.PROFILES, k, dataclasses.replace(v, graph=request.param))
    return request.param


def cubes(worlds: int = 2) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground(friction=1.0)
    s.add_box("c", size=(1.0,) * 3, pos=np.array([[3.0 * w, 0.0, 0.5001] for w in range(worlds)]), density=10.0,
              friction=1.0)
    return s


def counted(sim) -> list[int]:
    """The engine step count at each ``_good`` save from now on."""
    eng, at = sim.engine, []
    save = eng._good.save
    eng._good.save = lambda state, *a, **k: (at.append(eng.steps), save(state, *a, **k))[1]
    return at


def test_fast_checks_save_after_check_steps_only(graph) -> None:
    """profile="fast" (checks every 10 steps, a single MuJoCo entry, one 10 ms step per period): the first period,
    then after each check step: steps 0, 1, 11, 21 over 25 periods, and no health copy."""
    sim = alloy.build(cubes(), device="cpu", profile="fast")
    assert PS.sparse(sim.engine) and sim.engine.health.cfg.every == 10
    at = counted(sim)
    sim.step(25)
    assert at == [0, 1, 11, 21] and sim.check()["ok"]


def test_standard_checks_save_every_period() -> None:
    """The default (checks every step) keeps today's saves, every period."""
    sim = alloy.build(cubes(), device="cpu")
    assert not PS.sparse(sim.engine)
    n = sim.engine.clock.steps_per_period
    at = counted(sim)
    sim.step(5)
    assert at == [n * k for k in range(5)] and sim.physics_steps == 5 * n


def test_a_latched_world_freezes_at_its_last_checked_state(graph) -> None:
    """World 1 is kicked to over 20 m/s right after the check at step 10; the check at step 20 latches it INVALID. It
    freezes where it was checked (after step 10), not where its failing period started (after step 19, in flight);
    world 0 goes on."""
    sim = alloy.build(cubes(), device="cpu", profile="fast")
    sim.step(11)
    checked = sim.state(world=1)["c.pos"].copy()
    sim.step(1)
    sim.apply_wrench("c", force=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e6]]))
    sim.step(12)
    frozen = sim.state(world=1)["c.pos"]
    problems = sim.check()["problems"]
    assert problems and all(p["world"] == 1 for p in problems), problems
    np.testing.assert_allclose(frozen, checked, atol=1e-6, rtol=0)
    assert abs(float(sim.state(world=0)["c.pos"][2]) - 0.5) < 2e-3


ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def test_periods_that_never_end_on_a_check_save_every_period(graph) -> None:
    """WP5: a robot scene under profile="fast" steps 8 x 1.25 ms per period with checks every 10; no period ends on a
    check step (gcd 2), so it is not sparse and saves every period, as before."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    sim = alloy.build(s, device="cpu", profile="fast")
    assert sim.engine.clock.steps_per_period == 8 and sim.engine.health.cfg.every == 10
    at = counted(sim)
    sim.step(5)
    assert not PS.sparse(sim.engine) and at == [8 * k for k in range(5)]


class _Stepper:                                  # a stepper that can roll back (a coupled entry's)
    def rollback_requested(self, report) -> bool:
        return False


def test_rollback_or_regrow_is_never_sparse() -> None:
    """WP0: a stepper that can roll back, or an MPM grid that can regrow, restores the period start, so such a
    simulation keeps every period's copies; the decision is made each period (no cache)."""
    sim = alloy.build(cubes(), device="cpu", profile="fast")
    eng = sim.engine
    assert PS.sparse(eng)
    stepper = eng.stepper
    eng.stepper = _Stepper()
    assert not PS.sparse(eng)
    eng.stepper = stepper
    eng.parts.mpm_grid = object()
    assert not PS.sparse(eng)
    eng.parts.mpm_grid = None
    assert PS.sparse(eng)
