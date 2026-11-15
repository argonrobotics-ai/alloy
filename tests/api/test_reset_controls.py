# SPDX-License-Identifier: Apache-2.0
"""``sim.reset()`` puts joint targets and controls back to the start, as it puts back poses and removes drives.

Laws (CPU, on both CPU routes: ``mjc``, one MuJoCo C model per world, what users get, and ``warp``): after a run with
a target ramp, ``reset()`` and the same steps repeat the first steps of the build bit for bit (the arm no longer steps
toward the old target); after ``set_targets`` then ``settle``, ``reset()`` returns to the settled start with the
targets the settle ran with, so the run after it repeats the run after the settle bit for bit, a block resting on the
ground too; raw controls go back too. Before, the last targets stayed: the arm snapped from 0 back to 0.8 rad after
the reset. On ``mjc`` the run after a reset to a settled start was pushed from the float32 start, not the float64 state
the settle ended in, and drifted by 3e-8 rad within 20 steps.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body>
  <body name="link2" pos="0.5 0 0.1"><joint name="elbow" type="hinge" axis="0 1 0" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/><motor name="m" joint="elbow" gear="1"/></actuator></mujoco>"""


SOLVER = {"mjc": "MjcCoreSolver", "warp": "SolverMuJoCo"}      # the rigid solver each CPU route builds


@pytest.fixture(params=sorted(SOLVER))
def route(request, monkeypatch) -> str:
    """Both CPU routes, whatever the suite pinned."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", request.param)
    return request.param


def _sim(route: str) -> alloy.Sim:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(1.2, 0, 0.03), mass=0.2)    # settles onto the ground
    sim = alloy.build(scene, device="cpu")
    assert SOLVER[route] in [type(s).__name__ for s in sim.engine.solvers], route
    return sim


def _q(sim: alloy.Sim, steps: int) -> np.ndarray:
    traj = sim.run(steps * sim.dt, record=["arm/shoulder", "arm/elbow", "block"])
    return np.concatenate([traj["arm/shoulder.q"].ravel(), traj["arm/elbow.q"].ravel(), traj["block.pos"].ravel()])


@pytest.mark.compile_heavy
def test_reset_returns_targets_and_controls_to_the_start(route: str):
    sim = _sim(route)
    first = _q(sim, 20)
    sim.run(1.0, record=[], targets={"arm/shoulder": (0.0, 0.8)})
    sim.set_controls({"arm/m": 2.0})
    sim.reset()
    again = _q(sim, 20)
    assert np.array_equal(again, first)                       # was snapping toward 0.8 rad, the motor still on
    sim.close()


@pytest.mark.compile_heavy
def test_reset_after_settle_keeps_the_targets_the_settle_ran_with(route: str):
    sim = _sim(route)
    sim.set_targets({"arm/shoulder": 0.3})
    sim.settle(0.5)
    after_settle = _q(sim, 20)
    sim.set_targets({"arm/shoulder": -0.5})
    sim.step(20)
    sim.reset()
    assert np.array_equal(_q(sim, 20), after_settle)          # mjc: 3e-8 rad off, pushed from the float32 start
    sim.close()
