# SPDX-License-Identifier: Apache-2.0
"""Foot contact forces from the solver (``alloy.envs.loco.contact``) on both CPU routes: a standing Go2's feet carry
its weight, a lifted one reads zero, the sensors take the forces, and the trajectory is bit for bit the one without
the reader."""

from __future__ import annotations

import os

import numpy as np
import pytest

from alloy.envs import EnvError
from alloy.envs.loco import GO2, Go2VelocityConfig, go2_velocity
from alloy.envs.loco.contact import FootForces
from alloy.envs.loco.env import stand_action
from alloy.route.mjc_core_config import CPU_ROUTE_ENV

W, STEPS, LIFT = 2, 50, 0.2                    # 50 control steps of 20 ms = 1 s
SOLVER = {"mjc": "MjcCoreSolver", "warp": "SolverMuJoCo"}


def _env(contact: str):
    return go2_velocity(num_envs=W, device="cpu", backend="numpy",
                        config=Go2VelocityConfig(pushes=None, joint_jitter=0.0, contact=contact))


def _stand(env, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Zero actions for ``steps`` control steps; every step's body poses and joint positions."""
    out, state = [], env.sim.engine.state
    for _ in range(steps):
        env.step(stand_action(env))
        state = env.sim.engine.state
        out.append((state.body_q.numpy().copy(), state.joint_q.numpy().copy()))
    return out


def _measure(route: str) -> dict:
    """One run without the reader and one with it; then the robot of the second lifted by ``LIFT``."""
    plain = _env("height")
    plain.reset(seed=0)
    off = _stand(plain, STEPS)
    plain.close()
    env = _env("force")
    forces = env.sensors.reader                 # a FootForces, made by contact="force"
    before = np.array(forces.read())
    env.reset(seed=0)
    on = _stand(env, STEPS)
    model = env.sim.engine.model
    out = {"off": off, "on": on, "before": before, "standing": np.array(forces.read()),
           "norm": np.array(forces.read("norm")), "solver": type(env.sim.engine.solvers[0]).__name__,
           "weight": model.body_mass.numpy().reshape(W, -1).sum(1) * 9.81,
           "sensed": {k: np.array(env.state[k]) for k in ("feet.force", "feet.contact")}}
    standing_others = np.array(forces.read_others())
    env.sim.set_pose("go2", pos=env.sim.state(["go2"])["go2.pos"] + np.array([0.0, 0.0, LIFT], np.float32))
    env.step(stand_action(env))
    out["lifted"] = np.array(forces.read())
    out["lifted_height"] = np.array(env.state["feet.height"])
    out["lifted_contact"] = np.array(env.state["feet.contact"])
    out["standing_others"] = np.array(standing_others)
    env.sim.set_pose("go2", pos=(0.0, 0.0, 0.2), quat_wxyz=(0.0, 1.0, 0.0, 0.0))     # on its back
    env.sim.step(15)
    out["on_back_others"] = np.array(forces.read_others())
    env.close()
    return out


@pytest.fixture(scope="module", params=["warp", "mjc"])
def run(request) -> dict:
    old = os.environ.get(CPU_ROUTE_ENV)
    os.environ[CPU_ROUTE_ENV] = request.param
    try:
        out = _measure(request.param)
    finally:
        if old is None:
            os.environ.pop(CPU_ROUTE_ENV, None)
        else:
            os.environ[CPU_ROUTE_ENV] = old
    out["route"] = request.param
    return out


@pytest.mark.compile_heavy
def test_standing_feet_carry_the_weight(run):
    assert run["solver"] == SOLVER[run["route"]]
    force, weight = run["standing"], run["weight"]
    assert force.shape == (W, len(GO2.feet)) and (run["before"] == 0).all()
    assert 100.0 < weight[0] < 200.0                                   # a Go2: about 15 kg
    np.testing.assert_allclose(force.sum(1), weight, rtol=0.05, atol=0)
    assert (force > 5.0).all()
    assert (run["norm"] >= force).all() and (run["norm"] < 1.2 * force).all()   # mostly normal force


@pytest.mark.compile_heavy
def test_sensors_take_the_forces(run):
    np.testing.assert_array_equal(run["sensed"]["feet.force"], run["standing"])
    assert (run["sensed"]["feet.contact"] == 1).all()


@pytest.mark.compile_heavy
def test_lifted_robot_reads_zero(run):
    assert (run["lifted_height"] > 0.1).all()                          # the links followed the base
    assert (run["lifted"] == 0).all() and (run["lifted_contact"] == 0).all()


@pytest.mark.compile_heavy
def test_trajectory_is_the_same_with_the_reader(run):
    assert len(run["on"]) == len(run["off"]) == STEPS
    for (body_on, joint_on), (body_off, joint_off) in zip(run["on"], run["off"], strict=True):
        np.testing.assert_array_equal(body_on, body_off)
        np.testing.assert_array_equal(joint_on, joint_off)


def test_errors_say_what_to_pass():
    with pytest.raises(EnvError, match="worlds, feet"):
        FootForces(None, np.zeros(4, np.int32), None)


def test_other_parts_touch_nothing_standing_and_something_on_the_back(run: dict) -> None:
    """Standing, no non-foot shape carries a contact force; a robot on its back rests on some of them, and they
    carry a large share of its weight."""
    assert run["standing_others"].shape[0] == W and run["standing_others"].shape[1] > 10
    assert (run["standing_others"] == 0.0).all()
    on_back = run["on_back_others"]
    assert ((on_back > 1.0).sum(1) >= 1).all()
    assert (on_back.sum(1) > 0.5 * run["weight"]).all()


def test_randomized_robot_mass_shows_in_the_foot_forces() -> None:
    """With a per-world mass scale passed through the config, each world's feet carry its own weight."""
    config = Go2VelocityConfig(pushes=None, scene_randomize=[("go2", "mass_scale", 0.8, 1.2)])
    env = go2_velocity(num_envs=3, device="cpu", backend="numpy", config=config)
    env.reset(seed=0)
    for _ in range(STEPS):
        env.step(stand_action(env))
    scale = np.asarray(env.sim.randomized()["go2.mass_scale"])
    force = np.asarray(env.state["feet.force"]).sum(1)
    env.close()
    assert scale.min() >= 0.8 and scale.max() <= 1.2 and np.ptp(scale) > 0.01
    np.testing.assert_allclose(force / force[0], scale / scale[0], rtol=0.03)
