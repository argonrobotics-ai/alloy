# SPDX-License-Identifier: Apache-2.0
"""Row 4 bench envs (FrankaLift, LeapReorient): build, hold still, respond to actions, reset per world. CPU, tiny."""

from __future__ import annotations

import numpy as np
import pytest

from bench.scorecard.train.alloy_envs import make_franka_lift, make_leap_reorient
from bench.scorecard.train.alloy_envs.quat import angle_between, random_unit_quats

W = 2


@pytest.fixture(scope="module")
def franka():
    return make_franka_lift(W, device="cpu", backend="numpy")


@pytest.fixture(scope="module")
def leap():
    return make_leap_reorient(W, device="cpu", backend="numpy")


def _hold(env, steps, n):
    env.reset(seed=0)
    for _ in range(steps):
        env.step(np.zeros((W, n), np.float32))


def test_franka_holds_still_at_zero_action(franka):
    franka.reset(seed=0)
    start = franka.state["cube.pos"].copy()
    _hold(franka, 20, 8)
    assert np.abs(franka.state["cube.pos"] - start).max() < 0.05
    assert franka.sim.check()["ok"]


def test_franka_action_moves_arm_and_metrics_exist(franka):
    franka.reset(seed=0)
    q0 = franka.state["panda/joint1.q"].copy()
    a = np.zeros((W, 8), np.float32)
    a[:, 0] = 1.0
    for _ in range(10):
        franka.step(a)
    assert (franka.state["panda/joint1.q"] - q0).min() > 0.1
    assert set(franka.last_metrics) == {"goal_dist", "lifted", "success_2cm", "success_5cm"}


def test_franka_reset_one_world_keeps_other(franka):
    franka.reset(seed=0)
    for _ in range(5):
        franka.step(np.ones((W, 8), np.float32))
    q_other = float(franka.state["panda/joint1.q"][1])
    franka.reset(options={"reset_mask": np.array([True, False])})
    assert abs(float(franka.state["panda/joint1.q"][0])) < 0.2
    assert float(franka.state["panda/joint1.q"][1]) == pytest.approx(q_other)


def test_leap_holds_cube_at_zero_action(leap):
    leap.reset(seed=0)
    start = leap.state["leap/cube.pos"].copy()
    for _ in range(20):
        leap.step(np.zeros((W, 16), np.float32))
    assert np.abs(leap.state["leap/cube.pos"] - start).max() < 0.01
    assert leap.sim.check()["ok"]
    assert set(leap.last_metrics) == {"rot_err", "goal_reached", "dropped"}


def test_leap_action_moves_hand_and_goal_resamples_per_world(leap):
    leap.reset(seed=0)
    g0 = leap.goal.value.copy()
    leap.reset_worlds(np.array([True, False]))
    assert not np.allclose(leap.goal.value[0], g0[0])
    assert np.allclose(leap.goal.value[1], g0[1])
    q0 = leap.state["leap/if_mcp.q"].copy()
    for _ in range(5):
        leap.step(np.ones((W, 16), np.float32))
    assert np.abs(leap.state["leap/if_mcp.q"] - q0).max() > 0.1


def test_quaternion_helpers():
    from alloy.envs.backend import pick_backend

    xp = pick_backend("numpy", "cpu")
    q = random_unit_quats(xp, 64)
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(angle_between(xp, q, q), 0.0, atol=1e-6)  # exact at 0 in float32 (2 acos(dot) read ~1e-3)
    half = np.float32(0.15)  # a 0.3 rad turn about x, and its sign-flipped twin (the same rotation)
    turn = np.tile(np.array([[np.cos(half), np.sin(half), 0.0, 0.0]], np.float32), (4, 1))
    flip = np.tile(np.array([[0.0, 1.0, 0.0, 0.0]], np.float32), (4, 1))
    ident = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (4, 1))
    assert np.allclose(angle_between(xp, flip, ident), np.pi, atol=1e-3)
    assert np.allclose(angle_between(xp, turn, ident), 0.3, atol=1e-5)
    assert np.allclose(angle_between(xp, -turn, ident), 0.3, atol=1e-5)
