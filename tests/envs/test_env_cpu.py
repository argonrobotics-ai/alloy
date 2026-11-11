# SPDX-License-Identifier: Apache-2.0
"""The vector environment end to end on CPU (NumPy backend): spaces, a scripted grasp to success with same-step
autoreset, reset by world mask, action modes and shape errors, and recovery of a world the engine froze."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import ActionSpec, EnvError, Task, VectorEnv, make
from alloy.envs import checks as C
from alloy.envs.tasks import gripper_lift_scene

W = 3
SQUEEZE, LIFT = np.array([0.0, 1.0, 1.0], np.float32), np.array([1.0, 1.0, 1.0], np.float32)


@pytest.fixture(scope="module")
def env():
    e = make("GripperLift-v0", num_envs=W, device="cpu", backend="numpy")
    yield e
    e.close()


def _grasp(env: VectorEnv, steps: int, worlds: int = W) -> tuple:
    out = None
    for k in range(steps):
        out = env.step(np.tile(SQUEEZE if k < 25 else LIFT, (worlds, 1)))
    return out


@pytest.mark.compile_heavy
def test_spaces_and_names(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (W, 22) and obs.dtype == np.float32 and info == {}
    assert env.single_observation_space.shape == (22,) and env.observation_space.shape == (W, 22)
    assert env.single_action_space.shape == (3,) and env.action_space.shape == (W, 3)
    assert env.action_names() == ["gripper/lift", "gripper/close_left", "gripper/close_right"]
    assert env.obs_names()[0] == "gripper/lift.q" and len(env.obs_names()) == 22
    assert str(env.metadata["autoreset_mode"]).lower().endswith("same_step")   # Gymnasium's enum or the string


@pytest.mark.compile_heavy
def test_grasp_success_autoresets_same_step(env):
    env.task.end_on_success = True            # GripperLift-v0 holds to the end; here success ends the episode
    try:
        _success_ends_episodes(env)
    finally:
        env.task.end_on_success = False


def _success_ends_episodes(env: VectorEnv) -> None:
    env.reset()
    obs, r, term, trunc, info = _grasp(env, 40)
    s = env.state
    assert np.all(s["steps"] == 40) and np.all(C.rise("cube")(s) > 0.03) and info["terms"]["held"].all()
    assert np.all(r > 0.5) and not term.any() and not trunc.any()
    ended = np.zeros(W, bool)
    for k in range(40, 90):                                    # each world lifts past 8 cm and restarts
        obs, r, term, trunc, info = env.step(np.tile(LIFT, (W, 1)))
        if "_episode" not in info:
            continue
        m = info["_episode"] & ~ended
        assert term[m].all() and info["success"][m].all() and info["episode"]["success"][m].all()
        assert info["_final_obs"][m].all() and np.all(info["final_obs"][m, 8] > 0.1)   # cube z before the reset
        assert np.all(info["episode"]["l"][m] == k + 1) and np.all(obs[m, 8] < 0.031)  # obs after the reset
        assert np.all(env.state["steps"][m] == 0) and np.allclose(env.state["gripper/lift.q"][m], 0.0, atol=1e-6)
        ended |= m
        if ended.all():
            break
    assert ended.all()


@pytest.mark.compile_heavy
def test_success_without_ending(env):
    env.reset()
    obs, r, term, trunc, info = _grasp(env, 70)
    assert info["success"].all() and not term.any() and np.all(env.state["steps"] == 70)


@pytest.mark.compile_heavy
def test_reset_mask_only_touches_masked_worlds(env):
    env.reset()
    _grasp(env, 40)
    before = env.state["cube.pos"].copy()
    obs, info = env.reset(options={"reset_mask": np.array([False, True, False])})
    s = env.state
    assert s["steps"].tolist() == [40, 0, 40]
    assert np.allclose(s["cube.pos"][[0, 2]], before[[0, 2]]) and abs(s["cube.pos"][1, 2] - 0.03) < 1e-4
    assert np.allclose(s.start["cube.pos"][:, 2], 0.03, atol=1e-4)
    targets = env.sim.engine.control.joint_target_q.numpy()
    rows = env.layout.joint("gripper/lift").target_q[:, 0]
    assert targets[rows[1]] == 0.0 and targets[rows[0]] > 0.05            # world 1's lift target back to its start
    with pytest.raises(EnvError, match="reset mask"):
        env.reset(options={"reset_mask": np.array([True, False])})


@pytest.mark.compile_heavy
def test_action_errors(env):
    env.reset()
    with pytest.raises(EnvError, match=r"\(3, 3\)"):
        env.step(np.zeros((W, 2)))


@pytest.mark.compile_heavy
def test_frozen_world_is_truncated_and_recovers(env):
    """A world the engine marks INVALID is truncated, reset, and steps normally again (health record cleared)."""
    env.reset()
    _grasp(env, 3)
    eng = env.sim.engine
    qd = eng.state.joint_qd.numpy()
    qd[env.layout.world_of["qd"].tolist().index(2) + 2] = 500.0           # world 2's cube: 500 m/s up
    eng.state.joint_qd.assign(qd)
    obs, r, term, trunc, info = env.step(np.tile(SQUEEZE, (W, 1)))
    assert info["invalid"].tolist() == [False, False, True] and trunc.tolist() == [False, False, True]
    assert env.state["steps"].tolist() == [4, 4, 0]
    _grasp(env, 30)
    assert env.sim.health()["invalid_worlds"] == []
    assert C.rise("cube")(env.state)[2] > 0.0                              # world 2 lifts again: not frozen


@pytest.mark.compile_heavy
def test_absolute_actions_and_next_step_autoreset():
    task = Task(obs=["cube", C.height("cube")], max_steps=3,
                actions=ActionSpec(["gripper/close_left", "gripper/close_right"], mode="absolute"))
    env = VectorEnv(gripper_lift_scene(2), task, worlds=2, device="cpu", backend="numpy", autoreset="next_step")
    try:
        assert env.single_action_space.high.tolist() == pytest.approx([0.05, 0.05])
        env.reset()
        for _ in range(3):
            obs, r, term, trunc, info = env.step(np.full((2, 2), 0.2))      # clipped to the 5 cm limit
        assert np.allclose(info["targets"], 0.05) and trunc.all() and "final_obs" not in info
        obs, r, term, trunc, info = env.step(np.full((2, 2), 0.01))         # next_step: this step resets
        assert env.state["steps"].tolist() == [0, 0] and np.all(r == 0) and not trunc.any()
    finally:
        env.close()
