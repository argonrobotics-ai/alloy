# SPDX-License-Identifier: Apache-2.0
"""PickCube-v0 on the CPU: build, reset, step, cube at rest, success and grasp checks."""

import numpy as np
import pytest

from alloy.envs.pick_cube import grasped, make_pick_cube, pick_cube_task, tcp

pytestmark = pytest.mark.filterwarnings("ignore")
WORLDS = 3


@pytest.fixture(scope="module")
def env():
    e = make_pick_cube(num_envs=WORLDS, device="cpu", seed=0)
    e.reset(seed=0)
    return e


def _np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)


def test_steps_with_finite_outputs(env):
    obs, _ = env.reset(seed=1)
    assert env.action_space.shape[-1] == 8
    for _ in range(5):
        obs, reward, *_ = env.step(env.action_space.sample())
        assert _np(obs).shape[0] == WORLDS
        assert np.isfinite(_np(obs)).all() and np.isfinite(_np(reward)).all()
        assert _np(reward).shape == (WORLDS,)


def test_cube_rests_and_tcp_starts_above_it(env):
    env.reset(seed=2)
    z0 = _np(env.state["cube.pos"])[:, 2]
    for _ in range(25):                                  # 0.5 s at dt 0.02, zero (hold) actions
        env.step(np.zeros((WORLDS, 8), dtype=np.float32))
    z1 = _np(env.state["cube.pos"])[:, 2]
    assert np.abs(z1 - z0).max() < 1e-3
    assert np.allclose(z1, 0.02, atol=2e-3)
    point, cube = _np(tcp(env.state)), _np(env.state["cube.pos"])
    assert (point[:, 2] > cube[:, 2] + 0.05).all()
    assert (np.linalg.norm(point[:, :2] - cube[:, :2], axis=1) < 0.3).all()


def test_success_fires_on_goal_only(env):
    env.reset(seed=3)
    success = pick_cube_task().success
    assert not _np(success(env.state)).any()
    goal = _np(env.state["goal.pos"])
    for w in range(WORLDS):
        env.sim.set_pose("cube", pos=goal[w], world=w)
    assert _np(success(env.state)).all()


def test_not_grasped_at_start(env):
    env.reset(seed=4)
    assert not _np(grasped(env.state)).any()


def _scripted_env(worlds: int = 4):
    """PickCube with task-space hand actions (dx, dy, dz + the gripper), the hand held level: a scripted pick."""
    import dataclasses

    from alloy.envs.diffik import EEActionSpec
    from alloy.envs.pick_cube import ARM, GRIPPER, pick_cube_scene
    from alloy.envs.vector import VectorEnv

    spec = EEActionSpec(body="panda/hand", joints=ARM, extra=[GRIPPER], extra_delta=0.04, delta=0.02, orientation=True)
    task = dataclasses.replace(pick_cube_task(), actions=spec)
    return VectorEnv(pick_cube_scene(worlds), task, worlds=worlds, device="cpu", seed=0, backend="numpy",
                     autoreset="disabled")


def _drive(env, target, grip: float, steps: int) -> None:
    for _ in range(steps):
        s = env.state
        a = np.zeros((env.num_envs, env.writer.size), np.float32)
        a[:, :3] = np.clip((target(s) - tcp(s)) / 0.02, -1.0, 1.0)
        a[:, -1] = grip
        env.step(a)


@pytest.mark.compile_heavy
def test_a_scripted_pick_carries_the_cube_to_its_goal():
    """Open, go 6 cm above the cube, descend, close, carry to the goal: the cube counts as held only once it is off
    the table, success fires at the goal, and no world goes INVALID."""
    from alloy.envs.pick_cube import enclosed

    env = _scripted_env()
    env.reset(seed=0)
    above = lambda s: s["cube.pos"] + np.array([0.0, 0.0, 0.06], np.float32)  # noqa: E731
    _drive(env, tcp, 1.0, 8)
    _drive(env, above, 1.0, 60)
    _drive(env, lambda s: s["cube.pos"], 1.0, 40)
    _drive(env, tcp, -1.0, 10)
    s = env.state
    assert not _np(grasped(s)).any()                                   # closed on the cube, still on the table
    _drive(env, lambda s: s["goal.pos"], -1.0, 75)
    s = env.state
    done = _np(pick_cube_task().success(s)).astype(bool)
    assert done.sum() >= 3 and _np(grasped(s))[done].all() and _np(enclosed(s))[done].all()   # 4 of 4 on main
    assert env.sim.health()["invalid_worlds"] == []
    env.close()
