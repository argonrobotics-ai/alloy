# SPDX-License-Identifier: Apache-2.0
"""Random starts on CPU: jittered objects and joints differ per world and per reset, stay where they were put, are
reproducible from the seed, and a masked reset leaves the other worlds alone."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import EnvError, VectorEnv, jitter_body, jitter_joint
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task


def _yaw_deg(q_xyzw: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arctan2(q_xyzw[:, 2], q_xyzw[:, 3]))


@pytest.fixture(scope="module")
def env():
    task = gripper_lift_task()
    task.randomize = [jitter_body("cube", pos=(0.02, 0.0, 0.0), yaw_deg=30), jitter_joint("gripper/lift", by=0.02)]
    e = VectorEnv(gripper_lift_scene(4), task, worlds=4, device="cpu", backend="numpy")
    yield e
    e.close()


@pytest.mark.compile_heavy
def test_jitter_ranges_hold_and_seed(env):
    env.reset(seed=3)
    s = env.state
    x, yaw, lift = s["cube.pos"][:, 0].copy(), _yaw_deg(s["cube.quat_xyzw"]), s["gripper/lift.q"].copy()
    assert np.all(np.abs(x) <= 0.02) and np.ptp(x) > 1e-3 and np.all(np.abs(yaw) <= 30.0) and np.ptp(yaw) > 1.0
    assert np.all(np.abs(lift) <= 0.02) and np.ptp(lift) > 1e-3
    assert np.allclose(s.start["cube.pos"], s["cube.pos"])                        # rise() counts from here
    targets = env.sim.engine.control.joint_target_q.numpy()[env.layout.joint("gripper/lift").target_q[:, 0]]
    assert np.allclose(targets, lift)                                             # the servo holds the new start
    for _ in range(10):
        env.step(np.zeros((4, 3), np.float32))
    s = env.state
    assert np.allclose(s["cube.pos"][:, 0], x, atol=1e-3) and np.allclose(_yaw_deg(s["cube.quat_xyzw"]), yaw, atol=0.5)
    env.reset(seed=3)
    assert np.allclose(env.state["cube.pos"][:, 0], x) and np.allclose(env.state["gripper/lift.q"], lift)
    env.reset(seed=4)
    assert not np.allclose(env.state["cube.pos"][:, 0], x)


@pytest.mark.compile_heavy
def test_masked_reset_jitters_only_masked(env):
    env.reset(seed=0)
    before = env.state["cube.pos"].copy()
    env.reset(options={"reset_mask": np.array([True, False, False, True])})
    after = env.state["cube.pos"]
    assert np.allclose(after[[1, 2]], before[[1, 2]]) and not np.allclose(after[[0, 3], 0], before[[0, 3], 0])


def test_bad_jitters(env):
    with pytest.raises(EnvError, match="half-ranges"):
        jitter_body("cube", pos=(-0.1, 0, 0))
    with pytest.raises(EnvError, match="by >= 0"):
        jitter_joint("gripper/lift", by=-1)
    from alloy.envs.randomize import Randomizer

    with pytest.raises(EnvError, match="not a free object"):
        Randomizer([jitter_body("gripper/palm", pos=(0.01, 0, 0))], env.layout, env.sim.engine, env.xp)


@pytest.mark.compile_heavy
def test_masked_jitter_keeps_other_worlds_targets(env):
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.tile([1.0, 0.0, 0.0], (4, 1)).astype(np.float32))     # raise every lift target
    rows = env.layout.joint("gripper/lift").target_q[:, 0]
    before = env.sim.engine.control.joint_target_q.numpy()[rows].copy()
    env.reset(options={"reset_mask": np.array([True, False, False, False])})
    after = env.sim.engine.control.joint_target_q.numpy()[rows]
    assert np.allclose(after[1:], before[1:]) and abs(after[0] - env.state["gripper/lift.q"][0]) < 1e-6
