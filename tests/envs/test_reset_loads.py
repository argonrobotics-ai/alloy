# SPDX-License-Identifier: Apache-2.0
"""An env's reset removes what ``sim.reset()`` removes: held wrenches and running drives (audit T2-015).

``VectorEnv.reset()`` restored the engine state but left a ``sim.apply_wrench`` load on (the next episode's block was
pushed 3 cm with no command) and a ``sim.drive`` running (the belt jumped ahead of its start). Now a reset world starts
with no wrench (the other worlds keep theirs), and a reset of every world also stops every drive, as ``sim.reset()``
does; a drive moves its object in every world, so it keeps running through a reset of some worlds.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import ActionSpec, Task, VectorEnv

ARM = """<mujoco><worldbody><body name="b" pos="0 0 0.05">
  <joint name="x" type="slide" axis="1 0 0" range="-0.2 0.2" damping="20"/>
  <geom type="box" size="0.02 0.02 0.02" mass="2.0"/></body></worldbody>
  <actuator><position joint="x" kp="2000" kv="100"/></actuator></mujoco>"""
PUSH = np.array([[25.0, 0.0, 0.0], [25.0, 0.0, 0.0]])


def _env() -> VectorEnv:
    sc = alloy.Scene(worlds=2, dt=0.01)
    sc.add_ground()
    sc.add_box("block", size=(0.04, 0.04, 0.04), pos=(0.15, 0.0, 0.02), mass=0.5, friction=0.05)
    sc.add_box("belt", size=(0.10, 0.10, 0.02), pos=(-0.20, 0.0, 0.01), kinematic=True)
    sc.add_robot("arm", mjcf=ARM, pos=(0.0, 0.20, 0.01))
    task = Task(obs=["arm", "block"], actions=ActionSpec(["arm/x"]), max_steps=30)
    return VectorEnv(sc, task, worlds=2, device="cpu", backend="numpy")


def _steps(env: VectorEnv, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(n):
        env.step(np.zeros((2, 1), np.float32))
    return np.asarray(env.state["block.pos"])[:, 0], np.asarray(env.state["belt.pos"])[:, 0]


@pytest.mark.timeout(300)
def test_a_reset_of_every_world_removes_wrenches_and_drives():
    env = _env()
    env.reset(seed=0)
    env.sim.apply_wrench("block", force=PUSH)
    env.sim.drive("belt", velocity=(0.5, 0.0, 0.0))
    block, belt = _steps(env)
    assert np.all(block > 0.17) and np.all(belt > -0.18)               # pushed and driven in episode 1
    env.reset(seed=0)
    block, belt = _steps(env)
    assert np.allclose(block, 0.15, atol=1e-3) and np.allclose(belt, -0.20, atol=1e-3)
    env.close()


@pytest.mark.timeout(300)
def test_a_reset_world_starts_with_no_wrench_and_the_others_keep_theirs():
    env = _env()
    env.reset(seed=0)
    env.sim.apply_wrench("block", force=PUSH)
    _steps(env, 2)
    env.reset(options={"reset_mask": np.array([True, False])})
    block, _ = _steps(env)
    assert abs(block[0] - 0.15) < 1e-3 and block[1] > 0.17            # world 1 still pushed, world 0 not
    env.close()
