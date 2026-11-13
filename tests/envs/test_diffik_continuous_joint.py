# SPDX-License-Identifier: Apache-2.0
"""Gemini audit T2-020: hand-motion actions (``EEActionSpec``) work on an arm with a continuous (unlimited) hinge.

The writer wrapped the joints in an absolute ``ActionSpec`` without bounds, so a joint with no range failed the build
with "joints without limits" and a fix (``low=``, ``high=``) that ``EEActionSpec`` does not take. Inverse kinematics
sets the targets, so a joint without limits now gets an open range; a limited joint keeps its range. CPU."""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs.diffik import EEActionSpec

ARM = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="turret" pos="0 0 0.2">
      <joint name="j1" type="hinge" axis="0 0 1" damping="2"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
      <body name="hand" pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2"/>
        <geom type="sphere" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" kp="100" kv="10"/>
    <position joint="j2" kp="100" kv="10"/>
  </actuator>
</mujoco>"""


@pytest.mark.compile_heavy
def test_ee_actions_build_and_move_an_arm_with_a_continuous_joint():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM)
    spec = EEActionSpec(body="arm/hand", joints=["arm/j1", "arm/j2"])
    env = VectorEnv(scene, Task(obs=["arm"], actions=spec), worlds=2, device="cpu", backend="numpy")   # was: EnvError
    try:
        env.reset(seed=0)
        inner = env.writer.inner
        assert np.isinf(inner.low_np[0]) and np.isinf(inner.high_np[0])          # j1: open both ways
        assert np.allclose([inner.low_np[1], inner.high_np[1]], [-1.5, 1.5])      # j2 keeps its range
        start = env.state["arm/hand.pos"].copy()
        for _ in range(10):                                                       # sideways: j1 must turn
            env.step(np.tile(np.array([0.0, 1.0, 0.0], np.float32), (2, 1)))
        assert float(np.abs(env.state["arm/hand.pos"][:, 1] - start[:, 1]).min()) > 1e-3
    finally:
        env.close()
