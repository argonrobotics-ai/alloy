# SPDX-License-Identifier: Apache-2.0
"""``jitter_joint(slide, carry=[...])`` moves what the slide carries along the slide as each world has it at that reset
(audit T2-018).

The direction was read once, from world 0 at the build pose, so a slide on a turret that an earlier jitter (or a
per-world start) had turned moved its peg the wrong way: a turret at 90 degrees put the slider at y = 0.15 and the peg
at x = 0.15, 21 cm apart.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import alloy
from alloy.envs import ActionSpec, Task, VectorEnv
from alloy.envs import randomize as R

TURRET = """<mujoco><compiler angle="radian"/><worldbody><body name="turret" pos="0 0 0.20">
  <joint name="yaw" type="hinge" axis="0 0 1" range="-3.1416 3.1416" damping="5"/>
  <geom type="cylinder" size="0.03 0.02" mass="0.5"/>
  <body name="slider" pos="0 0 0.05"><joint name="reach" type="slide" axis="1 0 0" range="-0.25 0.25" damping="10"/>
    <geom type="box" size="0.022 0.022 0.022" mass="0.2" contype="0" conaffinity="0"/></body></body></worldbody>
  <actuator><position joint="yaw" kp="200" kv="20"/><position joint="reach" kp="500" kv="50"/></actuator></mujoco>"""


def _gap(yaw: R.Jitter) -> np.ndarray:
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=TURRET)
    scene.add_box("peg", size=(0.03, 0.03, 0.03), pos=(0.0, 0.0, 0.25), mass=0.05)   # centred in the slider
    task = Task(obs=["arm", "peg"], actions=ActionSpec(["arm/yaw", "arm/reach"]),
                randomize=[yaw, R.jitter_joint("arm/reach", at=0.15, carry=["peg"])])
    env = VectorEnv(scene, task, worlds=3, device="cpu", backend="numpy")
    gaps = []
    for seed in (0, 1):
        env.reset(seed=seed)
        slider, peg = np.asarray(env.state["arm/slider.pos"]), np.asarray(env.state["peg.pos"])
        assert np.allclose(np.linalg.norm(slider[:, :2], axis=1), 0.15, atol=1e-4)
        gaps.append(np.linalg.norm(slider - peg, axis=1))
    env.close()
    return np.concatenate(gaps)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("yaw", [R.jitter_joint("arm/yaw", at=math.pi / 2), R.jitter_joint("arm/yaw", by=1.2)],
                         ids=["turned", "random"])
def test_the_peg_stays_in_the_slider_of_a_turned_turret(yaw):
    assert np.all(_gap(yaw) < 1e-4)
