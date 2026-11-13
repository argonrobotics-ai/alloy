# SPDX-License-Identifier: Apache-2.0
"""What ``Task(on_build=...)`` does to the sim is the start every reset returns to (audit T2-012).

``on_build`` runs once after the build; its edits through the public Sim methods (``sim.set_pose``, ...) were kept
only when the task also settled (``settle > 0``): with the default ``settle=0`` the first ``env.reset()`` put a box
moved by ``sim.set_pose`` back where the scene had it.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import ActionSpec, Task, VectorEnv

SLIDER = """<mujoco><worldbody><body name="cart">
  <joint name="x" type="slide" axis="1 0 0" range="-0.5 0.5" damping="20"/>
  <geom type="box" size="0.02 0.02 0.02" mass="0.5"/></body></worldbody>
  <actuator><position joint="x" kp="2000" kv="100"/></actuator></mujoco>"""
MOVED = (0.25, 0.10, 0.02)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("settle", [0.0, 0.1])
def test_on_build_edits_survive_every_reset(settle):
    sc = alloy.Scene(dt=0.02)
    sc.add_ground()
    sc.add_robot("slider", mjcf=SLIDER, pos=(-0.4, 0.0, 0.1))
    sc.add_box("box", size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.15), mass=0.2)
    task = Task(obs=["box"], actions=ActionSpec(["slider/x"], mode="delta"), max_steps=20, settle=settle,
                on_build=lambda sim: sim.set_pose("box", pos=MOVED))
    env = VectorEnv(sc, task, worlds=2, device="cpu", backend="numpy")
    for _ in range(2):
        env.reset(seed=0)
        assert np.allclose(env.state["box.pos"][:, :2], MOVED[:2], atol=2e-3)
        assert np.all(env.state["box.pos"][:, 2] < 0.05)          # where on_build put it (resting), not 0.15 up
        env.step(np.zeros((2, 1), np.float32))
    env.close()
