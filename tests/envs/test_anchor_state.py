# SPDX-License-Identifier: Apache-2.0
"""An env reads an anchored object where the scene put it, as ``sim.state()`` does (audit T2-001).

``scene.add_object(name, card, pos=..., anchor="base")`` places the object's anchor point (here its base) at ``pos``:
``sim.state()["<name>.pos"]`` reads that point and ``"<name>.body_pos"`` the body frame's origin. The env's state,
observations and checks read the raw body origin instead: 25 mm above the base for the shipped ``aa_cell``, so a
height check said 0.075 m for an object standing at 0.05 m.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs import checks as C

ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="b" pos="0 0 0.05"><joint name="j" type="hinge" axis="0 0 1" range="-1 1"/>
  <geom type="sphere" size="0.02" mass="0.2"/></body></worldbody>
  <actuator><position joint="j" kp="50" kv="5"/></actuator></mujoco>"""


@pytest.fixture(scope="module")
def env():
    scene = alloy.Scene(worlds=2, dt=0.02)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(-0.3, 0.0, 0.0))
    scene.add_object("cell", "aa_cell", pos=(0.1, 0.0, 0.05), anchor="base")
    scene.add_object("tilted", "aa_cell", pos=(0.1, 0.2, 0.1), anchor="base", rpy=(0.6, 0.0, 0.0))
    e = VectorEnv(scene, Task(obs=["cell", "tilted"], max_steps=50), worlds=2, device="cpu", backend="numpy")
    yield e
    e.close()


@pytest.mark.compile_heavy
def test_pos_is_the_anchor_point_in_state_obs_and_checks(env):
    obs, _ = env.reset(seed=0)
    assert np.allclose(env.state["cell.pos"], [0.1, 0.0, 0.05], atol=1e-5)
    assert np.allclose(obs[:, :3], env.state["cell.pos"], atol=1e-6)
    assert np.allclose(C.height("cell")(env.state), 0.05, atol=1e-5)
    assert np.all(env.state["cell.body_pos"][:, 2] > 0.06)                 # the raw origin: 25 mm above the base


@pytest.mark.compile_heavy
def test_env_and_sim_read_the_same_points_as_objects_move_and_turn(env):
    env.reset(seed=0)
    for _ in range(10):                                                     # the tilted cell falls and turns
        env.step(np.zeros((2, env.writer.size), np.float32))
        sim = env.sim.state(["cell", "tilted"])
        for name in ("cell", "tilted"):
            assert np.allclose(env.state[f"{name}.pos"], sim[f"{name}.pos"], atol=1e-5)
            assert np.allclose(env.state[f"{name}.body_pos"], sim[f"{name}.body_pos"], atol=1e-5)
