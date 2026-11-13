# SPDX-License-Identifier: Apache-2.0
"""Tactile sensors under the API (audit T2-006, S1-003): a TactileSensor as a VectorEnv camera with masked resets,
and the Tactile plug-in taking Scene names ("<robot>/<part>"), with raw labels resolving exactly as before."""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera.tactile import TactileSensor
from alloy.plugins.tactile import Tactile

SLIDER = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="blk" pos="0 0 0.0315"><joint name="sx" type="slide" axis="1 0 0" range="-0.01 0.01"/>
  <geom type="box" size="0.008 0.008 0.010" mass="0.1"/></body></worldbody>
  <actuator><position joint="sx" kp="200" kv="20"/></actuator></mujoco>"""
FINGER = """<mujoco model="gripper_unit"><worldbody><body name="finger" pos="0 0 0.020">
  <joint name="pz" type="slide" axis="0 0 -1" range="0 0.05"/>
  <geom name="pad" type="box" size="0.02 0.02 0.005" mass="0.4"/></body></worldbody>
  <actuator><position joint="pz" kp="400" kv="20"/></actuator></mujoco>"""


def slider_sim(worlds: int = 2):
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("pad", size=(0.024, 0.024, 0.004), pos=(0.0, 0.0, 0.020), kinematic=True)
    scene.add_robot("slider", mjcf=SLIDER, pos=(0.0, 0.0, 0.0))
    return alloy.build(scene, device="cpu")


def test_tactile_sensor_is_a_vector_env_camera_with_masked_reset():
    """T2-006: Task(cameras={"gel": sensor}) raised "cannot sense a GpuScene"; a masked reset wiped every world."""
    from alloy.envs import Task, VectorEnv
    sim = slider_sim()
    sensor = TactileSensor("gel", attach_to="pad", pose=((0.0, 0.0, 0.002), (0.0, 0.0, 0.0, 1.0)), width=80)
    env = VectorEnv(sim, Task(obs=["slider"], cameras={"gel": sensor}, max_steps=50), backend="numpy")
    obs, _ = env.reset(seed=0)
    assert obs["gel"].shape == (2, 60, 80, 3) and obs["gel"].dtype == np.uint8
    for _ in range(3):
        obs, *_ = env.step(np.array([[0.004], [0.004]], np.float32))
    assert obs["gel"].shape == (2, 60, 80, 3)
    ss = sensor._bound[2]
    ss.anchor_body.fill_(7)  # stand-ins for two ongoing touches
    env.reset_worlds(np.array([True, False]))
    assert ss.anchor_body.numpy().tolist() == [-1, 7]  # only the reset world forgot its touch
    env.close()


def finger_scene(*robots: str, extra=None):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for k, name in enumerate(robots):
        scene.add_robot(name, mjcf=FINGER, pos=(0.2 * k, 0.0, 0.0))
    if extra:
        extra(scene)
    return scene


def test_tactile_plugin_takes_scene_names_of_two_robots_from_one_mjcf():
    """S1-003: "arm_left/finger" was rejected, and two robots from one MJCF clashed on the raw label."""
    tac = Tactile(Tactile.Config(bodies=("arm_left/finger", "arm_right/finger")))
    sim = alloy.build(finger_scene("arm_left", "arm_right"), device="cpu", plugins=[tac])
    want = [sim.index.body("arm_left/finger"), sim.index.body("arm_right/finger")]
    got = [int(tac.ctx.bodies(n).numpy()[0]) for n in ("arm_left/finger", "arm_right/finger")]
    assert got == want and want[0] != want[1]


def test_raw_labels_resolve_as_before():
    tac = Tactile(Tactile.Config(bodies=("gripper_unit/worldbody/finger",)))
    sim = alloy.build(finger_scene("arm_left"), device="cpu", plugins=[tac])
    raw = tac.ctx._raw_per_world("body", "gripper_unit/worldbody/finger").numpy().tolist()
    assert tac.ctx.bodies("gripper_unit/worldbody/finger").numpy().tolist() == raw == [sim.index.body("arm_left/finger")]
    assert tac.ctx.bodies("finger").numpy().tolist() == raw  # suffix match, as before
    with pytest.raises(alloy.BuildError, match="appears twice in world 0.*Scene name"):
        alloy.build(finger_scene("a", "b"), device="cpu",
                    plugins=[Tactile(Tactile.Config(bodies=("gripper_unit/worldbody/finger",)))])


def test_a_label_that_names_two_bodies_is_refused():
    """A Scene name ("hand": robot hand's root) that is also another body's raw label (".../hand") never picks."""
    other = FINGER.replace('model="gripper_unit"', 'model="o"').replace('name="finger"', 'name="hand"')
    scene = finger_scene("hand", extra=lambda s: s.add_robot("other", mjcf=other, pos=(0.4, 0.0, 0.0)))
    with pytest.raises(alloy.BuildError, match="Scene name of body .* raw label of"):
        alloy.build(scene, device="cpu", plugins=[Tactile(Tactile.Config(bodies=("hand",)))])
