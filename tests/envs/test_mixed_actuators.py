# SPDX-License-Identifier: Apache-2.0
"""One action can drive joints with MJCF ``<position>`` servos and MuJoCo ``<general>`` / ``<motor>`` actuators together
(audit T2-017).

``ActionSpec()`` picked one control kind for the whole scene: beside a Panda (``<general>`` servos) a gantry's
``<position>`` joints were the only columns, a ``<motor>`` wheel beside a ``<position>`` lift was dropped, naming
both kinds raised "no actuator named ...", and ``kind="position"`` on a Panda joint built and then never moved it.
Now "auto" (the default) gives each joint or actuator its own control, and a joint target on a ctrl-driven joint
raises with the fix.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.cards.library import default_root
from alloy.envs import ActionSpec, EnvError, Task, VectorEnv
from alloy.envs.diffik import EEActionSpec
from alloy.envs.mjcf import without_bodies
from alloy.envs.robots import gantry_gripper

BOT = """<mujoco><compiler angle="radian"/><worldbody><body name="chassis" pos="0 0 0.1">
  <joint name="wheel" type="hinge" axis="0 1 0" range="-10 10" damping="0.1"/>
  <geom type="cylinder" size="0.05 0.02" mass="1.0"/>
  <body name="arm" pos="0 0 0.1"><joint name="lift" type="slide" axis="0 0 1" range="0 0.2" damping="10"/>
    <geom type="box" size="0.02 0.02 0.02" mass="0.2"/></body></body></worldbody>
  <actuator><motor name="wheel_motor" joint="wheel" ctrlrange="-5 5"/>
    <position name="lift_servo" joint="lift" kp="400" kv="40"/></actuator></mujoco>"""
PANDA = [f"panda/actuator{k}" for k in range(1, 8)]


@pytest.fixture(scope="module")
def two_robots() -> alloy.Scene:
    panda = without_bodies(default_root() / "robots" / "emb_e2_panda" / "scene.xml", ["box", "cylinder", "mug"])
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_robot("panda", panda)
    scene.add_robot("helper", mjcf=gantry_gripper(), pos=(0.5, 0.0, 0.35))
    return scene


def _env(scene: alloy.Scene, actions=None) -> VectorEnv:
    return VectorEnv(scene, Task(obs=["panda", "helper"], actions=actions, max_steps=50), worlds=2, device="cpu",
                     backend="numpy")


def _q(env: VectorEnv, joint: str) -> float:
    return float(np.asarray(env.state[f"{joint}.q"])[0])


@pytest.mark.compile_heavy
def test_every_actuator_of_both_kinds_is_a_column(two_robots):
    env = _env(two_robots)
    names = env.action_names()
    env.close()
    assert len(names) == 13 and set(PANDA + ["panda/hand_finger", "helper/x"]) <= set(names)


@pytest.mark.compile_heavy
def test_named_columns_of_both_kinds_move_their_joints(two_robots):
    env = _env(two_robots, ActionSpec(["panda/actuator1", "helper/x"], mode="delta"))
    env.reset(seed=0)
    before = _q(env, "panda/joint1"), _q(env, "helper/x")
    for _ in range(10):
        env.step(np.ones((2, 2), np.float32))
    assert env.writer.kind == "mixed"
    assert _q(env, "panda/joint1") - before[0] > 0.05 and _q(env, "helper/x") - before[1] > 0.05
    env.close()


@pytest.mark.compile_heavy
def test_hand_motion_with_an_extra_of_the_other_kind(two_robots):
    spec = EEActionSpec(body="panda/hand", joints=PANDA, extra=["helper/x"])
    env = _env(two_robots, spec)
    env.reset(seed=0)
    z0, x0 = float(np.asarray(env.state["panda/hand.pos"])[0, 2]), _q(env, "helper/x")
    for _ in range(10):
        env.step(np.array([[0.0, 0.0, 1.0, 1.0]] * 2, np.float32))
    assert float(np.asarray(env.state["panda/hand.pos"])[0, 2]) - z0 > 0.01 and _q(env, "helper/x") - x0 > 0.05
    env.close()


@pytest.mark.compile_heavy
def test_a_joint_target_on_a_ctrl_driven_joint_raises_with_the_fix(two_robots):
    with pytest.raises(EnvError) as e:
        _env(two_robots, ActionSpec(["panda/joint1", "helper/x"], kind="position"))
    assert e.value.field == "kind" and "kind='auto'" in e.value.fix


@pytest.mark.timeout(300)
def test_a_motor_beside_a_position_servo_on_one_robot():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_robot("bot", mjcf=BOT)
    env = VectorEnv(scene, Task(obs=["bot"], max_steps=50), worlds=2, device="cpu", backend="numpy")
    assert env.action_names() == ["bot/lift", "bot/wheel_motor"]
    env.reset(seed=0)
    for _ in range(10):                                              # lift to its top, wheel at 0.25 N m
        _obs, _rew, term, trunc, info = env.step(np.array([[1.0, 0.05]] * 2, np.float32))
        assert not (term.any() or trunc.any() or info["invalid"].any())
    assert _q(env, "bot/lift") > 0.1 and abs(_q(env, "bot/wheel")) > 0.1
    env.close()
