# SPDX-License-Identifier: Apache-2.0
"""A world fork copies every control array the action writer drives (audit T2-017 follow-up).

``WorldOps`` copied the writer's one ``target`` array, read through ``writer.kind``: a writer that drives MJCF
``<position>`` targets and MuJoCo ``<motor>`` controls together (kind "mixed") has no single array, so a fork copied
neither and the forked world kept its own commands; a hand-motion writer (``EEWriter``) has no ``target`` at all.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.live.worldops import WorldOps

BOT = """<mujoco><compiler angle="radian"/><worldbody><body name="chassis" pos="0 0 0.1">
  <joint name="wheel" type="hinge" axis="0 1 0" range="-10 10" damping="0.1"/>
  <geom type="cylinder" size="0.05 0.02" mass="1.0"/>
  <body name="arm" pos="0 0 0.1"><joint name="lift" type="slide" axis="0 0 1" range="0 0.2" damping="10"/>
    <geom type="box" size="0.02 0.02 0.02" mass="0.2"/></body></body></worldbody>
  <actuator><motor name="wheel_motor" joint="wheel" ctrlrange="-5 5"/>
    <position name="lift_servo" joint="lift" kp="400" kv="40"/></actuator></mujoco>"""


@pytest.fixture(scope="module")
def env():
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_robot("bot", mjcf=BOT)
    e = VectorEnv(scene, Task(obs=["bot"], max_steps=500), worlds=3, device="cpu", backend="numpy")
    e.reset(seed=0)
    yield e
    e.close()


def _commands(env: VectorEnv, world: int) -> dict[str, np.ndarray]:
    """The world's rows of each control array the writer drives, read from the engine."""
    out = {}
    for kind, target, _rows, _cols in env.writer.groups:
        rows, ids = (np.asarray(x) for x in env.writer._world[kind])
        out[kind] = target.numpy()[rows[ids == world]].copy()
    return out


def _forked(env: VectorEnv, writer) -> tuple[dict, dict]:  # noqa: ANN001
    env.step(np.array([[0.05, 0.9], [-0.05, 0.1], [0.0, 0.5]], np.float32))       # wheel ctrl, lift target
    src = _commands(env, 0)
    assert set(src) == {"ctrl", "position"}
    assert not any(np.array_equal(v, _commands(env, 1)[k]) for k, v in src.items())
    real, env.writer = env.writer, writer
    try:
        WorldOps(env).fork(0, 1)
    finally:
        env.writer = real
    return src, _commands(env, 1)


@pytest.mark.timeout(300)
def test_a_fork_copies_targets_and_controls_of_a_mixed_writer(env):
    assert env.writer.kind == "mixed"
    env._frozen[1] = True                                                         # world 1 went INVALID earlier
    src, dst = _forked(env, env.writer)
    assert all(np.array_equal(src[k], dst[k]) for k in src)                       # both commands are world 0's
    assert not env._frozen[1]                                                     # health was reset with the copy


@pytest.mark.timeout(300)
def test_a_hand_motion_writer_forks_through_its_joint_writer(env):
    w = env.writer
    ee = SimpleNamespace(inner=w, kind=w.kind, _world=w._world, joints=w.joints)  # an EEWriter: no .target
    src, dst = _forked(env, ee)
    assert all(np.array_equal(src[k], dst[k]) for k in src)
