# SPDX-License-Identifier: Apache-2.0
"""A jitter that moves a container its liquid starts in, and leaves the liquid behind, warns at build.

``jitter_joint`` on the wrist that holds a jug turned the jug at each reset and left its water where it was, in mid-air:
14 % spilled before the first action, with no warning. ``carry=`` moves a liquid with a slide; a hinge cannot carry it.
"""
from __future__ import annotations

import logging

import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs.randomize import jitter_joint

WALL = '<geom type="box" size="{} {} 0.03" pos="{} {} 0.03" mass="0.02"/>'
JUG = f"""<mujoco><compiler angle="radian"/><worldbody>
  <body name="carriage" pos="0 0 0.05"><joint name="x" type="slide" axis="1 0 0" range="-0.2 0.2"/>
    <geom type="box" size="0.02 0.02 0.02" mass="1"/>
    <body name="hand" pos="0 0 0.05"><joint name="wrist" type="hinge" axis="1 0 0" range="-2 2"/>
      <geom type="box" size="0.04 0.04 0.003" mass="0.05"/>
      {WALL.format(0.003, 0.04, 0.04, 0)}{WALL.format(0.003, 0.04, -0.04, 0)}
      {WALL.format(0.04, 0.003, 0, 0.04)}{WALL.format(0.04, 0.003, 0, -0.04)}
    </body></body>
  <body name="other" pos="0.5 0 0.05"><joint name="z" type="slide" axis="0 0 1" range="-0.1 0.1"/>
    <geom type="box" size="0.02 0.02 0.02" mass="1"/></body></worldbody>
  <actuator><position joint="x" kp="2000" kv="150"/><position joint="wrist" kp="50" kv="5"/>
    <position joint="z" kp="2000" kv="150"/></actuator></mujoco>"""


@pytest.fixture(scope="module")
def scene() -> alloy.Scene:
    s = alloy.Scene(worlds=1, dt=0.01)
    s.add_ground()
    s.add_robot("pourer", mjcf=JUG)
    s.add_liquid("water", size=(0.06, 0.06, 0.03), pos=(0.0, 0.0, 0.122))       # in the jug, on its floor
    return s


def _warnings(scene: alloy.Scene, caplog, *jitters) -> list[str]:  # noqa: ANN001
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        env = VectorEnv(scene, Task(obs=["pourer"], randomize=list(jitters), max_steps=5), worlds=1, device="cpu",
                        backend="numpy")
        env.close()
    return [r.getMessage() for r in caplog.records if "jitter_joint" in r.getMessage()]


@pytest.mark.timeout(600)
def test_a_wrist_jitter_that_leaves_the_water_in_mid_air_warns(scene, caplog):
    said = _warnings(scene, caplog, jitter_joint("pourer/wrist", by=0.3))
    assert len(said) == 1 and "'water'" in said[0] and "cannot carry" in said[0], said


@pytest.mark.timeout(600)
def test_a_slide_jitter_names_carry_and_carrying_it_is_quiet(scene, caplog):
    said = _warnings(scene, caplog, jitter_joint("pourer/x", by=0.02))
    assert len(said) == 1 and "carry=['water']" in said[0], said
    assert not _warnings(scene, caplog, jitter_joint("pourer/x", by=0.02, carry=["water"]))


@pytest.mark.timeout(600)
def test_a_jitter_away_from_the_liquid_is_quiet(scene, caplog):
    assert not _warnings(scene, caplog, jitter_joint("pourer/z", by=0.05))
