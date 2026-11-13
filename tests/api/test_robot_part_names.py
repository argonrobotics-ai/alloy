# SPDX-License-Identifier: Apache-2.0
"""Every robot part keeps its own name: "<robot>/<name in its file>", also when that name holds "/" and when two
robots share a file.

Laws (CPU): a two-arm file whose parts are named "left/..." and "right/..." (as MuJoCo Menagerie's two-arm files are)
keeps both arms' bodies, joints and actuators, and a target moves only the joint it names; a tendon actuator belongs
to its robot ("<robot>/<actuator>"); an actuator name two robots share is no bare name: it raises, naming both.
Before, the name was cut at its last "/" ("left/waist" and "right/waist" were one "waist", one arm unreachable), a
tendon actuator had no robot, and the bare name drove the first robot alone.
"""

from __future__ import annotations

import pytest

import alloy.api as alloy

TWO_ARMS = """<mujoco model="two_arms"><compiler angle="radian"/><worldbody>
  <body name="left/base" pos="0 0.25 0.15"><joint name="left/waist" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
    <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025" mass="0.5"/></body>
  <body name="right/base" pos="0 -0.25 0.15"><joint name="right/waist" type="hinge" axis="0 0 1" range="-2 2"
    damping="1"/><geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025" mass="0.5"/></body></worldbody>
  <actuator><position name="left/waist" joint="left/waist" kp="80" kv="8"/>
    <position name="right/waist" joint="right/waist" kp="80" kv="8"/></actuator></mujoco>"""

TENDON_HAND = """<mujoco model="hand"><compiler angle="radian"/><worldbody><body name="palm" pos="0 0 0.15">
  <geom type="box" size="0.04 0.04 0.02" mass="0.3"/>
  <body name="f1" pos="-0.03 0 0.03"><joint name="j1" type="slide" axis="1 0 0" range="0 0.04" damping="1"/>
    <geom type="box" size="0.008 0.015 0.025" mass="0.05"/></body>
  <body name="f2" pos="0.03 0 0.03"><joint name="j2" type="slide" axis="-1 0 0" range="0 0.04" damping="1"/>
    <geom type="box" size="0.008 0.015 0.025" mass="0.05"/></body></body></worldbody>
  <tendon><fixed name="split"><joint joint="j1" coef="0.5"/><joint joint="j2" coef="0.5"/></fixed></tendon>
  <actuator><motor name="fingers" tendon="split" ctrlrange="-20 20" gear="10"/></actuator></mujoco>"""

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><position name="act_sh" joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _build(robots: dict[str, tuple[str, tuple[float, float, float]]]) -> alloy.Sim:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for name, (mjcf, pos) in robots.items():
        scene.add_robot(name, mjcf=mjcf, pos=pos)
    return alloy.build(scene, device="cpu")


@pytest.mark.compile_heavy
def test_a_name_with_a_slash_keeps_both_arms():
    sim = _build({"rob": (TWO_ARMS, (0, 0, 0))})
    names = sim.names()
    assert {"rob/left/waist", "rob/right/waist", "left/waist", "right/waist"} <= set(names["joints"])
    assert {"rob/left/base", "rob/right/base"} <= set(names["bodies"])
    assert {"rob/left/waist", "rob/right/waist"} <= set(names["actuators"])   # was ['rob/waist', 'waist']
    sim.set_targets({"rob/left/waist": 0.6})
    sim.step(100)
    st = sim.state(world=0)
    assert st["rob/left/waist.q"] == pytest.approx(0.6, abs=0.01)
    assert abs(float(st["rob/right/waist.q"])) < 1e-6                        # the other arm stays
    sim.set_controls({"rob/right/waist": -0.4})
    sim.step(100)
    assert sim.state(world=0)["rob/right/waist.q"] == pytest.approx(-0.4, abs=0.01)
    sim.close()


@pytest.mark.compile_heavy
def test_a_tendon_actuator_belongs_to_its_robot():
    sim = _build({"left": (TENDON_HAND, (0, 0.25, 0)), "right": (TENDON_HAND, (0, -0.25, 0))})
    acts = sim.names()["actuators"]
    assert {"left/fingers", "right/fingers"} <= set(acts) and "fingers" not in acts      # was ['fingers']
    sim.set_controls({"left/fingers": 15.0})
    sim.run(0.5, record=[])
    st = sim.state(world=0)
    assert float(st["left/j1.q"]) > 0.01 and abs(float(st["right/j1.q"])) < 1e-6
    sim.close()


@pytest.mark.compile_heavy
def test_an_actuator_name_two_robots_share_names_both():
    sim = _build({"left": (ARM, (-0.5, 0, 0.5)), "right": (ARM, (0.5, 0, 0.5))})
    assert "act_sh" not in sim.names()["actuators"]                       # was bound to the left arm alone
    with pytest.raises(alloy.UnknownNameError) as err:
        sim.set_controls({"act_sh": 0.8})
    assert "left/act_sh" in err.value.fix and "right/act_sh" in err.value.fix
    sim.close()
    one = _build({"arm": (ARM, (0, 0, 0.5))})                             # one robot: the bare name still works
    one.set_controls({"act_sh": 0.5})
    one.step(100)
    assert one.state(world=0)["arm/shoulder.q"] == pytest.approx(0.5, abs=0.02)
    one.close()
