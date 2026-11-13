# SPDX-License-Identifier: Apache-2.0
"""Each joint coordinate takes its own target: a body with several MJCF joints, and a fixed joint, which has none.

A body with several MJCF joints loads as one joint ("<a>_<b>") of one coordinate per MJCF joint. Laws (CPU, against
stock MuJoCo): a control on one of its actuators moves that coordinate, as MuJoCo moves it, for a two-hinge wrist and
a three-axis planar base; a target for the joint is one value per coordinate, each to its own servo (``<general>``
servos too), and a single number raises naming the count. A fixed joint has no coordinate: a target on it raises, and
the other targets of the call are not written. Before, every actuator of such a body drove its first coordinate (the
wrist tilted about the wrong axis, the base drove x when told y), only the last ``<general>`` servo of the body was
reachable, and a fixed joint's target landed on the next joint's coordinate.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import alloy.api as alloy

WRIST = """<mujoco model="wrist"><compiler angle="radian"/><worldbody><body name="wrist" pos="0 0 0.32">
  <joint name="wx" type="hinge" axis="1 0 0" range="-1.5 1.5" damping="1"/>
  <joint name="wy" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="1"/>
  <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.025" mass="1"/></body></worldbody>
  <actuator>{acts}</actuator></mujoco>"""
POSITION = '<position name="awx" joint="wx" kp="80" kv="8"/><position name="awy" joint="wy" kp="80" kv="8"/>'
GENERAL = ('<general name="gwx" joint="wx" biastype="affine" gainprm="80" biasprm="0 -80 -8"/>'
           '<general name="gwy" joint="wy" biastype="affine" gainprm="80" biasprm="0 -80 -8"/>')
BASE = """<mujoco model="base"><compiler angle="radian"/><worldbody><body name="base" pos="0 0 0.08">
  <joint name="jx" type="slide" axis="1 0 0" range="-2 2" damping="4"/>
  <joint name="jy" type="slide" axis="0 1 0" range="-2 2" damping="4"/>
  <joint name="jyaw" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1"/>
  <geom type="box" size="0.12 0.08 0.04" mass="4"/></body></worldbody>
  <actuator><position name="ax" joint="jx" kp="120" kv="20"/><position name="ay" joint="jy" kp="120" kv="20"/>
  <position name="ayaw" joint="jyaw" kp="60" kv="8"/></actuator></mujoco>"""
FIXED = """<mujoco model="arm"><compiler angle="radian"/><worldbody>
  <body name="base" pos="0 0 0.2"><geom type="box" size="0.05 0.05 0.05" mass="2"/>
    <body name="link1" pos="0 0 0.15"><joint name="j1" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      <geom type="box" size="0.03 0.03 0.03" mass="0.5"/></body></body></worldbody>
  <actuator><position name="a1" joint="j1" kp="100" kv="10"/></actuator></mujoco>"""


def _mujoco(xml: str, ctrl: dict[str, float], seconds: float) -> np.ndarray:
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for name, value in ctrl.items():
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = value
    mujoco.mj_step(m, d, nstep=int(round(seconds / m.opt.timestep)))
    return d.qpos.copy()


def _sim(xml: str) -> alloy.Sim:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("r", mjcf=xml)
    return alloy.build(scene, device="cpu")


@pytest.mark.compile_heavy
def test_a_control_moves_its_own_coordinate_as_mujoco_does():
    for xml, ctrl, joint in ((WRIST.format(acts=POSITION), {"awy": 0.6}, "r/wx_wy"),
                             (BASE, {"ay": 0.4}, "r/jx_jy_jyaw")):
        sim = _sim(xml)
        sim.set_controls({f"r/{k}": v for k, v in ctrl.items()})
        sim.run(1.0, record=[])
        q = np.asarray(sim.state(world=0)[f"{joint}.q"], np.float64)
        assert np.allclose(q, _mujoco(xml, ctrl, 1.0), atol=0.01), (joint, q)   # was the first coordinate moving
        sim.close()


@pytest.mark.compile_heavy
def test_a_target_takes_one_value_per_coordinate():
    for acts in (POSITION, GENERAL):
        sim = _sim(WRIST.format(acts=acts))
        with pytest.raises(alloy.SimulationError) as err:
            sim.set_targets({"r/wx_wy": 0.6})
        assert "2 values" in err.value.fix
        sim.set_targets({"r/wx_wy": [0.3, -0.4]})
        sim.run(1.5, record=[])
        q = np.asarray(sim.state(world=0)["r/wx_wy.q"], np.float64)
        assert np.allclose(q, [0.3, -0.4], atol=0.01), (acts[:9], q)          # <general>: wx was unreachable
        sim.close()


@pytest.mark.compile_heavy
def test_a_fixed_joint_takes_no_target():
    sim = _sim(FIXED)
    fixed = [n for n in sim.names()["joints"] if n.startswith("r/") and n != "r/j1"]
    assert fixed, sim.names()["joints"]
    with pytest.raises(alloy.SimulationError) as err:
        sim.set_targets({"r/j1": 0.5, fixed[0]: 0.7})
    assert "fixed" in err.value.message
    sim.step(50)
    assert abs(float(sim.state(world=0)["r/j1.q"])) < 0.05     # was driven to 0.7 by the fixed joint's target
    sim.set_targets({"r/j1": 0.5})
    sim.step(100)
    assert sim.state(world=0)["r/j1.q"] == pytest.approx(0.5, abs=0.02)
    sim.close()
