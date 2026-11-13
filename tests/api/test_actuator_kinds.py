# SPDX-License-Identifier: Apache-2.0
"""Joint targets and raw controls follow MuJoCo's meaning for every servo kind, checked against MuJoCo itself.

Laws (CPU): an ``<intvelocity>`` takes velocity targets and moves its joint as MuJoCo moves it for the same ctrl; a
position target on it, a position target on a ``<velocity>`` joint and a velocity target on a ``<position>`` joint
raise, naming the kind to use; a geared actuator's raw control means what MuJoCo's ctrl means (joint value = ctrl /
gear), while a joint target stays a joint value, for a ``<position>`` (driven from joint targets) and a ``<general>``
servo (driven by ctrl). Before, the ``<intvelocity>`` read a position target as a velocity and ran to its stop, the
mismatched kinds were silently ignored, and a gear of 2 moved the joint twice as far as MuJoCo.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import alloy.api as alloy

KINDS = """<mujoco model="kinds"><option timestep="0.00125" integrator="implicitfast"/><compiler angle="radian"/>
  <worldbody>
    <body name="b_pos" pos="0 -0.45 0.15"><joint name="j_pos" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025" mass="0.5"/></body>
    <body name="b_gear" pos="0 -0.15 0.15"><joint name="j_gear" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025" mass="0.5"/></body>
    <body name="b_vel" pos="0 0.15 0.15"><joint name="j_vel" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025" mass="0.5"/></body>
    <body name="b_intv" pos="0 0.45 0.15"><joint name="j_intv" type="hinge" axis="0 0 1" damping="0.5" range="-2 2"/>
      <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025" mass="0.5"/></body>
    <body name="b_gen" pos="0 0.75 0.15"><joint name="j_gen" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025" mass="0.5"/></body>
  </worldbody>
  <actuator>
    <position name="a_pos" joint="j_pos" kp="50" kv="5"/>
    <position name="a_gear" joint="j_gear" gear="2" kp="50" kv="5"/>
    <velocity name="a_vel" joint="j_vel" kv="10"/>
    <intvelocity name="a_intv" joint="j_intv" kp="50" kv="5" actrange="-2 2"/>
    <general name="a_gen" joint="j_gen" gear="2" gainprm="50" biastype="affine" biasprm="0 -50 -5"/>
  </actuator></mujoco>"""


def _mujoco_q(ctrl: dict[str, float], seconds: float) -> dict[str, float]:
    m = mujoco.MjModel.from_xml_string(KINDS)
    d = mujoco.MjData(m)
    for name, value in ctrl.items():
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = value
    mujoco.mj_step(m, d, nstep=int(round(seconds / m.opt.timestep)))
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)}


def _sim() -> alloy.Sim:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("r", mjcf=KINDS)
    return alloy.build(scene, device="cpu")


@pytest.mark.compile_heavy
def test_an_intvelocity_takes_velocity_and_mismatched_kinds_raise():
    sim = _sim()
    for joint, kind, use in (("r/j_intv", "position", "velocity"), ("r/j_vel", "position", "velocity"),
                             ("r/j_pos", "velocity", "position")):
        with pytest.raises(alloy.SimulationError) as err:
            sim.set_targets({joint: 0.5}, kind=kind)
        assert f"kind={use!r}" in err.value.fix, joint
    sim.set_targets({"r/j_intv": 0.5}, kind="velocity")      # raised before: "does not take velocity targets"
    sim.step(100)
    q = float(sim.state(world=0)["r/j_intv.q"])
    assert q == pytest.approx(_mujoco_q({"a_intv": 0.5}, 1.0)["j_intv"], abs=0.01)
    assert 0.4 < q < 0.5                                      # 0.5 rad/s for 1 s, less the servo's lag
    sim.close()


@pytest.mark.compile_heavy
def test_a_geared_control_means_what_mujoco_ctrl_means():
    ref = _mujoco_q({"a_gear": 0.5, "a_gen": 0.5}, 3.0)
    assert ref["j_gear"] == pytest.approx(0.25, abs=1e-3) and ref["j_gen"] == pytest.approx(0.25, abs=1e-3)
    sim = _sim()
    sim.set_controls({"r/a_gear": 0.5, "r/a_gen": 0.5})
    sim.step(300)
    st = sim.state(world=0)
    assert np.allclose([st["r/j_gear.q"], st["r/j_gen.q"]], [ref["j_gear"], ref["j_gen"]], atol=2e-3)   # was 0.5
    sim.set_targets({"r/j_gear": 0.1, "r/j_gen": 0.1})       # a joint target stays a joint value
    sim.step(300)
    st = sim.state(world=0)
    assert np.allclose([st["r/j_gear.q"], st["r/j_gen.q"]], 0.1, atol=2e-3)
    sim.close()
