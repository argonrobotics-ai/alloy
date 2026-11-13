# SPDX-License-Identifier: Apache-2.0
"""An actuator model holds a jittered start, and a jitter never writes a joint value into a motor (audit T2-011).

``ActuatorSpec`` (a PD / DC-motor model on ``motorized`` joints) restored its build pose at every reset, before the
task's ``jitter_joint`` moved the joint: a zero action then pulled a joint started at 1.0 rad back toward 0 (-9 N m),
and the jitter also wrote 1.0 into the ``<motor>``'s control, a torque of 1 N m added every step.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs import randomize as R
from alloy.envs.actuators import ActuatorSpec, motorized

ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="link" pos="0 0 0.25">
    <joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _env(mode: str, delay: int) -> VectorEnv:
    spec = ActuatorSpec(["arm/shoulder"], model="dc_motor", kp=60.0, kd=4.0, effort_limit=12.0,
                        saturation_effort=14.0, velocity_limit=8.0, mode=mode, delta=0.02, delay_steps=delay)
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=motorized(ARM))
    sim = alloy.build(scene, device="cpu", seed=0, plugins=[spec.plugin()])
    task = Task(obs=["arm"], actions=spec, randomize=[R.jitter_joint("arm/shoulder", at=1.0)], max_steps=200)
    return VectorEnv(sim, task, backend="numpy")


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("mode", "delay", "action"), [("delta", 0, 0.0), ("normalized", 2, 0.5), ("absolute", 0, 1.0)])
def test_a_jittered_start_is_held_with_no_phantom_torque(mode, delay, action):
    env = _env(mode, delay)
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.full((2, 1), action, np.float32))
    q = np.asarray(env.state["arm/shoulder.q"]).reshape(-1)
    torque = np.asarray(env.writer.torque)
    ctrl = np.asarray(env.xp.view(env.sim.engine.control.mujoco.ctrl))
    env.close()
    assert np.allclose(q, 1.0, atol=2e-3) and np.all(np.abs(torque) < 0.05) and np.all(ctrl == 0.0)
