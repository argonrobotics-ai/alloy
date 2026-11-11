# SPDX-License-Identifier: Apache-2.0
"""Small MuJoCo scenes for the C extension tests (alloy.route._mjc): an actuated two-joint arm with dry joint
friction whose box finger rests on a block, a free block, and a slab lying across a thin sheet's top edge."""

from __future__ import annotations

import mujoco
import pytest

from alloy.route import _mjc

CTRL = (0.3, 0.05)   # j1 turns the arm, dragging the block under the finger
ARM = """<mujoco><option timestep="0.002" cone="elliptic" integrator="{integrator}" impratio="10"/>
<worldbody><geom name="floor" type="plane" size="2 2 .1"/>
 <body name="base" pos="0 0 .275"><joint name="j1" type="hinge" axis="0 0 1" frictionloss="0.2" damping="1"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".03" mass="1"/>
  <body name="link2" pos=".3 0 0"><joint name="j2" type="hinge" axis="0 1 0" frictionloss="0.1" damping="0.5"/>
   <geom name="finger" type="box" size=".04 .04 .1" pos="0 0 -.12" mass=".3"/></body></body>
 <body name="block" pos=".3 0 .03"><freejoint/>
  <geom name="block" type="box" size=".03 .03 .03" mass=".1" condim="{condim}"/></body>
 <body name="sheet" pos="-.4 0 .1"><freejoint/><geom name="sheet" type="box" size=".005 .2 .1" mass=".5"/></body>
 <body name="slab" pos="-.4 0 .225" euler="0 0 0.3"><freejoint/>
  <geom name="slab" type="box" size=".15 .04 .02" mass=".2"/></body>
</worldbody><actuator><position joint="j1" kp="50" kv="2"/><position joint="j2" kp="30" kv="1"/></actuator>
</mujoco>"""


def c_step():
    """The C extension for a module of its own tests. Under ALLOY_NO_EXT (release CI's fallback leg) the module is
    skipped; elsewhere a missing extension fails the module, as before."""
    if _mjc.switched_off():
        pytest.skip(f"{_mjc.NO_EXT_ENV} switches the C step off", allow_module_level=True)
    return _mjc.require()


def arm(condim: int = 3, integrator: str = "implicitfast") -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(ARM.format(condim=condim, integrator=integrator))


def settled(m: mujoco.MjModel, steps: int = 150) -> mujoco.MjData:
    """The arm pressing on the block after `steps` plain mj_step calls."""
    d = mujoco.MjData(m)
    d.ctrl[:] = CTRL
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return d


def copy_state(m: mujoco.MjModel, d: mujoco.MjData) -> mujoco.MjData:
    out = mujoco.MjData(m)
    for f in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
        getattr(out, f)[:] = getattr(d, f)
    out.time = d.time
    return out
