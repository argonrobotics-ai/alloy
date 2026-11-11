# SPDX-License-Identifier: Apache-2.0
"""DirectSim's CPU route fast path (SCR-WP8-8): with ``MjcCoreSolver`` and no per-step hooks, a period is one
``step_many`` call, bitwise equal to the per-step loop; hooks, observers or a Newton collision pipeline keep the
loop."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.bully.direct import DirectSim
from alloy.route.mjc_core import MjcCoreSolver
from alloy.route.mjc_core_config import CPU_ROUTE_KEY
from alloy.scene.builtin_plugins import builtin_capabilities
from alloy.scene.noslip_route import NOSLIP_KEY

pytestmark = pytest.mark.cpu_route

XML = """<mujoco><option timestep="0.002" cone="elliptic" integrator="implicitfast" impratio="10"/>
<worldbody><geom name="floor" type="plane" size="2 2 .1"/>
 <body name="base" pos="0 0 .275"><joint name="j1" type="hinge" axis="0 0 1" frictionloss="0.2" damping="1"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".03" mass="1"/>
  <body name="link2" pos=".3 0 0"><joint name="j2" type="hinge" axis="0 1 0" frictionloss="0.1" damping="0.5"/>
   <geom name="finger" type="box" size=".04 .04 .1" pos="0 0 -.12" mass=".3"/></body></body>
 <body name="block" pos=".3 0 .03"><freejoint/><geom name="block" type="box" size=".03 .03 .03" mass=".1"/></body>
</worldbody><actuator><position joint="j1" kp="50" kv="2"/><position joint="j2" kp="30" kv="1"/></actuator>
</mujoco>"""
CTRL = (0.3, 0.05)
SUBSTEPS = 16


def make_sim(worlds: int = 2) -> DirectSim:
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(XML, ctrl_direct=True, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=worlds)
    model = b.finalize(device="cpu")
    cap = next(c for c in builtin_capabilities() if c.name == "mujoco")
    solver = cap.factory(model, {CPU_ROUTE_KEY: "mjc", NOSLIP_KEY: (50, 1e-10)})
    assert isinstance(solver, MjcCoreSolver)
    sim = DirectSim(model, solver, control_dt=SUBSTEPS * 0.002, substeps=SUBSTEPS, collide=False,
                    export_contacts=True)
    sim.control.mujoco.ctrl.assign(np.tile(CTRL, worlds).astype(np.float32))
    return sim


def test_fast_path_equals_the_per_step_loop() -> None:
    fast, slow = make_sim(), make_sim()
    calls = []
    slow.add_step_observer(lambda *args: calls.append(1))                # per-step observer: the loop
    assert fast.route_period_reason() is None
    assert slow.route_period_reason() is not None
    for _ in range(4):
        fast.step_period()
        slow.step_period()
    assert len(calls) == 4 * SUBSTEPS
    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
        assert np.array_equal(getattr(fast.state, name).numpy(), getattr(slow.state, name).numpy()), name
    assert fast.clock.ticks == slow.clock.ticks and fast.periods == slow.periods == 4
    assert np.array_equal(fast.contacts.rigid_contact_count.numpy(), slow.contacts.rigid_contact_count.numpy())
    hf, hs = fast.report()["health"], slow.report()["health"]
    assert np.array_equal(hf.flags, hs.flags) and np.array_equal(hf.max_pen_m, hs.max_pen_m)


def test_hooks_and_resets_keep_it_exact() -> None:
    a, b = make_sim(), make_sim()
    b.pre_substep.append(lambda state: None)                              # a hook: the per-step loop
    assert b.route_period_reason() == "pre_substep hooks or step observers need every step"
    for sim in (a, b):
        sim.step_period()
        sim.reset()
        sim.step_period()
        sim.step_period()
    assert np.array_equal(a.state.body_q.numpy(), b.state.body_q.numpy())
    assert a.clock.ticks == b.clock.ticks and a.periods == b.periods == 2
