# SPDX-License-Identifier: Apache-2.0
"""SCR-DC3-4: D20 energy includes passive spring energy 0.5 k (q - q_ref)^2 (Newton 1.8's core ``Model.joint_stiffness``
/ ``joint_rest_q``; fixed-tendon springs too), so a preloaded spring releasing is not an ENERGY_JUMP.

Rig: a 20 cm arm on a vertical hinge (gravity does no work), spring k = 50 N m/rad preloaded 0.6 rad (9 J) and
released at t = 0 under SolverMuJoCo, two worlds.
- With the spring term both worlds stay OK while the spring swings the arm (its energy moves between PE and KE).
- Without it (the old D20 sum) the release reads as an energy gain and the world goes INVALID: the rig is a real test.
- A real injection (3 E_scale of kinetic energy added to world 1 at step 40) is still caught: world 1 INVALID with
  ENERGY_JUMP (the cumulative S1 rule), world 0 OK.
- ``CheckPolicy.skip`` turns a check off publicly and rejects unknown names.
- Beside a free body (where Newton 1.8's legacy ``mujoco.dof_springref`` raises) the spring's rest is still read:
  ``joint_rest_q``, MuJoCo's springref - ref, the value MuJoCo's ``qpos_spring`` gets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.precision import health_setup
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags, Severity

K, PRELOAD, DT, STEPS = 50.0, 0.6, 5.0e-4, 400
POLICY = CheckPolicy(level="cheap", every=1, full_every=1)


def _rig() -> newton.Model:
    tmpl = newton.ModelBuilder()
    arm = tmpl.add_link(xform=wp.transform_identity(), mass=1.0, inertia=wp.mat33(np.eye(3) * 1e-3))
    tmpl.add_shape_box(arm, xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.01, hz=0.01)
    j = tmpl.add_joint_revolute(-1, arm, axis=(0.0, 0.0, 1.0))
    tmpl.add_articulation([j])
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(tmpl, 2)
    model = scene.finalize(device="cpu")
    model.joint_stiffness.assign(np.full(model.joint_dof_count, K, np.float32))
    model.joint_rest_q.assign(np.zeros(model.joint_coord_count, np.float32))
    model.joint_q.assign(np.full(model.joint_coord_count, PRELOAD, np.float32))
    return model


def _run(kick_world: int | None = None, policy: CheckPolicy = POLICY):
    model = _rig()
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    monitor = HealthMonitor(model, policy, solvers=[solver])
    swing = 0.0
    for k in range(STEPS):
        if kick_world is not None and k == 40:
            _kick(monitor, s0, kick_world)
        solver.step(s0, s1, control, None, DT)
        monitor.update(k, s0, s1, None)
        swing = max(swing, float(np.abs(s1.joint_q.numpy() - PRELOAD).max()))
        s0, s1 = s1, s0
    return monitor.poll(), swing


def _kick(monitor: HealthMonitor, state: newton.State, w: int) -> None:
    """Add 3 E_scale of kinetic energy to world ``w``'s arm. Its inertia about the hinge (link plus box shape) comes
    from the last check: KE = E - 0.5 k (q - q_ref)^2 = 0.5 I w^2."""
    e = float(monitor._acc.energy.numpy()[w])
    q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
    inertia = 2.0 * (e - 0.5 * K * float(q[w]) ** 2) / float(qd[w]) ** 2
    e_scale = float(monitor._ref.e_scale.numpy()[w])
    qd[w] = math.copysign(math.sqrt(float(qd[w]) ** 2 + 6.0 * e_scale / inertia), float(qd[w]))
    state.joint_qd.assign(qd)


def _energy_jump(report, w: int) -> bool:
    return bool(int(report.flags[w]) & int(HealthFlags.ENERGY_JUMP))


def test_preloaded_spring_release_is_not_an_energy_jump() -> None:
    report, swing = _run()
    assert swing > 0.5                                 # the spring really released (the arm swung through 0)
    assert report.invalid_worlds == () and not any(_energy_jump(report, w) for w in range(2))


def test_without_the_spring_term_the_release_reads_as_a_gain(monkeypatch) -> None:
    monkeypatch.setattr(health_setup, "_springs_on", lambda model: False)
    report, _ = _run()
    assert report.invalid_worlds == (0, 1) and all(_energy_jump(report, w) for w in range(2))


def test_real_injection_is_still_caught() -> None:
    report, _ = _run(kick_world=1)
    assert _energy_jump(report, 1) and int(report.severity[1]) == int(Severity.INVALID)
    assert not _energy_jump(report, 0) and int(report.severity[0]) == int(Severity.OK)


def test_skip_is_public_and_checked() -> None:
    report, _ = _run(kick_world=1, policy=CheckPolicy(level="cheap", every=1, full_every=1, skip=frozenset({"energy"})))
    assert not _energy_jump(report, 1)                 # the rig asked for no energy check
    with pytest.raises(ValueError, match="unknown check"):
        CheckPolicy(skip=frozenset({"energie"}))


TENDON_XML = """<mujoco><worldbody>
<body name="a"><joint name="j1" type="hinge" axis="0 0 1"/><geom type="box" size=".1 .01 .01" pos=".1 0 0"/>
 <body name="b" pos=".2 0 0"><joint name="j2" type="hinge" axis="0 0 1"/><geom type="box" size=".1 .01 .01" pos=".1 0 0"/>
 </body></body></worldbody>
<tendon><fixed name="t" stiffness="20" springlength="0.1 0.3"><joint joint="j1" coef="1"/><joint joint="j2" coef="-0.5"/>
</fixed></tendon></mujoco>"""


def test_fixed_tendon_spring_matches_mujoco() -> None:
    """A fixed tendon spring with a deadband: D20's term equals MuJoCo C's potential energy (``mj_energyPos``) at
    q = (0.6, 0.2) rad, where L = 0.5 is past the deadband's upper end 0.3: 0.5 * 20 * 0.2^2 = 0.4 J."""
    import mujoco

    tmpl = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(tmpl)
    tmpl.add_mjcf(TENDON_XML)
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(tmpl, 2)
    model = scene.finalize(device="cpu")
    state = model.state()
    state.joint_q.assign(np.tile(np.array([0.6, 0.2], np.float32), 2))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    monitor = HealthMonitor(model, POLICY, energy="on")
    monitor.reset_reference(state)
    out = wp.zeros(2, dtype=wp.float64, device="cpu")
    monitor._world_energy(state, out)
    mj = mujoco.MjModel.from_xml_string(TENDON_XML)
    d = mujoco.MjData(mj)
    d.qpos[:] = (0.6, 0.2)
    mj.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_forward(mj, d)
    np.testing.assert_allclose(out.numpy(), d.energy[0], rtol=1e-5)


FREE_ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="base" pos="0 0 1"><freejoint/>
 <geom type="box" size=".05 .05 .05" mass="1"/>
 <body name="link" pos=".1 0 0"><joint name="hinge" type="hinge" axis="0 1 0" stiffness="2" springref="0.7"/>
  <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass=".3"/></body></body></worldbody></mujoco>"""


def test_a_spring_beside_a_free_body_keeps_its_rest() -> None:
    """A hinge spring (k 2, springref 0.7 rad) under a free body: D20's spring row has q_ref 0.7, the joint_rest_q
    MuJoCo's qpos_spring gets (the legacy ``mujoco.dof_springref`` raises for this model, and a swallowed error read
    0)."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(FREE_ARM)
    model = b.finalize(device="cpu")
    with pytest.raises(AttributeError):
        model.mujoco.dof_springref                     # noqa: B018
    w, q, k, ref = health_setup._joint_springs(model)
    assert q.tolist() == [int(model.joint_q_start.numpy()[1])] and k.tolist() == [2.0]
    assert ref[0] == pytest.approx(0.7, abs=1e-6)
    mj = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True).mj_model
    assert mj.qpos_spring[mj.jnt_qposadr[1]] == pytest.approx(float(ref[0]), abs=1e-6)
