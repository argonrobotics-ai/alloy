# SPDX-License-Identifier: Apache-2.0
"""Compensated float32 joint coordinates (`alloy.scene.coordinate_carry`, SCR-SC10-3, coordinator rulings 2026-11-05
19:10Z and 22:33Z: default-on), through the MuJoCo entry factory, on MuJoCo Warp and MuJoCo C (CPU):
- every SolverMuJoCo gets it at construction where it can run; the entry kwarg False and ``uninstall`` take it off;
- an FR3-class position servo at q 2.2 rad freezes (kv / kp) x half-spacing / dt short of its target with a constant
  ghost velocity, and the carry lets it settle;
- a free body creeping below the float32 spacing stands still, and moves at its velocity with the carry (identity and
  rotated parent frames);
- off changes nothing; repeats are bitwise; reset zeroes the residuals; the guard refuses a non-integration change but
  not plain rounding on a joint with an MJCF ref; rk4 and update_data_interval 0 refuse;
- the residual is registered solver state (WP4's adapters: snapshots, rollback, freeze);
- a coupler's restart passes (the same input replayed, only the last pass kept) give the plain run bitwise, and its
  per-world reset (a restart notify, then reset(world_mask)) leaves the other worlds' residuals alone."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.scene.builtin_plugins import _mujoco
from alloy.scene.coordinate_carry import (CARRY_KEY, CarryError, carry_plan, guard_trips, install_coordinate_carry,
                                          uninstall)

DT = 1e-3
Q0, OFFSET = 2.2, 0.01           # rad: FR3 joint 4's angle in C18, and the servo's start error
KP, KV = 3500.0, 350.0           # FR3 joint 3/4 servo (menagerie fr3.xml), kv / kp = 0.1 s
STALL = (KV / KP) * float(np.spacing(np.float32(Q0))) / 2 / DT   # 11.9 urad at dt 1 ms
ROUTE = {"integrator": "implicitfast", "nconmax": 4, "njmax": 16}
PATHS = [pytest.param({}, id="mjwarp"), pytest.param({"use_mujoco_cpu": True}, id="mujoco_c")]


def _hinge() -> newton.Model:
    """One link (1 kg at 0.3 m, armature 0.2) on a hinge at Q0 under a position servo aimed OFFSET away, no gravity."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    body = b.add_link(mass=1.0, com=wp.vec3(0.3, 0.0, 0.0), inertia=wp.mat33(np.eye(3) * 1e-3))
    j = b.add_joint_revolute(-1, body, axis=(0.0, 1.0, 0.0), target_ke=KP, target_kd=KV, armature=0.2,
                             actuator_mode=newton.JointTargetMode.POSITION)
    b.add_articulation([j])
    m = b.finalize(device="cpu")
    m.joint_q.assign(np.array([Q0], np.float32))
    return m


def _free(parent_rot_z: float = 0.0, x0: float = 3.0, v: float = 1e-5, parent_x: float = 0.0) -> newton.Model:
    """A free 1 kg body at x0 (joint frame) drifting at v along the joint frame's x, no gravity; the joint's parent
    frame turned by parent_rot_z about z and moved parent_x along world x."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    body = b.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 1e-3))
    j = b.add_joint_free(body, parent_xform=wp.transform(wp.vec3(parent_x, 0.0, 0.0),
                                                         wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), parent_rot_z)))
    b.add_articulation([j])
    m = b.finalize(device="cpu")
    q = m.joint_q.numpy()
    q[:3] = (x0, 0.0, 0.0)
    m.joint_q.assign(q)
    qd = m.joint_qd.numpy()
    qd[0] = v
    m.joint_qd.assign(qd)
    return m


def _run(model: newton.Model, kw: dict, seconds: float, target: float | None = None, solver=None):
    """Step `seconds`; returns (final joint_q, final joint_qd, solver)."""
    solver = solver or _mujoco(model, {**ROUTE, **kw})
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    if target is not None:
        ctl.joint_target_q.assign(np.array([target], np.float32))
    for _ in range(int(round(seconds / DT))):
        solver.step(s0, s1, ctl, None, DT)
        s0, s1 = s1, s0
    return s0.joint_q.numpy(), s0.joint_qd.numpy(), solver


@pytest.mark.parametrize("path", PATHS)
def test_servo_freezes_short_and_the_carry_settles_it(path):
    target = float(np.float32(Q0 + OFFSET))
    q, qd, _ = _run(_hinge(), {**path, CARRY_KEY: False}, 2.0, target)
    assert 0.5 * STALL < target - q[0] <= 1.05 * STALL               # frozen 11.9 urad short
    assert abs(qd[0]) > 0.5 * STALL * KP / KV                        # with the ghost velocity it would need
    q, qd, solver = _run(_hinge(), {**path, CARRY_KEY: True}, 2.0, target)
    assert abs(target - q[0]) < 0.05 * STALL and abs(qd[0]) < 0.05 * STALL * KP / KV
    assert guard_trips(solver) == 0


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("rot", [0.0, math.pi / 2])
def test_free_body_creep_below_the_spacing_moves_only_with_the_carry(path, rot):
    T, v = 0.5, 1e-5                                                  # 1e-8 m per step at x 3 m (half-spacing 1.2e-7)
    q, _, _ = _run(_free(rot, v=v), {**path, CARRY_KEY: False}, T)
    assert abs(q[0] - 3.0) <= 2 * np.spacing(np.float32(3.0))      # frozen (a turned frame's round trip jitters 1 ulp)
    q, _, solver = _run(_free(rot, v=v), {**path, CARRY_KEY: True}, T)
    assert abs((q[0] - 3.0) - v * T) < 0.06 * v * T                   # 5 um within the 0.24 um spacing
    assert abs(q[1]) < 1e-6 and guard_trips(solver) == 0
    assert carry_plan(solver).counts == {"axes": 0, "free_positions": 3}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("rig", ["hinge", "free"])
def test_body_q_is_fk_of_the_carried_joint_q(path, rig):
    """The carry runs inside the step, before the FK that derives body_q (module docstring, Body poses): on every step,
    also those where a residual goes back into q, body_q is Newton's FK of the carried joint_q, bit for bit."""
    model = _hinge() if rig == "hinge" else _free(math.pi / 2)
    solver = _mujoco(model, {**ROUTE, **path, CARRY_KEY: True})
    s0, s1, ctl, fk = model.state(), model.state(), model.control(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    if rig == "hinge":
        ctl.joint_target_q.assign(np.array([np.float32(Q0 + OFFSET)], np.float32))
    carry, pushed = solver._alloy_carry, 0
    apply = carry.apply

    def counting_apply(state_out, dt):
        nonlocal pushed
        before = state_out.joint_q.numpy().copy()
        apply(state_out, dt)
        pushed += not np.array_equal(before, state_out.joint_q.numpy())

    carry.apply = counting_apply
    for _ in range(2000 if rig == "hinge" else 300):
        solver.step(s0, s1, ctl, None, DT)
        newton.eval_fk(model, s1.joint_q, s1.joint_qd, fk)
        assert np.array_equal(s1.body_q.numpy(), fk.body_q.numpy())
        s0, s1 = s1, s0
    assert pushed > 0 and guard_trips(solver) == 0                  # the carry added residuals back on the way


def test_default_on_at_construction_and_the_opt_outs():
    direct = newton.solvers.SolverMuJoCo(_hinge(), **ROUTE)
    assert hasattr(direct, "_alloy_carry") and "step" in vars(direct)
    assert uninstall(direct) and "step" not in vars(direct) and not hasattr(direct, "_alloy_carry")
    assert not uninstall(direct)
    rk4 = _mujoco(_hinge(), {**ROUTE, "integrator": "rk4"})          # SyncedMuJoCo: refused, runs without it
    assert not hasattr(rk4, "_alloy_carry")
    wrapped = newton.solvers.SolverMuJoCo(_hinge(), **ROUTE)
    wrapped.step = lambda *a, **k: None                              # something wrapped it after the carry
    with pytest.raises(CarryError, match="wrapped again"):
        uninstall(wrapped)


def test_off_leaves_the_solver_alone_and_repeats_are_bitwise():
    solver = _mujoco(_hinge(), {**ROUTE, CARRY_KEY: False})
    assert "step" not in vars(solver) and not hasattr(solver, "_alloy_carry")
    a = _run(_hinge(), {CARRY_KEY: True}, 0.3, Q0 + OFFSET)
    b = _run(_hinge(), {CARRY_KEY: True}, 0.3, Q0 + OFFSET)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    assert install_coordinate_carry(a[2]) == "already installed"


@pytest.mark.parametrize("path", PATHS)
def test_coupler_restart_passes_replay_the_residual(path):
    """Each step as a coupler runs it: an input update, a pass, then two restart passes from the same input."""
    target, T = Q0 + OFFSET, 0.6
    plain = _run(_hinge(), {**path, CARRY_KEY: True}, T, target)
    model = _hinge()
    solver = _mujoco(model, {**ROUTE, **path, CARRY_KEY: True})
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    ctl.joint_target_q.assign(np.array([target], np.float32))
    flags = int(newton.StateFlags.JOINT_Q) | int(newton.StateFlags.JOINT_QD)
    for _ in range(int(round(T / DT))):
        solver.coupling_notify_input_state_update(s0, flags, iteration_restart=False, dt=DT)
        for k in range(3):
            if k:
                solver.coupling_notify_input_state_update(s0, flags, iteration_restart=True, dt=DT)
            solver.step(s0, s1, ctl, None, DT)
        s0, s1 = s1, s0
    assert s0.joint_q.numpy().tobytes() == plain[0].tobytes()
    assert s0.joint_qd.numpy().tobytes() == plain[1].tobytes()


def test_coupler_world_reset_keeps_the_other_worlds():
    """SolverCoupled.reset notifies a restart before reset(world_mask): only the masked world's residual changes."""
    one = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(one)
    body = one.add_link(mass=1.0, com=wp.vec3(0.3, 0.0, 0.0), inertia=wp.mat33(np.eye(3) * 1e-3))
    one.add_articulation([one.add_joint_revolute(-1, body, axis=(0.0, 1.0, 0.0), target_ke=KP, target_kd=KV,
                                                 armature=0.2, actuator_mode=newton.JointTargetMode.POSITION)])
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(one, 2)
    model = b.finalize(device="cpu")
    model.joint_q.assign(np.full(2, Q0, np.float32))
    solver = _mujoco(model, {**ROUTE, CARRY_KEY: True})
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    ctl.joint_target_q.assign(np.full(2, Q0 + OFFSET, np.float32))
    flags = int(newton.StateFlags.JOINT_Q) | int(newton.StateFlags.JOINT_QD)
    for _ in range(200):
        solver.coupling_notify_input_state_update(s0, flags, iteration_restart=False, dt=DT)
        solver.step(s0, s1, ctl, None, DT)
        s0, s1 = s1, s0
    state, world = solver._alloy_carry, carry_plan(solver).world
    before = state.carry.numpy().copy()
    assert np.any(before[world == 1] != state.saved.numpy()[world == 1])   # the last exchange moved world 1's residual
    solver.coupling_notify_input_state_update(s0, flags, iteration_restart=True, dt=DT)
    solver.reset(s0, world_mask=wp.array([True, False, False], dtype=wp.bool, device="cpu"))
    solver.coupling_notify_input_state_update(s0, flags, iteration_restart=False, dt=DT)
    solver.step(s0, s1, ctl, None, DT)
    assert np.all(state.saved.numpy()[world == 0] == 0.0)
    assert state.saved.numpy()[world == 1].tobytes() == before[world == 1].tobytes()


def test_reset_zeroes_the_residuals():
    model = _free()
    q, _, solver = _run(model, {CARRY_KEY: True}, 0.01)
    carry = solver._alloy_carry.carry
    assert np.any(carry.numpy() != 0.0)
    solver.reset(model.state())
    assert np.all(carry.numpy() == 0.0)


def test_guard_refuses_a_change_that_is_not_rounding():
    model = _free(v=0.0)
    solver = newton.solvers.SolverMuJoCo(model, **ROUTE)
    inner = solver._mujoco_warp_step

    def jump():                    # stands in for a change MuJoCo did not integrate: qpos moved after its integration,
        inner()                    # before Newton reads it back (the carry runs between that read and the FK)
        qpos = solver.mjw_data.qpos.numpy()
        qpos[0, 0] += 1e-3
        solver.mjw_data.qpos.assign(qpos)
    solver._mujoco_warp_step = jump
    q, _, _ = _run(model, {}, 3 * DT, solver=solver)
    assert guard_trips(solver) == 3 and abs(q[0] - 3.003) < 1e-6
    assert np.all(solver._alloy_carry.carry.numpy() == 0.0)


def test_guard_scales_by_the_ref_offset():
    """An MJCF ref: Newton's q = qpos - ref starts at 0 while MuJoCo's qpos is 0.15 rad, so MuJoCo rounds at the 0.15 rad
    spacing (1.5e-8). A 5 urad/s creep (5e-9 per 1 ms step) freezes without the carry, and with it moves at its velocity
    without one guard trip (the guard scaled by |q| alone tripped on every step: SFP's ref'd connector slides)."""
    xml = """<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody><body>
      <joint name="h" type="hinge" axis="0 1 0" ref="0.15"/>
      <geom type="sphere" size="0.01" mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>"""

    def model():
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.add_mjcf(xml)
        m = b.finalize(device="cpu")
        m.joint_qd.assign(np.array([5e-6], np.float32))
        return m
    T = 0.5
    q, _, _ = _run(model(), {CARRY_KEY: False}, T)
    assert q[0] == 0.0
    q, _, solver = _run(model(), {}, T)
    assert carry_plan(solver).offset == pytest.approx([0.15])
    assert guard_trips(solver) == 0 and abs(q[0] - 5e-6 * T) < 1.5e-8


@pytest.mark.parametrize("path", PATHS)
def test_guard_scales_by_a_translated_parent_frame(path):
    """A free joint whose parent frame sits 3 m out: Newton's position starts at 0, MuJoCo's world position at 3 m, so
    MuJoCo rounds at 3 m (half-spacing 1.2e-7 m). A 10 um/s drift (1e-8 m per step) freezes without the carry, and with
    it moves at its velocity with no guard trip."""
    T, v = 0.5, 1e-5
    q, _, _ = _run(_free(x0=0.0, v=v, parent_x=3.0), {**path, CARRY_KEY: False}, T)
    assert abs(q[0]) <= 2 * np.spacing(np.float32(3.0))
    q, _, solver = _run(_free(x0=0.0, v=v, parent_x=3.0), path, T)
    assert carry_plan(solver).offset == pytest.approx([3.0, 3.0, 3.0])
    assert guard_trips(solver) == 0 and abs(q[0] - v * T) < 0.06 * v * T


_WELD_CHAIN = """<mujoco><compiler angle="radian"/><option integrator="implicitfast"/><worldbody>
  <body name="a" pos="0 0 1"><joint type="hinge" axis="0 1 0" damping="0.01"/>
    <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1" contype="0" conaffinity="0"/>
    <body name="b" pos="0.3 0 0"><joint type="hinge" axis="0 1 0" damping="0.01"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1" contype="0" conaffinity="0"/></body></body>
  <body name="c" pos="0.6 0 1"><freejoint/>
    <geom type="box" size="0.03 0.03 0.03" mass="0.5" contype="0" conaffinity="0"/></body></worldbody><equality><weld body1="b" body2="c"/></equality></mujoco>"""


@pytest.mark.parametrize("left_out", [(), ("mjw.cvel",), ("mjw.cdof_dot",)], ids=["all", "no_cvel", "no_cdof_dot"])
def test_weld_rows_previous_step_state_is_snapshotted(left_out):
    """A weld on a swinging hinge chain (qvel ~3-7 rad/s), resumed in a fresh solver from the Newton state and
    ``mujoco_state_arrays``: bitwise the unbroken run. MuJoCo Warp builds the weld rows' J-dot v term from the previous
    step's cvel and cdof_dot (make_constraint runs before com_vel), so leaving either out breaks it (WP4's check)."""
    from alloy.coupling.adapters import mujoco_state_arrays  # noqa: PLC0415

    def chain():
        b = newton.ModelBuilder()
        b.add_mjcf(_WELD_CHAIN)
        m = b.finalize("cpu")
        return m, newton.solvers.SolverMuJoCo(m, integrator="implicitfast", nconmax=4, njmax=64)

    def run(m, solver, s, n):
        s1, ctl = m.state(), m.control()
        for _ in range(n):
            solver.step(s, s1, ctl, None, 2e-3)
            s, s1 = s1, s
        return s

    fields = ("joint_q", "joint_qd", "body_q", "body_qd")
    m, solver = chain()
    s = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s)
    s = run(m, solver, s, 150)
    saved = {k: getattr(s, k).numpy().copy() for k in fields}
    arrays = {k: a.numpy().copy() for k, a in mujoco_state_arrays(solver).items()}
    assert {"mjw.cvel", "mjw.cdof_dot"} <= set(arrays) and np.abs(s.joint_qd.numpy()[:2]).min() > 1.0
    unbroken = run(m, solver, s, 50).body_q.numpy()
    m2, fresh = chain()
    r = m2.state()
    for k in fields:
        getattr(r, k).assign(saved[k])
    live = mujoco_state_arrays(fresh)
    for k, v in arrays.items():
        if k not in left_out:
            live[k].assign(v)
    same = np.array_equal(run(m2, fresh, r, 50).body_q.numpy(), unbroken)
    assert same == (not left_out), (left_out, "MuJoCo Warp may now recompute cvel / cdof_dot before make_constraint: "
                                              "drop them from adapters.MUJOCO_CORE" if left_out else "resume not bitwise")


def test_residual_is_registered_solver_state():
    from alloy.core.registry import state_arrays_for, state_world_maps_for
    from alloy.coupling.adapters import register_builtin_adapters

    register_builtin_adapters()
    solver = _mujoco(_free(), dict(ROUTE))
    arrays, maps = state_arrays_for(solver), state_world_maps_for(solver)
    for key in ("carry.residual", "carry.saved"):
        assert arrays[key] is getattr(solver._alloy_carry, "carry" if key.endswith("residual") else "saved")
        assert maps[key].numpy().tolist() == [0, 0, 0]


def test_refuses_without_the_round_trip_or_on_rk4():
    with pytest.raises(CarryError, match="update_data_interval"):
        install_coordinate_carry(newton.solvers.SolverMuJoCo(_hinge(), update_data_interval=0, **ROUTE))
    with pytest.raises(CarryError, match="update_data_interval"):   # rk4 entries step on SyncedMuJoCo
        _mujoco(_hinge(), {**ROUTE, "integrator": "rk4", CARRY_KEY: True})
    with pytest.raises(CarryError, match="rk4"):
        install_coordinate_carry(newton.solvers.SolverMuJoCo(_hinge(), **{**ROUTE, "integrator": "rk4"}))
