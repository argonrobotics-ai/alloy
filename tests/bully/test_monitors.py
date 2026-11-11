# SPDX-License-Identifier: Apache-2.0
"""BenchMonitors on hand-built states: each monitor fires on its defect and stays quiet on clean motion."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.bully.monitors import BenchMonitors
from alloy.bully.roles import KIND_FREE, KIND_NON_TARGET, KIND_ROBOT, body_roles
from alloy.bully.severity import classify_monitors
from alloy.bully.types import BullyLimits

LIM = BullyLimits()
DT = 1.0e-3


def _model(worlds: int = 2) -> newton.Model:
    """World template: body 0 robot (driven revolute link), body 1 free box, body 2 free box (non-target)."""
    b = newton.ModelBuilder()
    link = b.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    b.add_shape_sphere(link, radius=0.05)
    j = b.add_joint_revolute(-1, link, axis=(0.0, 1.0, 0.0), target_ke=100.0, target_kd=1.0,
                             parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    b.add_articulation([j])
    for x in (1.0, 3.0):
        body = b.add_body(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()), mass=1.0)
        b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    scene = newton.ModelBuilder()
    scene.replicate(b, copy_count=worlds)
    return scene.finalize(device="cpu")


def _monitors(model: newton.Model, **kw) -> tuple[BenchMonitors, newton.State]:
    mon = BenchMonitors(model, LIM, non_target_bodies=(2,), **kw)
    s = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s)
    mon.reset(s)
    return mon, s


def _moved(model: newton.Model, s: newton.State, body: int, dx=(0.0, 0.0, 0.0), v=(0.0, 0.0, 0.0)) -> newton.State:
    out = model.state()
    q, qd = s.body_q.numpy().copy(), s.body_qd.numpy().copy()
    q[body, :3] += dx
    qd[body, :3] = v
    out.body_q.assign(q)
    out.body_qd.assign(qd)
    return out


def test_undriven_jointed_body_is_passive():
    b = newton.ModelBuilder()
    door = b.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    b.add_shape_box(door, hx=0.4, hy=0.02, hz=0.8, xform=wp.transform((0.4, 0.0, 0.0), wp.quat_identity()))
    j = b.add_joint_revolute(-1, door, axis=(0.0, 0.0, 1.0))  # a hinge without a drive: not robot
    b.add_articulation([j])
    roles = body_roles(b.finalize(device="cpu"))
    assert roles.kind.tolist() == [0]
    # radius about the COM reaches the box corner: |(0.4, 0.02, 0.8)| = 0.894 m
    assert roles.radius[0] == pytest.approx(0.894, abs=0.01)


def test_roles_from_joint_graph():
    roles = body_roles(_model(), non_target=(2,))
    assert roles.kind[:3].tolist() == [KIND_ROBOT, KIND_FREE, KIND_FREE | KIND_NON_TARGET]
    assert roles.kind[3:].tolist() == roles.kind[:3].tolist()  # same template in world 1
    assert roles.world_start.tolist() == [0, 2, 4]  # the ledger covers the two passive boxes per world
    assert roles.radius[1] > 0.17


def test_clean_step_is_quiet():
    m = _model()
    mon, s = _monitors(m)
    mon.update(s, s, None, None, DT)
    rep = mon.report()
    assert classify_monitors(rep, LIM).tolist() == [0, 0]
    assert rep["e_gain"].max() <= 0.0 + 1e-9 and rep["nontarget_disp"].max() == 0.0


def test_energy_injection_detected_and_robot_contact_excuses_it():
    m = _model()
    mon, s = _monitors(m)
    kicked = _moved(m, s, 4, v=(3.0, 0.0, 0.0))  # world 1 box: +4.5 J of KE in one step, no wrench
    mon.update(s, kicked, None, None, DT)
    rep = mon.report()
    # E_scale = m_tot g L_char with m_tot = 3 kg (incl. the robot link and the box shape masses from density)
    expected = 0.5 * float(m.body_mass.numpy()[4]) * 9.0 / mon.e_scale[1]
    assert rep["e_gain"][1] == pytest.approx(expected, rel=1e-5)
    assert rep["e_gain"][0] <= 1e-9
    # E_scale ~ 19.5 kg * g * 3.3 m ~ 630 J, the kick is ~6 % of it: > 1e-2 per step (S2), < 0.5 cumulative (S1)
    assert classify_monitors(rep, LIM).tolist() == [0, 2]
    for _ in range(9):  # the cumulative ledger is net: a kick followed by an equal loss does not accumulate
        mon.update(kicked, s, None, None, DT)
        mon.update(s, kicked, None, None, DT)
    assert mon.report()["e_gain_cum"][1] == pytest.approx(expected, rel=1e-4)


def test_applied_wrench_work_is_not_a_gain():
    m = _model(1)
    mon, s = _monitors(m)
    mass = float(m.body_mass.numpy()[1])
    v = 2.0
    s.body_f.assign(np.array([[mass * v / DT, 0, 0, 0, 0, 0]] * 3, dtype=np.float32) * np.array([[0], [1], [0]]))
    kicked = _moved(m, s, 1, v=(v, 0.0, 0.0))  # F dt = m v: work F (0 + v)/2 dt = m v^2 / 2 = dKE
    mon.update(s, kicked, None, None, DT)
    assert abs(mon.report()["e_gain"][0]) < 1e-5


class _Walls:
    """Stand-in for a solver that reports its walls' work on its particles (``alloy_shape_work``, SCR-WP9-6)."""

    def __init__(self, n: int) -> None:
        self.work = np.zeros(n, dtype=np.float64)

    def alloy_shape_work(self, out: wp.array) -> None:
        out.assign(self.work)


def test_robot_wall_work_on_particles_is_not_a_gain():
    """A driven SPH cup gives the liquid energy through its walls, not through a wrench: the work the robot's shapes
    report is subtracted; a passive shape's work is an exchange inside the ledger and is not (WP9's test)."""
    m = _model()
    walls = _Walls(m.shape_count)
    mon, s = _monitors(m, work_sources=(walls,))
    kicked = _moved(m, s, 4, v=(3.0, 0.0, 0.0))  # world 1 box: +4.5 J of KE, no wrench
    gain = 0.5 * float(m.body_mass.numpy()[4]) * 9.0
    shape_body = m.shape_body.numpy()
    walls.work[np.flatnonzero(shape_body == 3)] = gain  # world 1 robot link did it through its boundary
    walls.work[np.flatnonzero(shape_body == 1)] = gain  # a world 0 passive box: internal, ignored
    mon.update(s, kicked, None, None, DT)
    rep = mon.report()
    assert abs(rep["e_gain"][1]) < 1e-6 and abs(rep["e_gain"][0]) < 1e-9
    walls.work[:] = 0.0  # without the report the same kick is a gain again
    mon.update(kicked, s, None, None, DT)
    mon.update(s, kicked, None, None, DT)
    assert mon.report()["e_gain"][1] == pytest.approx(gain / mon.e_scale[1], rel=1e-5)


def test_shape_work_sources_finds_the_solver_or_a_coupled_entry():
    from types import SimpleNamespace

    from alloy.bully.monitors import shape_work_sources

    walls = _Walls(1)
    assert shape_work_sources(walls) == (walls,)
    coupled = SimpleNamespace(_entries={"mujoco": SimpleNamespace(solver=object()), "sph": SimpleNamespace(solver=walls)})
    assert shape_work_sources(coupled) == (walls,)
    assert shape_work_sources(object()) == () and shape_work_sources(None) == ()


def test_nontarget_teleport_impossible_speed_and_nonfinite():
    m = _model()
    mon, s = _monitors(m)
    cur = _moved(m, s, 2, dx=(1e-4, 0.0, 0.0))  # world 0 non-target shifts 0.1 mm
    cur = _moved(m, cur, 4, dx=(1.0, 0.0, 0.0), v=(12.0, 0.0, 0.0))  # world 1 box: 1 m jump at 12 m/s, no contact
    mon.update(s, cur, None, None, DT)
    rep = mon.report()
    assert rep["nontarget_disp"][0] == pytest.approx(1e-4, rel=1e-2)  # float32 positions at x = 3 m
    assert rep["teleport"][1] == pytest.approx(1.0 / (10 * LIM.v_max * DT), rel=1e-4)
    assert rep["impossible_speed"][1] == pytest.approx(12.0)
    assert classify_monitors(rep, LIM).tolist() == [2, 1]
    q = cur.body_q.numpy()
    q[0, 0] = np.nan
    cur.body_q.assign(q)
    mon.update(s, cur, None, None, DT)
    assert mon.report()["nonfinite"].tolist() == [1.0, 0.0]


def test_joint_separation():
    m = _model(1)
    mon, s = _monitors(m)
    mon.update(s, _moved(m, s, 0, dx=(0.0, 0.0, 2e-3)), None, None, DT)
    assert mon.report()["joint_sep"][0] == pytest.approx(2e-3, rel=1e-3)


def _contacts(m: newton.Model, s0: int, s1: int, sep: float) -> newton.Contacts:
    c = newton.Contacts(4, 0, device="cpu")
    c.rigid_contact_count.assign(np.array([1], dtype=np.int32))
    c.rigid_contact_shape0.assign(np.array([s0, -1, -1, -1], dtype=np.int32))
    c.rigid_contact_shape1.assign(np.array([s1, -1, -1, -1], dtype=np.int32))
    c.rigid_contact_normal.assign(np.array([[1, 0, 0]] * 4, dtype=np.float32))
    c.rigid_contact_point0.assign(np.zeros((4, 3), np.float32))
    c.rigid_contact_point1.assign(np.array([[sep, 0, 0]] * 4, dtype=np.float32))  # body frames, bodies at rest
    return c


def test_contact_penetration_thin_part_and_robot_excuse():
    m = _model(1)
    shape_body = m.shape_body.numpy()
    s_robot, s_box = int(np.nonzero(shape_body == 0)[0][0]), int(np.nonzero(shape_body == 1)[0][0])
    mon, s = _monitors(m, thin_parts={1: 2e-3})
    q = s.body_q.numpy()
    # points at the body origins; separation along +x = (x1 - x0) - margins: choose body x so that pen = 1.5 mm
    c = _contacts(m, s_robot, s_box, sep=0.0)
    gap = float(q[1, 0] - q[0, 0])
    c.rigid_contact_point1.assign(np.array([[-gap - 1.5e-3, 0, 0]] * 4, dtype=np.float32))
    kicked = _moved(m, s, 1, v=(12.0, 0.0, 0.0))
    mon.update(s, kicked, None, None, DT, c)
    rep = mon.report()
    assert rep["max_pen"][0] == pytest.approx(1.5e-3, rel=1e-3)
    assert rep["pen_s1_ratio"][0] == pytest.approx(1.5e-3 / 1e-3, rel=1e-3)  # half the 2 mm part
    assert rep["impossible_speed"][0] == 0.0  # robot contact: excused
    assert rep["e_gain"][0] == 0.0  # robot contact: ledger step excused


def _speed_rows(mon: BenchMonitors) -> tuple[float, float]:
    rep = mon.report()
    return float(rep["impossible_speed"][0]), float(rep["impossible_speed_d12"][0])


def test_impossible_speed_free_fall_inside_the_ballistic_envelope_is_not_s1():
    """SCR-WP7-7: a contact kick counts at its own speed; contact-free flight is S1 only above
    max(v_max, |v_exit| + g t + 0.5 m/s). D12's old row (``impossible_speed_d12``) trips on the free fall."""
    m = _model(1)
    shape_body = m.shape_body.numpy()
    s_box, s_other = int(np.nonzero(shape_body == 1)[0][0]), int(np.nonzero(shape_body == 2)[0][0])
    mon, s = _monitors(m)
    kicked = _moved(m, s, 1, v=(6.0, 0.0, 0.0))
    mon.update(s, kicked, None, None, DT, _contacts(m, s_box, s_other, sep=0.0))  # box-box contact, no robot
    empty = newton.Contacts(4, 0, device="cpu")  # contact buffer known, nothing in it
    mon.update(kicked, kicked, None, None, DT, empty)  # the substep after: exit speed 6 m/s, t = 0
    mon.update(kicked, _moved(m, kicked, 1, v=(6.0, 0.0, -8.94)), None, None, 0.5, empty)  # 0.5 s later, |v| 10.77
    amended, d12 = _speed_rows(mon)
    assert amended == pytest.approx(10.77 * 10.0 / (6.0 + 9.81 * 0.5 + 0.5), rel=1e-3)  # inside the envelope
    assert amended < LIM.v_max < d12
    mon.update(kicked, _moved(m, kicked, 1, v=(13.0, 0.0, 0.0)), None, None, 0.01, empty)  # speeds up in flight
    amended, _ = _speed_rows(mon)
    assert amended == pytest.approx(13.0 * 10.0 / (6.0 + 9.81 * 0.51 + 0.5), rel=1e-3)
    assert amended > LIM.v_max  # beyond gravity: S1


def test_impossible_speed_contact_kick_trips_and_robot_throw_is_excused_in_flight():
    m = _model(1)
    shape_body = m.shape_body.numpy()
    s_robot, s_box = int(np.nonzero(shape_body == 0)[0][0]), int(np.nonzero(shape_body == 1)[0][0])
    s_other = int(np.nonzero(shape_body == 2)[0][0])
    empty = newton.Contacts(4, 0, device="cpu")
    mon, s = _monitors(m)
    mon.update(s, _moved(m, s, 1, v=(11.0, 0.0, 0.0)), None, None, DT, _contacts(m, s_box, s_other, sep=0.0))
    assert _speed_rows(mon)[0] == pytest.approx(11.0)  # a non-robot contact gave it 11 m/s: S1
    mon.reset(s)
    thrown = _moved(m, s, 1, v=(12.0, 0.0, 0.0))
    mon.update(s, thrown, None, None, DT, _contacts(m, s_robot, s_box, sep=0.0))  # robot contact: excused
    mon.update(thrown, thrown, None, None, DT, empty)  # one substep after the robot contact: still excused
    mon.update(thrown, _moved(m, s, 1, v=(12.0, 0.0, -0.981)), None, None, 0.1, empty)  # 0.1 s of fall
    amended, d12 = _speed_rows(mon)
    assert amended < LIM.v_max < d12
    mon.reset(s)
    mon.update(s, _moved(m, s, 1, v=(12.0, 0.0, 0.0)), None, None, DT, None)  # no contact buffer: fail closed
    assert _speed_rows(mon)[0] == pytest.approx(12.0)


def test_capacity_overflow_is_sticky_across_steps_and_ignores_iteration_limits():
    """Newton zeroes MJWarp's Data.overflow before every step: a row overflow in an early substep must still count,
    and the solver iteration-limit bits (ITERATIONS 1 << 9, LS_ITERATIONS 1 << 10) are not capacity overflows."""
    m = _model()
    bits = wp.zeros(m.world_count, dtype=int, device="cpu")
    mon, s = _monitors(m, overflow=bits)
    bits.assign(np.array([1, (1 << 9) | (1 << 10)]))  # world 0: NEFC; world 1: iteration limits only
    mon.update(s, s, None, None, DT)
    bits.zero_()  # what SolverMuJoCo.step does before the next step
    mon.update(s, s, None, None, DT)
    assert mon.report()["capacity_overflow"].tolist() == [1.0, 0.0]
    assert classify_monitors(mon.report(), LIM).tolist() == [1, 0]
    mon.reset(s)
    assert mon.report()["capacity_overflow"].tolist() == [0.0, 0.0]
