# SPDX-License-Identifier: Apache-2.0
"""SCR-SC0-1: rigid penetration on the native MuJoCo contact route, and "not measured" instead of 0.0.

- No contact source: ``max_pen_m`` is NaN and ``pen_measured`` False (never 0.0); ``HealthReport.ok()`` the same.
- Newton ``Contacts`` given: measured, also when there is no contact at all (0.0 is then a measurement).
- Native route (``SolverMuJoCo``, ``use_mujoco_contacts=True``, no Newton contacts): the monitor reads MuJoCo Warp's
  contacts on device. Two worlds, a 10 cm box pressed 0.5 mm (world 0) and 15 mm (world 1) into the ground: each
  world's ``max_pen_m`` equals -min(contact.dist) of ``mujoco.mj_forward`` (C, float64) at that world's ``qpos``.
  With ``CheckPolicy.native_pen_flags`` only world 1 trips PEN_FAIL (15 mm > pen_s1 = 1 cm); with it off (a rig's
  own choice; on by default since GAP-E1, audit ERR-1) the depth is measured and no PEN flag is raised. The GPU
  case also replays the step + check from a CUDA graph and gets the same record.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags, HealthReport, StepHealth
from alloy.testing.precision import write_row

HALF = 0.05
PRESS = (5.0e-4, 1.5e-2)  # [m] into the ground, world 0 / world 1
DT = 1.0e-6               # one tiny step: the bodies stay where the collision measured them
POLICY = CheckPolicy(level="cheap", every=1, full_every=1)
FLAGS = CheckPolicy(level="cheap", every=1, full_every=1, native_pen_flags=True)


def _boxes(device: str, half_z: float = HALF, press: tuple[float, ...] = PRESS) -> tuple[newton.Model, newton.State]:
    """One 10 cm x 10 cm box of half height ``half_z`` per world, pressed ``press[w]`` into the ground."""
    tmpl = newton.ModelBuilder()
    body = tmpl.add_body(xform=wp.transform((0.0, 0.0, half_z), wp.quat_identity()))
    tmpl.add_shape_box(body, hx=HALF, hy=HALF, hz=half_z)
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(tmpl, len(press))
    scene.add_ground_plane()
    model = scene.finalize(device=device)
    q = model.joint_q.numpy().reshape(len(press), 7)
    q[:, 2] = half_z - np.array(press)
    model.joint_q.assign(q.reshape(-1))
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state


def _reference_pen(solver) -> np.ndarray:
    """-min(contact.dist) per world from MuJoCo C at each world's qpos (0 when no contact penetrates)."""
    m = solver.mj_model
    out = []
    for qpos in solver.mjw_data.qpos.numpy():
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        dist = np.array([d.contact[i].dist for i in range(d.ncon)])
        out.append(max(0.0, -float(dist.min())) if dist.size else 0.0)
    return np.array(out)


def _native(device: str, graph: bool = False, policy: CheckPolicy = FLAGS) -> tuple[HealthReport, np.ndarray]:
    model, s0 = _boxes(device)
    s1, control = model.state(), model.control()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    monitor = HealthMonitor(model, policy, solvers=[solver])

    def step() -> None:
        solver.step(s0, s1, control, None, DT)
        monitor.update(0, s0, s1, None)

    step()
    if graph:
        with wp.ScopedCapture(device=device) as cap:
            step()
        wp.capture_launch(cap.graph)
    return monitor.poll(), _reference_pen(solver)


def _check_native(report: HealthReport, ref: np.ndarray) -> None:
    assert report.pen_measured is not None and report.pen_measured.all()
    np.testing.assert_allclose(report.max_pen_m, ref, atol=2.0e-6)
    assert ref[0] == pytest.approx(PRESS[0], abs=1.0e-4) and ref[1] == pytest.approx(PRESS[1], abs=1.0e-4)
    fail = [bool(int(f) & int(HealthFlags.PEN_FAIL)) for f in report.flags]
    assert fail == [False, True]


def test_no_source_reports_not_measured() -> None:
    model, s0 = _boxes("cpu")
    monitor = HealthMonitor(model, POLICY)
    monitor.update(0, s0, s0, None)
    report = monitor.poll()
    assert np.isnan(report.max_pen_m).all() and not report.pen_measured.any()
    assert not any(int(f) & int(HealthFlags.PEN_FAIL | HealthFlags.PEN_WARN) for f in report.flags)
    assert report.pen_text(0) == "not measured"
    ok = HealthReport.ok(3)
    assert np.isnan(ok.max_pen_m).all() and not ok.pen_measured.any() and ok.pen_text(2) == "not measured"


def test_newton_contacts_measure_zero() -> None:
    model, s0 = _boxes("cpu")
    q = s0.body_q.numpy()
    q[:, 2] += 0.5  # lifted clear of the ground: no contact, and 0.0 is a measurement
    s0.body_q.assign(q)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    pipe.collide(s0, contacts)
    monitor = HealthMonitor(model, POLICY)
    monitor.update(0, s0, s0, contacts)
    report = monitor.poll()
    assert report.pen_measured.all() and (report.max_pen_m == 0.0).all()


def test_newton_and_cloth_contacts_follow_the_same_dwell_rule() -> None:
    """GAP-E1 (ruling 2026-11-10 05:27Z, audit COUPLE-6: a body landing on a 2 mm cloth froze the world at first
    touch, 1.39 mm deep): contacts judged from a Newton ``Contacts`` (rigid pairs and cloth / soft contacts share
    ``note_pen``) follow the same rule as MuJoCo's own. A 2 mm sheet 0.5 / 1.4 / 2.5 mm into the ground: no flag;
    past half its thickness, PEN_WARN and PEN_FAIL only after the 0.1 s dwell; through it, PEN_FAIL at the fourth check.
    Without ``dt`` (callers that give none) past half still fails at the first check, as before."""
    press = (0.5e-3, 1.4e-3, 2.5e-3)
    model, s0 = _boxes("cpu", half_z=1.0e-3, press=press)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    pipe.collide(s0, contacts)
    monitor, seen = HealthMonitor(model, POLICY), []
    for k in range(4):                                                       # 30, 60, 90, 120 ms
        monitor.update(k, s0, s0, contacts, dt=0.03)
        report = monitor.poll()
        seen.append([(bool(int(f) & int(HealthFlags.PEN_WARN)), bool(int(f) & int(HealthFlags.PEN_FAIL)))
                     for f in report.flags])
    np.testing.assert_allclose(report.max_pen_m, press, atol=1.0e-4)
    shallow, half, through = zip(*seen)
    assert set(shallow) == {(False, False)}
    assert list(half) == [(True, False)] * 3 + [(True, True)]
    assert list(through) == [(True, False)] * 3 + [(True, True)]              # through: the fourth check
    no_dt = HealthMonitor(model, POLICY)
    no_dt.update(0, s0, s0, contacts)
    assert [bool(int(f) & int(HealthFlags.PEN_FAIL)) for f in no_dt.poll().flags] == [False, True, True]


def test_native_mujoco_contacts_cpu() -> None:
    _check_native(*_native("cpu"))


def test_native_flags_on_by_default_and_depth_measured_with_them_off() -> None:
    """GAP-E1 (audit ERR-1): the default policy raises the PEN flags from native contacts; a rig that turns them off
    (``native_pen_flags=False``) still gets the measured depth, without PEN flags."""
    pen = int(HealthFlags.PEN_FAIL | HealthFlags.PEN_WARN)
    report, ref = _native("cpu", policy=POLICY)
    np.testing.assert_allclose(report.max_pen_m, ref, atol=2.0e-6)
    assert not int(report.flags[0]) & pen and int(report.flags[1]) & int(HealthFlags.PEN_FAIL)
    off, ref = _native("cpu", policy=CheckPolicy(level="cheap", every=1, full_every=1, native_pen_flags=False))
    np.testing.assert_allclose(off.max_pen_m, ref, atol=2.0e-6)
    assert off.pen_measured.all() and not any(int(f) & pen for f in off.flags)


def _dwell_rig(policy: CheckPolicy):
    """The two pressed boxes with a check that stands for 30 ms of simulated time; ``check(z1)`` puts world 1's
    box centre at ``z1`` first and returns (PEN_WARN, PEN_FAIL) per world."""
    model, s0 = _boxes("cpu")
    s1, control = model.state(), model.control()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    monitor = HealthMonitor(model, policy, solvers=[solver])
    steps = iter(range(1000))

    def check(z1: float) -> list[tuple[bool, bool]]:
        q = s0.joint_q.numpy().reshape(2, 7)
        q[1, 2] = z1
        s0.joint_q.assign(q.reshape(-1))
        newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
        solver.step(s0, s1, control, None, DT)
        monitor.update(next(steps), s0, s1, None, dt=0.03)
        return [(bool(int(f) & int(HealthFlags.PEN_WARN)), bool(int(f) & int(HealthFlags.PEN_FAIL)))
                for f in monitor.poll().flags]

    return check


def test_native_pen_fail_needs_the_dwell_and_the_clock_resets_per_world() -> None:
    """GAP-E1 (ruling 2026-11-10 00:08Z): 15 mm deep is PEN_WARN at once and PEN_FAIL once it has stayed past 1 cm
    for ``native_pen_dwell_s`` (0.1 s: the 4th 30 ms check); coming out resets the clock; world 0 (0.5 mm) never
    flags. Past ``native_pen_hard`` x 1 cm it is PEN_FAIL at the fourth check in a row (rulings 06:35Z, 07:08Z)."""
    deep, clear = HALF - PRESS[1], HALF + 1.0e-3
    check = _dwell_rig(CheckPolicy(level="cheap", every=1, full_every=1))
    quiet, warn, fail = (False, False), (True, False), (True, True)
    assert [check(deep) for _ in range(3)] == [[quiet, warn]] * 3            # 30, 60, 90 ms past 1 cm
    assert check(clear) == [quiet, quiet]                                    # out: the clock resets
    assert [check(deep) for _ in range(3)] == [[quiet, warn]] * 3            # ... so three more checks only warn
    assert check(deep) == [quiet, fail]                                      # 120 ms: PEN_FAIL
    hard = _dwell_rig(CheckPolicy(level="cheap", every=1, full_every=1, native_pen_hard=1.2))
    assert [hard(deep) for _ in range(4)] == [[quiet, warn]] * 3 + [[quiet, fail]]   # 15 > 12 mm: 4th check


@pytest.mark.parametrize(("thick", "press"), [(5.0e-3, (2.0e-3, 3.0e-3, 6.0e-3)), (2.0e-3, (0.5e-3, 1.5e-3, 2.5e-3))])
def test_thin_part_limits_are_relative(thick: float, press: tuple[float, float, float]) -> None:
    """GAP-E1 (auditor, ruling 2026-11-10 00:52Z): one rule, the limit is min(1 cm, half the thinner part). A 5 mm
    plate and a 2 mm sheet pressed into the ground, three worlds: under half its thickness (PEN_WARN only past 1 mm,
    never PEN_FAIL), past half (PEN_WARN at once, PEN_FAIL after the 0.1 s dwell), through it (PEN_FAIL at the
    fourth check in a row).
    The absolute 1 cm / 2 cm limits alone would pass all three: every depth here is under 1 cm."""
    model, s0 = _boxes("cpu", half_z=thick / 2, press=press)
    s1, control = model.state(), model.control()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), solvers=[solver])
    seen = []
    for k in range(4):                                                       # 30, 60, 90, 120 ms
        solver.step(s0, s1, control, None, DT)
        monitor.update(k, s0, s1, None, dt=0.03)
        report = monitor.poll()
        seen.append([(bool(int(f) & int(HealthFlags.PEN_WARN)), bool(int(f) & int(HealthFlags.PEN_FAIL)))
                     for f in report.flags])
    np.testing.assert_allclose(report.max_pen_m, press, atol=2.0e-5)
    np.testing.assert_allclose(report.peak_pen_m, press, atol=2.0e-5)        # the record's deepest since its reset
    shallow, half, through = zip(*seen)
    assert set(shallow) == {(press[0] > 1.0e-3, False)}
    assert list(half) == [(True, False)] * 3 + [(True, True)]
    assert list(through) == [(True, False)] * 3 + [(True, True)]              # through: the fourth check


def test_at_once_limit_grows_with_a_large_body() -> None:
    """GAP-E1 (rulings 2026-11-10 03:13Z, 03:27Z): the at-once limit is max(2 cm, 10 % of the smaller body's extent
    along the normal); the ground sets none. A 1 m tall box pressed 6 cm and 11 cm, a 10 cm box pressed 3 cm: the
    tall box at 6 cm (under its 10 cm) is PEN_WARN and fails only after the 0.1 s dwell (the 1 cm limit stays
    absolute); at 11 cm it fails past the hard limit (4th check); so does the 10 cm box (limit 2 cm) at 3 cm."""
    flags = {}
    for name, half_z, press in (("tall", 0.5, (6.0e-2, 11.0e-2)), ("small", HALF, (3.0e-2, 3.0e-2))):
        model, s0 = _boxes("cpu", half_z=half_z, press=press)
        s1, control = model.state(), model.control()
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
        monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), solvers=[solver])
        rows = []
        for k in range(4):                                                   # 30, 60, 90, 120 ms
            solver.step(s0, s1, control, None, DT)
            monitor.update(k, s0, s1, None, dt=0.03)
            rows.append([bool(int(f) & int(HealthFlags.PEN_FAIL)) for f in monitor.poll().flags])
        flags[name] = list(zip(*rows))
    assert flags["tall"][0] == (False, False, False, True)                   # 6 cm: the dwell
    assert flags["tall"][1] == (False, False, False, True)                   # 11 cm > 10 cm: hard, 4th check
    assert flags["small"][0] == (False, False, False, True)                  # 3 cm > 2 cm: hard, 4th check


def test_a_designed_pair_is_relaxed_only_between_its_own_shapes() -> None:
    """GAP-E1 (DC5's review): ``relax_designed_overlap`` relaxes the thin-part limit for contacts between two shapes
    of one marked pair only. Three 2 mm sheets pressed 2.5 mm into the ground (through: PEN_FAIL at once): unmarked
    fails; a sheet marked in a pair whose partner is another sheet still fails against the ground; a sheet marked
    in a pair with the ground reads PEN_WARN only."""
    from alloy.precision.health_pen import relax_designed_overlap

    model, s0 = _boxes("cpu", half_z=1.0e-3, press=(2.5e-3,) * 3)
    s1, control = model.state(), model.control()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), solvers=[solver])
    body = model.shape_body.numpy()
    sheets, ground = np.flatnonzero(body >= 0), int(np.flatnonzero(body < 0)[0])
    mark = monitor._pair_mark.numpy()                       # the rig's labels repeat per world: mark by shape id
    mark[sheets[:2]] = 1                                    # entry 0: two sheets, not the ground
    mark[[sheets[2], ground]] = 2                           # entry 1: a sheet and the ground
    monitor._pair_mark.assign(mark)
    with pytest.raises(ValueError, match="no shape label ends with"):
        relax_designed_overlap(monitor, [(("no_such_shape", "ground_plane"), "renamed part")])
    assert relax_designed_overlap(monitor, [(("ground_plane",), "by label")]) == [[ground]]
    for k in range(4):                                                       # the hard limit holds for 4 checks
        solver.step(s0, s1, control, None, DT)
        monitor.update(k, s0, s1, None, dt=0.03)
    flags = [int(f) for f in monitor.poll().flags]
    assert [bool(f & int(HealthFlags.PEN_FAIL)) for f in flags] == [True, True, False]
    assert all(f & int(HealthFlags.PEN_WARN) for f in flags)


def test_thousands_of_at_once_contacts_in_one_world_still_fail() -> None:
    """WP5's review: the per-world level is a max, not a count, so no number of deep contacts can wrap it. 1,024
    2 mm sheets in ONE world, each pressed through (2.5 mm): over 4,096 contacts past their at-once limit (a count of
    4,096 x 2^20 wrapped an int32 to 0 and read clean) -> PEN_FAIL once the hard limit has held four checks. The
    sheets are massless shapes of one body (1 kg, set inertia): 6 dofs, 2 s on the CPU (1,024 bodies took 3-5 min)."""
    side, pitch = 32, 0.25
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0e-3 - 2.5e-3), wp.quat_identity()), mass=1.0,
                            inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1), lock_inertia=True)
    massless = newton.ModelBuilder.ShapeConfig(density=0.0)
    for i in range(side * side):
        at = wp.transform(((i % side) * pitch, (i // side) * pitch, 0.0), wp.quat_identity())
        builder.add_shape_box(body, xform=at, hx=HALF, hy=HALF, hz=1.0e-3, cfg=massless)
    builder.add_ground_plane()
    model = builder.finalize(device="cpu")
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, nconmax=10000, njmax=40000)
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), solvers=[solver])
    for k in range(4):                                                       # the hard limit holds for 4 checks
        solver.step(s0, s1, control, None, DT)
        monitor.update(k, s0, s1, None, dt=0.03)
    report = monitor.poll()
    assert int(solver.mjw_data.nacon.numpy()[0]) >= 4096
    assert int(report.flags[0]) & int(HealthFlags.PEN_FAIL) and report.max_pen_m[0] == pytest.approx(2.5e-3, abs=2e-5)


def test_step_health_has_no_padding() -> None:
    """Main red 16:45Z Nov 10 (test_graphs_match_eager_bitwise, 3 of 3 runs): padding bytes of the record are never
    written, so eager runs left stack garbage in them (bytes 52-55, 108-111, 180-183 of 184 after ERR-1's three
    fields) and graph replay zeros. Every byte is a field now, so the record's bytes are its values."""
    import ctypes

    ct = StepHealth.ctype
    assert sum(ctypes.sizeof(t) for _, t in ct._fields_) == ctypes.sizeof(ct), [
        (n, getattr(ct, n).offset) for n, _ in ct._fields_]
    rec = wp.zeros(2, dtype=StepHealth, device="cpu").numpy()
    assert sum(rec.dtype.fields[n][0].itemsize for n in rec.dtype.names) == rec.dtype.itemsize


def test_one_hard_factor_for_every_contact_source() -> None:
    """WP5's review: Newton / cloth contacts (``health_pen.HARD_TIMES``) and MuJoCo's own (``native_pen_hard``)."""
    from alloy.precision.health_pen import HARD_TIMES

    assert float(HARD_TIMES.val if hasattr(HARD_TIMES, "val") else HARD_TIMES) == CheckPolicy().native_pen_hard


def test_relax_thin_part_rule() -> None:
    model, _ = _boxes("cpu")
    monitor = HealthMonitor(model, FLAGS)
    box = int(np.flatnonzero(model.shape_body.numpy() >= 0)[0])
    monitor.relax_thin_part_rule([box])
    thick = monitor._thick.numpy()
    assert np.isinf(thick[box]) and np.isfinite(thick[np.flatnonzero(model.shape_body.numpy() >= 0)[1]])


@pytest.mark.gpu
def test_native_mujoco_contacts_gpu_graph() -> None:
    if not wp.get_cuda_device_count():
        pytest.skip("needs CUDA")
    eager, ref = _native("cuda:0")
    graph, ref_g = _native("cuda:0", graph=True)
    worst = float(max(np.abs(eager.max_pen_m - ref).max(), np.abs(graph.max_pen_m - ref_g).max()))
    write_row("SC0-1 native-route max_pen_m vs mj_forward",
              f"2 worlds pressed {PRESS[0] * 1e3:g} / {PRESS[1] * 1e3:g} mm: eager {eager.max_pen_m.tolist()}, "
              f"graph replay {graph.max_pen_m.tolist()}, C {ref.tolist()} m; max |diff| {worst:.2e} m; "
              f"PEN_FAIL world 1 only", "pass" if worst <= 2.0e-6 else "fail",
              "PYTHONPATH=. pytest tests/precision/test_health_pen_measured.py (spot L4)")
    _check_native(eager, ref)
    _check_native(graph, ref_g)
