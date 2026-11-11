# SPDX-License-Identifier: Apache-2.0
"""R1 rig twin pieces on CPU: synthetic Whitney model and dataset, metric extraction, judging, holder drive, LawSpec."""

from __future__ import annotations

import dataclasses
import math

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.bench.law import register_law, registered_laws
from alloy.bench.rigjudge import judge_rig
from alloy.bench.types import RigCard
from alloy.contact import rig_synth as RS
from alloy.contact.precision import size_l2
from alloy.contact.rig_holder import HolderDrive, HolderSetup, PegInertia, holder_damping
from alloy.contact.rig_r1 import (
    R1Geometry, R1Holder, R1Protocol, chamfered_hole_tris, curve_error, force_profile, insertion_metrics,
    r1_conditions,
)
from alloy.contact.rig_twins import R1_LAW, R1Twin, r1_pair
from alloy.core.errors import LawError
from alloy.testing.contact import blind_hole_tris, cylinder_tris, signed_volume, to_newton_mesh

G, HD, PR = R1Geometry(), R1Holder(), R1Protocol()
Z = np.arange(0.0, PR.z_end, 2e-5)


@pytest.fixture(scope="module")
def quick_data():
    return RS.r1_dataset(seed=0, quick=True)


def _pair(deg: float, off_mm: float = 0.05, holder: R1Holder = HD, mu: float = 0.35) -> RS.WhitneyPair:
    return RS.WhitneyPair.build(G, holder, math.radians(deg), off_mm * 1e-3, G.chamfer, mu)


# ----------------------------------------------------------------------------------------- geometry, grid ---
def test_conditions_grid() -> None:
    full, quick = r1_conditions(False), r1_conditions(True)
    assert len(full) == 24 and len({c.id for c in full}) == 24
    assert [c.id for c in quick] == ["t1_o0.05_ch", "t2_o0.05_ch"]
    assert quick[1].params == {"tilt": math.radians(2.0), "offset": 5e-5, "chamfer": G.chamfer}


def test_chamfered_hole_closed_and_outward() -> None:
    spec = G.hole_spec()
    verts, tris = chamfered_hole_tris(spec, G.chamfer)
    edges = {(int(a), int(b)) for t in tris for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    assert len(edges) == 3 * len(tris) and all((b, a) in edges for a, b in edges)   # closed, consistently wound
    plain = signed_volume(*blind_hole_tris(spec))
    rb, w = 0.5 * spec.bore_d, G.chamfer
    removed = math.pi * w / 3.0 * ((rb + w) ** 2 + (rb + w) * rb + rb ** 2) - math.pi * rb * rb * w
    assert plain - signed_volume(verts, tris) == pytest.approx(removed, rel=0.01)
    v0, t0 = chamfered_hole_tris(spec, 0.0)
    assert np.array_equal(t0, blind_hole_tris(spec)[1])
    with pytest.raises(ValueError):
        chamfered_hole_tris(spec, 0.02)


# ------------------------------------------------------------------------------------- Whitney model ---
def test_model_phase_sequence_and_two_point_geometry() -> None:
    p = _pair(2.0)
    P, phase = RS.quasi_static_force(p, Z)
    first = [int(np.argmax(phase == k)) for k in (RS.ONE_A, RS.TWO)]
    assert (phase == RS.FREE).any() and 0 < first[0] < first[1] and not (phase == RS.JAM).any()
    assert np.all(np.diff(phase[: first[1] + 1]) >= 0)          # free -> one-point -> two-point
    assert np.all(P >= 0.0) and np.all(P[phase == RS.FREE] == 0.0)
    h = Z[phase == RS.TWO] - p.edge
    two = RS._solve_two(p, h)
    engaged = (h + 0.5 * G.peg_d * np.sin(two.th)) / np.cos(two.th)   # axial length from the tip face to A
    lhs = G.peg_d * np.cos(two.th) + engaged * np.sin(two.th)           # A8: d cos θ + l sin θ = D
    assert np.max(np.abs(lhs - G.bore_d)) < 1e-12
    z_peak = Z[int(np.argmax(P))] - p.edge
    assert z_peak == pytest.approx(2.0 * G.clearance / math.radians(2.0), rel=0.25)   # peak of order 2c/θ0


def test_model_equilibrium_residuals() -> None:
    p = _pair(2.0)
    h = np.array([6e-3, 9e-3, 14e-3])
    s = RS._solve_two(p, h)
    fx, m = RS._wrench(p, s.x_t, s.th)
    ax, ay = RS._dirs(p, s.th)
    assert np.allclose(fx + s.fa * ax + s.fb, 0.0, atol=1e-9)
    assert np.allclose(-s.P + s.fa * ay + p.mu * s.fb, 0.0, atol=1e-9)
    assert np.allclose(RS._torque(p, s.th, fx, m, s.P, s.fa, s.fb, s.x_t, h), 0.0, atol=1e-10)
    a = RS._solve_one_a(p, h)
    assert np.allclose(RS._gap_a(p, a.x_t, a.th, h), 0.0, atol=1e-12)


def test_model_aligned_peg_is_free() -> None:
    P, phase = RS.quasi_static_force(_pair(0.0, 0.0), Z)
    assert np.all(phase == RS.FREE) and np.all(P == 0.0)
    with pytest.raises(ValueError):
        _pair(1.0, 0.2)                                          # offset beyond c/2: outside the model


def test_model_jam_and_stop() -> None:
    far = dataclasses.replace(HD, l_g=0.25)                      # compliance centre far above the tip: wedges
    p = _pair(2.0, holder=far, mu=0.5)
    _, phase = RS.quasi_static_force(p, Z)
    assert (phase == RS.JAM).any()
    cfg = RS.SynthConfig(holder=far)
    s = RS.trial_series(p, cfg, np.random.default_rng(0))
    m = insertion_metrics(s["depth"], s["force"], PR)
    assert m.jam == 1.0 and m.scalars() == {"jam": 1.0}
    assert m.jam_depth == pytest.approx(Z[int(np.argmax(phase == RS.JAM))], abs=0.05e-3)
    assert s["force"].max() == pytest.approx(PR.f_stop, abs=0.3)


# ------------------------------------------------------------------------------------------ dataset ---
def test_dataset_deterministic_and_split(quick_data) -> None:
    again, other = RS.r1_dataset(seed=0, quick=True), RS.r1_dataset(seed=1, quick=True)
    assert again.sha256 == quick_data.sha256 and other.sha256 != quick_data.sha256
    assert np.array_equal(again.trials[3].series["force"], quick_data.trials[3].series["force"])
    for cond in quick_data.conditions:
        trials = [t for t in quick_data.trials if t.condition_id == cond.id]
        assert [t.split for t in trials].count("calib") == 5 and len(trials) == 10
        assert list(trials[0].series)[0] == "t" and set(trials[0].series) == {"t", "depth", "force"}
    t1, t2 = (quick_data.split("valid")[i].scalars for i in (0, 5))
    assert set(t1) == {"sliding_force", "jam"} and set(t2) == {"sliding_force", "peak_force", "peak_depth", "jam"}
    assert quick_data.card.protocol_version.startswith("synthetic")
    assert RigCard(**dataclasses.asdict(quick_data.card)) == quick_data.card


def test_dataset_metrics_match_model(quick_data) -> None:
    vals = [t.scalars for t in quick_data.trials if t.condition_id == "t2_o0.05_ch"]
    P, _ = RS.quasi_static_force(_pair(2.0), Z)
    assert np.mean([v["peak_force"] for v in vals]) == pytest.approx(P.max(), rel=0.03)
    win = (Z >= PR.window[0]) & (Z <= PR.window[1])
    assert np.mean([v["sliding_force"] for v in vals]) == pytest.approx(P[win].mean(), rel=0.03)
    assert np.std([v["peak_depth"] for v in vals], ddof=1) < 0.03e-3


# ------------------------------------------------------------------------------- metric extraction ---
def _series(force_of_z, noise: float = 0.0, seed: int = 0, z_stop: float | None = None):
    t = np.arange(0.0, PR.duration(PR.speed), 0.01)
    z = np.clip(PR.speed * (t - PR.t_settle), 0.0, PR.stroke) - PR.approach
    if z_stop is not None:
        z = np.minimum(z, z_stop)
    f = force_of_z(z) + np.random.default_rng(seed).normal(0.0, noise, t.shape)
    return z, f


def _gauss_peak(z):
    return 2.0 + 5.0 * np.exp(-(((z - 9e-3) / 2e-3) ** 2)) * (z > 0)


def test_metrics_of_a_known_curve() -> None:
    m = insertion_metrics(*_series(_gauss_peak), PR)
    assert m.jam == 0.0 and m.sliding_force == pytest.approx(2.0, abs=1e-3)
    assert m.peak_force == pytest.approx(7.0, abs=0.01) and m.peak_depth == pytest.approx(9e-3, abs=1e-5)
    depths = [insertion_metrics(*_series(_gauss_peak, 0.05, s), PR).peak_depth for s in range(8)]
    assert np.std(depths) < 0.03e-3 and abs(np.mean(depths) - 9e-3) < 0.03e-3


def test_metrics_flat_jam_and_short_stroke() -> None:
    flat = insertion_metrics(*_series(lambda z: np.full_like(z, 1.5)), PR)
    assert flat.scalars() == {"sliding_force": pytest.approx(1.5), "jam": 0.0}
    rising = insertion_metrics(*_series(lambda z: 100.0 * np.maximum(z, 0.0)), PR)
    assert math.isnan(rising.peak_force)                          # a monotone rise has no two-point peak
    limit = insertion_metrics(*_series(lambda z: np.where(z > 8e-3, PR.f_stop, 0.5)), PR)
    assert limit.jam == 1.0 and limit.scalars() == {"jam": 1.0}
    short = insertion_metrics(*_series(lambda z: np.ones_like(z), z_stop=12e-3), PR)
    assert short.jam == 1.0 and short.jam_depth == pytest.approx(12e-3)


def test_profile_and_curve_error() -> None:
    z = np.array([1e-3, 1.01e-3, 0.99e-3, 5e-3, -1e-3, np.nan])
    f = np.array([1.0, 3.0, 2.0, 4.0, 9.0, 9.0])
    grid, prof = force_profile(z, f, PR)
    assert grid.shape == prof.shape and np.isnan(prof[int(3e-3 / PR.bin_m)])
    assert prof[int(1e-3 / PR.bin_m)] == pytest.approx(2.0)      # non-monotone samples average per bin
    real = np.array([1.0, 2.0, 4.0, np.nan])
    assert curve_error(real, real) == 0.0
    assert curve_error(real * 1.1, real) == pytest.approx(math.sqrt(np.mean((0.1 * real[:3]) ** 2)) / 4.0)
    assert math.isnan(curve_error(real, np.full(4, np.nan)))


# ---------------------------------------------------------------------------------------- judging ---
class _Units:
    units = R1Twin.units


def _sim_values(data, scale: dict[str, float]) -> tuple[dict[str, np.ndarray], list[str]]:
    conds = [c.id for c in data.conditions for _ in range(4)]
    out: dict[str, list[float]] = {m: [] for m in R1_LAW.metrics}
    for cid in conds:
        valid = [t.scalars for t in data.split("valid") if t.condition_id == cid]
        for m in R1_LAW.metrics:
            vals = [v[m] for v in valid if m in v]
            out[m].append(np.mean(vals) * scale.get(m, 1.0) if vals else math.nan)
    return {m: np.asarray(v) for m, v in out.items()}, conds


def test_judging_pass_fail_and_not_run(quick_data) -> None:
    values, conds = _sim_values(quick_data, {})
    verdicts, calib = judge_rig(R1_LAW, _Units(), values, quick_data, conds, None)
    got = {v.name: v.verdict for v in verdicts}
    assert got["peak_force@t1_o0.05_ch"] == "not_run" and got["peak_depth@t1_o0.05_ch"] == "not_run"
    judged = {k: v for k, v in got.items() if v != "not_run"}
    assert len(judged) == 6 and set(judged.values()) == {"pass"}, got     # TUR >= 3 for every judged metric
    assert calib["sliding_force@t2_o0.05_ch"] < 0.2
    off, _ = _sim_values(quick_data, {"sliding_force": 1.25, "peak_force": 0.85, "peak_depth": 1.02})
    off["jam"] = np.ones_like(off["jam"])
    bad = {v.name: v.verdict for v in judge_rig(R1_LAW, _Units(), off, quick_data, conds, None)[0]}
    assert all(bad[f"{m}@t2_o0.05_ch"] == "fail" for m in R1_LAW.metrics), bad
    nan, _ = _sim_values(quick_data, {})
    nan["sliding_force"][0] = math.nan
    assert {v.name: v.verdict for v in judge_rig(R1_LAW, _Units(), nan, quick_data, conds, None)[0]}[
        "sliding_force@t1_o0.05_ch"] == "fail"


# ------------------------------------------------------------------------------------- law, twin, drive ---
def test_law_spec_registered_on_import_and_dt_is_l2() -> None:
    import alloy.contact.laws  # noqa: F401,PLC0415  (registers R1 with A7/A8)
    assert {spec.id: f for spec, f in registered_laws("rig")}["R1.peg_insertion"] is R1Twin
    register_law(R1_LAW, R1Twin)                                  # idempotent; the registry is left as found
    l2 = size_l2(r1_pair(G), m_eff=G.peg_mass, design_force=50.0, contact_speed=0.05, control_dt=0.02)
    assert R1_LAW.dt == l2.dt and R1_LAW.solvers == ("mujoco",)
    assert R1_LAW.quick["duration_s"] == pytest.approx(PR.duration(R1_LAW.quick["stage_speed"]))
    assert R1_LAW.duration_s == pytest.approx(PR.duration(PR.speed))


def test_twin_card_and_worlds(quick_data) -> None:
    twin = R1Twin(device="cpu")
    twin.bind_calib(quick_data.card, quick_data.split("calib"))
    assert twin.mu == 0.35 and twin.holder == HD and twin.geometry.peg_d == G.peg_d
    cond = quick_data.conditions[1]
    a, b = twin.expand(cond, True, 4), twin.expand(cond, True, 4)
    assert a == b and a[0] == {"tilt": cond.params["tilt"], "offset": cond.params["offset"]}
    assert all(abs(s["tilt"] - cond.params["tilt"]) < 5 * HD.tilt_sigma for s in a)
    assert twin.speed == PR.quick_speed and twin.record_stride(0) == 6
    bad = dataclasses.replace(quick_data.card, holder={**quick_data.card.holder, "stage_speed": 2e-3})
    with pytest.raises(LawError):
        twin.bind_calib(bad, ())


def test_holder_damping_limits() -> None:
    peg = PegInertia(G.peg_mass, 3.26e-6, 2.8e-7, 0.03)
    d = holder_damping(HD, peg, R1_LAW.dt)
    assert all(v <= 1.0 + 1e-9 for v in d.margins.values()) and min(d.c_lat, d.c_ax, d.c_rot, d.c_spin) > 0.0
    with pytest.raises(LawError):
        holder_damping(dataclasses.replace(HD, k_z=1e6), peg, R1_LAW.dt)


def _free_pegs(tilts: tuple[float, ...]) -> newton.Model:
    b = newton.ModelBuilder()
    mesh = to_newton_mesh(*cylinder_tris(0.5 * G.peg_d, 0.5 * G.peg_len, 32))
    for tilt in tilts:
        b.begin_world()
        axis = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
        c = np.array([5e-5, 0.0, G.height + PR.approach]) + 0.5 * G.peg_len * axis
        body = b.add_body(xform=wp.transform(wp.vec3(*c), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)))
        b.add_shape_mesh(body, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(density=G.density,
                                                                                has_shape_collision=False))
        b.end_world()
    return b.finalize(device="cpu")


def _run_drive(proto: R1Protocol, steps: int = 1280):
    from alloy.bench.twin import DeviceTwinLog

    model, dt, speed = _free_pegs((0.0, math.radians(2.0))), R1_LAW.dt, PR.quick_speed
    drive = HolderDrive(model, np.array([0, 1]), HolderSetup(HD, proto, speed, dt, G.peg_len, G.height))
    solver = newton.solvers.SolverSemiImplicit(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    log = DeviceTwinLog(2, steps, dt, "cpu", stride=8)
    drive.record(s0, 0.0, 0, log)
    for k in range(1, steps + 1):
        s0.clear_forces()
        drive.apply(s0, (k - 1) * dt)
        solver.step(s0, s1, ctrl, None, dt)
        s0, s1 = s1, s0
        drive.record(s0, k * dt, log.column(k), log)
    return drive, log, speed


def test_holder_drive_tracks_stage_without_contact() -> None:
    proto = dataclasses.replace(PR, t_settle=0.02)
    drive, log, speed = _run_drive(proto)
    expect = np.clip(speed * (log.times() - proto.t_settle), 0.0, proto.stroke) - proto.approach
    assert np.max(np.abs(log.series("depth") - expect)) < 2e-6
    assert np.max(np.abs(log.series("force"))) < 0.05                # no contact: tared force stays ~0
    tilt = np.degrees(log.series("tilt"))
    assert np.max(np.abs(tilt[1] - 2.0)) < 2e-3 and np.max(tilt[0]) < 1e-3   # hanging equilibrium holds the tilt
    assert not drive.stopped().any()


def test_holder_drive_latches_on_force_limit() -> None:
    drive, log, _ = _run_drive(dataclasses.replace(PR, t_settle=0.02, f_stop=-1.0), steps=400)
    assert drive.stopped().all()                                      # latched at t = 0: the stage never moves
    assert np.max(np.abs(log.series("depth") + PR.approach)) < 2e-6
