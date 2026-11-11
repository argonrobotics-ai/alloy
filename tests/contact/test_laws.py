# SPDX-License-Identifier: Apache-2.0
"""WP3 laws A7.hydroelastic, A8.clearance, A8.push on CPU: registration, LawSpec fields, closed forms, world layout
and the twins' scalar maths on synthetic logs. The stepped runs need CUDA (tests/contact/test_laws_gpu.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.bench import law as L
from alloy.bench.runner import run_law
from alloy.bench.twin import DeviceTwinLog
from alloy.contact import law_twins as T
from alloy.contact.law_clearance import (ClearancePlan, ContactRows, closed_form_tilt, contact_length, free_width,
                                         line_root, mid_shift, tilt_at_depth, tilted_pose)
from alloy.contact.laws import A7_HYDROELASTIC, A8_CLEARANCE, A8_PUSH, register
from alloy.contact.precision import size_l2
from alloy.testing.contact import PegHoleSpec, blind_hole_tris, cylinder_tris, to_newton_mesh

G = PegHoleSpec()
LAWS = (A7_HYDROELASTIC, A8_CLEARANCE, A8_PUSH)


# --------------------------------------------------------------------------------------- registration / specs ---

def test_discovery_registers_a7_a8() -> None:
    status = L.discover(("alloy.contact.laws",))
    assert status == {"alloy.contact.laws": None}
    assert register() is True                                  # re-registering equal specs is a no-op
    have = {s.id: f for s, f in L.registered_laws()}
    assert {"A7.hydroelastic", "A8.clearance", "A8.push"} <= set(have)
    assert have["A7.hydroelastic"].inner_factory is T.HydroelasticSphereTwin
    assert have["A8.push"].inner_factory is T.PegPushTwin
    missing = {e.prefix for e, _ in L.missing_laws("analytic")}
    assert not {"A7", "A8"} & missing


def test_lawspec_fields() -> None:
    for spec in LAWS:
        assert spec.producer == "WP3" and spec.suite == "analytic" and L.law_prefix(spec.id) in ("A7", "A8")
        assert set(spec.stat) == set(spec.metrics) and all(t.source for t in spec.metrics.values())
        assert spec.substeps == 1 and not spec.sleeping
    assert A7_HYDROELASTIC.solvers == ("mujoco",) and A8_PUSH.solvers == ("mujoco",)
    assert A8_CLEARANCE.solvers == ("none",)                  # kinematic: no solver steps
    a7 = A7_HYDROELASTIC.metrics
    assert a7["exponent_err"].abs == 0.05 and a7["prefactor_rel_err"].abs == 0.10 and a7["settle_rel_err"].abs == 0.01
    a8 = A8_CLEARANCE.metrics
    assert a8["gap_err_m"].abs == 1e-6 and a8["tilt_err_deg"].abs == 0.01
    assert A8_PUSH.metrics["pen_max_m"].abs == pytest.approx(5e-5)      # D2 for the 0.1 mm radial clearance
    assert A7_HYDROELASTIC.dt == pytest.approx(0.02 / 2**5) and A8_PUSH.dt == pytest.approx(3.125e-4)
    assert A8_PUSH.duration_s == pytest.approx(0.7) and A8_PUSH.quick["duration_s"] == pytest.approx(0.7)
    assert A7_HYDROELASTIC.quick["worlds"] == len(T.DELTA_OVER_R)
    assert A7_HYDROELASTIC.quick["duration_s"] == A7_HYDROELASTIC.duration_s == T.A7_DURATION_S


# ---------------------------------------------------------------------------------------------------- A7 ---

def test_hydro_closed_form_against_cap_integral() -> None:
    assert T.series_stiffness(2e10, 2e10) == pytest.approx(1e10)
    assert T.series_stiffness(1e10, 3e10) == pytest.approx(7.5e9) and T.series_stiffness(0.0, 0.0) == 0.0
    R, k = 0.01, 1e10
    for frac in T.DELTA_OVER_R:
        delta = frac * R
        a = math.sqrt(2 * R * delta - delta * delta)
        r = np.linspace(0.0, a, 200001)
        h = delta - (R - np.sqrt(R * R - r * r))               # vertical columns under the sphere cap
        f = np.trapezoid(k * h * 2 * np.pi * r, r)
        assert f == pytest.approx(math.pi * k * delta**2 * (R - delta / 3), rel=1e-6)
        u = (2 * R * delta - delta**2 - r * r) / (4 * R - 2 * delta)   # iso-surface sdf_sphere = sdf_plate = -u
        f_iso = np.trapezoid(2 * k * u * 2 * np.pi * r, r)            # p = kh·u = 2·k_eff·u (equal kh)
        assert f_iso == pytest.approx(math.pi * k * delta**2 * (R - delta / 2), rel=1e-6)
        small = T.hydro_sphere_force(R, k, delta)
        assert 0.0 <= small / f_iso - 1.0 <= frac / 2 / (1 - frac / 2) + 1e-9   # ≤ 2.6 % over the law's range
    d = np.array(T.DELTA_OVER_R) * R
    assert T.power_law_exponent(d, T.hydro_sphere_force(R, k, d)) == pytest.approx(2.0, abs=1e-9)
    assert math.isnan(T.power_law_exponent(np.array([1e-4, -1e-4]), np.array([1.0, 2.0])))


def _a7_log(twin: T.HydroelasticSphereTwin, delta: np.ndarray, creep: float = 0.0, steps: int = 400
            ) -> DeviceTwinLog:
    """Synthetic run: δ_w(t) = delta_w·(1 + creep·t) over the law's duration."""
    log = DeviceTwinLog(len(delta), steps, T.A7_DURATION_S / steps, "cpu")
    depth = delta[:, None] * (1.0 + creep * log.times()[None, :])
    log.channel("z").assign((twin.geometry.radius - depth).astype(np.float32))
    return log


def test_a7_scalars_judge_the_closed_form() -> None:
    twin = T.HydroelasticSphereTwin()
    twin.worlds = list(twin.conditions(quick=True))
    exact = twin.param("delta_closed")
    out = twin.scalars(_a7_log(twin, exact))
    assert out["exponent_err"][0] < 1e-3 and np.all(out["prefactor_rel_err"] < 1e-3)
    assert np.all(out["settle_rel_err"] < 1e-6)
    assert set(twin.monitors["delta_history_m"]) == {"0.5s", "1s", "1.5s", "2s"}
    out = twin.scalars(_a7_log(twin, 1.2 * exact))          # 20 % deeper: prefactor 1/1.44, fails the 10 % bound
    assert np.all(out["prefactor_rel_err"] > 0.10) and out["exponent_err"][0] < 1e-3
    out = twin.scalars(_a7_log(twin, exact, creep=0.5))     # still sinking 50 %/s: 1.3 % over 0.05 s, not settled
    assert np.all(out["settle_rel_err"] > 0.01)


def test_a7_world_layout_cpu() -> None:
    twin = T.HydroelasticSphereTwin()
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    for cond in twin.conditions(quick=True)[:2]:
        b.begin_world(gravity=twin.world_gravity(cond))
        twin.build_world(b, cond, {})
        b.end_world()
    assert b.shape_count == 4 and b.body_count == 2
    assert [int(t) for t in b.shape_type[:2]] == [int(newton.GeoType.BOX), int(newton.GeoType.SPHERE)]
    assert [int(j) for j in b.joint_type] == [int(newton.JointType.PRISMATIC)] * 2      # guided indenter
    condim = b.custom_attributes["mujoco:condim"].values
    assert all(int(condim[s]) == 1 for s in range(4))                                    # frictionless rows
    assert all(f & int(newton.ShapeFlags.HYDROELASTIC) for f in b.shape_flags)
    assert b.shape_material_kh[:2] == [twin.geometry.kh] * 2
    assert {(0, 1), (2, 3)} <= {tuple(sorted(p)) for p in b.shape_collision_filter_pairs}
    spec = twin.pair_spec()
    assert (spec.shape_a, spec.shape_b, spec.mode, spec.ccd) == (0, 1, "hydroelastic", "off")
    assert twin.layer_kwargs()["hydro_iso_per_world"] < 419_430 / 8      # all quick worlds fit one pipeline
    assert twin.settings_record("mujoco")["precision_pair"]["kh"] == twin.geometry.kh


# ---------------------------------------------------------------------------------------------------- A8 ---

def test_tilt_closed_form() -> None:
    d, D = G.peg_d, G.bore_d
    assert math.degrees(closed_form_tilt(d, D, 10e-3)) == pytest.approx(1.1600876, abs=1e-6)   # CT-5's value
    th = tilt_at_depth(d, D, 10e-3)
    assert free_width(d, D, 10e-3, th) == pytest.approx(0.0, abs=1e-15)
    assert closed_form_tilt(d, D, contact_length(d, 10e-3, th)) == pytest.approx(th, abs=1e-12)
    with pytest.raises(ValueError):
        closed_form_tilt(d, d - 1e-4, 10e-3)


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _radial_excess(phi: float, theta: float, shift: float, depth: float, n: int = 1440) -> float:
    """Largest radial excess over the bore radius of the exact cylinder's bottom rim and top-plane crossing."""
    pose = tilted_pose(G, phi, theta, shift, depth).astype(np.float64)
    c, q = pose[:3], pose[3:]
    a, e1, e2 = (_quat_rot(q, np.eye(3)[i]) for i in (2, 0, 1))
    psi = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = 0.5 * G.peg_d * (np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2)
    bottom = c - 0.5 * G.peg_len * a + ring
    top = c + ((G.height - c[2] - ring[:, 2]) / a[2])[:, None] * a + ring
    pts = np.vstack([bottom, top])
    return float(np.max(np.hypot(pts[:, 0], pts[:, 1])) - 0.5 * G.bore_d)


@pytest.mark.parametrize("phi", [0.0, 1.0, 4.0])
def test_tilted_pose_touches_at_the_closed_form(phi: float) -> None:
    depth = 10e-3
    th = tilt_at_depth(G.peg_d, G.bore_d, depth)
    assert abs(_radial_excess(phi, th, mid_shift(G, th, depth), depth)) < 1e-9
    assert _radial_excess(phi, 0.999 * th, mid_shift(G, 0.999 * th, depth), depth) < -5e-8
    shifts = np.linspace(-1e-4, 2e-4, 601)
    assert min(_radial_excess(phi, 1.002 * th, s, depth, 360) for s in shifts) > 0.0   # no shift fits above θ_max


def test_clearance_plan_layout() -> None:
    plan = ClearancePlan.for_run(quick=True)
    poses, phi = plan.poses()
    n0 = len(plan.centred_depths)
    assert poses.shape == (n0 + plan.directions * plan.tilts, 7) and phi.shape == (poses.shape[0],)
    assert np.allclose(poses[:n0, :2], 0.0) and np.allclose(poses[:n0, 3:], [0, 0, 0, 1])
    assert np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-6)
    assert plan.thetas()[0] < plan.theta_closed < plan.thetas()[-1]
    assert ClearancePlan.for_run(quick=False).directions == 36


def test_side_gaps_and_line_root() -> None:
    mid = np.array([[1e-3, 0, 0], [-1e-3, 0, 0], [-2e-3, 0, 0], [0, 1e-3, 0], [0, -1e-3, 0]], float)
    rows = ContactRows(np.array([0, 0, 0, 1, 1]), np.array([2e-5, 3e-5, 1e-5, -4e-6, 7e-6]), mid)
    plus, minus = rows.side_gaps(np.array([0.0, math.pi / 2]))
    assert plus.tolist() == [2e-5, -4e-6] and minus.tolist() == [1e-5, 7e-6]
    assert rows.world_min(3).tolist() == [1e-5, -4e-6, math.inf]
    x = np.linspace(0.0, 1.0, 9)
    root, rms = line_root(x, 0.4 - x)
    assert root == pytest.approx(0.4) and rms < 1e-12
    assert math.isnan(line_root(x, x - 0.4)[0]) and math.isnan(line_root(x[:2], x[:2])[0])


def test_push_plan_pair_and_dt() -> None:
    plan = T.PushPlan()
    assert [plan.scale(t) for t in (0.0, 0.1, 0.35, 0.6, 0.7)] == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    spec = T.push_pair()
    assert (spec.mode, spec.shape_a, spec.shape_b) == ("sdf", 0, 1)
    assert spec.pair_stiffness == pytest.approx(2e6) and spec.resolved_max_penetration() == pytest.approx(5e-5)
    m = T.peg_mass()
    assert m == pytest.approx(7800 * math.pi * 0.006**2 * 0.03, rel=1e-3)
    assert T.push_dt() == size_l2(spec, m_eff=m, design_force=50.0, contact_speed=0.05, control_dt=0.02).dt
    assert T.PegPushTwin().conditions(quick=True)[0].params["force"] == 50.0


def test_push_world_layout_cpu() -> None:
    """``add_peg_hole`` without baked SDFs (sdf mode allows it on the builder): template order and filtering."""
    meshes = (to_newton_mesh(*blind_hole_tris(G), compute_inertia=False),
              to_newton_mesh(*cylinder_tris(0.5 * G.peg_d, 0.5 * G.peg_len, G.segments)))
    b = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, gap=5e-4, margin=0.0)
    for w in range(2):
        b.begin_world()
        hole, peg = T.add_peg_hole(b, meshes, cfg, T.peg_start_xform(G, 5e-5, 0.1 * w, 1e-5 * w), 0.99)
        b.end_world()
        assert (hole, peg) == (2 * w, 2 * w + 1)
    assert b.shape_body[:4] == [-1, 0, -1, 1]
    assert {(0, 1), (2, 3)} <= {tuple(sorted(p)) for p in b.shape_collision_filter_pairs}
    solimp = b.custom_attributes["mujoco:geom_solimp"].values
    assert all(abs(float(solimp[s][0]) - 0.99) < 1e-6 for s in range(4))


@pytest.mark.parametrize("spec", LAWS, ids=lambda s: s.id)
def test_laws_not_run_without_cuda(spec) -> None:
    """Texture SDFs need CUDA: on CPU each law is ``not_run`` with the reason (SB-12 leaves it out), never a pass."""
    factory = {s.id: f for s, f in L.registered_laws()}[spec.id]
    res = run_law(spec, factory(), None, device="cpu", solver=spec.solvers[0], quick=True)
    assert res.verdict == "not_run" and "CUDA" in res.monitors["not_run_reason"], res.monitors
    assert {v.name for v in res.verdicts} == set(spec.metrics)
