# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD: the ROD_VBD emitter and its analytic laws on CPU (quick). A11.torsion.rod_vbd is A11.torsion on
Newton's Cosserat rod with SolverVBDRod; the emitter checks pin the per-joint stiffnesses, masses and clamps the
laws rely on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.bench.runner import run_law
from alloy.scene import analytic_rods_vbd as V
from alloy.scene.analytic_rods import rod_card
from alloy.scene.emitters.rod_vbd import build_rod_vbd, damping_time, joint_stiffness


def _card(links: int = 11, zeta: float = 0.0):
    return V.vbd_card(rod_card(0.4, 0.02, links, radius=0.003, damping_ratio=zeta))


def test_emitter_layout_stiffness_mass_and_clamp():
    rod = _card(zeta=0.05)
    b = newton.ModelBuilder()
    h = build_rod_vbd(b, rod, [(0.0, 0.0, 1.0), (rod.length, 0.0, 1.0)], clamp="both")
    model = b.finalize(device="cpu")
    n = rod.links
    assert len(h.bodies) == n and len(h.joints) == n - 1
    ls = np.asarray(h.lengths)
    spacing = rod.length / (n - 1)
    assert ls[0] == pytest.approx(0.5 * spacing, rel=1e-6, abs=0) and ls.sum() == pytest.approx(rod.length, rel=1e-6,
                                                                                                    abs=0)
    k = joint_stiffness(rod, spacing)
    assert k["twist"] * spacing == pytest.approx(rod.GJ, rel=1e-12, abs=0)     # (N - 1) joints in series: L / GJ
    assert k["bend"] * spacing == pytest.approx(rod.EI, rel=1e-12, abs=0)
    dofs = np.concatenate([np.arange(6) + q for q in model.joint_qd_start.numpy()[list(h.joints)]])
    ke = model.joint_target_ke.numpy()[dofs].reshape(n - 1, 6)                # shear, shear, stretch, bend, bend, twist
    assert ke[:, 5] == pytest.approx(k["twist"], rel=1e-6, abs=0) and ke[:, 2] == pytest.approx(k["stretch"], rel=1e-6,
                                                                                                abs=0)
    kd = model.joint_target_kd.numpy()[dofs].reshape(n - 1, 6)
    assert kd[:, 3] == pytest.approx(k["bend"] * damping_time(rod), rel=1e-6, abs=0)
    m = model.body_mass.numpy()[list(h.bodies)]
    lin = rod.density * math.pi * rod.radius ** 2
    assert m.sum() == pytest.approx(lin * rod.length, rel=1e-6, abs=0)          # caps carry no mass
    ia = model.body_inertia.numpy()[list(h.bodies)][:, 2, 2]
    assert ia == pytest.approx(0.5 * m * rod.radius ** 2, rel=1e-6, abs=0)      # physical axial inertia, no armature
    flags = model.body_flags.numpy()[list(h.bodies)]
    kin = int(newton.BodyFlags.KINEMATIC)
    assert flags[0] & kin and flags[-1] & kin and not (flags[1:-1] & kin).any()


def test_emitter_refuses_bad_clamp_and_one_segment():
    with pytest.raises(ValueError, match="clamp"):
        build_rod_vbd(newton.ModelBuilder(), _card(), None, clamp="middle")
    with pytest.raises(ValueError, match="2 segments"):
        build_rod_vbd(newton.ModelBuilder(), _card(links=1), None)


def test_local_offset_maps_a_world_offset_into_the_body_frame():
    s = math.sqrt(0.5)
    assert V.local_offset((0.0, s, 0.0, s), (0.0, 0.0, 1.0)) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)


@pytest.mark.law
def test_a11_torsion_on_rod_vbd_quick_cpu():
    """T L / (G J) within 1 % at 0.5 and 1.5 rad on the emitted rod, physical inertia; the seed bend dies out."""
    res = run_law(V.LAW_TORSION_VBD, V.TorsionVBDTwin(), None, device="cpu", quick=True)
    assert res.verdict == "pass", res.verdicts
    mon = res.monitors
    assert mon["length_ratio"] == pytest.approx([1.0, 1.0], rel=1e-6, abs=0)
    assert max(mon["axis_drift_end_m"]) < 0.1 * min(mon["seed_drift_peak_m"])


def test_axial_twist_reads_the_relative_rotation_about_local_z():
    import warp as wp

    q0 = np.array([0, 0, 0, *wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)])
    rel = np.array(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 2.5))
    q1 = np.array([0, 0, 0, *wp.mul(wp.quat(*q0[3:7]), wp.quat(*rel))])
    assert V.axial_twist(q0, q1) == pytest.approx(2.5, abs=1e-6)


@pytest.mark.law
def test_xb4_twist_hook_on_the_gamma5_rod():
    """RERUN.md's `rerun.py twist` contract at its largest torque (2e-3 N m): static, theta G J / (T L) within 1e-4
    (the gap closes at 1.7 %; MuJoCo 4.0 reads 0.983). The default path RERUN.md names runs this route."""
    from alloy.scene.analytic_rods import torsion_case

    L, r = 0.5, 0.002
    lin = 1200.0 * math.pi * r ** 2
    EI = lin * 9.81 * L ** 3 / 5.0
    rod = {"length_m": L, "EI_Nm2": EI, "GJ_Nm2": EI / 1.3, "links": 30, "mass_per_length_kg_m": lin, "radius_m": r}
    out = torsion_case(rod, 2e-3, "cpu")
    assert out["ok"] and out["route"].startswith("ROD_VBD")
    assert out["theta_rad"] * (EI / 1.3) / (2e-3 * L) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.law
def test_a11_cantilever_on_rod_vbd_quick_cpu():
    """The clamped self-weight cantilever (gamma 0.5 and 5, 30 segments) settles within 1 % of the same discrete
    chain's exact static tip drop; measured -0.069 % / -0.038 % with the emitter's bend + twist damping."""
    res = run_law(V.LAW_CANTILEVER_VBD, V.CantileverVBDTwin(), None, device="cpu", quick=True)
    assert res.verdict == "pass", res.verdicts
    m = res.monitors
    errs = [a / b - 1.0 for a, b in zip(m["tip_drop_m"], m["chain_exact_m"], strict=True)]
    assert max(abs(e) for e in errs) < 2e-3, errs                  # MuJoCo 4.0's stiff-edge row: 8.1e-4


@pytest.mark.law
def test_a11_hanging_chain_and_catenary_on_rod_vbd_quick_cpu():
    """Pinned ends (world ball joints). The hanging chain's omega1 is within 2 % of 1.2024 sqrt(g / L) (measured
    0.25 % / 0.05 %). The catenary sags within 1 % of the inextensible catenary for the realised span (measured
    +0.09 % / +0.16 %; MuJoCo 4.0: 9.5e-3), its supports held to a micrometre."""
    hang = run_law(V.LAW_HANGING_VBD, V.HangingChainVBDTwin(), None, device="cpu", quick=True)
    assert hang.verdict == "pass", hang.verdicts
    cat = run_law(V.LAW_CATENARY_VBD, V.CatenaryVBDTwin(), None, device="cpu", quick=True)
    assert cat.verdict == "pass", cat.verdicts
    m = cat.monitors
    assert max(m["support_drop_m"]) < 1e-5
    assert max(abs(a / b - 1.0) for a, b in zip(m["sag_sim_m"], m["sag_ref_m"], strict=True)) < 5e-3


def test_pins_hold_the_end_nodes_and_keep_forward_kinematics():
    from alloy._vendor import newton

    rod = V.vbd_card(rod_card(0.4, 1e-6, 10, radius=0.002), chain=True)
    b = newton.ModelBuilder()
    h = build_rod_vbd(b, rod, [(0.0, 0.0, 1.0), (0.4, 0.0, 1.0)], pin="both")
    q_built = np.array([list(b.body_q[i]) for i in h.bodies])
    model = b.finalize(device="cpu")
    s0 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    assert np.abs(s0.body_q.numpy()[list(h.bodies)] - q_built).max() < 1e-5
    pins = [j for j, lab in enumerate(model.joint_label) if "_pin" in str(lab)]
    assert len(pins) == 2 and all(model.joint_parent.numpy()[j] == -1 for j in pins)


@pytest.mark.law
def test_a10_capstan_on_rod_vbd_quick_cpu():
    """WP2's capstan rope over the drum, contacts from Newton's pipeline, 4 iterations at CHAIN's 5e-4 s: the hold/slip
    threshold lands at 1.00 of exp(mu theta) at mu 0.2 and 0.4 (bracket 0.99-1.01; bound 2 %)."""
    res = run_law(V.LAW_CAPSTAN_VBD, V.CapstanVBDTwin(), None, device="cpu", quick=True)
    assert res.verdict == "pass", res.verdicts
    per = res.monitors["per_condition"]
    assert all(abs(c["threshold_ratio"] - 1.0) <= 0.01 for c in per.values()), per
