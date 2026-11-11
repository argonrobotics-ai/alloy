# SPDX-License-Identifier: Apache-2.0
"""WP2's laws through the WP6 runner on CPU (quick): A11 (hanging chain, catenary, cantilever, torsion), A12
(buckling), A10 (capstan on the routed rope), R9 (rope twin on its synthetic dataset); plus the reference solutions
they rest on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench.runner import run_law
from alloy.route import rodref, roperef
from alloy.scene import analytic_buckling as AB
from alloy.scene import analytic_capstan as AC
from alloy.scene import analytic_rods as AR
from alloy.scene import analytic_torsion as AT
from alloy.testing import scene as S

LIN = 1200.0 * math.pi * 0.002**2


def _record(res) -> None:
    for v in res.verdicts:
        S.record(f"{res.law}.{v.name}", float(v.error), v.verdict, bound=float(v.bound), solver=res.solver,
                 device=res.device, wall_s=round(res.wall_s, 1))


# --- references ----------------------------------------------------------------------------------------------------

def test_supported_sag_limp_limit_is_the_catenary():
    assert roperef.supported_sag(1.0, 0.8, LIN, 1e-9) == pytest.approx(rodref.catenary_sag(0.8, 1.0), rel=1e-5)


def test_supported_sag_stiff_rope_sags_more_than_the_catenary():
    assert roperef.supported_sag(1.0, 0.8, LIN, 1e-2) > rodref.catenary_sag(0.8, 1.0)


def test_hanging_mode_limp_limit_and_damping_projection():
    m = roperef.hanging_modes(1.0, LIN, 1e-9)
    assert m.omega == pytest.approx(rodref.hanging_chain_w1(1.0), rel=1e-5) and m.zeta == 0.0
    damped = roperef.hanging_modes(1.0, LIN, 1e-9, pivot_damping=1e-4)
    assert damped.zeta > 0.0 and damped.omega == m.omega


def test_buckling_and_capstan_thresholds():
    f = np.array(AB.GRID)
    assert AB.critical_factor(f, f > 1.01) == pytest.approx(1.0125)
    assert math.isnan(AB.critical_factor(f, f < 1.0))                  # non-monotone -> NaN
    r = np.array(AC.RATIOS)
    assert AC.threshold_ratio(r, r > 1.0) == pytest.approx(1.0)
    assert math.isnan(AC.threshold_ratio(r, np.ones_like(r, bool)))   # all slip: not bracketed


def test_capstan_rope_is_the_routed_chain():
    """A10's rope is what the router emits for it: the CHAIN link rule, links no shorter than a diameter (no bead
    chain), hinges on both leg/drum junctions, and a thin-rope R/ρ that keeps exp(μθ·R/ρ) inside the ±1 % bracket."""
    from alloy.route import checks as ck
    from alloy.route.types import RouterConfig

    cfg, w = RouterConfig(), AC.wrap()
    c = ck.rod_bending_length(AC.DENSITY, math.pi * AC.ROPE_R**2, AC.EI)
    assert w.links == ck.chain_links(w.length, c, AC.ROPE_R)[0] and w.links <= cfg.chain_links_max
    assert w.link >= 2.0 * AC.ROPE_R and w.arc == w.links - 2 - 2 * AC.LEG_LINKS
    assert w.leg == pytest.approx((AC.LEG_LINKS + 0.5) * w.link)
    pts = np.array(AC.wrap_points())
    assert np.allclose(np.hypot(pts[20:20 + w.arc + 1, 0], pts[20:20 + w.arc + 1, 2]), w.rw)   # hinges on rw
    assert AC.DRUM_R / w.rw > 0.99
    for mu in (0.2, 0.4):
        assert math.exp(-mu * math.pi * (1.0 - AC.DRUM_R / w.rw)) > AC.RATIOS[2]   # above the 0.99 hold ratio


@pytest.mark.parametrize("pin", [False, True], ids=["free_root", "pinned_root"])
def test_chain_armature_floor_on_every_rotational_dof(pin):
    """The route's CHAIN rule (emitters.chain.ARMATURE_FLOOR): every rotational dof, the root's included, carries at
    least the floor whatever the integrator; the free root's linear dofs carry none."""
    from alloy._vendor import newton

    from alloy.scene.emitters import chain as CH

    for integ in ("implicitfast", "discrete"):
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        rod = AR.rod_card(0.5, 3e-6, 12, radius=1e-3, density=7850.0, integrator=integ)
        h = CH.build_chain(b, rod, None, CH.ChainOptions(capability="mujoco", dt=5e-4, pin=pin))
        for j in h.joints:
            q0, (lin, ang) = b.joint_qd_start[j], b.joint_dof_dim[j]
            assert all(b.joint_armature[d] >= CH.ARMATURE_FLOOR for d in range(q0 + lin, q0 + lin + ang)), (integ, j)
            assert all(b.joint_armature[d] == 0.0 for d in range(q0, q0 + lin)), (integ, j)


def test_capstan_slip_speed_is_a_fraction_of_the_smallest_overload():
    m = AC.rope_mass()
    for mu in (0.2, 0.4):
        accel = 0.01 * AC.T_IN * math.exp(mu * math.pi) / m
        assert AC.slip_speed(mu, 0.2) == pytest.approx(AC.SLIP_FRACTION * accel * (0.2 - AC.EASE_S))


def test_rope_fits_invert_their_references():
    from alloy.scene.twins import rope as RP

    p = RP.ROPES["usb_cable"]
    drop = rodref.elastica_tip_drop(RP.CANT_L, p["lin_mass"], p["EI"])
    assert RP.fit_ei(drop, RP.CANT_L, p["lin_mass"]) == pytest.approx(p["EI"], rel=1e-6)
    t = np.linspace(0.0, 10.0, 5001)
    x = np.exp(-0.1 * t / 1.6) * np.cos(2 * math.pi * t / 1.6)
    assert RP.peak_log_decrement(t, x) == pytest.approx(0.1, rel=1e-3)


def test_twist_angle_reads_rotation_about_x_past_pi():
    """A11.torsion's reading: an off-axis point at +z rotated about +x by theta sits at (-sin, cos); unwrapped in time,
    so a twist past pi is not folded back."""
    theta = np.linspace(0.0, 4.0, 81)[None, :]
    off = (np.zeros_like(theta), -0.01 * np.sin(theta), 1.0 + 0.01 * np.cos(theta))
    axis = (np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta))
    np.testing.assert_allclose(AT.twist_angle(off, axis), theta, atol=1e-12)


# --- laws (CPU quick) ----------------------------------------------------------------------------------------------

@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
@pytest.mark.parametrize("spec,twin", [(AR.LAW_HANGING, AR.HangingChainTwin), (AR.LAW_CATENARY, AR.CatenaryTwin),
                                       (AR.LAW_CANTILEVER, AR.CantileverTwin), (AT.LAW_TORSION, AT.TorsionTwin),
                                       (AB.LAW, AB.BucklingTwin)],
                         ids=lambda x: getattr(x, "id", ""))
def test_analytic_rod_law_passes(spec, twin):
    res = run_law(spec, twin(), None, device="cpu", quick=True)
    _record(res)
    assert res.monitors.get("finite") and res.verdict == "pass", [(v.name, v.error, v.bound) for v in res.verdicts]


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_a10_capstan():
    res = run_law(AC.LAW, AC.CapstanTwin(), None, device="cpu", quick=True)
    _record(res)
    assert res.monitors.get("finite") and res.verdict == "pass", res.monitors.get("per_condition")
    for cond in res.monitors["per_condition"].values():           # creep stays well under the slip speed
        assert cond["creep_max_m_s"] < 0.5 * cond["slip_speed_m_s"], cond


@pytest.mark.slow
@pytest.mark.timeout(1500)
def test_r9_rope_twin_on_synthetic_data():
    from alloy.scene.twins import rope as RP

    res = run_law(RP.LAW, RP.RopeTwin(), RP.dataset(), device="cpu", quick=True)
    _record(res)
    assert res.verdict == "pass", [(v.name, v.verdict, v.error, v.bound) for v in res.verdicts]


def test_link_frames_keep_sub_microradian_bends():
    """WP8: a float32 `cos > 1 - 1e-12` transport dropped A12's 1e-4 L imperfection (2e-7 to 9e-6 rad per link),
    so the column came out straight. The float64 transport keeps every link's own frame and the tip offset."""
    from alloy._vendor import newton

    from alloy.scene.emitters.chain import ChainOptions, build_chain, link_frames

    _, quats, _ = link_frames(AB.imperfect_column(), AB.L, 30)
    assert len({tuple(q) for q in quats}) == 30
    _, straight, _ = link_frames(None, 1.0, 30)
    assert len({tuple(q) for q in straight}) == 1
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    h = build_chain(b, AB.rod_card(AB.L, AB.EI, 30, radius=AB.RADIUS, density=AB.DENSITY), AB.imperfect_column(),
                    ChainOptions(capability="mujoco", clamp=True))
    m = b.finalize(device="cpu")
    st = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, st)
    tip = st.body_q.numpy()[h.bodies[-1], :3].astype(np.float64)
    card = AB.IMPERFECTION * AB.L * (1.0 - np.cos(0.5 * np.pi * tip[2] / AB.L))
    assert tip[0] == pytest.approx(card, rel=0.01) and card > 1e-5
