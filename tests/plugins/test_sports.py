# SPDX-License-Identifier: Apache-2.0
"""Sports plug-ins on CPU: model.py against its sources, config checks, readings, and the laws."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene
from alloy.plugins.sports import BallAero, GoalNet, PressureBall
from alloy.plugins.sports import laws as BL
from alloy.plugins.sports import laws_net as NL
from alloy.plugins.sports import model as M


def test_shell_law_and_air_coefficients():
    d = 1e-5                                                     # small overlap: k = 2πR·p + k_s
    k0 = float(M.shell_force(d)) / d
    assert k0 == pytest.approx(2 * math.pi * M.R_BALL * M.P_TEST + M.SHELL_K, rel=1e-3)
    d = 0.02                                                     # the gas compresses: above the patch-area force
    assert float(M.shell_force(d)) > M.P_TEST * math.pi * (2 * M.R_BALL * d - d * d) + M.SHELL_K * d
    assert float(M.cd(5e4)) == pytest.approx(M.CD_SUB) and float(M.cd(3e5)) == pytest.approx(M.CD_SUPER)
    assert M.reynolds(14.0) == pytest.approx(2.0e5, rel=0.05)    # Goff & Carré: v ≈ 14 m/s ↔ Re ≈ 2e5
    assert float(M.cl(0.5)) == pytest.approx(0.35, abs=0.01)     # Goff & Carré: CL ≈ 0.35 at Sp ≈ 0.5
    assert 2 * M.R_BALL * math.pi == pytest.approx(0.69) and 0.41 <= M.M_BALL <= 0.45   # IFAB Law 2


def test_fitted_constants_meet_the_fifa_targets():
    rigid = M.drop_rebound(surface="rigid", dt=1e-5)
    assert rigid["rebound_m"] == pytest.approx(1.45, abs=0.005)          # turf manual: 1.45 ± 0.03 m
    assert rigid["v_in"] == pytest.approx(6.25, abs=0.15)                # FIFA: 6.25 ± 0.15 m/s at impact (in air)
    assert 0.006 < rigid["contact_s"] < 0.010
    turf = M.drop_rebound(surface="turf", dt=1e-5)
    assert turf["rebound_m"] == pytest.approx(0.80, abs=0.01)            # middle of FIFA's 0.60-1.00 m
    assert M.roll_distance(M.ramp_exit_speed()) == pytest.approx(7.0, abs=0.05)   # FIFA band 4.0-10.0 m
    lo, mid, hi = (M.drop_rebound(p=p, dt=1e-5)["rebound_m"] for p in BL.PRESSURES)
    assert lo < mid < hi                                                 # restitution rises with pressure


def test_registered_and_validated():
    names = [p.name for p in ap.available()]
    assert {"pressure_ball", "ball_aero", "goal_net"} <= set(names)
    with pytest.raises(PluginError, match="collider_mu"):
        PressureBall(colliders=("a", "b"), collider_mu=(0.5,))
    with pytest.raises(PluginError, match="radius"):
        PressureBall(radius=0.0)
    with pytest.raises(PluginError, match="re_lo"):
        BallAero(re_lo=3e5, re_hi=2e5)
    with pytest.raises(PluginError, match="perpendicular"):
        GoalNet(across=(1.0, 0.0, 0.0), up=(1.0, 0.0, 1.0))
    with pytest.raises(PluginError, match="strain0"):
        GoalNet(strain0=0.7)


def test_ball_that_collides_in_the_solver_is_refused():
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.harness.scene import body_xform  # noqa: PLC0415

    def build(b):
        body = b.add_body(xform=body_xform(0.0, 0.0, 0.5), label="ball")
        b.add_shape_sphere(body, radius=M.R_BALL, cfg=newton.ModelBuilder.ShapeConfig(density=80.0))

    run = run_scene(TestScene(build=build, worlds=1, periods=1, dt=BL.DT, ground=False), [PressureBall()], "cpu",
                    periods=1)
    assert run.error is not None and "has_shape_collision=False" in str(run.error)


@pytest.mark.law
@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["pressure_ball", "ball_aero", "goal_net"])
def test_quick_check_passes_every_law(name):
    report = ap.check(name, quick=True)
    assert report.passed, report.summary()


@pytest.mark.timeout(300)
def test_touch_reading_marks_finished_contacts():
    run = run_scene(BL.drop_scene((M.P_TEST,), True), [PressureBall(pitch=False, colliders=(BL.PLATE,))], "cpu",
                    periods=36)
    assert run.error is None, run.error
    touch = run.trace.plugin("pressure_ball", "touch")[-1, 0, 0]      # [1 + C, 2]
    assert touch[0, 0] == 0.0 and touch[1, 1] < 0.0                  # never on the pitch; plate contact finished
    assert np.isfinite(touch).all()


def test_net_mode_closed_form_and_substeps():
    discrete, continuum = NL.mode_frequency(NL.MODE)
    assert discrete == pytest.approx(continuum, rel=0.01)           # a 24 x 17-cell net is close to the membrane
    run = run_scene(NL.net_scene(), [GoalNet()], "cpu", periods=2)
    assert run.error is None, run.error
    net = run.sim.plugins["goal_net"]
    for dt in (0.02 / 64, 0.02 / 32, 0.02 / 128):                  # counted from each exchange step (WP13 S2)
        n = net.substeps_for(dt)
        assert n % 2 == 0 and net.node_rate() * dt / n <= 0.4 and net.damping_rate() * dt / n <= 1.0


@pytest.mark.timeout(300)
def test_net_rests_on_the_ground():
    """The slack gathers at the bottom but no node goes below the ground (WP13 S1)."""
    run = run_scene(NL.net_scene(), [GoalNet()], "cpu", periods=25)
    assert run.error is None, run.error
    z = run.trace.plugin("goal_net", "nodes")[-1, :, :, 2]
    assert float(z.min()) >= -1e-6


@pytest.mark.timeout(300)
def test_pitch_off_with_zero_turf_damping_stays_finite():
    """WP13 S4: with the pitch off, its turf terms must not run (turf_c 0 used to give 0/0)."""
    ball = PressureBall(pitch=False, turf_c=0.0, colliders=(BL.PLATE,))
    run = run_scene(BL.drop_scene((M.P_TEST,), True), [ball], "cpu", periods=34)
    assert run.error is None, run.error
    assert np.isfinite(run.trace.plugin("pressure_ball", "turf_u")).all()
    assert np.isfinite(run.trace.body_q).all()
