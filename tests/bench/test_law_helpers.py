# SPDX-License-Identifier: Apache-2.0
"""Pure helpers of the WP6 laws, against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench import report as R
from alloy.bench.analytic import coulomb, neohookean, pendulum, restitution


def test_elliptic_period_small_angle_and_90deg() -> None:
    # T0 = 2 pi sqrt(I/(m g d)); at 90 deg, T = 4 sqrt(I/(m g d)) K(sin 45 deg) with K = 1.854075 (bench A2)
    i, m, d, g = 2.0, 1.0, 1.0, 9.81
    t0 = 2 * math.pi * math.sqrt(i / (m * g * d))
    assert pendulum.elliptic_period(i, m, d, 1e-6, g) == pytest.approx(t0, rel=1e-9)
    assert pendulum.elliptic_period(i, m, d, math.pi / 2, g) == pytest.approx(4 * math.sqrt(i / (m * g * d)) * 1.854075,
                                                                               rel=1e-6)


def test_crossing_period_of_a_cosine() -> None:
    t = np.linspace(0.0, 2.0, 2001)
    assert pendulum.crossing_period(t, np.cos(2 * math.pi * t / 0.7)) == pytest.approx(0.7, rel=1e-6)
    assert math.isnan(pendulum.crossing_period(t, np.ones_like(t)))


def test_onset_from_grid() -> None:
    off = np.array([-0.15, -0.05, 0.05, 0.15])
    assert coulomb.onset_from_grid(off, np.array([False, False, True, True])) == pytest.approx(0.0)
    assert coulomb.onset_from_grid(off[::-1], np.array([True, True, False, False])) == pytest.approx(0.0)  # unsorted
    assert math.isnan(coulomb.onset_from_grid(off, np.zeros(4, bool)))
    assert math.isnan(coulomb.onset_from_grid(off, np.ones(4, bool)))



def test_slip_ratios_need_a_hold_then_a_slip() -> None:
    t = np.arange(0.0, 1.0, 1.25e-3)
    r = coulomb.push_ratio(t, 1.0)
    assert r[0] == 0.0 and coulomb.push_ratio(coulomb.T_EASE, 1.0) == pytest.approx(coulomb.R_LO)
    assert r[-1] == pytest.approx(coulomb.R_HI, abs=1e-3)
    slips_at_one = np.where(r > 1.0, (r - 1.0) * 1.0, 0.0)       # holds until the ramp passes 1.0, then moves
    creeps = np.where(t > coulomb.T_EASE, (t - coulomb.T_EASE) * 0.01, 0.0)  # moving from the ramp start
    never = np.zeros_like(t)
    out = coulomb.slip_ratios(t, np.stack([slips_at_one, creeps, never]), 1.0)
    assert 1.0 < out[0] < 1.001 and math.isnan(out[1]) and math.isnan(out[2])

def test_first_rebound_apex() -> None:
    t = np.linspace(0, 1, 1001)
    z = np.where(t < 0.3, 1.0 - t, 0.7 + 0.5 * (t - 0.3) - (t - 0.3) ** 2)  # impact at 0.3, apex at 0.55: 0.7625
    assert restitution.first_rebound(t, z) == pytest.approx(0.7625, abs=1e-6)
    assert math.isnan(restitution.first_rebound(t, 1.0 - t))  # never rebounds


def test_neohookean_incompressible_limit() -> None:
    # lambda -> inf: P = mu (a - a^-2) (bench A15)
    assert neohookean.uniaxial_stress(0.8, mu=1.0, lam=1e9) == pytest.approx(0.8 - 0.8**-2, rel=1e-6)
    assert neohookean.uniaxial_stress(0.8) < 0 < neohookean.uniaxial_stress(1.2)


def test_is_report(tmp_path) -> None:
    (tmp_path / "x.json").write_text('{"schema": "alloy.bench.report/1"}')
    (tmp_path / "y.json").write_text("not json")
    assert R.is_report(tmp_path / "x.json") and not R.is_report(tmp_path / "y.json")


def test_a5_kinetic_gated_on_default_route_recipe_is_a_known_gap() -> None:
    from alloy.bench import default_route as DR
    from alloy.bench.law import KNOWN_GAPS

    slide, recipe = coulomb.CoulombSlideTwin(), coulomb.CoulombSlideRecipeTwin()
    routed = {**DR.mujoco_default_kwargs(), **DR.route_fill_ins()}
    assert dict(slide.settings_for("mujoco").kwargs) == routed  # impratio: the route's fill-in (SCR-WP3-10)
    assert slide.settings_for("mujoco").mujoco_solimp is None  # Newton's solimp
    assert recipe.settings_for("mujoco").kwargs["impratio"] == 300.0
    assert all(c.params["tilted"] for c in recipe.conditions(True)) and coulomb.LAW_SLIDE_RECIPE.id in KNOWN_GAPS
    spread = coulomb.CoulombSlideSpreadTwin()
    assert dict(spread.settings_for("mujoco").kwargs) == routed
    assert coulomb.LAW_SLIDE_SPREAD.id in KNOWN_GAPS and len(spread.conditions(True)) == coulomb.N_DIR
    assert coulomb.CoulombOnsetTwin().settings_for("mujoco").kwargs["impratio"] == 300.0  # static metrics keep it


class _SepLog:
    """Stub DeviceTwinLog: spheres 0 and 1 drift apart along x at 0.5 m per sample."""

    def __init__(self, n_worlds: int, n: int) -> None:
        self._t = np.arange(n, dtype=np.float64)
        self._x = np.tile(0.5 * self._t, (n_worlds, 1))

    def series(self, name: str) -> np.ndarray:
        if name == "s0x":
            return self._x
        return np.zeros_like(self._x)

    def times(self) -> np.ndarray:
        return self._t


def test_a4_collide_window_ends_at_separation_and_drift_stays_inside_it() -> None:
    from alloy.bench.analytic import momentum
    from alloy.bench.types import Condition

    twin = momentum.MomentumTwin()
    twin.worlds = [Condition("collide", {}), Condition("tumble", {})]
    ends = twin.window_end(_SepLog(2, 6))  # separation 0, 0.5, 1.0, ... m
    assert ends.tolist() == [2, 5]  # collide stops at 1 m apart; tumble keeps the whole run
    v = np.zeros((2, 6, 3))
    v[:, :, 0] = 1.0
    v[:, 4, 0] = 2.0  # a jump after the collide window: counted only in the world without a window
    assert momentum.drift(v, ends).tolist() == [0.0, 1.0]


def test_a4_collide_is_reconditioned_and_the_small_lever_world_is_informational() -> None:
    from alloy.bench.analytic import momentum
    from alloy.bench.law import INFORMATIONAL, KNOWN_GAPS, not_gated, registered_laws

    (p0, v0, _), (p1, _, _) = momentum.MomentumTwin.scenes["collide"][:2]
    (q0, w0, _), (q1, _, _) = momentum.SmallLeverCollideTwin.scenes["collide_small_lever"][:2]
    assert (p0[1], q0[1]) == (0.2, 0.02) and v0 == w0  # 10x lever, same speed
    assert round(p0[1] - p1[1], 9) == round(q0[1] - q1[1], 9) == 0.02  # same impact offset
    assert momentum.LAW_SMALL_LEVER.metrics == momentum.LAW.metrics  # tolerance unchanged
    assert momentum.SmallLeverCollideTwin.window_sep == {}  # the old world keeps the whole run
    assert momentum.LAW_SMALL_LEVER.id in INFORMATIONAL and momentum.LAW_SMALL_LEVER.id not in KNOWN_GAPS
    assert not_gated(momentum.LAW_SMALL_LEVER.id).startswith("informational, not gated:")
    assert not_gated(momentum.LAW.id) == ""
    assert momentum.LAW_SMALL_LEVER.id in {spec.id for spec, _ in registered_laws()}
