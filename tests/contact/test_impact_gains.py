# SPDX-License-Identifier: Apache-2.0
"""Impact-matched native gains (alloy.contact.impact_gains, alloy.scene.native_gains; physics rulings 2026-11-11:
SAMPLE-SWEEP's restitution-0 rebound and SAMPLE-CLASSIC C4; 2026-11-13 option A and 2026-11-14 option (1): only
for one-point contacts (balls, no resting floor) in free-body entries, rk4 and implicitfast on rk4's or its own gains;
every other contact and mode writes the static rule bit for bit).

- A MuJoCo contact lets go at zero push, so e(zeta) is the zero-force release (``release_speed``): exp(-2) = 0.135 at
  zeta 1, where the zero-overlap rule Alloy used gave e 0. ``zeta_for_restitution`` inverts it, floored at e 0.02.
- ke = K / m makes the row a K spring during an impact (it was K (1 - imp) / m: K at rest, 5 % of K on impact).
- Both capped for stability at the chosen step and never under today's rule (``static_gains``), which the
  rigid-step planner and step 2c keep deciding on, so no step choice changes.

Rows (CPU, MuJoCo C, before -> after): C4's 6.8 kg ball into a static wall at 1.71 m/s, 14.3 / 14.9 / 14.8 -> 7.7 /
3.9 / 2.7 mm at auto 3.33 / fixed 1.67 / 0.417 ms (a K spring: 4.5 mm undamped); 1 kg 5.7 -> 3.9 / 3.9 / 1.1 mm. A
4.1 kg steel ball at e 0 rebounds at 0.018-0.019 (0.3125 ms) and 0.050-0.055 (1.25 ms) from 1-5 m/s, against 0.12-0.15;
a 4 g ball keeps 0.11-0.15 (it sits at the stability cap: never softer).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.contact import impact_gains as IG
from alloy.scene import native_gains as NG

K = 1.0e6                                   # the default pair (two 2e6 N/m materials in series)
STEPS = (0.0003125, 0.000625, 0.00125, 0.0016667, 0.0033333)
MODES = ("rk4", "implicitfast", "own")


def _release_numeric(zeta: float, n: int = 100_000, t_end: float = 20.0) -> float:
    """The same release, integrated: x'' = -x - 2 zeta x' from (0, 1) to the first zero of the push."""
    h = t_end / n
    x, v = 0.0, 1.0
    for _ in range(n):
        a = -x - 2.0 * zeta * v
        if x > 0.0 and -a <= 0.0:              # push = x + 2 zeta v (= -a) reaches zero
            return -v
        v += h * a
        x += h * v
    return float("nan")


@pytest.mark.parametrize("zeta", [0.1, 0.255, 0.5, 1.0, 2.0, 3.3])
def test_release_speed_is_the_unilateral_spring_damper(zeta):
    assert IG.release_speed(zeta) == pytest.approx(_release_numeric(zeta), abs=2e-3)


def test_zeta_for_restitution_inverts_the_release_with_its_floor():
    assert IG.release_speed(1.0) == pytest.approx(math.exp(-2.0))
    for e in (0.0, 0.01, 0.05, 0.3, 0.5, 0.8, 0.95):
        assert IG.release_speed(IG.zeta_for_restitution(e)) == pytest.approx(max(e, IG.E_FLOOR), abs=1e-9)
    assert IG.zeta_for_restitution(0.5) == pytest.approx(0.2553, abs=1e-4)
    assert IG.zeta_for_restitution(1.0) == 0.0
    with pytest.raises(ValueError):
        IG.zeta_for_restitution(1.2)


def test_impact_gains_are_stable_and_never_softer_than_todays():
    for m in np.geomspace(1e-3, 100.0, 15):
        for e in (0.0, 0.2, 0.5, 0.8):
            for dt in STEPS:
                for mode in MODES:
                    ke, kd, zeta = NG.impact_gains(K, m, e, dt=dt, zeta_min=0.0, integrator=mode)
                    ke0, kd0 = NG.static_gains(K, m, e, dt=dt, integrator=mode)
                    at = (m, e, dt, mode)
                    assert ke >= ke0 * (1 - 1e-12) and zeta >= NG.contact_zeta(e) - 1e-12, at
                    assert ke <= max(K / m, ke0) * (1 + 1e-12), at
                    assert ke <= NG.stable_ke(zeta, dt, integrator=mode) * (1 + 1e-9), at
                    assert kd == pytest.approx(2.0 * zeta * math.sqrt(ke)), at


def test_heavy_bodies_get_the_card_on_impact_and_light_ones_keep_todays():
    ke, _, zeta = NG.impact_gains(K, 6.8, 0.5, dt=0.00125, integrator="rk4")
    assert ke * 6.8 / K == pytest.approx(1.0) and IG.release_speed(zeta) == pytest.approx(0.5)
    ke, _, zeta = NG.impact_gains(K, 0.005, 0.0, dt=0.00125, integrator="rk4")
    assert (ke, zeta) == pytest.approx((NG.static_gains(K, 0.005, 0.0, dt=0.00125, integrator="rk4")[0], 1.0))


def test_the_planning_rule_writes_todays_gains():
    with NG.planning_rule():
        assert NG.native_gains(K, 6.8, 0.5, dt=0.00125, integrator="rk4") == NG.static_gains(K, 6.8, 0.5, dt=0.00125,
                                                                                               integrator="rk4")
        assert not NG.impact_mode("rk4")
    assert NG.native_gains(K, 6.8, 0.5, dt=0.00125, integrator="rk4") != NG.static_gains(K, 6.8, 0.5, dt=0.00125,
                                                                                           integrator="rk4")


def _main_capped_ratio(K: float, m: float, e: float, dt: float, zeta_min: float, mode: str) -> float:
    """``capped_ratio`` as it read before impact-matched gains: realised / card static stiffness."""
    return min(1.0, NG.stable_ke(NG.contact_zeta(e, zeta_min), dt, NG.DEFAULT_IMP, mode) / (K * (1.0 - NG.DEFAULT_IMP)
                                                                                            / m))


@pytest.mark.parametrize("mode", ["implicitfast", "discrete", "implicit"])
def test_modes_without_a_handled_friction_pull_keep_the_static_rule_bit_for_bit(mode):
    """Robot, kinematic and chain entries ("implicitfast", MuJoCo's default; "discrete" names the chain's integrator)
    write the static rule: ke, kd, zeta and the stiffness ratio exactly as before impact-matched gains."""
    assert not NG.impact_mode(mode) and NG.impact_mode("rk4") and NG.impact_mode("own")
    for m in np.geomspace(1e-3, 100.0, 9):
        for e in (0.0, 0.5, 0.8):
            for zmin in (0.0, NG.ZETA_CCD, NG.ZETA_MULTIPOINT):
                for dt in (None, *STEPS):
                    got = NG.impact_gains(K, m, e, dt=dt, zeta_min=zmin, integrator=mode)
                    want = (*NG.static_gains(K, m, e, dt=dt, zeta_min=zmin, integrator=mode), NG.contact_zeta(e, zmin))
                    assert got == want, (m, e, zmin, dt)
                    assert NG.native_gains(K, m, e, dt=dt, zeta_min=zmin, integrator=mode) == want[:2]
                    if dt is not None:
                        assert NG.capped_ratio(K, m, e, dt, zeta_min=zmin, integrator=mode) == _main_capped_ratio(
                            K, m, e, dt, zmin, mode), (m, e, zmin, dt)
    assert NG.contact_zeta(0.0) == 1.0 and NG.contact_zeta(0.5) == pytest.approx(0.2155, abs=1e-4)


@pytest.mark.parametrize("mode", ["rk4", "own"])
def test_contacts_with_a_resting_floor_keep_the_static_rule_in_free_body_entries(mode):
    """Option (1), physics 2026-11-14: only one-point contacts (no resting floor: balls) take impact gains; a box,
    capsule, cylinder or mesh (multi-point or CCD floor), and a pair row with a floored dynamic side, keep the static
    rule bit for bit, also in rk4 and capped entries."""
    assert NG.impact_applies(mode, 0.0) and not NG.impact_applies(mode, NG.ZETA_CCD)
    assert not NG.impact_applies("implicitfast", 0.0)
    for m in (0.1, 2.0, 27.0, 100.0):
        for e in (0.0, 0.5):
            for zmin in (NG.ZETA_CCD, NG.ZETA_MULTIPOINT, 1.0):
                got = NG.impact_gains(K, m, e, dt=0.00125, zeta_min=zmin, integrator=mode)
                want = (*NG.static_gains(K, m, e, dt=0.00125, zeta_min=zmin, integrator=mode), NG.contact_zeta(e, zmin))
                assert got == want, (m, e, zmin)
            assert NG.impact_gains(K, m, e, dt=0.00125, integrator=mode)[0] >= NG.static_gains(
                K, m, e, dt=0.00125, integrator=mode)[0]
