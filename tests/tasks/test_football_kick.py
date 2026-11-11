# SPDX-License-Identifier: Apache-2.0
"""task_football_kick on CPU: kicks score or miss as aimed, the flight matches model.py, the net holds a slow ball."""

from __future__ import annotations

import math

import pytest

from alloy.tasks import football_kick as F


def test_leg_energy_balance():
    i_hip, c = F.leg_inertia()
    assert i_hip == pytest.approx(F.LEG_MASS * F.LEG_L ** 2 / 3 + F.BOOT_MASS * F.LEG_L ** 2)
    assert 0.0 < c < F.LEG_L
    assert F.foot_speed(0.0) == 0.0
    assert F.foot_speed(1.3) > F.foot_speed(0.8) > 0.0
    assert F.foot_speed(1.3) == pytest.approx(math.sqrt(2 * (0.5 * F.LEG_K * 1.3 ** 2 + (F.LEG_MASS + F.BOOT_MASS)
                                                             * 9.81 * c * (1 - math.cos(1.3))) / i_hip) * F.LEG_L)


@pytest.mark.timeout(300)
def test_kicks_score_or_miss_as_aimed():
    kicks = (F.KICKS[0], F.KICKS[2], F.KICKS[3], F.KICKS[5])        # straight, wide, curler, weak roller
    row = F.run(kicks, "cpu", net=False, seconds=1.4)
    assert row["success"] == [True, False, True, True], row
    assert abs(row["cross_yz"][1][0]) > 0.5 * F.GOAL_WIDTH          # wide: crosses outside the post
    assert row["launch_spin_rps"][2][2] > 1.0                       # the curler spins about the vertical
    ratio = row["launch_mps"][0] / row["foot_mps"][0]
    assert 1.3 < ratio < 1.9                                        # (1 + e)/(1 + m/M_eff), e ≈ 0.89, M_eff ≈ 3.7 kg
    checked = [c for c in row["flight_check_mm"] if c is not None]
    assert checked and max(checked) < 10.0                          # crossing vs model.flight from the launch state


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_net_stops_the_shot():
    row = F.run((F.KICKS[5],), "cpu", net=True, seconds=1.8)        # the slow roller ends in the net
    assert row["success"] == [True] and row["held_in_goal"] == [True], row
