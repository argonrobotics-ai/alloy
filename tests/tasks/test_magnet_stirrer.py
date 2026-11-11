# SPDX-License-Identifier: Apache-2.0
"""task_magnet_stirrer on CPU: the bar follows the drive below its decoupling speed and spins out above it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import magnet_stirrer as S


def test_decoupling_speed_is_the_pull_out_balance():
    b = 1e-7 * S.M_DRIVE / S.H_DRIVE ** 3
    omega = S.decoupling_rpm() * 2 * math.pi / 60
    assert S.K_DRAG * omega ** 2 == pytest.approx(S.M_BAR * b, rel=1e-12)
    assert 900.0 < S.decoupling_rpm() < 1300.0                     # a 25 mm bar in water: about 1,000 rpm


def test_stirs_below_decoupling_and_spins_out_above():
    rpm = S.decoupling_rpm()
    targets = np.array([300.0, 0.85 * rpm, 1.3 * rpm])
    row = S.run(targets, "cpu")
    assert row["success"] == [True, True, False], row
    assert row["spin_ratio"][0] == pytest.approx(1.0, abs=0.01) and row["spin_ratio"][2] < 0.5
    assert max(row["off_axis_mm_max"][:2]) < 2.0
