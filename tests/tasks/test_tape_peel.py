# SPDX-License-Identifier: Apache-2.0
"""task_tape_peel: scoring on synthetic series, the run length, and the task end to end (slow, CPU, 50 mm strip)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import tape_peel as T


def test_score_needs_tape_off_and_the_row():
    r = T.rig(0.05)
    front = np.linspace(0.0, 0.05, 50)[:, None].repeat(2, axis=1)
    width = np.full_like(front, 0.025)
    force = np.full_like(front, 380.0 * 0.025)
    force[:, 1] *= 1.2                                   # world 1: 20 % over the row
    bonded = np.zeros_like(front)
    row = T._score({"peel_force": force, "width": width, "front": front, "bonded": bonded}, r, 380.0)
    assert row["success"] == 1 and row["tape_off"] == 2
    bonded[-1, 0] = 1e-6                                 # world 0: a bond left
    row = T._score({"peel_force": force, "width": width, "front": front, "bonded": bonded}, r, 380.0)
    assert row["success"] == 0


def test_run_length_covers_the_strip():
    r = T.rig(0.1)
    assert T.periods(90, r) * r.control_dt >= (r.length - r.tab) / T.SPEED
    assert T.periods(180, r) * r.control_dt >= 2.0 * (r.length - r.tab) / T.SPEED


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("angle", [90, 180])
def test_task_tape_peel(angle):
    """Rows (CPU, 1 world, 50 mm strip, ~2 min each): 90° F/b 394.1 N/m vs 380 (+3.7 %), 180° 469.3 vs 470
    (-0.2 %); tape off in both."""
    row = T.run(angle, 1, "cpu", length=0.05)
    assert row["success"] == 1, row
