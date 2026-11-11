# SPDX-License-Identifier: Apache-2.0
"""task_label_apply: the script's phases, the press force, scoring, and the task end to end (slow, CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.bonds import adhesive
from alloy.tasks import label_apply as L


def test_script_phases():
    x, z, rel = L.targets(0.0)
    assert (x, z, rel) == (0.0, 0.0, 0.0)
    x, z, rel = L.targets(L.PHASES[1])
    assert x == pytest.approx(L.CARRY_X) and z == pytest.approx(-(L.LIFT - L.CLEAR)) and rel == 0.0
    x, z, rel = L.targets(0.5 * (L.PHASES[2] + L.PHASES[3]))
    assert z < -L.LIFT and rel == 1.0                    # pressing, vacuum off
    assert L.targets(L.PHASES[4])[1] == pytest.approx(0.0)
    assert L.PHASES[3] - L.PHASES[2] >= L.HOLD * adhesive(L.CARD).dwell - 1e-12


def test_press_force_is_twice_p_form_over_the_label():
    total = L.press_force() + L.RIG.link_mass * 9.81
    assert total / (L.LABEL[0] * L.LABEL[1]) == pytest.approx(2.0 * adhesive(L.CARD).p_form)


def test_score():
    nx, ny = int(round(L.LABEL[0] / L.RIG.dx)) + 1, int(round(L.LABEL[1] / L.RIG.dx)) + 1
    state = np.ones((2, nx * ny), np.int32)
    state[1, 0] = 0                                      # world 1: a corner not bonded
    q = np.zeros((2, nx * ny, 3))
    row = L._score(state, q)
    assert row["success"] == 1 and row["corners_ok"] == 1


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_task_label_apply():
    """Row (CPU, 1 world, ~2 min): all 187 vertices bonded, corners bonded, centre 0.04 mm off the target."""
    row = L.run(1, "cpu")
    assert row["success"] == 1, row
    assert row["ga_trapped_air_pass"] == 1 and row["trapped_area_mm2_max"] == 0.0, row     # the pad drains the air


def test_ga_criteria_report_wet_out_trapped_air_and_the_unscored_barcode():
    nx, ny = int(round(L.LABEL[0] / L.RIG.dx)) + 1, int(round(L.LABEL[1] / L.RIG.dx)) + 1
    read = {"wet": np.full((2, nx * ny), 0.9), "trapped_area": np.array([0.0, 25e-6]), "air_overflow": np.zeros(2),
            "air_vanished": np.zeros(2)}
    read["wet"][1] = 0.5
    row = L.ga_criteria(read)
    assert row["ga_wet_out_pass"] == 1 and row["ga_trapped_air_pass"] == 1
    assert row["trapped_area_mm2_max"] == pytest.approx(25.0) and "not scored" in row["barcode"]
