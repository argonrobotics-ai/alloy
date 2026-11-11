# SPDX-License-Identifier: Apache-2.0
"""melt_chocolate (alloy/tasks/melt_chocolate.py): a short CPU run of the hot-plate melt (light test; the full 20-minute
task runs on the GPU smoke)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.thermal.phase_materials import material
from alloy.tasks import cook_common as C
from alloy.tasks import melt_chocolate as T

pytestmark = pytest.mark.law


def test_start_values_match_what_the_plugin_writes():
    mat = material(T.MATERIAL)
    got = C.mpm_values(mat, T.T_START[0])
    assert got == {"mpm:viscosity": pytest.approx(mat.mu_solid), "mpm:yield_stress": pytest.approx(mat.yield_solid),
                   "mpm:friction": pytest.approx(mat.friction_solid),
                   "mpm:young_modulus": pytest.approx(mat.young_solid)}
    assert C.mpm_values(material("beef_lean"), 278.0) == {}          # no flow law: the task's own values


def test_per_world_draws_are_seeded_and_in_range():
    a, b = C.draws(3, 8, *T.T_PLATE, stream=1), C.draws(3, 8, *T.T_PLATE, stream=1)
    assert np.array_equal(a, b) and not np.array_equal(a, C.draws(4, 8, *T.T_PLATE, stream=1))
    assert (a >= T.T_PLATE[0]).all() and (a <= T.T_PLATE[1]).all()


def test_short_melt_starts_at_the_plate_and_stays_below_it():
    row = T.run(2, "cpu", seed=1, periods=6)
    assert row["neighbor_overflow_max"] == 0, row
    assert 0.0 < row["melted_min"] < T.MELTED, row                  # 72 real s: the bottom layer has started to melt
    assert row["hottest_over_plate_K_max"] <= T.HOT_TOL, row       # no particle hotter than its plate
    assert 0.9 < row["height_ratio_min"] and row["height_ratio_max"] < 1.01, row   # a solid block, on the plate
    assert row["contact_share_min"] > 0.95 and row["isolated_share_max"] == 0.0, row   # its whole base on the plate
