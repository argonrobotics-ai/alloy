# SPDX-License-Identifier: Apache-2.0
"""cook_sauce (alloy/tasks/cook_sauce.py): the spoon's loop in the pan, the layers, the hob rule, the score, and a
short CPU run of the SPH starch sauce (light test; the full task runs on the GPU smoke)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.thermal.phase_materials import material
from alloy.tasks import cook_sauce as T

pytestmark = pytest.mark.law


def test_batch_and_spoon_fit_the_pan():
    mat = material(T.MATERIAL)
    grams = np.prod(T.COUNTS) * mat.density * (2.0 * T.RADIUS) ** 3 * 1e3
    assert grams == pytest.approx(590.0, rel=0.01)                            # Gym-Anything's batch
    period = T.SPOON.length / T.SPOON.speed
    path = np.stack([T.SPOON.centre(x) for x in np.linspace(0.0, period, 2001)])
    assert np.allclose(path[0], path[-1], atol=1e-12)                          # a closed loop
    assert path[:, 2].min() - T.BLADE_HALF[2] == pytest.approx(T.SPOON.floor_clear)
    assert np.abs(path[:, 0]).max() + T.BLADE_HALF[0] == pytest.approx(0.5 * T.INNER[0] - T.CLEAR)
    assert path[:, 2].max() - T.BLADE_HALF[2] > T.DEPTH                        # out of the sauce on the return
    assert 0.5 * T.INNER[1] - T.BLADE_HALF[1] >= 4.0e-3 and T.CLEAR > 2.0 * T.RADIUS


def test_layers_are_equal_counts_by_height():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.0, 0.04, (2, 300, 3))
    value = q[..., 2] / 0.04                                                   # 0 at the floor, 1 at the top
    lay = T.layers(q, value)
    assert lay.shape == (2, 3) and np.all(np.diff(lay, axis=1) > 0.2)
    assert lay[:, 0] == pytest.approx(1.0 / 6.0, abs=0.05) and lay[:, 2] == pytest.approx(5.0 / 6.0, abs=0.05)


def test_hob_turns_off_at_the_boil_once_cooked_or_out_of_time():
    on = np.array([True, True, True, False])
    boiled = np.array([True, True, False, True])
    cooked = np.array([0.96, 0.90, 0.99, 0.99])
    on_s = np.array([120.0, 150.0, T.HOB_MAX_S, 100.0])
    assert T.hob_off(on, boiled, cooked, on_s).tolist() == [True, False, True, False]


def test_score_is_one_on_success_else_their_product_without_the_unjudged_credits():
    """Audit AB-gym-4: Gym-Anything's closing credits for the pan on the hob (+ 0.10) and the hob off (+ 0.15) belong
    to criteria the port does not judge, so a failed world's score takes the closing factor 0.75 alone."""
    ones = np.ones(3)
    ok, least = np.array([True, False, False]), np.array([0.99, 0.475, 0.99])
    got = T.score(ok, least, np.array([1.0, 1.0, 0.9]), np.array([1.0, 0.8, 1.0]))
    assert got == pytest.approx([1.0, 0.9 * 0.5 * 0.8 * 0.75, 0.9 * 0.9 * 0.75])
    assert T.score(np.zeros(3, bool), ones, ones, ones) == pytest.approx(0.9 * 0.75 * ones)


def test_short_cook_delivers_the_hob_and_writes_the_viscosity():
    row = T.run(2, "cpu", seed=0, periods=3)
    assert row["neighbor_overflow_max"] == 0 and row["heater_idle_J_max"] == 0.0, row
    assert row["hob_ratio"] == pytest.approx([1.0, 1.0], rel=1e-9), row   # η·P·t into the sauce
    assert row["spilt_max"] == 0.0 and row["burnt_cm2_max"] == 0.0, row
    assert row["cooked_min"] < 1e-3 and row["hob_on_at_end"] == 2, row     # cold milk: raw starch, hob still on
    ga = row["ga"]                         # audit AB-gym-4: 4 of Gym-Anything's 6 judged, 2 listed as not judged
    assert ga["judged"] == "4 of 6" and ga["criteria"]["vessel_in_place"] is None, ga
    assert ga["criteria"]["appliance_off"] is None and set(ga["not_judged"]) == {"vessel_in_place", "appliance_off"}
    assert ga["criteria"]["cooked"] == [False, False] and ga["criteria"]["not_boiled_on"] == [True, True], ga
    assert row["success"] == 0 and row["boil_on_s_max"] == 0.0 and "not Gym-Anything's" in ga["summary"], row
    lo, hi = row["viscosity_Pa_s"]                                         # the raw sauce between 20 and 4 °C
    mat = material(T.MATERIAL)
    assert mat.viscosity(T.T_START[1] + 1.0, 1.0, 0.0) < lo <= hi < mat.viscosity(T.T_START[0] - 1.0, 1.0, 0.0), row
