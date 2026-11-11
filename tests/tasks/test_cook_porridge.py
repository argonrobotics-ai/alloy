# SPDX-License-Identifier: Apache-2.0
"""cook_porridge (alloy/tasks/cook_porridge.py): the spoon's fold loop, the portions, and a short CPU run of the SPH
reheat (light test; the full 2-minute task runs on the GPU smoke)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import cook_porridge as T

pytestmark = pytest.mark.law


def test_spoon_loop_sweeps_the_floor_and_walls_and_leaves_the_porridge():
    period = T.LOOP / T.SPEED
    t = np.linspace(0.0, period, 2001)
    path = np.stack([T.spoon_centre(x) for x in t])
    assert np.allclose(path[0], path[-1], atol=1e-12) and 2.0 < period < 3.0     # a closed loop, 2.6 s
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert step.max() <= T.SPEED * (t[1] - t[0]) * (1.0 + 1e-9)                    # constant speed, no jumps
    assert path[:, 2].min() - T.BLADE_HALF[2] == pytest.approx(T.FLOOR_CLEAR)      # 1.5 mm over the floor
    assert np.abs(path[:, 0]).max() + T.BLADE_HALF[0] == pytest.approx(0.5 * T.INNER[0] - T.CLEAR)
    assert path[:, 2].max() - T.BLADE_HALF[2] > T.DEPTH                           # out of the porridge on the return
    assert 0.5 * T.INNER[1] - T.BLADE_HALF[1] >= 4.0e-3 and np.allclose(path[:, 1], 0.0)
    floor = np.isclose(path[:, 2], T.Z_LOW)
    ends = path[floor, 0].min(), path[floor, 0].max()                              # both floor strokes, wall to wall
    assert ends == pytest.approx((-T.X_END, T.X_END), abs=1e-4)
    down = np.isclose(np.abs(path[:, 0]), T.X_DOWN) & ~floor                       # descents keep 8 mm off the wall
    assert down.any() and T.X_DOWN < T.X_END


def test_portions_are_equal_halves_by_x_then_height():
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, (1, 400, 3))
    temp = np.where(q[..., 0] < np.median(q[0, :, 0]), 300.0, 340.0) + np.where(q[..., 2] > 0.0, 1.0, 0.0)
    part = T.portions(q, temp)[0]
    assert part[0] == pytest.approx(300.0, abs=1.0) and part[3] == pytest.approx(341.0, abs=1.0)
    assert part[1] > part[0] and part[3] > part[2]                        # the upper layer of each half is warmer


def test_short_reheat_delivers_the_hob_and_drives_the_spoon():
    row = T.run(2, "cpu", seed=0, periods=3)
    assert row["neighbor_overflow_max"] == 0 and row["heater_idle_J_max"] == 0.0, row
    assert row["hob_ratio"] == pytest.approx([1.0, 1.0], rel=1e-9), row   # η·P·t into the porridge
    assert row["spoon_err_mm_max"] < 1e-3, row                            # the kinematic blade follows the chords
    assert row["burnt_max"] < 1.0 and row["evaporated_in_solver_max"] == 0, row
    assert row["spilt_max"] == 0.0 and row["burnt_cm2_max"] == 0.0, row   # judged from the state, not assumed
    ga = row["ga"]                                                        # cold porridge after 60 ms: not hot yet
    assert ga["criteria"]["hot_through"] == [False, False] and ga["criteria"]["contained"] == [True, True], ga
    assert row["success"] == 0 and ga["judged"] == "3 of 5", row


def test_the_verdict_judges_three_of_gym_anythings_five_criteria_and_says_so():
    """Audit AB-gym-4 (coordinator rule 2026-11-09): every Gym-Anything criterion is judged from the simulation state
    or listed as not judged (None, with the reason); success counts the judged ones only; the verdict says how many
    of theirs that is and that the scene is smaller than theirs."""
    hot = np.array([True, True, True, False])
    ok, ga = T.verdict(hot, burnt_cm2=np.array([0.0, 2.25, 0.0, 0.0]), out=np.array([0.0, 0.0, 0.02, 0.0]))
    assert ok.tolist() == [True, False, False, False]
    assert ga["criteria"] == {"hot_through": [True, True, True, False], "not_scorched": [True, False, True, True],
                              "contained": [True, True, False, True], "vessel_in_place": None, "appliance_off": None}
    assert ga["judged"] == "3 of 5" and ga["judged_criteria"] == ["hot_through", "not_scorched", "contained"]
    assert set(ga["not_judged"]) == {"vessel_in_place", "appliance_off"}
    assert "3 of Gym-Anything's 5 criteria judged" in ga["summary"] and "smaller than Gym-Anything's" in ga["summary"]
    assert "smaller scene than Gym-Anything's" in T.__doc__ and "Not judged" in T.__doc__


def test_spilt_counts_porridge_outside_the_pot_only():
    hx, hy = 0.5 * T.INNER[0], 0.5 * T.INNER[1]
    q = np.zeros((1, 100, 3)) + (0.0, 0.0, 0.02)                         # all in the pot
    assert T.spilt(q).tolist() == [0.0]
    q[0, 0] = (hx + 0.02, 0.0, 0.01)                                     # over the wall, beside the pot
    q[0, 1] = (0.0, 0.0, -0.05)                                          # below the floor
    q[0, 2] = (0.0, hy - 0.001, 0.2)                                     # thrown up over the opening: not outside
    assert T.spilt(q) == pytest.approx([0.02])


def test_a_criterion_given_as_none_must_come_with_its_reason():
    from alloy.tasks.cook_common import ga_verdict

    with pytest.raises(ValueError, match="not_judged"):
        ga_verdict("x/y", {"a": np.array([True]), "b": None}, {}, "scene")
    ok, ga = ga_verdict("x/y", {"a": np.array([True, False]), "b": None}, {"b": "static"}, "scene")
    assert ok.tolist() == [True, False] and ga["judged"] == "1 of 2" and ga["criteria"]["b"] is None
