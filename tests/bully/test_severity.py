# SPDX-License-Identifier: Apache-2.0
"""SPEC §4.3 severity ladder, row by row; BullyLimits == CheckPolicy; classifier and badness."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.bully.severity import (FLAG_ROWS, MONITOR_ROWS, MONITORS, NONFINITE_BADNESS, S1, S2, badness,
                                  classify, classify_flags, classify_monitors, world_cause)
from alloy.bully.types import BullyLimits
from alloy.core.errors import BullyError
from alloy.precision.types import CheckPolicy, HealthFlags, HealthReport, Severity

LIM = BullyLimits()

# §4.3 rows: flag -> (bully severity or None, StepHealth severity)
SPEC_FLAG_TABLE = {
    "NONFINITE_Q": (S1, Severity.INVALID), "NONFINITE_QD": (S1, Severity.INVALID),
    "NONFINITE_PARTICLE": (S1, Severity.INVALID), "CONTACT_OVERFLOW": (S1, Severity.INVALID),
    "ROW_OVERFLOW": (S1, Severity.INVALID), "PEN_FAIL": (S1, Severity.INVALID),
    "PAIR_PEN_FAIL": (S1, Severity.INVALID), "SPEED_FAIL": (S1, Severity.INVALID),
    "PEN_WARN": (S2, Severity.DEGRADED), "PAIR_PEN_WARN": (S2, Severity.DEGRADED),
    "ENERGY_JUMP": (S2, Severity.DEGRADED), "JOINT_FAIL": (S2, Severity.DEGRADED),
    "FAR_FROM_ORIGIN": (None, Severity.OK), "UNCONVERGED": (None, Severity.DEGRADED),
    "FALLBACK_USED": (None, Severity.OK), "DET_DOWNGRADED": (None, Severity.OK),      # notices (ruling 2026-11-10)
    "LATCHED": (None, Severity.INVALID),
}
if "RUNG_PINNED" in HealthFlags.__members__:  # SCR-JOB2-1: WP5 adds the bit; the row applies once it exists
    SPEC_FLAG_TABLE["RUNG_PINNED"] = (None, Severity.DEGRADED)
if "STRAIN_RANGE" in HealthFlags.__members__:  # ruling 2026-11-06T14:17Z: WP5 adds the bit (soft tet strain > 60 %)
    SPEC_FLAG_TABLE["STRAIN_RANGE"] = (S2, Severity.DEGRADED)
for _name in ("START_PEN", "SPIN"):  # WP5 adds the bits (SCR-WP5-14 start overlap; spin > 0.5 rad / step): flag only
    if _name in HealthFlags.__members__:
        SPEC_FLAG_TABLE[_name] = (None, Severity.DEGRADED)
if "CLOTH_STRETCH" in HealthFlags.__members__:  # SCR-WP5-17: WP5 adds the bit (cloth stretch > 50 %), like STRAIN_RANGE
    SPEC_FLAG_TABLE["CLOTH_STRETCH"] = (S2, Severity.DEGRADED)
if "SELF_CONTACT_OVERFLOW" in HealthFlags.__members__:  # gap #13 (WP1): VBD self-contact rows over their buffers
    SPEC_FLAG_TABLE["SELF_CONTACT_OVERFLOW"] = (S2, Severity.DEGRADED)
if "FAST_CONTACT" in HealthFlags.__members__:  # GAP-E2: a body closed on a static shape past the contact gap in a step
    SPEC_FLAG_TABLE["FAST_CONTACT"] = (S2, Severity.DEGRADED)
if "TUNNEL" in HealthFlags.__members__:  # GAP-E2: a body passed through a static shape (INVALID)
    SPEC_FLAG_TABLE["TUNNEL"] = (S1, Severity.INVALID)
# §4.3 bully-monitor rows: monitor -> (severity, threshold)
SPEC_MONITOR_TABLE = {
    "nonfinite": (S1, 0.5), "capacity_overflow": (S1, 0.5), "teleport": (S1, 1.0), "max_speed": (S1, 20.0), "impossible_speed": (S1, 10.0),
    "pen_s1_ratio": (S1, 1.0), "e_gain_cum": (S1, 0.5), "max_pen": (S2, 1e-3), "e_gain": (S2, 1e-2),
    "joint_sep": (S2, 1e-3), "nontarget_disp": (S2, 5e-5), "contact_free_accel": (None, np.inf),
    "impossible_speed_d12": (None, np.inf),
    # target rows (SB-7 per class, coordinator 2026-11-05): counts the target's sim reports, S1 at the first one
    "cut_pool_overflow": (S1, 0.5), "inverted_tets": (S1, 0.5), "cloth_pass_through": (S1, 0.5),
    "cloth_crossings": (S1, 0.5), "cut_face_deviation": (None, np.inf),
}


def _health(flags: int, sev: Severity, w: int = 1) -> HealthReport:
    base = HealthReport.ok(w)
    return HealthReport(np.full(w, int(sev)), np.full(w, flags), base.first_bad_step, base.max_pen_m,
                        base.max_pair_pen_m, base.max_speed_mps, base.invalid_worlds, base.det_mode,
                        base.det_reasons)


def test_every_health_flag_has_exactly_one_row():
    assert {f.name for f in FLAG_ROWS} == {f.name for f in HealthFlags}
    assert set(SPEC_FLAG_TABLE) == {f.name for f in HealthFlags}


@pytest.mark.parametrize("name", sorted(SPEC_FLAG_TABLE))
def test_flag_row(name):
    level, step_sev = SPEC_FLAG_TABLE[name]
    assert FLAG_ROWS[HealthFlags[name]] == level
    sev = classify_flags(_health(int(HealthFlags[name]), step_sev))[0]
    if step_sev == Severity.INVALID:  # D3: INVALID <=> S1 (fail closed), incl. LATCHED
        assert sev == S1
    else:
        assert sev == (0 if level is None else int(level))


def test_pre_registered_flag_rows():
    """Rows for bits WP5 is adding: STRAIN_RANGE and CLOTH_STRETCH are S2 (model-limit reports; S1 not redefined);
    RUNG_PINNED, START_PEN and SPIN flag-only; SELF_CONTACT_OVERFLOW (gap #13, WP1) S2, a capacity report."""
    from alloy.bully.severity import _ADDED_FLAG_ROWS

    assert _ADDED_FLAG_ROWS == {"RUNG_PINNED": None, "STRAIN_RANGE": S2, "START_PEN": None, "SPIN": None,
                                "CLOTH_STRETCH": S2, "SELF_CONTACT_OVERFLOW": S2, "FAST_CONTACT": S2, "TUNNEL": S1}


def test_energy_jump_and_far_from_origin_s1_levels_arrive_as_invalid():
    for name in ("ENERGY_JUMP", "FAR_FROM_ORIGIN"):
        assert classify_flags(_health(int(HealthFlags[name]), Severity.INVALID))[0] == S1


@pytest.mark.parametrize("name", sorted(SPEC_MONITOR_TABLE))
def test_monitor_row_just_above_and_below(name):
    level, thr = SPEC_MONITOR_TABLE[name]
    row = MONITORS[name]
    assert row.severity == level
    assert row.threshold(LIM) == pytest.approx(thr)
    if level is None:
        return
    above, below = np.array([thr * 1.001 if thr else 1e-9]), np.array([thr * 0.999])
    assert classify_monitors({name: above}, LIM)[0] == int(level)
    assert classify_monitors({name: below}, LIM)[0] == 0
    assert classify_monitors({name: np.array([np.nan])}, LIM)[0] == int(level)  # NaN trips


def test_monitor_rows_cover_the_table_once():
    assert [r.name for r in MONITOR_ROWS] == list(dict.fromkeys(r.name for r in MONITOR_ROWS))
    assert set(SPEC_MONITOR_TABLE) == set(MONITORS)


def test_bully_limits_equal_check_policy():
    cp = CheckPolicy()
    assert LIM.pen_limit == cp.pen_limit_m
    assert LIM.pen_s1 == cp.pen_s1_m
    assert LIM.speed_fail == cp.speed_fail_mps
    assert LIM.e_gain_s2_rel == cp.energy_jump_rel
    assert LIM.e_gain_s1_rel == cp.energy_s1_rel
    # bully-only numbers from §4.2 / §4.3
    assert (LIM.v_max, LIM.rest_disp_max, LIM.joint_sep_s2) == (10.0, 5e-5, 1e-3)


def test_worst_severity_wins_and_unknown_monitor_raises():
    h = _health(int(HealthFlags.PEN_WARN), Severity.DEGRADED, w=3)
    extra = {"max_speed": np.array([25.0, 0.0, 0.0]), "nontarget_disp": np.array([0.0, 1e-3, 0.0])}
    assert classify(h, extra, LIM).tolist() == [1, 2, 2]
    with pytest.raises(BullyError):
        classify_monitors({"made_up": np.zeros(1)}, LIM)


def test_badness_formula_and_cap():
    extra = {"max_speed": np.array([10.0, np.inf]), "max_pen": np.array([2e-3, 0.0]), "nonfinite": np.array([0, 1])}
    b = badness(extra, LIM)
    # world 0: ratios 0.5 (S1, not tripped) and 2.0 (S2) -> 2.0 + 0.1 * 0.5
    assert b[0] == pytest.approx(2.05)
    # world 1: two tripped S1 rows at the cap, each 10 + cap (SCR-WP7-6)
    assert b[1] == pytest.approx(1.1 * (10.0 + NONFINITE_BADNESS))
    ok = HealthReport.ok(2)
    assert world_cause(ok, extra, LIM, 0) == ("max_pen", 2e-3, 1e-3, 2)


def test_badness_ranks_an_s1_world_above_an_s2_outlier():
    """SCR-WP7-6: a bystander S2 ratio of 371 (the 1-h B1 run) caps at 10; a just-tripped S1 row scores 10 + r."""
    extra = {"nontarget_disp": np.array([371 * 5e-5, 0.0]), "teleport": np.array([0.0, 1.01])}
    b = badness(extra, LIM)
    assert b[0] == pytest.approx(10.0) and b[1] == pytest.approx(11.01)
    assert b[1] > b[0]


def test_world_cause_is_the_row_that_set_the_severity():
    # teleport ratio 1.2 is S1, max_pen ratio 5 is only S2: the S1 row names the finding, not the larger ratio
    extra = {"teleport": np.array([1.2]), "max_pen": np.array([5e-3])}
    assert world_cause(HealthReport.ok(1), extra, LIM, 0)[::3] == ("teleport", 1)
    # a world tripped only by a health flag is named by the flag; an INVALID world without an S1 row fails closed
    assert world_cause(_health(int(HealthFlags.PEN_WARN), Severity.DEGRADED), {}, LIM, 0) == ("PEN_WARN", 1.0, 0.5, 2)
    assert world_cause(_health(int(HealthFlags.ENERGY_JUMP), Severity.INVALID), {}, LIM, 0)[0] in ("INVALID",)
