# SPDX-License-Identifier: Apache-2.0
"""OSS-7 rest drift, penetration, health and energy findings from synthetic run results (no simulation): each
finding's level, numbers, objects and fix."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.check import events, motion
from alloy.check.contacts import PairDepth
from alloy.check.objects import ObjectIndex
from alloy.check.run import RunResult, Sample, checkpoints, over_budget, pick_samples, sample_periods
from alloy.precision.types import FirstFailure, HealthFlags, HealthReport, Severity

W = 2   # worlds; one body ("mug") per world, plus a static table without bodies


def _arr(a):
    return SimpleNamespace(numpy=lambda: np.asarray(a))


def _built():
    return SimpleNamespace(model=SimpleNamespace(body_world=_arr([0, 1]), particle_world=_arr([])))


def _index():
    names = ["table", "mug"]
    return ObjectIndex(names, {"table": "static", "mug": "dynamic"}, {"table": 0, "mug": 1},
                       np.array([0, 1, 0, 1], np.int32), np.array([1, 1], np.int32), np.zeros(0, np.int32), W,
                       {"mug": (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0)})


def _sample(t, z, x=0.0, quat=(0.0, 0.0, 0.0, 1.0), v=0.0):
    q = np.array([[x, 0.0, z, *quat], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]])   # world 1 stays put
    qd = np.array([[v, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], np.float64)
    return Sample(t, q, qd, np.zeros((0, 3)))


def _run(samples, **kw):
    return RunResult(samples=samples, control_dt=0.02, **kw)


def test_sample_periods_cover_settle_mid_end():
    assert sample_periods(100, 0.02) == {12, 50, 100}
    assert sample_periods(4, 0.02) == {1, 2, 4}


def test_budget_rule_has_a_hard_cap():
    assert not over_budget(30.0, 3, 25.0) and over_budget(30.0, 4, 25.0)      # at least 4 periods ...
    assert over_budget(101.0, 1, 25.0) and not over_budget(1e6, 1, None)       # ... unless 4 x the budget is gone


def test_samples_of_a_run_stopped_early():
    cps = checkpoints(100, 0.02)
    assert {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 50, 64, 96, 100} == cps
    taken = {k: _sample(k * 0.02, 0.1) for k in [0, *sorted(c for c in cps if c <= 30), 37]}   # stopped at 37
    picked = [round(x.t / 0.02) for x in pick_samples(taken, 37, 0.02)]
    assert picked == [0, 8, 16, 37]                 # settle mark 9 -> 8, middle 18 -> 16, end 37
    assert [round(x.t / 0.02) for x in pick_samples(taken, 40, 0.02)] == [0, 8, 16, 37]   # error: last taken


def test_at_rest_gives_no_findings():
    s = [_sample(t, 0.1) for t in (0.0, 0.25, 1.0, 2.0)]
    found, metrics = motion.motion_findings(_run(s), _built(), _index())
    assert found == [] and metrics["mug"]["drift_m"] == 0.0


def test_settling_drop_says_lower_it_with_settled_pose():
    s = [_sample(0.0, 0.1), _sample(0.25, 0.0958), _sample(1.0, 0.0958), _sample(2.0, 0.0958)]
    found, _ = motion.motion_findings(_run(s), _built(), _index())
    assert len(found) == 1
    f = found[0]
    assert (f.check, f.level, f.objects) == ("rest_drift", "warn", ("mug",))
    assert "lower mug by 4.2 mm" in f.fix and "world 0" in f.message
    assert f.suggestion["pose"][2] == pytest.approx(0.0958)


def test_fall_and_tip_are_errors_unless_the_scene_moves():
    tipped = (0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4))      # 90 deg about y
    s = [_sample(0.0, 0.1), _sample(0.25, 0.0), _sample(1.0, -0.2), _sample(2.0, -0.2, quat=tipped)]
    found, _ = motion.motion_findings(_run(s), _built(), _index())
    levels = {f.message.split()[1]: f.level for f in found}
    assert levels["fell"] == "error" and levels["tipped"] == "error"
    found, _ = motion.motion_findings(_run(s), _built(), _index(), expect_rest=False)
    assert {f.level for f in found} == {"info"}


def test_late_sliding_is_named_with_friction_fix():
    s = [_sample(0.0, 0.1), _sample(0.25, 0.1), _sample(1.0, 0.1), _sample(2.0, 0.1, x=0.003, v=0.003)]
    found, _ = motion.motion_findings(_run(s), _built(), _index())
    slide = [f for f in found if "slides" in f.message]
    assert slide and slide[0].level == "warn" and "mu_static" in slide[0].fix
    assert slide[0].value == pytest.approx(0.003)


def _health(flags_w0: int, ff_flags: int, pen=np.nan, speed=0.0):
    sev = np.array([Severity.INVALID if ff_flags else Severity.OK, Severity.OK], np.int32)
    n = lambda v: np.array([v, np.nan])   # noqa: E731
    i = lambda v: np.array([v, -1])       # noqa: E731
    ff = FirstFailure(i(30), np.array([ff_flags, 0], np.uint32), n(0.31), n(pen), i(0), i(1), i(-1), n(0.0), i(-1),
                      n(speed), i(1), n(0.0), n(0.0), n(0.0), i(-1), n(np.nan))
    invalid = (0,) if ff_flags else ()
    return HealthReport(sev, np.array([flags_w0, 0], np.uint32), i(30), np.array([1e-4, 2e-4]), np.zeros(W),
                        np.array([speed, 0.0]), invalid, None, (), np.ones(W, bool), first_failure=ff)


def test_health_invalid_world_names_objects_and_fix():
    h = _health(int(HealthFlags.SPEED_FAIL | HealthFlags.LATCHED), int(HealthFlags.SPEED_FAIL), speed=31.0)
    found, metrics = events.health_findings(_run([], health=h), _index())
    assert metrics["invalid_worlds"] == [0]
    (f,) = found
    assert f.level == "error" and f.worlds == (0,) and set(f.objects) == {"table", "mug"}
    assert "31 m/s" in f.message and "t=0.31 s" in f.message and "overlaps" in f.fix


def test_run_exception_is_a_health_error():
    found, _ = events.health_findings(_run([], error="CouplingError: boom", error_t=0.4), _index())
    assert found[0].level == "error" and "t=0.4 s" in found[0].message


def test_penetration_levels_and_attribution():
    pair = PairDepth("mug", "table", 0.004, 1, (0.0, 0.0, -1.0))
    res = _run([], pen_max=np.array([2e-4, 4e-3]), pen_t=np.array([0.1, 0.5]), pair_pen_max=np.zeros(W),
               attributions=[(0.5, pair)])
    found, metrics = events.penetration_findings(res, _index())
    (f,) = found
    assert f.level == "warn" and f.objects == ("mug", "table") and f.worlds == (1,)
    assert "4 mm between mug and table" in f.message and metrics["max_pen_m"] == pytest.approx(4e-3)
    res.pen_max = np.array([np.nan, 0.02])
    assert events.penetration_findings(res, _index())[0][0].level == "error"
    res.attributions = [(0.5, PairDepth("mug", "mug", 0.02, 1, (0.0, 0.0, 1.0)))]      # its own parts
    (f,) = events.penetration_findings(res, _index())[0]
    assert f.level == "info" and "inside mug" in f.message and f.objects == ("mug",)
    res.pen_max = np.array([np.nan, np.nan])
    assert events.penetration_findings(res, _index()) == ([], {"max_pen_m": None})


def test_penetration_peak_and_end_depth():
    # F17: a landing sinks 3 mm for a moment and rests at 10 um; sim.health() reports the end depth
    pair = PairDepth("table", "mug", 0.003, 1, (0.0, 0.0, 1.0))
    res = _run([], pen_max=np.array([2e-4, 3.05e-3]), pen_t=np.array([0.1, 0.2]), pen_end=np.array([1e-5, 1e-5]),
               pair_pen_max=np.zeros(W), attributions=[(0.2, pair)])
    (f,), metrics = events.penetration_findings(res, _index())
    assert f.level == "warn" and metrics["end_pen_m"] == pytest.approx(1e-5)
    assert "3.05 mm between table and mug at t=0.2 s" in f.message
    assert "; 10 um at the end of the run, the depth sim.health() reports" in f.message
    assert f.fix.startswith("it sank for a moment") and "start mug on its support" in f.fix
    res.pen_end = np.array([1e-5, 2e-3])                        # still deep at the end: a resting contact
    (f,) = events.penetration_findings(res, _index())[0]
    assert "2 mm at the end of the run" in f.message and f.fix.startswith("stiffen the contact between table and mug")
    res.pen_end = None                                          # check_sim: no end value, the depth is now
    (f,) = events.penetration_findings(res, _index())[0]
    assert "end of the run" not in f.message and f.fix.startswith("stiffen")


def test_energy_gain_levels():
    mon = {"e_gain_cum": np.array([0.0, 0.7]), "e_gain": np.array([0.0, 0.02])}
    found, metrics = events.energy_findings(_run([], monitors=mon, e_scale=np.ones(W)), _index())
    assert found[0].level == "error" and "70 %" in found[0].message and metrics["gain_cum_rel"] == 0.7
    mon = {"e_gain_cum": np.array([0.0, 0.1]), "e_gain": np.array([0.0, 0.02])}
    assert events.energy_findings(_run([], monitors=mon), _index())[0][0].level == "warn"
    mon = {"e_gain_cum": np.array([0.0, 0.01]), "e_gain": np.array([0.0, 0.02])}     # a one-step blip only
    assert events.energy_findings(_run([], monitors=mon), _index())[0][0].level == "info"
    assert events.energy_findings(_run([]), _index()) == ([], {})


def test_a_sagging_cable_is_judged_by_its_centroid():
    built = SimpleNamespace(model=SimpleNamespace(body_world=_arr([0, 0, 1, 1]), particle_world=_arr([])))
    idx = ObjectIndex(["table", "cable"], {"table": "static", "cable": "dynamic"}, {"table": 0, "cable": 1},
                      np.zeros(0, np.int32), np.ones(4, np.int32), np.zeros(0, np.int32), W, {})

    def smp(t, sag):           # two segments per world; world 0's middle sags by `sag`, its end stays put
        q = np.tile([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], (4, 1))
        q[0, 2] -= sag
        return Sample(t, q, np.zeros((4, 6)), np.zeros((0, 3)))

    s = [smp(0.0, 0.0), smp(0.25, 0.008), smp(1.0, 0.008), smp(2.0, 0.008)]
    found, metrics = motion.motion_findings(_run(s), built, idx)
    assert metrics["cable"]["settle_m"] == pytest.approx(0.004)      # the centroid moves half the segment's 8 mm
    assert [f.level for f in found] == ["info"] and "while settling" in found[0].message


def test_a_run_cut_short_cannot_judge_late_drift():
    s = [_sample(0.0, 0.1), _sample(0.02, 0.1), _sample(0.04, 0.1), _sample(0.08, 0.1, x=0.003, v=0.1)]
    found, _ = motion.motion_findings(_run(s), _built(), _index())
    assert found and {f.level for f in found} == {"info"} and all("too short to judge" in f.message for f in found)
