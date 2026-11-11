# SPDX-License-Identifier: Apache-2.0
"""GA forklift criteria: for each task a passing synthetic episode, and for each criterion a minimal mutation that fails
exactly that criterion just past its threshold (and passes just inside it). CPU-only, numpy."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_forklift_ga_synth as S  # noqa: E402
from alloy.tasks import forklift_ga as ga
from alloy.tasks.forklift_ga_types import contact_episodes_from_series
from alloy.tasks.forklift_site import SITE, rack_location

# task -> (function, good-episode factory, kwargs, time-limit key)
TASKS = {
    "start-and-park": (ga.start_and_park, S.good_park, {}),
    "ramp-down-loaded": (ga.ramp_down_loaded, S.good_ramp_down, {}),
    "put-away-rack": (ga.put_away_rack, S.good_put_away, {}),
    "load-trailer": (ga.load_trailer, S.good_load_trailer, dict(door="door1", side="L")),
    "load-bales": (ga.load_bales, S.good_load_bales, {}),
}


def run(task: str, ep):
    fn, _, kw = TASKS[task]
    return fn(ep, SITE, **kw)


def fails(task: str, ep) -> set[str]:
    return set(run(task, ep).failed())


def good(task: str):
    return TASKS[task][1]()


@pytest.mark.parametrize("task", list(TASKS))
def test_good_episode_passes(task):
    v = run(task, good(task))
    assert v.success and v.score == 1.0 and not v.failed()


@pytest.mark.parametrize("task", list(TASKS))
def test_fault_ends_without_success(task):
    v = run(task, replace(good(task), fault="non-finite state"))
    assert not v.success and not v.failed()
    assert v.score == (0.0 if task == "load-bales" else 1.0)


@pytest.mark.parametrize("task", list(TASKS))
def test_operator_criteria_are_not_judged_without_a_humanoid(task):
    by = run(task, good(task)).by_name()
    for name in ("operator_seated", "head_clear_of_guard"):
        assert by[name].passed is None and by[name].note


# criteria not judged on a passing synthetic episode, and (judged, all) per task
NOT_JUDGED = {
    "start-and-park": ["operator_seated", "head_clear_of_guard", "time_limit", "parking_brake_set",
                       "direction_neutral", "engine_off", "key_off"],
    "ramp-down-loaded": ["operator_seated", "head_clear_of_guard", "time_limit"],
    "put-away-rack": ["operator_seated", "head_clear_of_guard", "time_limit"],
    "load-trailer": ["operator_seated", "head_clear_of_guard", "time_limit"],
    "load-bales": ["operator_seated", "head_clear_of_guard", "time_limit"],
}
ALL = {"start-and-park": 16, "ramp-down-loaded": 14, "put-away-rack": 15, "load-trailer": 14, "load-bales": 11}


@pytest.mark.parametrize("task", list(TASKS))
def test_judged_counts_and_success_on_a_passing_episode(task):
    v = run(task, good(task))
    assert v.success and v.score == 1.0
    assert sorted(v.not_judged()) == sorted(NOT_JUDGED[task])
    assert v.judged == (ALL[task] - len(NOT_JUDGED[task]), ALL[task])
    assert v.passed_count == (v.judged[0], *v.judged)
    assert v.summary().startswith(f"{v.judged[0]} passed of {v.judged[0]} judged; {v.judged[0]} of {ALL[task]} ")
    assert "not judged: operator_seated" in v.summary()


def test_start_and_park_command_echoes_can_fail_but_never_pass():
    by = run("start-and-park", good("start-and-park")).by_name()
    for name in ("parking_brake_set", "direction_neutral", "engine_off", "key_off"):
        assert by[name].passed is None
    bad = S.last_signal("park", 0.0)(good("start-and-park"))
    assert run("start-and-park", bad).by_name()["parking_brake_set"].passed is False
    assert not run("start-and-park", bad).success


@pytest.mark.parametrize("task", list(TASKS))
def test_a_failing_judged_criterion_fails_success(task):
    v = run(task, S.with_roll(good(task), 5, 31.0))
    assert not v.success and v.failed() == ["roll_under_6deg"]
    assert v.passed_count[0] == v.judged[0] - 1


@pytest.mark.parametrize("task", list(TASKS))
def test_time_limit_is_judged_from_the_plan_finishing(task):
    limit = SITE.limits.time_limit_s[task]
    done = run(task, replace(good(task), plan_known=True, plan_done_t=limit - 1.0))
    cut = run(task, replace(good(task), plan_known=True, plan_done_t=None))
    assert done.by_name()["time_limit"].passed is True and done.success
    assert cut.by_name()["time_limit"].passed is False and not cut.success


def test_every_criterion_not_judged_is_no_success():
    from alloy.tasks.forklift_ga_types import Criterion, make_verdict
    c = Criterion("x", 0, None, "-", "-", "-")
    assert not make_verdict("t", [c], None).success


# ------------------------------------------------------------------------------------------------ shared rules
def _lift(n):
    def f(ep):
        fz = np.array(ep.wheel_fz)
        fz[3:3 + n, 1] = 0.0
        return replace(ep, wheel_fz=fz)
    return f


def _time(limit):
    return lambda ep: replace(ep, t=np.linspace(0.0, limit, ep.n), plan_known=True,
                              plan_done_t=limit if limit <= SITE.limits.time_limit_s[_TASK[0]] else None)


_TASK = [""]


def _seated(n):
    def f(ep):
        ep = replace(ep, humanoid=True)
        ep = S.with_speed(ep, slice(5, 5 + n), 1.0)
        return S.with_signal(ep, "seat_switch_closed", False, slice(5, 5 + n))
    return f


def _head(force):
    return lambda ep: replace(ep, humanoid=True, head_contact_n=np.where(np.arange(ep.n) == 7, force, 0.0))


SHARED = [  # (criterion, inside, outside)
    ("no_strike", lambda e: S.with_contact(e, peak_n=2000.0, impulse_ns=50.0), lambda e: S.with_contact(e,
                                                                                                        peak_n=2000.1)),
    ("no_strike", S.identity, lambda e: S.with_contact(e, peak_n=100.0, impulse_ns=50.1)),
    ("no_wheel_lift", _lift(2), _lift(3)),
    ("roll_under_6deg", lambda e: S.with_roll(e, 5, 5.9), lambda e: S.with_roll(e, 5, 6.1)),
    ("roll_under_6deg", S.identity, lambda e: S.with_roll(e, 5, 31.0)),
    ("no_dropped_load", lambda e: S.with_floor(e, "bale_0", 1), lambda e: S.with_floor(e, "bale_0", 2)),
    ("speed_limit", lambda e: S.with_speed(e, 10, 2.444), lambda e: S.with_speed(e, 10, 2.445)),
    ("operator_seated", _seated(8), _seated(9)),
    ("head_clear_of_guard", _head(130.0), _head(130.1)),
    ("time_limit", None, None),  # per task limit
]


@pytest.mark.parametrize("task", list(TASKS))
@pytest.mark.parametrize("case", range(len(SHARED)))
def test_shared_rules_flip_at_their_thresholds(task, case):
    name, inside, outside = SHARED[case]
    if name == "time_limit":
        limit = SITE.limits.time_limit_s[task if task != "load-trailer" else "load-trailer"]
        _TASK[0] = task
        inside, outside = _time(limit), _time(limit + 0.1)
    assert fails(task, inside(good(task))) == set(), name
    assert fails(task, outside(good(task))) == {name}, name


def test_building_vs_yard_speed_rule_and_ramp_zone():
    ep = S.with_speed(S.good_ramp_down(), 10, 2.5)  # off the ramp, above 8.8 km/h: the yard rule fails speed_limit only
    assert fails("ramp-down-loaded", ep) == {"speed_limit"}
    assert run("ramp-down-loaded", ep).by_name()["speed_limit"].value == pytest.approx(9.0)


def test_strike_exemptions_set_down_on_beams_floors_and_leaning():
    base = S.good_put_away()
    beam = ("carried:pallet", "A0_beam0_1_0.62")
    ok = dict(mover="load:carried", structure="rack", parts=beam, onset_speed=0.10, peak_n=25000.0, impulse_ns=900.0)
    assert fails("put-away-rack", S.with_contact(base, **ok)) == set()
    assert fails("put-away-rack", S.with_contact(base, **{**ok, "onset_speed": 0.1001})) == {"no_strike"}
    assert fails("put-away-rack", S.with_contact(base, **{**ok, "peak_n": 25001.0})) == {"no_strike"}
    # a truck, or a load not on a beam, gets no beam set-down exemption
    assert fails("put-away-rack", S.with_contact(base, **{**ok, "mover": "truck"})) == {"no_strike"}
    wall = dict(mover="load:carried", structure="wall", parts=("carried:pallet", "wall_w"), onset_speed=0.09)
    assert fails("put-away-rack", S.with_contact(base, peak_n=1000.0 * 9.81, impulse_ns=900.0, **wall)) == set()
    assert fails("put-away-rack", S.with_contact(base, peak_n=1000.0 * 9.81 + 1, impulse_ns=900.0,
                                                 **wall)) == {"no_strike"}
    fast = {**wall, "onset_speed": 0.10}
    assert fails("put-away-rack", S.with_contact(base, peak_n=1000.0, impulse_ns=10.0, **fast)) == set()
    assert fails("put-away-rack", S.with_contact(base, peak_n=1000.0, impulse_ns=51.0, **fast)) == {"no_strike"}


def test_trailer_floor_set_down_bound_and_truck_driving_on_it():
    base = S.good_load_trailer()  # the heaviest load, 800 kg: bound = m (g + B v)
    bound = 800.0 * (9.81 + SITE.limits.set_down_decel)
    mk = dict(mover="load:carried", structure="trailer_floor", impulse_ns=999.0, onset_speed=0.5, onset_normal=0.10)
    assert fails("load-trailer", S.with_contact(base, peak_n=bound - 1, **mk)) == set()
    assert fails("load-trailer", S.with_contact(base, peak_n=bound + 1, **mk)) == {"no_strike"}
    assert fails("load-trailer", S.with_contact(base, peak_n=bound - 1, **{**mk,
                                                                           "onset_normal": 0.1001})) == {"no_strike"}
    assert fails("load-trailer", S.with_contact(base, peak_n=1e6, **{**mk, "mover": "truck"})) == set()  # driven on


def test_contact_episodes_from_series_groups_runs_and_integrates_impulse():
    t = np.arange(10) * 0.5
    f = np.array([0, 0, 100, 300, 0, 0, 50, 50, 50, 50.0])
    eps = contact_episodes_from_series("truck", "wall", t, f, np.arange(10.0), parts=("a", "b"))
    assert len(eps) == 2 and eps[0].peak_n == 300 and eps[0].impulse_ns == pytest.approx(200.0)
    assert eps[0].onset_speed == 2.0 and not eps[0].open and eps[1].open and eps[1].t_end is None


# ------------------------------------------------------------------------------------------------ start-and-park
PARK = [  # (criterion, inside, outside, fails inside, other criteria failing outside)
    ("in_parking_bay", S.truck_xy(x=3.46), S.truck_xy(x=3.461), set(), set()),
    ("in_parking_bay", S.truck_xy(y=27.161), S.truck_xy(y=27.159), set(), set()),
    ("in_parking_bay", S.truck_xy(y=29.339), S.truck_xy(y=29.341), set(), set()),
    ("stationary_2s", S.rest_for(16), S.rest_for(15), set(), set()),
    ("parking_brake_set", S.last_signal("park", 0.51), S.last_signal("park", 0.5), set(), set()),
    ("direction_neutral", S.identity, S.last_signal("direction", 1), set(), set()),
    ("engine_off", S.identity, S.last_signal("engine_running", True), set(), set()),
    ("key_off", S.identity, S.last_signal("key_on", True), set(), set()),
    # the 20 mm bound is looser than the 5 mm tilt-to-floor bound, so a raised fork also fails that one
    ("forks_lowered", S.tips_z(0.0199), S.tips_z(0.0201), {"forks_tilted_to_floor"}, {"forks_tilted_to_floor"}),
    ("forks_tilted_to_floor", S.tips_z(0.0049), S.tips_z(0.0051), set(), set()),
    ("forks_tilted_to_floor", S.identity, lambda e: replace(e, blade_pitch=np.full(e.n, 1e-4)), set(), set()),
]


@pytest.mark.parametrize("case", range(len(PARK)))
def test_start_and_park_criteria(case):
    name, inside, outside, in_fail, also = PARK[case]
    ep = S.good_park()
    if name == "forks_lowered":
        ep = replace(ep, blade_pitch=np.full(ep.n, -0.01))
    assert fails("start-and-park", inside(ep)) == in_fail, name
    assert fails("start-and-park", outside(ep)) == {name} | also, name


# ------------------------------------------------------------------------------------------------ ramp-down-loaded
CX, CY = 15.5, 6.5


def _pallet(dx=0.0, dz=0.0, tilt=0.0):
    def f(ep):
        R = S.rot_y(math.radians(tilt))
        return S.with_load(ep, "carried", S.pallet_corners((CX + dx, CY, dz), R), 1000.0)
    return f


def _tip_back(d):
    return lambda ep: S.set_tips(ep, (CX - d, CY), (CX - d, CY), 0.05)


def _ramp_speed(v):
    def f(ep):
        pos = np.array(ep.truck_pos)
        pos[3] = (10.0, 0.0, 0.0)
        return S.with_speed(replace(ep, truck_pos=pos), 3, v)
    return f


def _ramp_floor(xy):
    return lambda ep: S.with_floor(ep, "carried", 2, xy=xy)


RAMP = [
    ("in_drop_zone", _pallet(dx=0.889), _pallet(dx=0.892), set(), set()),
    ("in_drop_zone", _pallet(tilt=4.9), _pallet(tilt=5.1), {"resting_on_floor"}, {"resting_on_floor"}),
    ("resting_on_floor", _pallet(dz=0.0099), _pallet(dz=0.0101), set(), set()),
    ("backed_out", _tip_back(1.1097), _tip_back(1.1095), set(), set()),
    ("forks_lowered", S.tips_z(0.1499), S.tips_z(0.1501), set(), set()),
    ("stationary", S.rest_for(8), S.rest_for(7), set(), set()),
    ("slow_on_ramp", _ramp_speed(1.666), _ramp_speed(1.668), set(), set()),
    ("no_dropped_load", _ramp_floor((CX, CY)), _ramp_floor((12.0, CY)), set(), set()),
]


@pytest.mark.parametrize("case", range(len(RAMP)))
def test_ramp_down_criteria(case):
    name, inside, outside, in_fail, also = RAMP[case]
    ep = S.good_ramp_down("east")
    assert fails("ramp-down-loaded", inside(ep)) == in_fail, name
    assert fails("ramp-down-loaded", outside(ep)) == {name} | also, name


def test_ramp_down_west_zone_and_ramp_speed_zone_is_ga_literal():
    west = S.good_ramp_down("west")
    v = ga.ramp_down_loaded(west, SITE, drop_zone="west")
    assert v.success
    assert not ga.ramp_down_loaded(west, SITE, drop_zone="east").success
    pos = np.array(west.truck_pos)
    pos[3] = (12.1, 0.0, 0.0)  # 2.1 m off the ramp's centreline: inside GA's zone (|dx| < full width 2.13 m)
    assert ga.ramp_down_loaded(S.with_speed(replace(west, truck_pos=pos), 3, 2.0), SITE, "west").by_name()[
        "slow_on_ramp"].passed is False
    pos[3] = (12.2, 0.0, 0.0)  # past it
    assert ga.ramp_down_loaded(S.with_speed(replace(west, truck_pos=pos), 3, 2.0), SITE, "west").by_name()[
        "slow_on_ramp"].passed


# ------------------------------------------------------------------------------------------------ put-away-rack
LOC = rack_location(SITE, "A-e-01-1-0")


def _put(overhang=0.076, dy=0.0, dz=0.0, tilt=0.0):
    def f(ep):
        c = S.rack_pallet_centre(LOC, overhang) + np.array([0.0, dy, dz])
        return S.with_load(ep, "carried", S.pallet_corners(c, S.rot_y(math.radians(tilt))), 1000.0)
    return f


def _beams(keep=None, extra=None):
    def f(ep):
        labels = ep.beam_contacts[:keep] if keep is not None else ep.beam_contacts
        return replace(ep, beam_contacts=labels + ((extra,) if extra else ()))
    return f


def _stored_moved(d):
    def f(ep):
        name = next(n for n in ep.loads if n.startswith("stored_"))
        arr = np.array(ep.loads[name])
        arr[-1] += np.array([0.0, d, 0.0])
        return replace(ep, loads=dict(ep.loads, **{name: arr}))
    return f


def _tips_at(d, z=0.05):
    c = S.rack_pallet_centre(LOC)
    return lambda ep: S.set_tips(ep, (c[0] + d, c[1]), (c[0] + d, c[1]), z)


PUT = [
    ("on_both_beams", S.identity, _beams(keep=1), set()),
    ("on_both_beams", S.identity, _beams(extra="A0_beam1_1_0.62"), set()),
    ("on_both_beams", S.identity, _beams(extra="A0_beam0_2_0.62"), set()),
    ("at_beam_level", _put(dz=0.029), _put(dz=0.031), set()),
    ("front_overhang", _put(overhang=0.0511), _put(overhang=0.0509), set()),
    ("front_overhang", _put(overhang=0.1009), _put(overhang=0.1011), set()),
    ("side_clearance", _put(dy=-0.0515), _put(dy=-0.0517), set()),
    ("side_clearance", _put(dy=0.1531), _put(dy=0.1533), set()),
    ("level", _put(tilt=1.99), _put(tilt=2.01), set()),
    ("stored_loads_undisturbed", _stored_moved(0.0299), _stored_moved(0.0301), set()),
    ("forks_withdrawn_lowered", _tips_at(1.001), _tips_at(0.999), set()),
    ("forks_withdrawn_lowered", _tips_at(1.5, 0.2999), _tips_at(1.5, 0.3001), set()),
]


@pytest.mark.parametrize("case", range(len(PUT)))
def test_put_away_criteria(case):
    name, inside, outside, also = PUT[case]
    ep = S.good_put_away()
    assert fails("put-away-rack", inside(ep)) == set(), name
    assert fails("put-away-rack", outside(ep)) == {name} | also, name


def test_put_away_on_a_double_row_west_face_and_upper_level():
    for loc_name in ("B-w-03-2-1", "B-e-07-1-0", "C-w-02-1-1"):
        loc = rack_location(SITE, loc_name)
        ep = S.good_put_away(loc_name)
        assert ga.put_away_rack(ep, SITE, loc_name).success, loc_name
        assert loc.z in (1.6, 3.2)
    wrong = ga.put_away_rack(S.good_put_away("B-w-03-1-0"), SITE, "B-w-03-2-0")
    assert set(wrong.failed()) == {"on_both_beams", "at_beam_level"}
