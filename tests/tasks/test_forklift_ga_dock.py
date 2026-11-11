# SPDX-License-Identifier: Apache-2.0
"""GA forklift criteria of the dock tasks: load-trailer (with the red-restraint variant) and load-bales, each criterion
flipped by a minimal mutation just past its threshold."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_forklift_ga_synth as S  # noqa: E402
from alloy.tasks import forklift_ga as ga  # noqa: E402
from alloy.tasks.forklift_site import SITE, docked_trailer_frame  # noqa: E402


def run(task, ep):
    kw = dict(door="door1", side="L") if task == "load-trailer" else {}
    return (ga.load_trailer if task == "load-trailer" else ga.load_bales)(ep, SITE, **kw)


def fails(task, ep) -> set[str]:
    return set(run(task, ep).failed())


# ------------------------------------------------------------------------------------------------ load-trailer
FR = docked_trailer_frame(SITE, "door1", 0.2)
AHEAD_X = SITE.trailer.L - 0.03 - SITE.pallet.L / 2
LY = SITE.trailer.W / 2 - 0.05 - SITE.pallet.W / 2
LX = AHEAD_X - SITE.pallet.L - 0.02


def _lt(dx=0.0, dy=0.0, dz=0.0, tilt=0.0):
    return lambda ep: S.with_load(ep, "carried", S.trailer_pallet(FR, LX + dx, LY + dy, tilt, dz), 800.0)


def _shift_trailer(d):
    return lambda ep: S.with_trailer(ep, "door1", 0.2, shift=d)


LT = [
    ("in_trailer_on_floor", _lt(dz=0.0149), _lt(dz=0.0151), set()),
    ("gap_to_pallets_ahead", _lt(dx=0.0249), _lt(dx=0.0251), set()),
    ("gap_to_pallets_ahead", _lt(dx=-0.0299), _lt(dx=-0.0301), set()),
    ("gap_to_side_wall", _lt(dy=0.0499), _lt(dy=0.0501), {"in_trailer_on_floor"}),
    ("gap_to_side_wall", _lt(dy=-0.0749), _lt(dy=-0.0751), set()),
    # a pallet tilted past 2 deg across its width has a corner more than 15 mm off the floor too
    ("level", _lt(tilt=1.6), _lt(tilt=2.01), {"in_trailer_on_floor"}),
    ("truck_back_on_dock", S.truck_xy(y=18.99), S.truck_xy(y=19.01), set()),
    ("trailer_movement", _shift_trailer(0.0249), _shift_trailer(0.0251), set()),
]


@pytest.mark.parametrize("case", range(len(LT)))
def test_load_trailer_criteria(case):
    name, inside, outside, also = LT[case]
    ep = S.good_load_trailer()
    assert fails("load-trailer", inside(ep)) == set(), name
    assert fails("load-trailer", outside(ep)) == {name} | also, name


def test_load_trailer_level_value_flips_at_two_degrees():
    ep = S.good_load_trailer()
    assert run("load-trailer", _lt(tilt=1.99)(ep)).by_name()["level"].passed
    assert not run("load-trailer", _lt(tilt=2.01)(ep)).by_name()["level"].passed


def test_load_trailer_dropped_load_only_counts_away_from_its_slot():
    ep = S.good_load_trailer()
    slot = FR.origin + FR.R @ np.array([LX, LY, 0.0])
    assert fails("load-trailer", S.with_floor(ep, "carried", 2, xy=slot[:2])) == set()
    assert fails("load-trailer", S.with_floor(ep, "carried", 2, xy=slot[:2] + [0.0,
                                                                               0.4])) == set()  # 0.3 m margin + slot
    assert fails("load-trailer", S.with_floor(ep, "carried", 2, xy=(50.0, 50.0))) == {"no_dropped_load"}
    far = FR.origin + FR.R @ np.array([2.0, LY, 0.0])  # 2 m behind the slot
    assert fails("load-trailer", S.with_floor(ep, "carried", 2, xy=far[:2])) == {"no_dropped_load"}


def test_load_trailer_right_side_mirrors_left():
    ep = S.good_load_trailer(side="R", door="door2")
    assert ga.load_trailer(ep, SITE, door="door2", side="R").success
    assert "gap_to_side_wall" in ga.load_trailer(ep, SITE, door="door2", side="L", behind="trailer_1R").failed()


# ------------------------------------------------------------------------------------------------ red-restraint variant
def good_unsecured() -> object:
    ep = S.base_episode((5.0, 15.0, 0.0), math.pi / 2, tip_z=0.4)
    ep = S.set_tips(ep, (5.0, 16.0), (5.0, 16.0), 0.4)
    return S.with_load(ep, "carried", S.pallet_corners((5.0, 16.8, 0.4), np.eye(3)), 800.0)


def _traj_y(idx, y):
    def f(ep):
        pos = np.array(ep.truck_pos)
        pos[idx, 1] = y
        return replace(ep, truck_pos=pos)
    return f


def _carried_tip_dist(d):
    return lambda ep: S.set_tips(ep, (5.0, 16.8 - d), (5.0, 16.8 - d), 0.4)


UNSECURED = [
    ("stayed_off_the_leveler", _traj_y(10, 15.7615), _traj_y(10, 15.7617)),
    ("waiting_at_the_door", _traj_y(-1, 13.7001), _traj_y(-1, 13.6999)),
    ("stopped_short", S.rest_for(16), S.rest_for(15)),
    ("stopped_short", S.last_signal("park", 0.51), S.last_signal("park", 0.5)),
    ("stopped_short", S.identity, S.last_signal("direction", -1)),
    ("load_on_forks", _carried_tip_dist(1.2191), _carried_tip_dist(1.2193)),
]


@pytest.mark.parametrize("case", range(len(UNSECURED)))
def test_load_trailer_unsecured_variant_criteria(case):
    name, inside, outside = UNSECURED[case]
    ep = good_unsecured()
    v = lambda e: set(ga.load_trailer(e, SITE, door="door1", side="L", secured=False).failed())
    assert v(ep) == set()
    assert v(inside(ep)) == set(), name
    assert v(outside(ep)) == {name}, name


def test_unsecured_variant_shares_the_site_rules():
    ep = S.with_speed(good_unsecured(), 10, 2.445)
    assert ga.load_trailer(ep, SITE, door="door1", side="L", secured=False).failed() == ["speed_limit"]


# ------------------------------------------------------------------------------------------------ load-bales
BFR = docked_trailer_frame(SITE, "door3", 0.2)
BX = SITE.trailer.L - 0.02 - SITE.bale.size[0] / 2


def _bale(name="bale_0", lx=BX, ly=0.411, tilt=0.0):
    return lambda ep: S.with_load(ep, name, S.bale_corners(BFR, lx, ly, tilt), 500.0)


def _past(y):
    return lambda ep: S.set_tips(ep, (12.3, y), (12.3, y), 0.1)


def _drop(name):
    return lambda ep: S.with_floor(ep, name, 2)


def _third_bale(ep):
    return S.with_load(ep, "bale_2", S.bale_corners(BFR, 3.0, 0.0), 500.0)


def _without(name):
    return lambda ep: replace(ep, loads={k: v for k, v in ep.loads.items() if k != name})


def _bale_trailer(d):
    return lambda ep: S.with_trailer(ep, "door3", 0.2, shift=d)


BALES = [
    ("two_bales_at_nose", _bale(tilt=-4.9), _bale(tilt=-5.1)),  # leaning away from the other bale
    ("two_bales_at_nose", _bale(lx=BX + 0.0249), _bale(lx=BX + 0.0251)),
    ("two_bales_at_nose", _bale(lx=BX - 0.0549), _bale(lx=BX - 0.0551)),
    ("two_bales_at_nose", _bale("bale_1", ly=-0.411 + 0.0649), _bale("bale_1", ly=-0.411 + 0.0651)),
    ("two_bales_at_nose", S.identity, _third_bale),
    ("two_bales_at_nose", S.identity, _without("bale_1")),
    ("truck_back_on_dock", _past(19.999), _past(20.001)),
    ("truck_back_on_dock", S.truck_xy(y=18.379), S.truck_xy(y=18.381)),
    ("trailer_movement", _bale_trailer(0.0249), _bale_trailer(0.0251)),
    ("no_dropped_load", _drop("nose_dunnage_0a"), _drop("bale_0")),
]


@pytest.mark.parametrize("case", range(len(BALES)))
def test_load_bales_criteria(case):
    name, inside, outside = BALES[case]
    ep = S.good_load_bales()
    assert fails("load-bales", inside(ep)) == set(), name
    assert fails("load-bales", outside(ep)) == {name}, name


def _nose(door, **kw):
    base = dict(mover="load:bale_0", structure="trailer_wall", parts=("bale_0:box", f"{door}:nose"), impulse_ns=900.0)
    return lambda ep: S.with_contact(ep, **{**base, **kw})


NOSE = [  # (contact kwargs, is a strike)
    (dict(onset_speed=0.05, onset_normal=0.05, peak_n=4905.0), False),  # resting against the wall: below its weight
    (dict(onset_speed=0.05, onset_normal=0.05, peak_n=4905.1), True),  # pressed harder than its weight
    (dict(onset_speed=0.5, onset_normal=0.10, peak_n=1999.0), False),  # a fast one, but under the strike force
    (dict(onset_speed=0.5, onset_normal=0.10, peak_n=2001.0), True),
    (dict(onset_speed=0.5, onset_normal=0.0999, peak_n=2001.0), False),  # approached slower than 0.10 m/s, weight bound
]


@pytest.mark.parametrize("case", range(len(NOSE)))
def test_load_bales_nose_wall_rule(case):
    kw, strike = NOSE[case]
    got = fails("load-bales", _nose("door3", **kw)(S.good_load_bales()))
    assert got == ({"no_strike"} if strike else set())


def test_nose_wall_rule_is_scoped_to_the_target_trailer():
    kw = dict(onset_speed=0.5, onset_normal=0.5, peak_n=1999.0, impulse_ns=60.0)
    assert fails("load-bales", _nose("door3", **kw)(S.good_load_bales())) == set()
    assert fails("load-bales", _nose("door4", **kw)(S.good_load_bales())) == {"no_strike"}


def _row_bales(ep, frame_free=True):
    """Both bales still at the bale row on the dock floor, the truck backed out: an untouched scene."""
    for k, y in enumerate((5.0, 7.0)):
        ep = S.with_load(ep, f"bale_{k}", S.box_corners((21.0, y, 0.6096), np.eye(3), SITE.bale.size), 500.0)
    return replace(ep, loads={k: v for k, v in ep.loads.items() if "dunnage" not in k})


def test_load_bales_reward_untouched_scene_scores_zero_and_success_one():
    done = ga.load_bales(S.good_load_bales(), SITE)
    assert done.success and done.score == 1.0 and done.details["reward"]["progress"] == pytest.approx(1.0)
    idle = ga.load_bales(_row_bales(S.good_load_bales()), SITE)
    assert not idle.success and idle.score == 0.0 and idle.details["reward"]["placement"] == 0.0
    assert ga.load_bales(replace(S.good_load_bales(), fault="x"), SITE).score == 0.0


def test_load_bales_reward_partial_progress_and_workmanship():
    ep = S.good_load_bales()
    start = {"bale_0": (BFR.origin[0], BFR.origin[1]), "bale_1": (21.0, 6.0)}
    # bale_1 still on the trailer floor halfway in, bale_0 placed
    half = S.with_load(ep, "bale_1", S.bale_corners(BFR, 6.0, 0.0), 500.0)
    v = ga.load_bales(half, SITE, bale_start_xy=start)
    assert not v.success and 0.0 < v.score < 0.9
    assert v.details["reward"]["bales"]["bale_0"] == 1.0 and 0.0 < v.details["reward"]["bales"]["bale_1"] < 1.0
    hit = S.with_contact(half, mover="truck", structure="wall", peak_n=9000.0, impulse_ns=200.0, onset_speed=0.4)
    worse = ga.load_bales(hit, SITE, bale_start_xy=start)
    assert 0.0 < worse.score < v.score
    assert worse.details["reward"]["factors"]["no_strike"] < 1.0


def test_load_bales_bale_on_a_floor_earns_no_placement():
    ep = S.with_floor(S.good_load_bales(), "bale_0", 2)
    ep = S.with_load(ep, "bale_0", S.box_corners((BFR.origin[0], 25.0, 0.6096 + 0.2), S.rot_z(math.pi / 2),
                                                 SITE.bale.size))
    v = ga.load_bales(ep, SITE)
    assert "no_dropped_load" in v.failed() and v.score < 0.9
