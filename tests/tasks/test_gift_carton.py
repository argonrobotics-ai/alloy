# SPDX-License-Identifier: Apache-2.0
"""Open-flaps (alloy/tasks/gift_carton.py, gift_flap_path.py, gift_carton_flaps.py; docs/notes/paper-pack.md). CPU.

The carton's flaps sit on Jenkins crease hinges (a dry-friction slider for the plastic set in series with an elastic
spring, 0.8 N·m/m plastic moment, 45 deg elastic range). Laws: closed flaps hold against gravity (the slider's
armature keeps MuJoCo's soft dry friction from creeping); a flap pushed past its elastic range and let go springs back
by that range, so one pushed to 120 deg falls back below upright and one pushed to 200 deg stays open; and the
scripted hook opens all four flaps with the carton and its contents in place (the reference task's checks).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks.gift_carton import FLAPS, CartonSpec, carton_mjcf, flap_angles
from alloy.tasks.gift_carton_flaps import (
    GA_FLAP,
    NOT_JUDGED,
    ORDER,
    WRIST_LIMIT,
    FlapsSpec,
    build,
    read,
    run,
    unwrap_yaw,
)
from alloy.tasks.gift_flap_path import Push, flap_frame, tip_offset, tool_yaw, waypoints


def test_closed_flaps_hold_against_gravity():
    """The carton alone, 2 s: the sprung major keeps its 12 deg set less its 3 deg elastic sag; the other flaps rest
    on their supports (wall tops, the minors) within 0.5 deg of closed."""
    import alloy

    s = CartonSpec()
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("carton", mjcf=carton_mjcf(s))
    sim = alloy.build(scene, device="cpu")
    sim.run(2.0)
    x = sim.state(["carton/carton"] + [f"carton/{f}" for f in FLAPS], world=0)
    a = flap_angles(s, x["carton/carton.quat_wxyz"], {f: x[f"carton/{f}.quat_wxyz"] for f in FLAPS})
    assert a["major_p"] == pytest.approx(s.sprung_deg - 3.0, abs=1.0), a
    assert all(abs(a[f]) < 0.5 for f in ("major_m", "minor_p", "minor_m")), a
    assert sim.check()["ok"]


def _push_one(target_deg: float) -> tuple[float, float]:
    """The loaded carton and the hook; the sprung major pushed to ``target_deg`` and let go. (Largest angle reached,
    angle 1.5 s after the hook left) [deg]. Empty (0.33 kg), the carton lifts and tips under the ~3 N push, as the
    reference's does."""
    s = FlapsSpec(push=Push(target_deg=target_deg))
    sim, _ = build(s, worlds=1, device="cpu")
    sim.run(0.3)
    poses, ang = read(sim, s)
    p0 = np.array([0.0, 0.0, s.carton.height + 0.45])
    cur = {k: 0.0 for k in ("gantry_x", "gantry_y", "gantry_z", "wrist_yaw")}
    yaw = tool_yaw(flap_frame(s.carton, "major_p", poses[0])[2])
    peak = 0.0
    for _, tip, sec in waypoints(s.carton, "major_p", poses[0], math.radians(ang[0, 0]), s.push):
        palm = tip - tip_offset(yaw) - p0
        new = {"gantry_x": palm[0], "gantry_y": palm[1], "gantry_z": palm[2], "wrist_yaw": yaw}
        sim.run(sec, targets={k: (cur[k], new[k]) for k in cur})
        cur = new
        peak = max(peak, float(read(sim, s)[1][0, 0]))
    sim.run(1.5)
    assert sim.check()["ok"]
    return peak, float(read(sim, s)[1][0, 0])


@pytest.mark.timeout(180)
@pytest.mark.parametrize("target", [120.0, 200.0])
def test_a_pushed_flap_springs_back_by_its_elastic_range(target):
    """Let go, the flap returns by about the crease's 45 deg elastic range: 35-50 deg, as its weight closes it (below
    level) or opens it (past level) a few degrees and MuJoCo's soft dry friction slips a little under load (measured:
    120 deg push 44-46 deg back, 200 deg push 37 deg). Pushed to 120 deg it falls back below upright (the reference's
    control: 70.9 deg); to 200 deg it stays open."""
    peak, rest = _push_one(target)
    assert 35.0 <= peak - rest <= 50.0, (peak, rest)
    assert (rest < 90.0) if target < 150.0 else (rest > 120.0), (peak, rest)


def _flap_yaws() -> list[float]:
    """The hook's wanted yaw per flap, in ORDER, for the carton at its nominal pose."""
    pose = (np.zeros(3), np.eye(3))
    yaws = [float(tool_yaw(flap_frame(CartonSpec(), f, pose)[2])) for f in ORDER]
    return [y - 0.02 * (k == 1) for k, y in enumerate(yaws)]       # a settled carton is never exactly square


def test_the_wrist_yaw_target_stays_inside_the_joint_range_over_the_four_flaps():
    """The target is the wanted yaw modulo 2 pi nearest the current one INSIDE the wrist's +-3.3 rad (the shortest
    turn alone accumulated to 4.6 rad, past the stop: a clamped target left a flap shut)."""
    assert WRIST_LIMIT == pytest.approx(3.3)
    cur = 0.0
    for want in _flap_yaws() + [3.0, -3.0, 3.2, -3.2, 6.0, -6.0]:
        new = float(unwrap_yaw(want, cur))
        assert abs(new) <= WRIST_LIMIT, (want, cur, new)
        assert math.isclose(math.sin(new), math.sin(want), abs_tol=1e-9)
        assert math.isclose(math.cos(new), math.cos(want), abs_tol=1e-9)
        cur = new
    assert float(unwrap_yaw(3.0, -3.0)) == pytest.approx(3.0 - 2 * math.pi)     # shortest turn is inside: taken
    assert float(unwrap_yaw(-3.0 + 2 * math.pi, 1.0)) == pytest.approx(3.28, abs=0.01)  # 3.28 in range: nearest


def test_the_old_shortest_turn_rule_left_the_range_over_the_four_flaps():
    """Documents the bug: the same yaws unwrapped by the shortest turn alone reach past +-3.3 rad."""
    cur, peak = 0.0, 0.0
    for want in _flap_yaws():
        cur = cur + (want - cur + math.pi) % (2 * math.pi) - math.pi
        peak = max(peak, abs(cur))
    print("old peak", peak, "yaws", _flap_yaws())
    assert peak > WRIST_LIMIT


@pytest.mark.timeout(300)
def test_the_scripted_hook_opens_all_four_flaps():
    """The task's demo (1 world): every flap at least 90 deg open through the 3 s window, the carton moved under
    30 mm and tilted under 10 deg, the contents inside and moved under 10 mm, the hook clear."""
    out = run(FlapsSpec(), device="cpu", worlds=1)
    assert out["check_ok"], out["problems"]
    v = out["worlds"][0]
    assert out["success"], v
    assert out["ga_judged"] == "8 of 8"
    assert min(v[f"{f}_min_deg"] for f in FLAPS) > 150.0, v
    assert set(v["ga_criteria"]) == GA_NAMES and v["ga_criteria"]["hands_off_window"] is True
    assert v["ga_not_judged"] == {} and v["port_checks"] == {} and v["hands_off_contact_frames"] == 0
    assert v["ga_notes"] and v["ga_judged_n"] == 8 and v["ga_total_n"] == 8


GA_NAMES = {"flap_major_ypos_open", "flap_major_yneg_open", "flap_minor_xpos_open", "flap_minor_xneg_open",
            "carton_in_place", "carton_upright", "contents_in_place", "hands_off_window"}


def test_the_names_and_the_unjudged_set_are_pinned():
    """The reference's open-flaps criteria (seam_open dropped): four flaps, carton and contents, hands-off window.
    The hands-off window (a reaction-force test in the reference) is judged by distance (gift_hands_off.py); nothing is
    left out."""
    assert {k + "_open" for k in GA_FLAP.values()} | {"carton_in_place", "carton_upright", "contents_in_place",
                                                      "hands_off_window"} == GA_NAMES
    assert set(GA_FLAP) == set(FLAPS)
    assert NOT_JUDGED == {}


def test_the_flap_verdict_judges_eight_of_eight_and_a_touch_in_the_window_fails_it():
    from alloy.tasks.gift_carton_flaps import ga_report
    row = {f"{f}_open": True for f in FLAPS} | dict(carton_in_place=True, carton_upright=True, contents_in_place=True,
                                                    hands_off_window=True)
    v = ga_report(row)
    assert (v["ga_judged"], v["success"], v["port_checks"], v["ga_not_judged"]) == ("8 of 8", True, {}, {})
    assert any("distance test" in n for n in v["ga_notes"])
    assert not ga_report(row | {"hands_off_window": False})["success"]
    assert ga_report(row | {"hands_off_window": False})["ga_criteria"]["hands_off_window"] is False


def test_open_carton_adds_the_seam_and_judges_nine_of_nine():
    """The taped carton's verdict (gift_carton_tape.py): the reference's open-carton has seam_open besides."""
    from alloy.tasks.gift_carton_flaps import ga_report
    row = {f"{f}_open": True for f in FLAPS} | dict(carton_in_place=True, carton_upright=True,
                                                    contents_in_place=True, hands_off_window=True)
    v = ga_report(row, {"seam_open": True}, ("a note",))
    assert set(v["ga_criteria"]) == GA_NAMES | {"seam_open"} and v["ga_judged"] == "9 of 9" and v["success"]
    assert v["ga_criteria"]["hands_off_window"] is True and v["port_checks"] == {} and "a note" in v["ga_notes"]
    assert not ga_report(row, {"seam_open": False})["success"]
