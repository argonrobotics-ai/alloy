# SPDX-License-Identifier: Apache-2.0
"""The hands-off window of the gift tasks (alloy/tasks/gift_hands_off.py): box separation, the tool's boxes, and the
window verdict of bag-candle and open-flaps / open-carton on synthetic frames (numpy only, no simulation)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from alloy.tasks.gift_bag_candle import CandleSpec, hands_off_window
from alloy.tasks.gift_carton import CartonSpec, _hinges
from alloy.tasks.gift_carton_flaps import ITEMS, FlapsSpec, window_contacts
from alloy.tasks.gift_gantry import gantry_mjcf, tool_boxes
from alloy.tasks.gift_hands_off import (WorldBox, boxes_overlap, contact_track, join_frames, points_in_box,
                                        rot_from_quat, window_verdict)

I3 = np.eye(3)


def _box(c, half, R=I3):
    return WorldBox(np.asarray(c, float), np.asarray(R, float), tuple(half))


def _yaw(deg):
    h = math.radians(deg) / 2
    return rot_from_quat(np.array([math.cos(h), 0, 0, math.sin(h)]))


def test_boxes_overlap_by_margin_and_across_edges():
    a, b = _box((0, 0, 0), (0.01,) * 3), _box((0.024, 0, 0), (0.01,) * 3)       # 4 mm apart
    assert boxes_overlap(a, b, 0.005) and not boxes_overlap(a, b, 0.003) and not boxes_overlap(a, b)
    assert not boxes_overlap(_box((0.031, 0, 0), (0.01,) * 3), a, 0.005)         # 11 mm apart
    # edge to edge: a cube turned 45 deg about z reaches 14.1 mm along x; centres 25 mm apart leave 0.9 mm
    r = _box((0.025, 0, 0), (0.01,) * 3, _yaw(45.0))
    assert not boxes_overlap(a, r) and boxes_overlap(a, r, 0.002)
    assert boxes_overlap(a, _box((0.0, 0.0, 0.0), (0.001,) * 3, _yaw(33.0)))     # inside


def test_boxes_overlap_agrees_with_sampled_points_on_random_boxes():
    """Any sampled point of one box inside the other forces an overlap; boxes far apart never overlap."""
    rng = np.random.default_rng(1)
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, 5)] * 3), -1).reshape(-1, 3)
    for _ in range(200):
        q = rng.normal(size=4)
        a = _box(rng.uniform(-0.1, 0.1, 3), rng.uniform(0.01, 0.06, 3), rot_from_quat(q))
        b = _box(rng.uniform(-0.1, 0.1, 3), rng.uniform(0.01, 0.06, 3), rot_from_quat(rng.normal(size=4)))
        pts = a.centre + (g * np.asarray(a.half)) @ a.rot.T
        if points_in_box(pts, b):
            assert boxes_overlap(a, b)
        far = _box(b.centre + 1.0, b.half, b.rot)
        assert not boxes_overlap(a, far)


def test_the_frames_and_worlds_broadcast():
    pos = np.zeros((4, 2, 3))
    pos[2, 1] = (0.015, 0.0, 0.0)
    q = np.tile([1.0, 0, 0, 0], (4, 2, 1))
    tool = WorldBox.of_body(pos, q, (0, 0, 0), (0.01,) * 3)
    hit = contact_track([tool], [_box((0.04, 0, 0), (0.01,) * 3)], [], floor_z=None)
    assert hit.shape == (4, 2) and hit.tolist() == [[False, False], [False, False], [False, True], [False, False]]
    assert contact_track([tool], [], [], floor_z=-1.0).sum() == 0 and contact_track([tool], [], [], 0.0).all()


def test_the_window_verdict_needs_a_complete_window_and_no_touch():
    hit = np.zeros((301, 3), bool)
    hit[150, 1] = True
    v = window_verdict(hit, 300, 0.01)
    assert [x["hands_off_window"] for x in v] == [True, False, True]
    assert v[1]["hands_off_first_contact_s"] == 1.5 and v[1]["hands_off_contact_frames"] == 1
    assert not any(x["hands_off_window"] for x in window_verdict(hit[:200], 300, 0.01))        # too short
    j = join_frames([{"a": np.arange(3)}, {"a": np.arange(3) + 2}, {"a": np.arange(3) + 4}])
    assert j["a"].tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_the_tool_boxes_are_the_mjcf_geoms():
    """The boxes that the window test uses are the gantry MJCF's collision geoms (same sizes and places)."""
    for tool, width in (("hook", 0.052), ("fingers", 0.024)):
        root = ET.fromstring(gantry_mjcf((0, 0, 1), tool=tool, tine_width=width))
        geoms = [(g, b) for b in root.iter("body") for g in b.findall("geom")
                 if b.get("name") in ("wrist", "finger_l", "finger_r") and g.get("contype") != "0"
                 and g.get("type") == "box"]
        want = tool_boxes(tool, width)
        assert len(geoms) == len(want)
        for (g, b), (body, centre, half) in zip(geoms, want):
            assert b.get("name") == body and np.allclose([float(x) for x in g.get("size").split()], half)
            pos = [float(x) for x in (g.get("pos") or "0 0 0").split()]
            assert np.allclose(pos, centre)


# ---- bag-candle: synthetic window frames, 301 frames x 2 worlds, the fingers far aside ----------------------------

def _bag_frames(F=301, W=2):
    f = {}
    bag = np.zeros((F, W, 4, 3))
    bag[..., :, :] = [[-0.1, -0.05, 0.0], [0.1, -0.05, 0.0], [0.1, -0.05, 0.26], [-0.1, 0.05, 0.26]]
    f["bag.points"], f["board.points"] = bag + (0, 0, 0.0035), np.zeros((F, W, 2, 3)) + (0, 0, 0.0035)
    f["candle.pos"] = np.tile([0.0, 0.0, 0.0585], (F, W, 1))
    f["candle.quat_wxyz"] = np.tile([1.0, 0, 0, 0], (F, W, 1))
    for n, x in (("l", 0.5), ("r", 0.6)):
        f[f"gripper/finger_{n}.pos"] = np.tile([x, 0.0, 0.55], (F, W, 1))
        f[f"gripper/finger_{n}.quat_wxyz"] = np.tile([1.0, 0, 0, 0], (F, W, 1))
    return f


def test_bag_candle_clear_window_passes_and_a_one_frame_touch_in_the_middle_fails():
    s = CandleSpec()
    f = _bag_frames()
    assert [v["hands_off_window"] for v in hands_off_window(s, f)] == [True, True]
    f["gripper/finger_l.pos"][150, 1] = (0.1 + 0.004 + 0.003, -0.05, 0.13)   # world 1, frame 150: 3 mm off a vertex
    v = hands_off_window(s, f)
    assert [x["hands_off_window"] for x in v] == [True, False]
    assert v[1]["hands_off_first_contact_s"] == 1.5 and v[1]["hands_off_contact_frames"] == 1


def test_bag_candle_sees_the_carton_the_mat_and_an_incomplete_window():
    s = CandleSpec()
    f = _bag_frames()
    f["gripper/finger_r.pos"][10, 0] = (0.045 + 0.004 + 0.004, 0.0, 0.0585)     # 4 mm from the carton's +x face
    f["gripper/finger_r.pos"][20, 1] = (0.15, 0.05, 0.15 + 0.003 + 0.004)       # tip 4 mm over the mat, off the bag
    f["gripper/finger_r.pos"][30, 1] = (0.5, 0.5, 0.15 + 0.004)                 # tip 4 mm over the bare counter
    v = hands_off_window(s, f)
    assert v[0]["hands_off_window"] is False and v[0]["hands_off_first_contact_s"] == 0.1
    assert v[1]["hands_off_window"] is False and v[1]["hands_off_first_contact_s"] == 0.2
    assert v[1]["hands_off_contact_frames"] == 2
    f["gripper/finger_r.pos"][20, 1] = (0.15, 0.05, 0.15 + 0.003 + 0.009)       # 9 mm over the mat: clear
    assert hands_off_window(s, f)[1]["hands_off_first_contact_s"] == 0.3
    assert not any(x["hands_off_window"] for x in hands_off_window(s, {k: a[:200] for k, a in _bag_frames().items()}))


# ---- carton tasks ------------------------------------------------------------------------------------------------

def _carton_frames(s: FlapsSpec, F=301, W=2):
    tr = {}
    quat = np.tile([1.0, 0, 0, 0], (F, W, 1))
    tr["carton/carton.pos"], tr["carton/carton.quat_wxyz"] = np.zeros((F, W, 3)), quat
    for f, h in _hinges(s.carton).items():
        tr[f"carton/{f}.pos"], tr[f"carton/{f}.quat_wxyz"] = np.tile(h["p"], (F, W, 1)), quat
    for k, name in enumerate(ITEMS):
        tr[f"{name}.pos"] = np.tile([0.05 * k - 0.2, 0.0, 0.05], (F, W, 1))
        tr[f"{name}.quat_wxyz"] = quat
    tr["hook/wrist.pos"] = np.tile([0.0, 0.0, s.carton.height + 0.5], (F, W, 1))     # tine 0.3 m lower: 0.2 m clear
    tr["hook/wrist.quat_wxyz"] = quat
    return tr


def test_open_flaps_clear_window_passes_and_one_touching_frame_fails_it():
    s = FlapsSpec()
    tr = _carton_frames(s)
    assert [v["hands_off_window"] for v in window_contacts(s, tr)] == [True, True]
    tr["hook/wrist.pos"][150, 1] = (0.0, 0.0, 0.30 + 0.004 + s.carton.height + 0.005)   # tine 1 mm above the wall top
    v = window_contacts(s, tr)
    assert [x["hands_off_window"] for x in v] == [True, False]
    assert v[1]["hands_off_first_contact_s"] == 1.5 and v[1]["hands_off_contact_frames"] == 1


def test_open_flaps_sees_a_flap_a_content_item_and_the_counter():
    s = FlapsSpec()
    tr = _carton_frames(s)
    top = _hinges(s.carton)["major_p"]["p"]                 # the closed major lies at the hinge's height, +y side
    tr["hook/wrist.pos"][5, 0] = (0.0, 0.1, top[2] + 0.305)             # tine 3 mm over the flap board
    tr["hook/wrist.pos"][6, 1] = (1.0, 1.0, 0.30 + 0.004)                # 1 mm over the counter, far from the carton
    v = window_contacts(s, tr)
    assert v[0]["hands_off_first_contact_s"] == 0.05 and v[1]["hands_off_first_contact_s"] == 0.06
    tr = _carton_frames(s)
    tr["hook/wrist.pos"][7, 0] = (0.05 * 3 - 0.2, 0.0, 0.05 + 0.5 * 0.07 + 0.30 + 0.004)   # onto a tea carton
    assert window_contacts(s, tr)[0]["hands_off_first_contact_s"] == 0.07


def test_a_short_window_fails():
    s = FlapsSpec()
    tr = {k: a[:100] for k, a in _carton_frames(s).items()}
    assert not any(v["hands_off_window"] for v in window_contacts(s, tr))
