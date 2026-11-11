# SPDX-License-Identifier: Apache-2.0
"""The kraft gift bag and the bag-candle checks (alloy/tasks/gift_bag_mesh.py, gift_bag_checks.py). CPU.

Mesh: the library OBJ is the generator's output, the card's hash matches it, the surface is an oriented open box (one
rim boundary, every other edge shared by two triangles wound opposite ways) of Uline S-7098's size, and every fold
line runs along mesh edges. Checks: the bag's rest shape with a board on its bottom and a carton set on the board
passes every check; each failure the reference task guards against (a tilted carton, a carton outside, a pinched
mouth, a toppled bag, a board under the bag) fails its own check.
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
import pytest

from alloy.tasks.gift_bag_checks import GA_CRITERIA, NOT_JUDGED, evaluate, layout
from alloy.tasks.gift_bag_mesh import INCH, OBJ_PATH, BagMeshSpec, bag_mesh, obj_text

ROOT = Path(__file__).resolve().parents[2]
S = BagMeshSpec()
CANDLE = np.array([0.09, 0.09, 0.11])
BOARD_H = 0.00051


def _edges(t: np.ndarray) -> collections.Counter:
    return collections.Counter((int(a), int(b)) for tri in t for a, b in ((tri[0], tri[1]), (tri[1], tri[2]),
                                                                           (tri[2], tri[0])))


def test_library_mesh_is_the_generators_and_the_card_hash_matches():
    text = (ROOT / OBJ_PATH).read_text()
    assert text == obj_text(bag_mesh(S))
    card = (ROOT / "library/objects/gift_bag_kraft.yaml").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() in card


def test_mesh_is_an_oriented_open_box_of_the_vendor_size():
    m = bag_mesh(S)
    v, t = m.vertices, m.triangles
    assert np.allclose(v.max(0) - v.min(0), [8 * INCH, 4.5 * INCH, 10.25 * INCH])
    directed = _edges(t)
    assert max(directed.values()) == 1                          # no edge twice the same way: consistent winding
    und = collections.Counter()
    for (a, b), n in directed.items():
        und[(min(a, b), max(a, b))] += n
    boundary = {e for e, n in und.items() if n == 1}
    assert max(und.values()) == 2
    assert {i for e in boundary for i in e} == set(m.rim.tolist())
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    assert np.all(((v[t].mean(1) - [0, 0, 0.5 * S.height]) * n).sum(1) > 0)      # every normal points out
    folds = [(min(a, b), max(a, b)) for line in m.creases for a, b in zip(line[:-1], line[1:])]
    assert len(set(folds)) == len(folds) and all(e in und for e in folds)
    assert len(m.creases) == 11                                 # 4 side folds, 2 gusset centres, bottom, 4 feet


def _state(tilt_deg: float = 0.0):
    """The rest-shape bag (optionally tipped about x), a board on its bottom panel, a carton on the board."""
    m = bag_mesh(S)
    bag = m.vertices.copy() + [0.0, 0.0, 0.0032]
    board = np.array([[x, y, 0.0032 + 135e-6 + 0.5 * BOARD_H] for x in np.linspace(-0.1, 0.1, 9)
                      for y in np.linspace(-0.05, 0.05, 5)])
    pos = np.array([0.0, 0.0, board[0, 2] + 0.5 * BOARD_H + 0.001 + 0.5 * CANDLE[2]])
    L = layout(S, (0.0, 0.0), (0.2, 0.15), BOARD_H, CANDLE, m.bottom, m.rim, m.triangles, 0.0031)
    if tilt_deg:
        a = np.radians(tilt_deg)
        R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        bag, board, pos = bag @ R.T, board @ R.T, R @ pos
    return L, bag, board, pos


Q_UP = np.array([1.0, 0.0, 0.0, 0.0])


def test_a_carton_set_on_the_board_in_the_open_bag_passes():
    L, bag, board, pos = _state()
    out = evaluate(L, bag, board, pos, Q_UP, bottom_mark=bag[L.bottom].mean(0))
    assert out["success"], out
    assert out["bottom_squared"] and out["base_at_rest"] and out["bottom_flat_fraction"] == 1.0
    assert out["mouth_area_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert out["least_gap_mm"] == pytest.approx(S.depth * 1e3, abs=0.1)


@pytest.mark.parametrize("case, check", [("tilted carton", "candle_upright"), ("carton outside", "candle_in_bag"),
                                         ("carton hovering", "candle_on_board"), ("pinched mouth", "mouth_open"),
                                         ("toppled bag", "standing"), ("board under the bag", "board_in_bag"),
                                         ("bag off the mat", "on_mat"),
                                         ("bag lifted", "bottom_squared"), ("bag sliding", "base_at_rest")])
def test_each_failure_fails_its_check(case, check):
    L, bag, board, pos = _state()
    q = Q_UP
    if case == "tilted carton":
        a = np.radians(20.0)
        q = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])
    elif case == "carton outside":
        pos = pos + [0.3, 0.0, 0.0]
    elif case == "carton hovering":
        pos = pos + [0.0, 0.0, 0.02]
    elif case == "pinched mouth":
        z = bag[:, 2] > 0.6 * S.height
        bag[z, 1] *= 0.3                                         # faces pressed to 30 % of the depth up top
    elif case == "toppled bag":
        L, bag, board, pos = _state(tilt_deg=30.0)
    elif case == "board under the bag":
        board = board - [0.0, 0.0, 0.005]
    elif case == "bag off the mat":
        bag, board, pos = bag + [0.35, 0, 0], board + [0.35, 0, 0], pos + [0.35, 0, 0]
    mark = bag[L.bottom].mean(0)
    if case == "bag lifted":
        bag = bag + [0.0, 0.0, 0.01]                             # 10 mm up: no panel within 3 mm of the mat plane
    elif case == "bag sliding":
        bag = bag + [0.002, 0.0, 0.0]                            # 2 mm in the last 2 s, over 1.33 mm
    out = evaluate(L, bag, board, pos, q, bottom_mark=mark)
    assert out[check] is False, (case, out)
    assert out["ga_criteria"][check] is False
    assert not out["success"]


@pytest.mark.timeout(180)
def test_the_scripted_gripper_sets_each_carton_down_square_and_upright_at_the_bag():
    """The demo's script alone (bag and board left out, rigid only), 4 worlds with the reference's start spread
    (±20 mm, ±30 deg): the gantry turns its wrist to each carton, picks it by friction, carries it above the rim's
    height while turning it square, and sets it down upright at the bag's spot; the fingers end clear of it."""
    from alloy.tasks.gift_bag_candle import CandleSpec, build, plan
    from alloy.tasks.gift_bag_checks import candle_corners
    from alloy.tasks.gift_gantry import drive, fingers_clear, schedule

    s = CandleSpec()
    sim, st = build(s, worlds=4, seed=0, device="cpu", with_bag=False)
    assert np.abs(np.degrees(st[:, 2])).max() > 10.0                # the seed turns some cartons well off square
    rows = {r["phase"]: np.asarray(r["candle_z"]) for r in drive(sim, schedule(plan(s), st))}
    assert np.all(rows["carry"] - 0.5 * s.candle[2] > 0.29)         # carried with its bottom above the 260 mm rim
    c = sim.state(["candle"])
    for w in range(4):
        assert np.allclose(c["candle.pos"][w], [0.0, 0.0, 0.003 + 0.055], atol=3e-3), w
        q = c["candle.quat_wxyz"][w]
        assert abs(q[0]) > np.cos(np.radians(3.0) / 2), (w, q)        # square to the bag and upright, within 3 deg
    corners = np.stack([candle_corners(c["candle.pos"][w], c["candle.quat_wxyz"][w], s.candle) for w in range(4)])
    assert all(fingers_clear(sim, {"candle": corners}))
    assert sim.check()["ok"]


@pytest.mark.timeout(180)
def test_the_open_bag_stands_open_on_the_mat():
    """The open kraft bag left alone on the mat (public API, 0.4 s): it stands, square and open (the paper's bending
    and the 90 deg folds hold the walls up), moving under 2 mm."""
    import alloy
    from alloy.tasks.gift_bag_candle import MAT, CandleSpec, _heights, _layout
    from alloy.tasks.gift_bag_checks import Limits, mouth, standing

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("mat", size=MAT, pos=(0.0, 0.0, 0.5 * MAT[2]), static=True, friction=0.9)
    scene.add_object("bag", "gift_bag_kraft", pos=(0.0, 0.0, _heights(CandleSpec())["bag"]))
    sim = alloy.build(scene, device="cpu")
    assert sim.routes()["bag"]["route"] == "cloth_film"
    p0 = sim.state(world=0)["bag.points"].copy()
    sim.run(0.4)
    p = sim.state(world=0)["bag.points"]
    assert np.linalg.norm(p - p0, axis=1).max() < 2e-3
    L = _layout(CandleSpec())
    assert mouth(L, p, Limits())["mouth_open"] and standing(L, p, Limits())["standing"]
    assert sim.check()["ok"]


GA_NAMES = {"hands_off_window", "mouth_open", "standing", "bottom_squared", "on_mat", "base_at_rest", "board_in_bag",
            "candle_in_bag", "candle_upright", "candle_on_board", "bag_not_torn"}


def test_the_verdict_names_every_ga_criterion_and_lists_the_unjudged_with_a_reason():
    L, bag, board, pos = _state()
    out = evaluate(L, bag, board, pos, Q_UP, bottom_mark=bag[L.bottom].mean(0), hands_off=True)
    assert set(GA_CRITERIA) == GA_NAMES and set(out["ga_criteria"]) == GA_NAMES
    unjudged = {k for k, v in out["ga_criteria"].items() if v is None}
    assert unjudged == {"bag_not_torn"} == set(out["ga_not_judged"]) == set(NOT_JUDGED)
    assert all(isinstance(r, str) and len(r) > 20 for r in out["ga_not_judged"].values())
    assert (out["ga_judged"], out["ga_judged_n"], out["ga_total_n"]) == ("10 of 11", 10, 11)
    assert out["port_checks"] == {} and out["ga_criteria"]["hands_off_window"] is True and out["ga_notes"]
    assert out["success"] and all(out["ga_criteria"][k] is True for k in GA_NAMES - unjudged)
    assert all(out[k] == out["ga_criteria"][k] for k in GA_NAMES - unjudged)    # top-level bools agree


def test_without_the_two_second_mark_base_at_rest_is_not_judged_and_success_ignores_it():
    L, bag, board, pos = _state()
    out = evaluate(L, bag, board, pos, Q_UP, hands_off=True)
    assert out["ga_criteria"]["base_at_rest"] is None and out["ga_judged"] == "9 of 11"
    assert out["success"]
    bare = evaluate(L, bag, board, pos, Q_UP)                       # no window recorded: not judged, with a reason
    assert bare["ga_criteria"]["hands_off_window"] is None and "hands_off_window" in bare["ga_not_judged"]


def test_success_needs_the_port_checks_and_never_counts_a_none_as_true():
    L, bag, board, pos = _state()
    out = evaluate(L, bag, board, pos, Q_UP, port_checks={"a_port_check": False}, hands_off=True)
    assert not out["success"] and out["ga_criteria"]["candle_in_bag"]
    assert not evaluate(L, bag, board, pos, Q_UP, hands_off=False)["success"]
    from alloy.tasks.gift_gantry import ga_verdict
    assert not ga_verdict({"a": None, "b": None}, {"a": "x", "b": "y"}, {})["success"]       # nothing judged
    assert ga_verdict({"a": True, "b": None}, {"b": "y"}, {"p": True})["success"]
    assert not ga_verdict({"a": True, "b": False}, {}, {})["success"]
    with pytest.raises(ValueError):
        ga_verdict({"a": None}, {}, {})                                       # a None needs its reason


def test_the_window_is_cut_so_the_state_two_seconds_before_the_end_can_be_marked():
    from alloy.tasks.gift_bag_candle import split_window
    phases = [("aside", 0.6, {}), ("hands_off", 3.0, {})]
    cut, mark = split_window(phases)
    assert [(n, round(t, 3)) for n, t, _ in cut] == [("aside", 0.6), ("hands_off", 1.0), ("hands_off_last", 2.0)]
    assert mark == "hands_off"
    short, mark = split_window([("aside", 0.6, {}), ("hands_off", 1.5, {})])
    assert len(short) == 2 and mark == "aside"
