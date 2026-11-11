# SPDX-License-Identifier: Apache-2.0
"""OSS-19 geometry and layout without the engine: primitive meshes, ray casts, resting orientation, clearance,
work area, rows, and dropping objects into a container (they land on its floor, never on a wall)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.compose import geom
from alloy.compose.body import Body
from alloy.compose.layout import CLEARANCE, EDGE, GAP, Layout, Placed


def _box(name: str, hx: float, hy: float, hz: float) -> Body:
    v, f = geom.primitive_mesh("box", (hx, hy, hz))
    return Body.from_mesh(name, v, f, np.zeros(3))


def _table() -> Placed:
    return Placed("table", "table", _box("table", 0.6, 0.4, 0.009), 0.0, np.array([0.0, 0.0, -0.009]))


def _tray(hx: float = 0.1, hy: float = 0.05, w: float = 0.001) -> Body:
    """An open box 2hx x 2hy m (default 0.2 x 0.1), 20 mm deep: 2w floor and walls (five boxes in one mesh)."""
    parts = [((0, 0, w), (hx, hy, w)), ((hx - w, 0, 0.01), (w, hy, 0.01)),
             ((w - hx, 0, 0.01), (w, hy, 0.01)), ((0, hy - w, 0.01), (hx, w, 0.01)),
             ((0, w - hy, 0.01), (hx, w, 0.01))]
    vs, fs, n = [], [], 0
    for c, h in parts:
        v, f = geom.primitive_mesh("box", h)
        vs.append(v + np.asarray(c))
        fs.append(f + n)
        n += len(v)
    return Body.from_mesh("tray", np.vstack(vs), np.vstack(fs), np.array([0.0, 0.0, 0.006]))


def test_ray_casts_and_polygons():
    v, f = geom.primitive_mesh("cylinder", (0.01, 0.05))
    tris = v[f]
    assert geom.ray_down(tris, np.array([[0.0, 0.0], [0.5, 0.0]])).tolist() == pytest.approx([0.05, -math.inf])
    assert geom.ray_up(tris, np.array([[0.0, 0.0]]))[0] == pytest.approx(-0.05)
    sq = geom.hull2d(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]))
    assert len(sq) == 4 and geom.polygon_area(sq) == pytest.approx(1.0)
    assert geom.inside_depth(sq, np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert geom.separated(sq, sq + [2.0, 0.0], 0.5) and not geom.separated(sq, sq + [1.2, 0.0], 0.5)


def test_quaternions_round_trip():
    q = geom.quat_from_matrix(geom.yaw_matrix(0.7) @ geom.rotation_to(np.array([1.0, 0, 0]), np.array([0, 0, -1.0])))
    assert np.allclose(geom.matrix_from_quat(q) @ [1, 0, 0], geom.yaw_matrix(0.7) @ [0, 0, -1])


def test_resting_orientation():
    tall = Body.from_mesh("pencil", *geom.primitive_mesh("cylinder", (0.0035, 0.085)), com=np.zeros(3))
    assert tall.reoriented and tall.size[2] == pytest.approx(0.007, abs=1e-4) and tall.size[0] > 0.16
    flat = _box("plate", 0.05, 0.02, 0.003)
    assert not flat.reoriented and flat.rest == (0.0, 0.0, 0.0, 1.0)
    edge = _box("knife", 0.0004, 0.1, 0.02)                   # authored standing on its edge
    assert edge.reoriented and edge.size[2] == pytest.approx(0.0008, abs=1e-6)
    ball = Body.from_mesh("ball", *geom.primitive_mesh("sphere", (0.02,)), com=np.zeros(3))
    assert not ball.reoriented


def test_objects_keep_clearance_and_rest_on_the_table():
    lay = Layout(_table(), "normal", np.random.default_rng(1))
    cube = _box("cube", 0.015, 0.015, 0.015)
    lay.plan_area([cube] * 6)
    placed = [lay.place(cube, "cube", f"cube#{i}") for i in range(6)]
    assert all(p is not None for p in placed)
    for p in placed:
        assert p.pos[2] == pytest.approx(0.015 + GAP)
        assert geom.depths(_table().outline, p.outline).min() >= EDGE - 1e-9
    for i, p in enumerate(placed):
        for q in placed[i + 1:]:
            assert geom.separated(p.outline, q.outline, CLEARANCE["normal"] - 1e-9)


def test_work_area_grows_with_tidiness():
    cube = _box("cube", 0.02, 0.02, 0.02)
    sizes = {}
    for clutter in ("cluttered", "tidy"):
        lay = Layout(_table(), clutter)
        lay.plan_area([cube] * 10)
        sizes[clutter] = float(np.prod(lay.area[1] - lay.area[0]))
    assert sizes["tidy"] > 2 * sizes["cluttered"]


def test_row_slots_line_up_along_x():
    lay = Layout(_table(), "normal")
    pencil = Body.from_mesh("pencil", *geom.primitive_mesh("cylinder", (0.0035, 0.085)), com=np.zeros(3))
    lay.plan_area([pencil] * 4, "row")
    placed = [lay.place(pencil, "pencil", f"p{i}") for i in range(4)]
    ys = [p.pos[1] for p in placed]
    xs = sorted(p.pos[0] for p in placed)
    assert max(ys) - min(ys) < 1e-9 and all(b - a > 0.007 for a, b in zip(xs, xs[1:]))


def test_items_in_a_tray_land_on_its_floor():
    lay = Layout(_table(), "cluttered", np.random.default_rng(3))
    tray = lay.place(_tray(), "tray", "tray")
    assert tray is not None and tray.pos[2] == pytest.approx(GAP)
    nut = _box("nut", 0.008, 0.008, 0.004)
    for i in range(4):
        p = lay.place(nut, "nut", f"nut#{i}", target="tray")
        assert p is not None and p.target == "tray"
        assert p.pos[2] == pytest.approx(tray.pos[2] + 0.002 + 0.004 + GAP, abs=1e-6)   # floor top + half height
    assert lay.remove("tray")[0].key == "tray" and not lay.placed


def test_a_tight_fit_goes_in_centred_and_square():
    """A 200 x 100 mm board in a 203 x 114 mm bag fits only centred and square to it: the last tries are that pose."""
    lay = Layout(_table(), "normal", np.random.default_rng(5))
    bag = lay.place(_tray(0.1015, 0.057, 0.0005), "bag", "bag")
    board = lay.place(_box("board", 0.1, 0.05, 0.00025), "board", "board", target="bag")
    assert board is not None and board.target == "bag"
    assert np.allclose(board.pos[:2], bag.pos[:2], atol=1e-9)
    assert math.cos(2 * (board.yaw - bag.yaw)) == pytest.approx(1.0)                  # square to the bag
    assert board.pos[2] == pytest.approx(bag.pos[2] + 0.001 + 0.00025 + GAP, abs=1e-6)


def test_no_room_returns_none():
    lay = Layout(_table(), "tidy", np.random.default_rng(0))
    big = _box("slab", 0.4, 0.3, 0.01)                         # 0.8 x 0.6 m on a 1.2 x 0.8 m table
    assert lay.place(big, "slab", "slab") is not None
    assert lay.place(big, "slab", "slab#1", tries=20) is None
