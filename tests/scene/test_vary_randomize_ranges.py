# SPDX-License-Identifier: Apache-2.0
"""alloy vary moves an object's randomized start coordinates with it (``alloy.scene.variants.place._move_ranges``),
AUDIT S2-018 (Gemini 0.1.2): ``Scene.randomize`` x / y / z draws an absolute coordinate per world, and vary moved the
start pose but left the range where it was, so every world put the object back (0.000 mm built shift for a reported
90 mm move) onto what vary had placed there (3 of 12 variants: two boxes 50 mm into each other). The laws: each
range keeps its width and its centre moves as the object does, in API items, card placements and particle starts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.scene.variants import vary
from alloy.scene.variants.place import _move_ranges, edit_poses
from alloy.scene.variants.xform import Move

HALF = 0.001                    # m: b's randomized x is pose x +- 1 mm


def _shelf() -> alloy.Scene:
    """The audit's mixed recipe: b randomized in x around 0, a beside it (not randomized), on a narrow table."""
    s = alloy.Scene(worlds=2, dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.32, 0.08, 0.40), pos=(0.0, 0.0, 0.20), static=True)
    s.add_box("b", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.43), mass=0.2)
    s.add_box("a", size=(0.06, 0.06, 0.06), pos=(0.07, 0.0, 0.43), mass=0.2)
    s.randomize("b", "x", -HALF, HALF)
    return s


def test_the_range_moves_with_the_object_and_the_built_worlds_follow():
    """Four variants (CPU, 2 worlds each): b's range is centred on its new x with its width kept, b builds within it
    in every world, and no variant starts with an overlap. ~15 s."""
    for v in vary(_shelf(), 4, pos=0.10, textures="keep", seed=0):
        (row,) = v.items["b"]["randomize"]
        x = float(v.items["b"]["pose"][0])
        assert 0.5 * (row["lo"] + row["hi"]) == pytest.approx(x, abs=1e-12)
        assert row["hi"] - row["lo"] == pytest.approx(2.0 * HALF, abs=1e-12)
        sim = alloy.build(v, device="cpu", seed=0)
        bx = sim.state()["b.pos"][:, 0]
        assert np.all(np.abs(bx - x) <= HALF + 1e-6), (bx, x)
        assert sim.check()["ok"], sim.check()["problems"]


def test_a_turned_group_moves_the_range_centre_as_the_object():
    rows = [{"field": "pose.0", "lo": 0.09, "hi": 0.11}, {"field": "material.density", "lo": 1.0, "hi": 2.0}]
    mv = Move(("b",), (0.01, 0.02), math.pi / 2.0, (0.0, 0.0, 0.0))
    _move_ranges(rows, np.array([0.1, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]), mv)
    # the centre (0.10, 0, 0.4) turns to (0, 0.10, 0.4) and shifts by (0.01, 0.02): x 0.01, width kept
    assert (rows[0]["lo"], rows[0]["hi"]) == pytest.approx((0.0, 0.02), abs=1e-12)
    assert rows[1] == {"field": "material.density", "lo": 1.0, "hi": 2.0}


def test_card_placements_and_particle_starts_move_too():
    """A scene card's placement rows (``raw_placement``: compose and library scenes) and a cloth or liquid's start
    rows (``start_random``) move as an API item's rows do."""
    item = {"pose": None, "raw_placement": {"pose": [0.25, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                                            "randomize": [{"field": "pose.1", "lo": -0.01, "hi": 0.01}]},
            "start_random": [{"field": "pose.0", "lo": 0.2, "hi": 0.3}, {"field": "pose.2", "lo": 0.1, "hi": 0.2}]}
    scene = SimpleNamespace(items={"x": item})
    base = SimpleNamespace(poses={"x": np.array([0.25, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])})
    edit_poses(scene, [Move(("x",), (0.05, -0.02), 0.0, (0.0, 0.0, 0.0))], base)
    assert item["raw_placement"]["pose"][:3] == pytest.approx([0.30, -0.02, 0.1])
    (r,) = item["raw_placement"]["randomize"]
    assert (r["lo"], r["hi"]) == pytest.approx((-0.03, -0.01))
    x, z = item["start_random"]
    assert (x["lo"], x["hi"]) == pytest.approx((0.25, 0.35)) and (z["lo"], z["hi"]) == (0.1, 0.2)
