# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the remaining public helpers (SPEC §6.2: every public function ships a test)."""

from __future__ import annotations

import math

import pytest

from alloy.cards import meshgen as mg
from alloy.cards import types as ct
from alloy.cards.defaults import DEFAULT_CONTACT_STIFFNESS, fill_material, is_routing_leaf
from alloy.cards.geometry import box_extents, curve_extents, curve_length
from alloy.cards.library import CardLibrary, default_root, pair_key
from alloy.cards.loader import load_any
from alloy.core.errors import CardError

LIB = CardLibrary(assets=None)


def test_load_any_dispatches_on_schema(tmp_path) -> None:
    assert isinstance(load_any(default_root() / "materials" / "water.yaml"), ct.MaterialCard)
    assert isinstance(load_any(default_root() / "scenes" / "demo_pour.yaml"), ct.SceneCard)
    bad = tmp_path / "x.yaml"
    bad.write_text("name: x\n")
    with pytest.raises(CardError) as exc:
        load_any(bad)
    assert exc.value.field == "schema"


def test_index_helpers() -> None:
    assert pair_key("b", "a") == "a__b"
    assert LIB.has_pair("stainless_303", "pmma_cast") and not LIB.has_pair("water", "honey")
    assert LIB.path_of("objects", "peg_d12").name == "peg_d12.yaml"
    assert [n for n, _ in LIB.iter_cards("scenes")] == list(LIB.names("scenes"))


def test_defaults_helpers() -> None:
    m = LIB.material("water")
    filled = fill_material(m)
    assert filled.surface.contact_stiffness.value == DEFAULT_CONTACT_STIFFNESS and fill_material(filled) == filled
    assert is_routing_leaf("loads.bend_force") and is_routing_leaf("part_materials.grip.elastic.youngs")
    assert not is_routing_leaf("material.failure.tear_energy") and not is_routing_leaf("material.fluid.sound_speed")


def test_curve_and_box_extents() -> None:
    c = ct.CurveSource(((0.0, 0.0, 0.0), (0.3, 0.4, 0.0), (0.3, 0.4, 1.0)), 0.001)
    assert curve_length(c) == pytest.approx(1.5) and curve_extents(c) == pytest.approx((1.5, 0.002, 0.002))
    assert box_extents((0.0, 0.0, 0.0), (0.1, -0.3, 0.2)) == pytest.approx((0.3, 0.2, 0.1))


def test_ear_clip_non_convex_and_ray_angles() -> None:
    l_shape = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    tris = mg.ear_clip(l_shape)
    area = sum(abs((l_shape[b][0] - l_shape[a][0]) * (l_shape[c][1] - l_shape[a][1]) -
                   (l_shape[b][1] - l_shape[a][1]) * (l_shape[c][0] - l_shape[a][0])) / 2 for a, b, c in tris)
    assert len(tris) == 4 and area == pytest.approx(3.0)
    angles = mg.ray_angles(4, [0.0, math.pi / 2 + 1e-12, 2 * math.pi - 1e-12, 1.0])
    assert angles == pytest.approx([0.0, 1.0, math.pi / 2, math.pi, 1.5 * math.pi])
    with pytest.raises(ValueError):
        mg.lathe([(0.1, 0.0), (0.0, 1.0)])
    with pytest.raises(ValueError):
        mg.box_pocket((0.01, 0.01, 0.01), (0.02, 0.005, 0.005))
