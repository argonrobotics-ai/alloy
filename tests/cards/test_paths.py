# SPDX-License-Identifier: Apache-2.0
"""Dotted card paths: get, set, unit kind, leaf iteration, errors."""

from __future__ import annotations

import pytest

from alloy.cards.library import CardLibrary
from alloy.cards.paths import CardView, PathError, get_value, iter_q_leaves, kind_for_path, with_value

LIB = CardLibrary(assets=None)


def _view(name: str) -> CardView:
    c = LIB.object(name)
    return CardView(c, LIB.material(c.material), {k: LIB.material(v) for k, v in c.part_materials.items()},
                    (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0))


def test_get_and_set() -> None:
    v = _view("usb_a_plug")
    assert get_value(v, "material.density") == 7850.0
    assert get_value(v, "part_materials.insert.density") == LIB.material("pbt").density.value
    assert get_value(v, "pose.2") == 0.1 and get_value(v, "loads.grip_force") == 20.0
    assert get_value(v, "precision_features.0.clearance") == pytest.approx(5e-4)
    v2 = with_value(v, "part_materials.insert.density", 1400.0)
    assert get_value(v2, "part_materials.insert.density") == 1400.0
    assert v2.parts["insert"].density.note == v.parts["insert"].density.note
    assert get_value(with_value(v, "pose.0", 0.5), "pose.0") == 0.5
    assert get_value(with_value(_view("peg_d12"), "shape.size.1", 0.03), "shape.size.1") == 0.03


@pytest.mark.parametrize("path", ["material.colour", "loads", "pose.9", "part_materials.ghost.density", "shape.size.x",
                                  "material.density.value", "", "loads..grip_force", "name"])
def test_bad_paths(path: str) -> None:
    with pytest.raises(PathError):
        get_value(_view("usb_a_plug"), path)


@pytest.mark.parametrize("path, kind", [("material.elastic.youngs", "pressure"), ("loads.tolerance", "length"),
                                        ("pose.3", "length"), ("shape.size.0", "length"), ("shape.radius", "length"),
                                        ("material.fluid.viscosity", "viscosity"), ("material.surface.mu_static", "dimensionless"),
                                        ("part_materials.x.density", "density"), ("precision_features.0.clearance", "length")])
def test_kind_for_path(path: str, kind: str) -> None:
    assert kind_for_path(path) == kind


def test_iter_q_leaves() -> None:
    paths = [p for p, _ in iter_q_leaves(LIB.material("cotton_woven_aniso"))]
    assert "elastic.cloth.bend_weft" in paths and "density" in paths
