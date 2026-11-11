# SPDX-License-Identifier: Apache-2.0
"""Audit GAPS-11: the build warns when a rigid body is small enough to pass between a sheet's vertices.

Cloth and film sheets touch rigid shapes at their vertices only, so a ball under about a cell across passes through
with no health flag (known gap cloth-small-body-passes-through; its table gives the measured ratio). The warning
(``alloy.scene.vbd_contact.small_body_warnings``) names, once per sheet, the rigid primitives whose smallest
dimension is under ``SMALL_BODY_CELLS`` of its median rest edge, and the cell size that would hold the smallest.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest

import alloy
from alloy.cards.types import MeshSource, Primitive
from alloy.scene import vbd_contact as VC

CELL = 0.005           # about: a 6 cm sheet at resolution 13 is built with 4.6 mm cells


def _scene(radius: float | None, cloth: bool = True) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.2, 0.2, 0.04), pos=(0, 0, 0.02), static=True)
    if cloth:
        scene.add_cloth("sheet", size=(0.06, 0.06), pos=(0, 0, 0.045), resolution=13)
    if radius is not None:
        scene.add_sphere("bead", radius, pos=(0, 0, 0.06), mass=0.002)
    return scene


def _warnings(scene: alloy.Scene, caplog) -> list[str]:  # noqa: ANN001 - pytest's caplog
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        alloy.build(scene, device="cpu").close()
    out = [r.getMessage() for r in caplog.records if "can pass through" in r.getMessage()]
    caplog.clear()
    return out


def test_a_bead_smaller_than_the_cells_is_named_with_the_cell_size_that_holds_it(caplog):
    (line,) = _warnings(_scene(0.003), caplog)
    cell = float(re.search(r"of its cells \(([0-9.]+) mm each\)", line).group(1))
    assert line.startswith("sheet and bead (6 mm across) can pass through each other")
    assert 4.0 < cell < 5.5                                                  # the built sheet's median rest edge
    assert f"cells of {6.0 / VC.SMALL_BODY_CELLS:.3g} mm or less" in line and "finer resolution" in line


def test_a_body_larger_than_the_rule_or_a_scene_without_a_sheet_says_nothing(caplog):
    big = 0.5 * VC.SMALL_BODY_CELLS * CELL + 0.001
    assert _warnings(_scene(big), caplog) == []
    assert _warnings(_scene(0.003, cloth=False), caplog) == []


def test_many_small_bodies_make_one_sentence_per_sheet_naming_the_smallest_four(caplog):
    scene = _scene(None)
    for i in range(7):                                   # 2 to 8 mm across: all under 2 cells (9.2 mm)
        scene.add_sphere(f"bead{i}", 0.001 + 0.0005 * i, pos=(0.02 * i - 0.06, 0.05, 0.06), density=2000.0)
    (line,) = _warnings(scene, caplog)
    assert "bead0 (2 mm across), bead1 (3 mm across), bead2 (4 mm across), bead3 (5 mm across) and 3 more can" in line
    assert "cells of 1 mm or less" in line


def test_the_smallest_extent_of_each_primitive_is_its_thinnest_full_width():
    assert VC.smallest_extent(Primitive("sphere", (0.003,))) == pytest.approx(0.006)
    assert VC.smallest_extent(Primitive("box", (0.05, 0.001, 0.02))) == pytest.approx(0.002)
    assert VC.smallest_extent(Primitive("cylinder", (0.002, 0.03))) == pytest.approx(0.004)
    assert VC.smallest_extent(Primitive("cone", (0.002, 0.03))) == pytest.approx(0.004)
    assert VC.smallest_extent(Primitive("plane", ())) == np.inf
    assert VC.smallest_extent(MeshSource("part.obj", "")) == np.inf            # meshes are not checked


def test_the_cell_is_the_median_rest_edge():
    v = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], float) * CELL
    assert VC.median_edge(v, np.array([0, 1, 2, 1, 3, 2])) == pytest.approx(CELL)   # 4 sides and a diagonal twice
    assert VC.median_edge(v, np.zeros((0, 3), int)) == np.inf
