# SPDX-License-Identifier: Apache-2.0
"""create: a generated mesh is turned onto the face it would rest on (camera tilt, rounded bases, toppling shapes)."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from alloy.create.level import level, resting_normal  # noqa: E402


def _tilted(mesh, degrees: float, axis=(1.0, 0.0, 0.0)):
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), axis))
    return mesh


def _rests_flat(mesh) -> bool:
    """The face the mesh would rest on (centre of mass above it) points straight down."""
    return bool(np.isclose(resting_normal(mesh) @ [0.0, 0.0, -1.0], 1.0, atol=1e-6))


@pytest.mark.parametrize("degrees", [8.0, 20.0, 30.0])
def test_a_cup_tilted_by_the_camera_stands_on_its_base(degrees):
    cup = _tilted(trimesh.creation.cylinder(radius=0.04, height=0.1, sections=48), degrees, (0.3, 1.0, 0.0))
    done = level(cup)
    assert done.applied and done.degrees == pytest.approx(degrees, abs=0.5)
    assert _rests_flat(cup) and cup.extents[2] == pytest.approx(0.1, abs=1e-6)


def test_an_upright_mesh_is_left_alone():
    box = trimesh.creation.box((0.1, 0.1, 0.05))
    before = box.vertices.copy()
    assert not level(box).applied and np.array_equal(box.vertices, before)


def test_a_domed_base_rolls_back_upright():
    cup = trimesh.creation.cylinder(radius=0.04, height=0.1, sections=48)
    bevel = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
    bevel.apply_translation((0.0, 0.0, -0.03))
    rounded = trimesh.util.concatenate([cup, bevel]).convex_hull           # a cup on a shallow dome
    _tilted(rounded, 25.0)
    level(rounded)
    assert _rests_flat(rounded) and abs(rounded.center_mass[0] - rounded.bounds.mean(axis=0)[0]) < 0.01


def test_a_tall_thin_shape_tilted_far_falls_onto_its_side():
    stick = _tilted(trimesh.creation.cylinder(radius=0.01, height=0.2, sections=24), 70.0)
    done = level(stick)
    assert done.applied and _rests_flat(stick) and stick.extents[2] < 0.03            # lying down, as dropped
