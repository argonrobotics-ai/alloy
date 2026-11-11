# SPDX-License-Identifier: Apache-2.0
"""alloy.io geometry helpers: poses, rotations, volumes, OBJ files, number text."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.io.geometry import (compose, inverse, is_identity, matrix_quat, mesh_volume, quat_matrix, read_obj,
                               shape_volume, tessellate, write_obj)
from alloy.io.model import IOShape
from alloy.io.urdf_write import rpy
from alloy.io.usd_text import num

RNG = np.random.default_rng(7)


def _random_pose():
    q = RNG.normal(size=4)
    return (*RNG.normal(size=3), *(q / np.linalg.norm(q)))


def test_compose_inverse_is_identity():
    for _ in range(20):
        p = _random_pose()
        assert is_identity(compose(p, inverse(p)), 1e-12)
        assert is_identity(compose(inverse(p), p), 1e-12)


def test_matrix_quat_round_trip():
    for _ in range(50):
        q = np.asarray(_random_pose()[3:])
        np.testing.assert_allclose(quat_matrix(matrix_quat(quat_matrix(q))), quat_matrix(q), atol=1e-12)


def _rpy_matrix(r, p, y):
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_urdf_rpy_reproduces_rotation():
    for _ in range(50):
        q = _random_pose()[3:]
        np.testing.assert_allclose(_rpy_matrix(*rpy(q)), quat_matrix(q), atol=1e-9)


@pytest.mark.parametrize("kind,size,exact", [
    ("box", (0.1, 0.2, 0.3), 0.048), ("sphere", (0.1,), 4 / 3 * math.pi * 1e-3),
    ("capsule", (0.05, 0.1), math.pi * 0.0025 * 0.2 + 4 / 3 * math.pi * 0.05 ** 3),
    ("cylinder", (0.05, 0.1), math.pi * 0.0025 * 0.2), ("cone", (0.05, 0.1), math.pi * 0.0025 * 0.2 / 3),
    ("ellipsoid", (0.1, 0.05, 0.02), 4 / 3 * math.pi * 1e-4)])
def test_tessellated_volume_close_to_analytic(kind, size, exact):
    shape = IOShape("s", kind, size)
    assert shape_volume(shape) == pytest.approx(exact, rel=1e-12)
    mesh = tessellate(shape, segments=48)
    assert mesh_volume(mesh) == pytest.approx(exact, rel=0.02)
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    assert hi[2] - lo[2] > hi[0] - lo[0] or kind in ("box", "sphere", "ellipsoid")   # axial shapes along +Z


def test_obj_round_trip(tmp_path):
    mesh = tessellate(IOShape("b", "box", (0.1, 0.2, 0.3)))
    digest = write_obj(mesh, tmp_path / "m" / "box.obj")
    back = read_obj(tmp_path / "m" / "box.obj")
    assert len(digest) == 64
    np.testing.assert_allclose(back.vertices, mesh.vertices, rtol=1e-8)
    np.testing.assert_array_equal(back.faces, mesh.faces)


def test_num_is_float32_shortest():
    assert num(0.029999999329447746) == "0.03"
    assert num(-0.0) == "0.0"
    assert float(num(1.2345678e-7)) == pytest.approx(1.2345678e-7, rel=1e-6)
