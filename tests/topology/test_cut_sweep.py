# SPDX-License-Identifier: Apache-2.0
"""Opened-face deviation geometry (alloy/topology/cut_sweep.py), CPU: exact point-triangle distance, the swept
surface of a sampled edge, the per-world episode start, and the vectorised quaternion rotate it uses."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.topology.blade_contact import quat_rotate
from alloy.topology.cut_sweep import blade_slots, point_triangle_distance, swept_distance

A, B, C = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


@pytest.mark.parametrize(("p", "d"), [
    ((0.2, 0.2, 0.5), 0.5),                         # over the interior: the height
    ((0.2, 0.2, -0.3), 0.3),                        # either side
    ((-1.0, -1.0, 0.0), np.sqrt(2.0)),              # past vertex A
    ((0.5, -0.4, 0.3), 0.5),                        # beside edge AB: distance to the edge line
    ((1.0, 1.0, 0.0), np.sqrt(0.5)),                # beside the hypotenuse
])
def test_point_triangle_distance_by_region(p, d) -> None:
    assert point_triangle_distance(np.array(p), A, B, C) == pytest.approx(d, abs=1e-12)


def test_point_triangle_distance_matches_dense_sampling() -> None:
    rng = np.random.default_rng(7)
    u, v = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    keep = u + v <= 1.0
    u, v = u[keep], v[keep]
    for _ in range(20):
        a, b, c, p = rng.normal(size=(4, 3))
        pts = a + u[:, None] * (b - a) + v[:, None] * (c - a)
        brute = np.linalg.norm(pts - p, axis=1).min()
        exact = float(point_triangle_distance(p, a, b, c))
        span = max(np.linalg.norm(b - a), np.linalg.norm(c - a))
        assert exact <= brute + 1e-12
        assert brute - exact <= span / 200.0


def test_a_degenerate_triangle_is_measured_by_its_edges() -> None:
    a, b, c = np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])    # collinear
    assert point_triangle_distance(np.array([1.5, 0.3, 0.4]), a, b, c) == pytest.approx(0.5)
    assert point_triangle_distance(np.array([3.0, 0.0, 0.0]), a, a, a) == pytest.approx(3.0)   # a point


def _descent() -> np.ndarray:
    """[1, 2, 2, 3] an edge along y (y in [-1, 1]) sampled at z = 0, then z = -1."""
    e = np.array([[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0]]])
    return e[None]


def test_swept_distance_is_the_distance_to_the_swept_square() -> None:
    e = _descent()
    pts = np.array([[0.3, 0.5, -0.5], [0.0, 0.0, 0.4], [0.0, 1.5, -0.5], [-0.2, -0.9, -0.99]])
    d = swept_distance(pts, np.repeat(e, len(pts), axis=0), np.zeros(len(pts), np.int64))
    np.testing.assert_allclose(d, [0.3, 0.4, 0.5, 0.2], atol=1e-12)


def test_swept_distance_counts_samples_from_the_episode_start_only() -> None:
    p = np.array([[0.3, 0.5, -0.5]])
    assert swept_distance(p, _descent(), np.array([1]))[0] == pytest.approx(np.hypot(0.3, 0.5))  # the z = -1 edge
    assert swept_distance(p, _descent()[:, :1], np.array([0]))[0] == pytest.approx(np.hypot(0.3, 0.5))


def test_blade_slots_pad_worlds_without_blades() -> None:
    np.testing.assert_array_equal(blade_slots(np.array([0, 2, 0]), 3), [[0, 2], [-1, -1], [1, -1]])
    np.testing.assert_array_equal(blade_slots(np.zeros(0, np.int64), 2), [[-1], [-1]])


def test_quat_rotate_broadcasts_row_by_row() -> None:
    rng = np.random.default_rng(3)
    q = rng.normal(size=(5, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(2, 3))
    out = quat_rotate(q[:, None, :], v)
    assert out.shape == (5, 2, 3)
    for i in range(5):
        for k in range(2):
            np.testing.assert_allclose(out[i, k], quat_rotate(q[i], v[k]), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), np.broadcast_to(np.linalg.norm(v, axis=1), (5, 2)))
