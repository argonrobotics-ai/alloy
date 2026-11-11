# SPDX-License-Identifier: Apache-2.0
"""Vendored patch 0011-box-triangle-plane-exit: a mesh triangle that dips slightly into a box face reports that dip.

Newton 1.8 restricts a box's exits from a one-sided mesh triangle to the triangle's front side and discarded an exit
that leaned a little behind it. For a cube mesh resting slightly tilted on a wide box, the shallow exit (lift the
triangle out of the top face) leans that way on some side triangles, and the next exit crossed the whole box: a 62 mm
contact for a 0.5 mm dip. The patch projects such an exit onto the triangle's plane.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.geometry.penetration import _box_triangle_depth

HALF = (0.075, 0.125, 0.04)          # the wide box
SIDE, DIP, TILT = 0.01, 0.0005, -0.02  # a 2 cm cube mesh, its lowest corner 0.5 mm inside the top face, tilted 0.02 rad


@wp.kernel
def _one_sided_depths(half: wp.vec3, tri: wp.array2d(dtype=wp.vec3), depth: wp.array(dtype=float),
                      direction: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    d, n = _box_triangle_depth(half, tri[i, 0], tri[i, 1], tri[i, 2], True)
    depth[i] = d
    direction[i] = n


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _resting_cube_triangles(axis: tuple[float, float, float]) -> np.ndarray:
    """(12, 3, 3) triangles of the tilted cube mesh in the box's frame, its lowest corner DIP inside the top face."""
    cube = newton.Mesh.create_box(SIDE, SIDE, SIDE, duplicate_vertices=False, compute_normals=False,
                                  compute_uvs=False, compute_inertia=False)
    v = np.asarray(cube.vertices, dtype=float) @ _rotation(axis, TILT).T
    v += np.array([-0.02, 0.03, HALF[2] - DIP - v[:, 2].min()])
    return v[np.asarray(cube.indices).reshape(-1, 3)]


@pytest.mark.parametrize("axis", [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.6, 0.8, 0.0)])
def test_a_triangle_dipping_into_a_box_face_reports_the_dip(axis) -> None:
    tris = _resting_cube_triangles(axis)
    n = len(tris)
    depth, direction = wp.zeros(n, dtype=float, device="cpu"), wp.zeros(n, dtype=wp.vec3, device="cpu")
    wp.launch(_one_sided_depths, dim=n, device="cpu",
              inputs=[wp.vec3(*HALF), wp.array(tris, dtype=wp.vec3, device="cpu"), depth, direction])
    depth, direction = depth.numpy(), direction.numpy()
    dips = HALF[2] - tris[:, :, 2].min(axis=1)              # how far each triangle's lowest corner is inside the face
    inside = dips > 1e-6
    assert inside.sum() >= 2                                 # the bottom face and the side faces at the low corner
    # Every such triangle leaves through the top face: its own dip (to 0.1 mm), never a path across the box.
    assert np.all(depth[inside] > 0.0) and np.all(depth[inside] <= dips[inside] + 1e-4), (depth[inside], dips[inside])
    assert np.all(direction[inside][:, 2] > np.cos(2.0 * TILT)), direction[inside]
    assert np.all(depth[~inside] <= 1e-6)                    # a triangle above the face is not in contact
