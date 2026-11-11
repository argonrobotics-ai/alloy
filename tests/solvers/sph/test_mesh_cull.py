# SPDX-License-Identifier: Apache-2.0
"""Boundary cull radius of mesh shapes whose origin is not their AABB centre (found by SC1, 2026-11-03): the cull
measures from the shape origin, so the radius must enclose the mesh about that origin."""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton
from alloy.solvers.sph.layout import origin_radius


def _tall_box_mesh(hx: float, hy: float, height: float) -> newton.Mesh:
    """Closed box mesh standing on its origin (z from 0 to height), like a bottle with its origin at the base."""
    v = np.array([[x, y, z] for z in (0.0, height) for y in (-hy, hy) for x in (-hx, hx)], dtype=np.float32)
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
                  [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]], dtype=np.int32)
    return newton.Mesh(v, f.reshape(-1))


def test_origin_radius_encloses_an_off_centre_mesh():
    b = newton.ModelBuilder()
    b.add_shape_mesh(-1, mesh=_tall_box_mesh(0.035, 0.035, 0.21))
    b.add_shape_box(-1, hx=0.01, hy=0.02, hz=0.03)
    m = b.finalize(device="cpu")
    newton_r = m.shape_collision_radius.numpy()
    r = origin_radius(m)
    far = float(np.linalg.norm([0.035, 0.035, 0.21]))
    assert newton_r[0] < 0.6 * far  # Newton's radius is about the AABB centre (the bug's cause)
    assert r[0] >= far - 1e-6
    assert r[1] == newton_r[1]  # primitives are centred on their origin: unchanged
