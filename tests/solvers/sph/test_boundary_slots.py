# SPDX-License-Identifier: Apache-2.0
"""A particle's full row of 4 boundary slots (boundary_slots._shape_insert), CPU. The case from a 15 mm cup of 24 slabs
at 2 mm: a particle in the floor corner at the inner vertex between the 30 and 45 deg slabs had the floor, the 30 deg
slab and the 15 and 60 deg slabs (1.7-1.9 mm off) in its slots; the 45 deg slab, 0.1 mm off, was dropped as facing the
same way as the 30 deg one, and the particle stepped into it and out through the cup's outer face."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.boundary_slots import _shape_insert

NEAR = 0.4e-3  # 0.1 h at 2 mm particles
FLOOR = 99


@wp.kernel(enable_backward=False)
def _insert(shape: wp.array2d[wp.int32], phi: wp.array2d[wp.float32], face: wp.array2d[wp.int32],
            nrm: wp.array2d[wp.vec3], s: int, p: float, n: wp.vec3, near: float, out: wp.array[wp.int32]):
    out[0] = _shape_insert(shape, phi, face, nrm, 0, 4, s, p, n, near)


def _wall(deg: float) -> tuple[float, float, float]:
    a = math.radians(deg)
    return (math.cos(a), math.sin(a), 0.0)


def _row(slots, new) -> tuple[int, list[int]]:
    """Insert ``new`` = (shape, phi, normal) into the full row ``slots`` (phi-ascending); (kept, shapes after)."""
    d = "cpu"
    shape = wp.array(np.array([[s for s, _, _ in slots]], np.int32), dtype=wp.int32, device=d)
    phi = wp.array(np.array([[p for _, p, _ in slots]], np.float32), dtype=wp.float32, device=d)
    face = wp.array(np.full((1, 4), -1, np.int32), dtype=wp.int32, device=d)
    nrm = wp.array(np.array([[n for _, _, n in slots]], np.float32), dtype=wp.vec3, device=d)
    out = wp.zeros(1, dtype=wp.int32, device=d)
    wp.launch(_insert, dim=1, inputs=[shape, phi, face, nrm, new[0], new[1], wp.vec3(*new[2]), NEAR, out], device=d)
    ph = phi.numpy()[0]
    assert np.all(np.diff(ph) >= 0.0), ph
    return int(out.numpy()[0]), shape.numpy()[0].tolist()


def test_the_nearer_of_two_slabs_at_an_inner_vertex_is_kept():
    slots = [(FLOOR, 0.0, (0.0, 0.0, 1.0)), (2, 1.0e-6, _wall(30)), (1, 1.67e-3, _wall(15)), (4, 1.88e-3, _wall(60))]
    kept, after = _row(slots, (3, 1.0e-4, _wall(45)))
    assert kept == 1 and after == [FLOOR, 2, 3, 1]                   # the farthest duplicate (60 deg) goes


def test_the_floor_takes_a_corner_slot_from_four_slabs():
    slots = [(2, 1.0e-4, _wall(30)), (3, 2.0e-4, _wall(45)), (1, 5.0e-4, _wall(15)), (4, 6.0e-4, _wall(60))]
    kept, after = _row(slots, (FLOOR, 1.0e-3, (0.0, 0.0, 1.0)))
    assert kept == 1 and after == [2, 3, 1, FLOOR]


def test_a_farther_duplicate_is_dropped():
    slots = [(FLOOR, 0.0, (0.0, 0.0, 1.0)), (2, 1.0e-6, _wall(30)), (3, 1.0e-4, _wall(45)), (1, 1.67e-3, _wall(15))]
    kept, after = _row(slots, (4, 1.88e-3, _wall(60)))
    assert kept == 0 and after == [FLOOR, 2, 3, 1]


@pytest.mark.parametrize("deg", [90.0, 180.0])
def test_walls_within_a_cfl_step_are_never_evicted(deg):
    """Four slots within ``near`` stay; a farther wall facing another way finds no room."""
    slots = [(FLOOR, 0.0, (0.0, 0.0, 1.0)), (2, 0.0, _wall(30)), (3, 1.0e-4, _wall(45)), (1, 2.0e-4, _wall(15))]
    kept, after = _row(slots, (7, 3.0e-3, _wall(deg)))
    assert kept == 0 and after == [FLOOR, 2, 3, 1]
