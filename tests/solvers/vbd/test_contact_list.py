# SPDX-License-Identifier: Apache-2.0
"""SCR-WP10-6: the `vbd` entry sizes SolverVBD's per-body soft-contact list from the model
(alloy.solvers.vbd.contact_list), so a body under a draped cloth keeps its contacts in its list instead of overflowing
Newton's 256 entries into a scan of every soft contact."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import make_vbd
from alloy.solvers.vbd.contact_list import LIST_MAX, LIST_MIN, list_size, records_per_body

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_body_contact_lanes import _model as _box_on_cloth  # noqa: E402  (the lane test's draped box)
from test_body_contact_lanes import _run as _run_lanes  # noqa: E402


def _cloth_and_boxes(n: int, shapes: int, worlds: int = 1) -> newton.Model:
    """``worlds`` copies of an n x n cloth (n-1 cells a side) and one free body with ``shapes`` boxes."""
    tpl = newton.ModelBuilder()
    body = tpl.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), mass=0.2)
    for k in range(shapes):
        tpl.add_shape_box(body, xform=wp.transform((0.01 * k, 0.0, 0.0), wp.quat_identity()), hx=0.01, hy=0.01, hz=0.01)
    tpl.add_cloth_grid(pos=wp.vec3(0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=n - 1, dim_y=n - 1,
                       cell_x=0.01, cell_y=0.01, mass=1.0e-3, particle_radius=1.0e-3)
    b = newton.ModelBuilder()
    for _ in range(worlds):
        b.add_world(tpl)
    b.color()
    return b.finalize(device="cpu")


def test_the_bound_is_a_bodys_shapes_times_its_worlds_particles_edges_and_triangles() -> None:
    m = _cloth_and_boxes(5, shapes=3, worlds=2)
    per_world = (m.particle_count + m.edge_count + m.tri_count) // 2
    assert records_per_body(m) == 3 * per_world
    assert list_size(m) == max(LIST_MIN, 1 << (3 * per_world - 1).bit_length())


def test_the_list_stays_within_newtons_floor_the_cap_and_the_memory_budget() -> None:
    small, large = _cloth_and_boxes(3, shapes=1), _cloth_and_boxes(30, shapes=8)
    assert list_size(small) == LIST_MIN                              # 9 particles: Newton's 256
    assert records_per_body(large) > LIST_MAX and list_size(large) == LIST_MAX
    assert list_size(large, budget=4 * large.body_count * 1000) == 512   # the largest power of two the budget holds
    assert list_size(large, budget=0) == LIST_MIN


def test_make_vbd_sizes_the_list_and_an_explicit_size_wins() -> None:
    m, _ = _box_on_cloth()
    assert make_vbd(m, {"iterations": 1}).body_particle_contact_buffer_pre_alloc == list_size(m) > 529
    explicit = {"iterations": 1, "rigid_body_particle_contact_buffer_size": 256}
    assert make_vbd(m, explicit).body_particle_contact_buffer_pre_alloc == 256


def test_the_sized_list_holds_the_draped_box_and_equals_a_large_list_bitwise() -> None:
    """The lane test's box (529 contacts) no longer overflows at the entry's own size, and its run equals the
    4,096-entry reference bit for bit (the same list, in the same order, on the CPU)."""
    sized, ref = _run_lanes(4, list_size(_box_on_cloth()[0])), _run_lanes(4, 4096)
    assert sized["count"] == ref["count"] == 529
    for key in ("force", "torque", "body_q", "body_qd", "particle_q"):
        np.testing.assert_array_equal(sized[key], ref[key])
