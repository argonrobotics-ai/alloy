# SPDX-License-Identifier: Apache-2.0
"""The world digest covers State.particle_q_lo (SCR-WP11-F32): VBD's sub-ulp position remainder is state, so two
states equal in particle_q / particle_qd but different there are different worlds. Rigid-only digests do not move."""

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.precision.digest import world_digests


def _cloth_model():
    b = newton.ModelBuilder()
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=3, dim_y=3,
                     cell_x=0.05, cell_y=0.05, mass=0.01)
    return b.finalize(device="cpu")


def test_a_change_in_particle_q_lo_alone_changes_the_digest():
    model = _cloth_model()
    a, b = model.state(), model.state()
    assert a.particle_q_lo is not None
    same = world_digests(model, a)
    assert np.array_equal(same, world_digests(model, b))
    lo = b.particle_q_lo.numpy()
    lo[4, 2] = 1.0e-9
    b.particle_q_lo.assign(lo)
    assert np.array_equal(a.particle_q.numpy(), b.particle_q.numpy())
    assert not np.array_equal(same, world_digests(model, b))


def test_a_rigid_only_digest_is_unchanged_by_the_new_array():
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = b.finalize(device="cpu")
    state = model.state()
    with_array = world_digests(model, state)
    state.particle_q_lo = None                       # as on a tree before the array existed
    assert np.array_equal(with_array, world_digests(model, state))
