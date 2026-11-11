# SPDX-License-Identifier: Apache-2.0
"""Soft tet strain beyond the model range (alloy/topology/soft_strain.py; coordinator ruling 2026-11-06T14:17Z), CPU."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.topology.geometry import INVERTED_STRAIN
from alloy.topology.meshes import grid_tet_mesh
from alloy.topology.soft_strain import SOFT_STRAIN_RANGE, SoftStrain

CELL = 5.0e-3


def _blocks(worlds: int) -> newton.Model:
    v, t = grid_tet_mesh(2, 1, 1, CELL)
    tmpl = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    tmpl.add_soft_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                       vertices=[wp.vec3(*q) for q in v], indices=t.ravel().tolist(), density=1000.0, k_mu=5.0e4,
                       k_lambda=5.0e4, k_damp=0.0)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.replicate(tmpl, worlds)
    return b.finalize(device="cpu")


def _deform(model: newton.Model, world: int, f: np.ndarray) -> newton.State:
    """A state with world ``world``'s particles mapped by x -> f x about its first particle."""
    s = model.state()
    x = s.particle_q.numpy()
    sel = model.particle_world.numpy() == world
    x0 = x[sel][0]
    x[sel] = x0 + (x[sel] - x0) @ np.asarray(f, np.float32).T
    s.particle_q.assign(x)
    return s


@pytest.mark.parametrize(("f", "strain"), [(np.diag([1.7, 1.0, 1.0]), 0.7), (np.diag([1.0, 0.45, 1.0]), 0.55),
                                           (np.diag([1.2, 0.8, 1.1]), 0.2)])
def test_the_largest_principal_strain_is_read_per_world(f, strain) -> None:
    m = _blocks(3)
    out = SoftStrain(m).update(_deform(m, 1, f)).numpy()
    np.testing.assert_allclose(out, [0.0, strain, 0.0], atol=1e-5)
    assert (out[1] > SOFT_STRAIN_RANGE) == (strain > SOFT_STRAIN_RANGE)


def test_a_rotation_is_no_strain_and_an_inverted_tet_is_out_of_range() -> None:
    m = _blocks(2)
    c, s = np.cos(0.7), np.sin(0.7)
    np.testing.assert_allclose(SoftStrain(m).update(_deform(m, 0, [[c, -s, 0], [s, c, 0], [0, 0, 1]])).numpy(), 0.0,
                               atol=1e-5)
    out = SoftStrain(m).update(_deform(m, 1, np.diag([1.0, 1.0, -1.0]))).numpy()
    assert out[0] < 1e-5 and out[1] == pytest.approx(float(INVERTED_STRAIN))


def test_tets_without_material_are_left_out_and_each_update_starts_over() -> None:
    m = _blocks(1)
    probe = SoftStrain(m)
    s = _deform(m, 0, np.diag([2.0, 1.0, 1.0]))
    assert probe.update(s).numpy()[0] == pytest.approx(1.0, abs=1e-5)
    mat = m.tet_materials.numpy()
    mat[:, :2] = 0.0                                          # every tet dormant (as a split tet's whole)
    m.tet_materials.assign(mat)
    assert probe.update(s).numpy()[0] == 0.0
