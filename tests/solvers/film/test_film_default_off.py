# SPDX-License-Identifier: Apache-2.0
"""FILM-6 default off: SolverFilm without film attributes (or with them all at their defaults) steps bitwise as
Newton's SolverVBD (ruling 2026-11-03: every current VBD/sheet scene unchanged). CPU, small."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.film import SolverFilm
from alloy.solvers.film.config import register_custom_attributes
from alloy.solvers.film.rigs import grid_mesh

pytestmark = pytest.mark.compile_heavy


def _towel_like(film_attrs: bool) -> newton.Model:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    if film_attrs:
        register_custom_attributes(b)
    v, t = grid_mesh(6, 6, 0.02, plane="xz")
    b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.5), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(),
                     vertices=v.astype(np.float32), indices=t.reshape(-1).tolist(), density=0.3, tri_ke=660.0,
                     tri_ka=0.0, tri_kd=1e-3, edge_ke=2e-5, edge_kd=1e-8, particle_radius=3e-4)
    for p in (0, 6 * 7):
        b.particle_mass[p] = 0.0
        b.particle_flags[p] = 0
    b.color()
    return b.finalize(device="cpu")


def _run(model: newton.Model, cls, steps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    s0, s1 = model.state(), model.state()
    solver = cls(model, iterations=4)
    for _ in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, None, None, 1.0e-3)
        s0, s1 = s1, s0
    return s0.particle_q.numpy(), s0.particle_qd.numpy()


@pytest.mark.parametrize("film_attrs", [False, True], ids=["no_attributes", "default_attributes"])
def test_solver_film_is_vbd_without_film_items(film_attrs: bool) -> None:
    q_ref, qd_ref = _run(_towel_like(film_attrs), newton.solvers.SolverVBD)
    model = _towel_like(film_attrs)
    q, qd = _run(model, SolverFilm)
    assert np.array_equal(q, q_ref) and np.array_equal(qd, qd_ref)


def test_film_off_builds_no_film_arrays() -> None:
    solver = SolverFilm(_towel_like(True), iterations=2)
    assert not solver.film_topology.any and solver.film_state_arrays() == {}
