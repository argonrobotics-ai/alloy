# SPDX-License-Identifier: Apache-2.0
"""The film projection's reaction on its pinned vertices (``strain.vertex_reaction``, ``SolverFilm.projection_reaction``;
docs/known_gaps/film-projection-force.md). A constraint moves only its free vertices, so the momentum the projection
gives the free film comes through the pins: each step the pins' reaction sums to minus the projection's momentum change
of the free vertices, over dt (the XPBD reaction λ∇C). A sheet with no pinned vertex reads none (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.materials import HDPE_15UM
from alloy.solvers.film.rigs import FilmRig, grid_mesh, pin

pytestmark = pytest.mark.compile_heavy
LOAD = 0.005                                    # kg on the bottom row: past the membrane's 2.6 mN, the projection acts


def _strip(pinned: bool = True) -> tuple[FilmRig, np.ndarray]:
    v, t = grid_mesh(8, 16, 5e-3, plane="xz")                                  # 40 x 80 mm, hanging in x-z
    top = np.flatnonzero(np.abs(v[:, 2] - v[:, 2].max()) < 1e-9)
    bottom = np.flatnonzero(np.abs(v[:, 2] - v[:, 2].min()) < 1e-9)

    def build(b):
        p0, _ = add_film_mesh(b, v, t, HDPE_15UM, pos=(0.0, 0.0, 0.1))           # near 0: float32 shifts
        if pinned:
            pin(b, p0 + top)
            for i in p0 + bottom:
                b.particle_mass[i] += LOAD / len(bottom)
        return {}

    return FilmRig(build, dt=1.0e-3), top


def test_pins_reaction_is_minus_the_projections_momentum_change() -> None:
    rig, top = _strip()
    s, m = rig.solver, rig.model.particle_mass.numpy()
    free = np.setdiff1d(np.arange(len(m)), top)
    shift = {}
    project = s._project

    def watched(state_out, dt, contacts=None):                                  # the projection's own shift
        x0 = state_out.particle_q.numpy().copy()
        project(state_out, dt, contacts)
        shift["dp"] = (m[free, None] * (state_out.particle_q.numpy()[free] - x0[free])).sum(axis=0) / dt ** 2

    s._project = watched
    rig.run(0.05)
    acted = 0
    for _ in range(40):                                                         # the load falls onto the cap and rings
        rig.step()
        r = s.projection_reaction.numpy()
        assert np.all(r[free] == 0.0)                                           # only pinned vertices take one
        f = r[top].sum(axis=0)
        scale = max(float(np.abs(shift["dp"]).max()), 1e-9)
        assert np.abs(f + shift["dp"]).max() <= 5e-3 * scale + 1e-9                  # dp from float32 positions
        acted += int(scale > 1e-6)
    assert acted >= 30                                                          # the projection carried the load


def test_unpinned_sheet_has_no_reaction() -> None:
    rig, _ = _strip(pinned=False)
    rig.run(0.05)
    assert np.all(rig.solver.projection_reaction.numpy() == 0.0)
