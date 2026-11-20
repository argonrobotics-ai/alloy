# SPDX-License-Identifier: Apache-2.0
"""Seam twins in VBD's colours. Newton's colouring leaves springs out, so the film moves a twin end that shares its
partner's colour to a free one (``alloy.solvers.film.seam_vbd.recolor_twins``). VBD keeps two colour records: the
groups (which particles each colour solves) and ``particle_colors`` (to which particles the spring, body-contact and
self-contact kernels add an element's force while a colour is solved). A twin moved in the groups only got its seam
spring under its old colour, a colour late or after it was solved: on the tearable gift bag 374 twin ends, and the
standing bag gained energy each step until the run went UNCONVERGED."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

H = 0.01                      # m, cell
N = 4                         # cells per side of each sheet
SEAM_KE = 100.0 * 50.0        # N/m, the topology's seam ratio (100) times the sheets' tri_ke
GAP = 2.0e-3                  # m, the seam opened before the step
DT = 1.25e-3                  # s, the film route's step at a 10 ms control period


def _two_sheets():
    """Two N x N sheets side by side, their shared edge duplicated (the twins) and joined by zero-length springs,
    each twin in its partner's colour (``_colour_twins_alike``: Newton's graph leaves springs out, so nothing keeps
    them apart), then the film's recolouring. Returns the model, the twin rows and how many ends it moved."""
    import warp as wp

    from alloy._vendor import newton
    from alloy.solvers.film.seam_vbd import recolor_twins

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x0 in (0.0, N * H):
        b.add_cloth_grid(pos=wp.vec3(x0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=N, dim_y=N,
                         cell_x=H, cell_y=H, mass=1.0e-4, tri_ke=50.0, tri_ka=50.0, tri_kd=0.0, edge_ke=1.0e-3)
    x = np.asarray(b.particle_q)
    per = (N + 1) ** 2
    a = np.flatnonzero(np.isclose(x[:per, 0], N * H))
    c = per + np.flatnonzero(np.isclose(x[per:, 0], N * H))
    a, c = a[np.argsort(x[a, 1])], c[np.argsort(x[c, 1])]
    for i, j in zip(a.tolist(), c.tolist()):
        b.add_spring(i, j, SEAM_KE, 0.0, 0.0)
    b.color()
    m = b.finalize(device="cpu")
    _colour_twins_alike(m, a, c, per)
    moved = recolor_twins(m, SimpleNamespace(a=a.astype(np.int64), b=c.astype(np.int64)))
    return m, a, c, moved


def _colour_twins_alike(m, a, c, per: int) -> None:
    """Keep the first sheet's colours; colour the second greedily over its triangles with each twin ``c`` given its
    partner's colour first. Both records are written, as Newton's builder writes them."""
    import warp as wp

    color = m.particle_colors.numpy().astype(np.int64)
    color[per:] = -1
    color[c] = color[a]
    nbr = [set() for _ in range(m.particle_count)]
    for tri in m.tri_indices.numpy().tolist():
        for v in tri:
            nbr[v].update(u for u in tri if u != v)
    for v in range(per, m.particle_count):
        if color[v] < 0:
            used = {int(color[u]) for u in nbr[v]}
            color[v] = next(k for k in range(m.particle_count) if k not in used)
    dt = m.particle_color_groups[0].dtype
    m.particle_color_groups = [wp.array(np.flatnonzero(color == g).astype(np.int32), dtype=dt, device=m.device)
                               for g in range(int(color.max()) + 1)]
    m.particle_colors.assign(color.astype(np.int32))


def _opened(iterations: int):
    """The sheets with the second one moved GAP away from the seam, and a VBD solver; no gravity, no contact."""
    from alloy._vendor.newton.solvers import SolverVBD

    m, a, c, _ = _two_sheets()
    s0, s1 = m.state(), m.state()
    x = s0.particle_q.numpy().copy()
    x[(N + 1) ** 2:, 0] += GAP
    s0.particle_q.assign(x)
    return m, a, c, s0, s1, SolverVBD(m, iterations=iterations)


def test_a_recoloured_twin_carries_its_new_colour_in_both_records():
    m, a, c, moved = _two_sheets()
    assert moved > 0, "the sheets must need a recolouring for this law to mean anything"
    group = np.full(m.particle_count, -1)
    for g, ids in enumerate(m.particle_color_groups):
        group[ids.numpy()] = g
    assert (group[a] != group[c]).all()
    assert (m.particle_colors.numpy() == group).all()


@pytest.mark.compile_heavy
def test_one_step_at_ten_iterations_closes_an_opened_seam():
    """Both twins feel their seam spring in the colour that solves them, so one step of ten iterations closes a 2 mm
    opening to 13 um; with the moved twins' spring added under their old colour it stayed 1.4 mm open."""
    m, a, c, s0, s1, solver = _opened(10)
    solver.step(s0, s1, m.control(), None, DT)
    x1 = s1.particle_q.numpy()
    assert np.abs(x1[a] - x1[c]).max() < 0.05 * GAP, np.abs(x1[a] - x1[c]).max()


@pytest.mark.compile_heavy
def test_seam_springs_add_no_energy_at_five_iterations():
    """No gravity, no contact: the only energy is what the opened seam stored (k GAP^2 / 2 a spring, 0.05 J), so no
    step may hold more kinetic energy than that. At five iterations it stays under 1e-3 J; with the moved twins'
    spring added under their old colour it reached 5e11 J within 20 steps."""
    m, a, c, s0, s1, solver = _opened(5)
    stored = 0.5 * SEAM_KE * GAP ** 2 * len(a)
    mass = m.particle_mass.numpy()
    for _ in range(20):
        solver.step(s0, s1, m.control(), None, DT)
        s0, s1 = s1, s0
        v = s0.particle_qd.numpy()
        assert 0.5 * float((mass * (v * v).sum(axis=1)).sum()) < stored
