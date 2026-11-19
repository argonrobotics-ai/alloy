# SPDX-License-Identifier: Apache-2.0
"""Surface tension on small liquid pieces (csf_sparse): a free cluster at rest gains no energy from tension.

Law (physics 2026-11-17): free clusters of 7 / 19 / 33 particles at rest in zero gravity, 1500 steps at the capillary
step, kinetic energy at most 10 % of the surface energy the cluster holds (sigma A / m, A of the sphere of its
volume). CSF alone gave 1,120 / 59 / 0.7 %; with the pair force 0.1 / 0.5 / 0.6 %. CPU, about 2 min for all three."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.solvers.sph import csf_sparse
from alloy.solvers.sph.drops import WATER, _solver, drop_points


def _kinetic_peak(radius_cells: float, steps: int = 1500) -> tuple[int, float, float]:
    """(particles, peak kinetic energy per kg, surface energy per kg) of a free cluster at rest."""
    m, s = _solver(drop_points(radius_cells, 2.0 * WATER.r), WATER, device="cpu")
    n, sp = m.particle_count, 2.0 * WATER.r
    radius = (3.0 * n * sp**3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    e_surf = WATER.sigma * 4.0 * math.pi * radius**2 / (n * WATER.rho0 * sp**3)
    a, c, peak = m.state(), m.state(), 0.0
    for k in range(steps):
        s.step(a, c, None, None, s.config.dt_max)
        a, c = c, a
        if k % 10 == 9:
            peak = max(peak, 0.5 * float((a.particle_qd.numpy().astype(np.float64) ** 2).sum(axis=1).mean()))
    return n, peak, e_surf


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("radius_cells", "n"), [(1.0, 7), (1.5, 19), (2.0, 33)])
def test_a_free_cluster_at_rest_gains_no_energy_from_tension(radius_cells, n):
    count, peak, e_surf = _kinetic_peak(radius_cells)
    assert count == n
    assert peak <= 0.10 * e_surf, f"{n} particles: peak {peak:.3g} J/kg, {peak / e_surf:.1%} of {e_surf:.3g}"


def test_the_band_reads_a_full_neighbourhood():
    """The band sits between a 19-particle cluster's largest count (18) and a resolved interior's (about 26)."""
    assert 18.0 <= csf_sparse.CSF_BAND[0] < csf_sparse.CSF_BAND[1] <= 26.0
