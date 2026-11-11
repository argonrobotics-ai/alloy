# SPDX-License-Identifier: Apache-2.0
"""Soft seam twin stiffness vs VBD convergence (alloy/topology/emit.py ``SOFT_SEAM_KE_RATIO``; coordinator GO
2026-11-06, [internal note] scr/WP10.md), CPU.

A pre-split block must carry load like the solid one at 10 VBD iterations. Twins 100x stiffer than the material
converge so slowly that the TOP-2 block (8 x 4 x 4 cells of 2.5 mm, agar-like, clamped at the bottom, one seam at
x = 10 mm) sagged 9x the solid block under 1 g after 0.3 s.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.testing.topology import clamp_particles
from alloy.topology.emit import SOFT_SEAM_KE_RATIO, add_presplit_soft
from alloy.topology.meshes import grid_tet_mesh
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import TopologyBudget

CELLS, CELL = (8, 4, 4), 2.5e-3


def _top_sag(plane_x: float, ratio: float = SOFT_SEAM_KE_RATIO) -> float:
    """[m] mean top-face sag after 0.3 s at 1 g (10 VBD iterations, dt 0.5 ms); plane_x outside the block = solid."""
    v, t = grid_tet_mesh(*CELLS, CELL)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    add_presplit_soft(b, v, t, SplitPlan(planes=(((1.0, 0.0, 0.0), plane_x),)),
                      dict(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                           density=1010.0, k_mu=1.72e4, k_lambda=1.55e5, k_damp=0.0, particle_radius=1.0e-3),
                      budget=TopologyBudget(spare_vertices=1000), seam_ke_ratio=ratio)
    b.color()
    m = b.finalize(device="cpu")
    x0 = m.particle_q.numpy()
    clamp_particles(m, np.flatnonzero(x0[:, 2] <= 1e-9))
    solver = newton.solvers.SolverVBD(m, iterations=10)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(600):
        solver.step(s0, s1, ctrl, None, 5.0e-4)
        s0, s1 = s1, s0
    top = np.isclose(x0[:, 2], CELLS[2] * CELL)
    return float((x0[top, 2] - s0.particle_q.numpy()[top, 2]).mean())


@pytest.fixture(scope="module")
def solid() -> float:
    return _top_sag(1.0)


def test_a_pre_split_block_sags_within_10_percent_of_the_solid_one(solid) -> None:
    assert solid == pytest.approx(9.2e-6, rel=0.05)                     # E 5e4 Pa block: ~9 um at 1 g
    assert _top_sag(0.5 * CELLS[0] * CELL) / solid - 1.0 < 0.10       # measured +7 %


def test_twins_100x_stiffer_than_the_material_converge_too_slowly(solid) -> None:
    assert _top_sag(0.5 * CELLS[0] * CELL, ratio=100.0) / solid > 3.0   # measured 9.3x (the old ratio)
