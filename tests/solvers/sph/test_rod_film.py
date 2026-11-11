# SPDX-License-Identifier: Apache-2.0
"""Resting film on a convex rod stays attached (SCR-SC1-3 (b), coordinator law 2026-11-03): zero gravity, a film three
spacings thick around an analytic rod of radius 2-10 spacings, seeded as a Cartesian lattice and as rings; the gap
between the wall and the innermost particles' surfaces (10th percentile of the centre distance minus r) stays within
0.5 spacing. SC1 withdrew its lift-off report as a measurement artifact (a drained gap band); this pins the resting
behaviour. Flowing films are SCR-SC1-3 (a)'s laws (test_wall_viscosity.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.sph.build import add_fluid_points
from alloy.solvers.sph.config import BOUNDARY_SDF, DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH

R_PARTICLE = 1.0e-3
LAYERS = 3
LENGTH_CELLS = 16


def film_points(rod: float, dx: float, length: float, seed: str) -> np.ndarray:
    """A film of LAYERS spacings around a rod of radius ``rod`` along z: lattice cells or staggered rings."""
    r = 0.5 * dx
    if seed == "cart":
        n = int(math.ceil((rod + LAYERS * dx) / dx)) + 1
        ax = (np.arange(-n, n + 1) + 0.5) * dx
        az = (np.arange(int(length / dx)) + 0.5) * dx - 0.5 * length
        pts = np.stack(np.meshgrid(ax, ax, az, indexing="ij"), -1).reshape(-1, 3)
        rho = np.hypot(pts[:, 0], pts[:, 1])
        return pts[(rho >= rod + r) & (rho < rod + r + LAYERS * dx)]
    pts = []
    for layer in range(LAYERS):
        rl = rod + r + layer * dx
        na = max(int(round(2.0 * math.pi * rl / dx)), 3)
        t = 2.0 * math.pi * (np.arange(na) + 0.5 * (layer % 2)) / na
        for k in range(int(length / dx)):
            z = (k + 0.5) * dx - 0.5 * length
            pts.append(np.c_[rl * np.cos(t), rl * np.sin(t), np.full(na, z)])
    return np.concatenate(pts)


def film_gap(rod_cells: float, seed: str, steps: int = 400, dt: float = 1.0e-3) -> tuple[float, float]:
    """(gap at the start, largest gap over ``steps``) in spacings, measured on the middle half of the rod."""
    cfg = DFSPHConfig(particle_radius=R_PARTICLE)
    dx = cfg.spacing
    rod, length = rod_cells * dx, LENGTH_CELLS * dx
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    b.add_shape_cylinder(-1, xform=wp.transform_identity(), radius=rod, half_height=length,
                         custom_attributes={"sph:shape_boundary": BOUNDARY_SDF})
    add_fluid_points(b, cfg, film_points(rod, dx, length, seed), 1000.0, 1.0e-3)
    b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()

    def gap(q: np.ndarray) -> float:
        mid = np.abs(q[:, 2]) < 0.25 * length
        return float(np.percentile(np.hypot(q[mid, 0], q[mid, 1]) - rod, 10) - 0.5 * dx) / dx

    g0, worst = gap(a.particle_q.numpy()), 0.0
    for k in range(steps):
        s.step(a, c, None, None, dt)
        a, c = c, a
        if (k + 1) % 50 == 0:
            worst = max(worst, gap(a.particle_q.numpy()))
    return g0, worst


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("seed", ["cart", "rings"])
@pytest.mark.parametrize("rod_cells", [2.0, 4.0, 10.0])
def test_resting_film_on_convex_rod_stays_attached(rod_cells, seed):
    g0, worst = film_gap(rod_cells, seed)
    assert worst <= 0.5, f"film lifted off a rod of {rod_cells} spacings ({seed}): gap {worst:.3f} (start {g0:.3f})"

