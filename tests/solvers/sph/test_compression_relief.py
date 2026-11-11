# SPDX-License-Identifier: Apache-2.0
"""Compression relief (pressure.py, DFSPHConfig.compression_relief_speed): a particle left in a wall's 1 um projection
skin (density ratio 1.5-2.5) must leave it at a bounded speed, not in one step. Found as the flat-impact blow-up: a
72 mm block dropped 20 mm flat into a 48 mm box glass (T4, r 1 mm) pinned corner particles in the skin and the
uncapped solve shot them and their neighbours out at 57 m/s (500 m/s converged), docs/notes/wp9-results.md."""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest

from alloy.solvers.sph.build import add_box_tank, add_fluid_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH

DT = 1.3e-4  # the step a CFL controller picks right after an impact at a few m/s (r = 1 mm)


def _pressed_column(relief: float | None, steps: int = 30) -> tuple[float, float, np.ndarray]:
    """6x6x8 column (r = 1 mm) two spacings clear of the side walls, its bottom layer pressed into the floor's
    projection skin (z = 1 um); returns the first step's max density ratio, the max speed over the steps and the
    final positions."""
    cfg = DFSPHConfig(particle_radius=1.0e-3, compression_relief_speed=relief)
    s_ = cfg.spacing
    n, gap = 6, 2
    w = (n + 2 * gap) * s_
    g = (np.arange(n) + 0.5 + gap) * s_
    pts = np.array([(x, y, (k + 0.5) * s_) for k in range(8) for y in g for x in g])
    pts[pts[:, 2] < s_, 2] = 1.0e-6
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    b.begin_world()
    add_box_tank(b, (0.0, 0.0, 0.0), (w, w, 18 * s_), wall=cfg.h)
    add_fluid_points(b, cfg, pts, 1000.0, 1.0e-3, (0.0, 0.0, 0.0))
    b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()
    ratio0, vmax = 0.0, 0.0
    for k in range(steps):
        s.step(a, c, None, None, DT)
        a, c = c, a
        if k == 0:
            ratio0 = float(s._w.ratio.numpy().max())
        vmax = max(vmax, float(np.linalg.norm(a.particle_qd.numpy(), axis=1).max()))
    return ratio0, vmax, a.particle_q.numpy()


def test_skin_particles_leave_at_the_relief_speed():
    """Capped: the pressed layer (ratio ~2.5) lifts out of the skin below the relief speed (measured 0.07 m/s) and
    stays above the floor; uncapped, the same layer is shot out at 3.6 m/s in the first step."""
    ratio0, vmax, q = _pressed_column(0.5)
    assert ratio0 > 2.0, ratio0  # the case under test: deep compression from the wall skin
    assert vmax < 0.5, vmax
    assert np.all(np.isfinite(q)) and q[:, 2].min() > 0.0, q[:, 2].min()
    _r, vmax_off, _q = _pressed_column(None, steps=2)
    assert vmax_off > 2.0, vmax_off


def test_relief_is_inactive_in_a_resting_column():
    """A primed resting column never reaches the cap (8-20 % at normal steps): bitwise the uncapped solver."""
    out = []
    for relief in (0.5, None):
        cfg = DFSPHConfig(particle_radius=2.0e-3, compression_relief_speed=relief)
        s_ = cfg.spacing
        b = newton.ModelBuilder()
        register_custom_attributes(b)
        b.begin_world()
        add_box_tank(b, (0.0, 0.0, 0.0), (6 * s_, 6 * s_, 14 * s_), wall=cfg.h)
        g = (np.arange(6) + 0.5) * s_
        add_fluid_points(b, cfg, np.array([(x, y, (k + 0.5) * s_) for k in range(8) for y in g for x in g]),
                         1000.0, 1.0e-3, (0.0, 0.0, 0.0))
        b.end_world()
        m = b.finalize(device="cpu")
        s = SolverDFSPH(m, cfg)
        a, c = m.state(), m.state()
        s.prime(a, 1.25e-3)
        for _ in range(20):
            s.step(a, c, None, None, 1.25e-3)
            a, c = c, a
        out.append(a.particle_q.numpy())
    assert np.array_equal(out[0], out[1])


def test_relief_speed_must_be_positive_or_none():
    DFSPHConfig(compression_relief_speed=None).validate()
    with pytest.raises(ValueError, match="compression_relief_speed"):
        DFSPHConfig(compression_relief_speed=0.0).validate()

