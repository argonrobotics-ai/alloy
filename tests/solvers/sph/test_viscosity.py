# SPDX-License-Identifier: Apache-2.0
"""Weiler 2018 implicit viscosity (SPEC §3.8 viscosity "weiler2018", router rule mu > 1e-2 Pa s): decay of a shear
mode against nu k^2, exact conservation of linear and angular momentum under rigid motion, a stable viscous column."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.solvers.sph.build import add_fluid_block
from alloy.solvers.sph.config import register_custom_attributes
from alloy.solvers.sph.kernels import laplacian_lattice_factor
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.sph import default_config, run, water_column

MU, DT = 1.0, 1.25e-3  # honey-like 1 Pa s, nu = 1e-3 m^2/s


def _blob(counts: tuple[int, int, int], viscosity: str):
    cfg = default_config(viscosity=viscosity)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    add_fluid_block(b, cfg, (0.0, 0.0, 0.0), counts, viscosity=MU)
    b.end_world()
    m = b.finalize(device="cpu")
    return m, SolverDFSPH(m, cfg), cfg


def test_lattice_factor_cubic():
    assert laplacian_lattice_factor(2.0e-3, 4.0e-3, "cubic") == pytest.approx(0.698, abs=2e-3)


def test_auto_selects_weiler_above_threshold():
    sc = water_column(3, 3, 3)
    sc.model.sph.viscosity.fill_(0.5)
    assert SolverDFSPH(sc.model, default_config()).viscosity == "weiler2018"


def test_shear_mode_decays_at_nu_k2():
    """v_x = U sin(k z) in a zero-g blob (12 x 12 x 24 cells, kh = 0.52): the interior amplitude after 10 implicit
    steps follows (1 + nu k^2 dt)^-10 within 30 % in the exponent; XSPH alone barely damps it."""
    out = {}
    for visc in ("weiler2018", "xsph"):
        m, s, cfg = _blob((12, 12, 24), visc)
        q = m.particle_q.numpy()
        k = 2.0 * np.pi / (24 * cfg.spacing)
        v = np.zeros_like(q)
        v[:, 0] = 0.01 * np.sin(k * q[:, 2])
        a, b = m.state(), m.state()
        a.particle_qd.assign(v)
        hi = np.array([12, 12, 24]) * cfg.spacing
        inner = np.all((q > 2 * cfg.h) & (q < hi - 2 * cfg.h), axis=1)

        def amp(st):
            basis = np.sin(k * st.particle_q.numpy()[inner, 2])[:, None]
            return np.linalg.lstsq(basis, st.particle_qd.numpy()[inner, 0], rcond=None)[0][0]

        a0 = amp(a)
        a, b = run(s, a, b, 10, DT)
        out[visc] = amp(a) / a0
    nu = MU / 1000.0
    expect = -10.0 * np.log1p(nu * k * k * DT)
    assert 0.7 < np.log(out["weiler2018"]) / expect < 1.3
    assert out["xsph"] > 0.99


def test_rigid_motion_conserves_momentum():
    m, s, cfg = _blob((10, 10, 10), "weiler2018")
    q = m.particle_q.numpy()
    c = q.mean(axis=0)
    v = np.cross(np.array([0.0, 0.0, 2.0]), q - c) + np.array([0.01, 0.0, 0.0])
    a, b = m.state(), m.state()
    a.particle_qd.assign(v)
    mass = m.particle_mass.numpy()[:, None]
    p0, l0 = (mass * v).sum(0), (mass * np.cross(q - c, v)).sum(0)
    a, b = run(s, a, b, 10, DT)
    q1, v1 = a.particle_q.numpy(), a.particle_qd.numpy()
    p1, l1 = (mass * v1).sum(0), (mass * np.cross(q1 - q1.mean(0), v1)).sum(0)
    assert np.allclose(p1, p0, rtol=1e-4, atol=1e-10)
    assert l1[2] == pytest.approx(l0[2], rel=1e-4)


def test_viscous_column_rests():
    sc = water_column(6, 6, 16, cfg=default_config(viscosity="weiler2018"))
    sc.model.sph.viscosity.fill_(MU)
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    s.prime(a, DT)
    a, b = run(s, a, b, 100, DT)
    r = s.report()
    assert r.escaped[0] == 0 and r.max_speed[0] < 0.05
