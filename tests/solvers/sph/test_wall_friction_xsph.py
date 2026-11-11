# SPDX-License-Identifier: Apache-2.0
"""SCR-OSS3-2 (coordinator 2026-11-06 09:32Z): lone-particle wall friction and the symmetric XSPH weight.

- A lone particle sliding on a floor is held off it by the wall-contact pressure (never projected); with
  ``wall_friction`` it slows at mu g and stops (before: it slid forever at its landing speed).
- XSPH with Monaghan's symmetric weight m_j / rho_bar_ij conserves momentum: a water block hitting a free box in zero
  gravity keeps the total momentum to float rounding (m_j / rho_j drifted 3.4e-3 of P0 after the impact)."""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.build import add_fluid_block, add_fluid_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH


def _slider(friction: bool):
    cfg = DFSPHConfig(particle_radius=2.0e-3, wall_friction=friction)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    floor = b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.005), wp.quat_identity()), hx=2.0, hy=2.0,
                            hz=0.005)
    add_fluid_points(b, cfg, np.array([[0.0, 0.0, 0.002]]), velocity=(2.6, 0.0, 0.0))
    b.end_world()
    m = b.finalize(device="cpu")
    return m, SolverDFSPH(m, cfg), float(m.shape_material_mu.numpy()[floor])


@pytest.mark.parametrize("friction", [True, False])
def test_lone_particle_on_a_floor_slows_at_mu_g(friction):
    m, sol, mu = _slider(friction)
    a, b = m.state(), m.state()
    dt = 1.25e-3
    vx = []
    for _ in range(320):
        sol.step(a, b, None, None, dt)
        a, b = b, a
        vx.append(float(a.particle_qd.numpy()[0, 0]))
    t = dt * np.arange(1, 321)
    if friction:
        expect = np.maximum(2.6 - mu * 9.81 * t, 0.0)
        late = t >= 0.05  # it falls the 2 mm onto the floor first; the landing's normal impulse makes up mu g t0
        assert np.abs(np.array(vx)[late] - expect[late]).max() < 0.01
        assert vx[-1] == 0.0
    else:
        assert np.allclose(vx, 2.6)


def test_xsph_conserves_momentum_through_an_impact():
    cfg = DFSPHConfig(particle_radius=1.0e-3, integrate_bodies="dynamic")
    s = cfg.spacing
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    liq = add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (6, 6, 6), velocity=(0.3, 0.0, 0.0))
    h = 3 * s
    body = b.add_body(xform=wp.transform(wp.vec3(8 * s + h, h, h), wp.quat_identity()))
    b.add_shape_box(body, hx=h, hy=h, hz=h, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0))
    b.end_world()
    m = b.finalize(device="cpu")
    sol = SolverDFSPH(m, cfg)
    a, bb = m.state(), m.state()
    mp = m.particle_mass.numpy().astype(np.float64)[liq.start:liq.stop, None]
    mb = float(m.body_mass.numpy()[0])

    def momentum(st):
        v = st.particle_qd.numpy().astype(np.float64)[liq.start:liq.stop]
        return (mp * v).sum(axis=0) + mb * st.body_qd.numpy()[0, :3].astype(np.float64)

    P0 = momentum(a)
    drift = 0.0
    for _ in range(200):  # 0.1 s: the block hits the box at ~0.013 s and spreads over it
        sol.step(a, bb, None, None, 5.0e-4)
        a, bb = bb, a
        drift = max(drift, float(np.linalg.norm(momentum(a) - P0) / np.linalg.norm(P0)))
    assert drift < 1.0e-6
    assert mb * float(a.body_qd.numpy()[0, 0]) > 0.3 * P0[0]  # the box took a real share
