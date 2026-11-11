# SPDX-License-Identifier: Apache-2.0
"""TOP-5: crumble conversion momentum (linear ≤ 1e-6, angular ≤ 2 %), plus sampling, parking and an MPM step.

A free rigid chunk with linear and angular velocity is crumbled into its spare MPM particles by a manual trigger.
Momentum is compared on the host in float64. Angular momentum is taken about the COM: the body's is R·I·Rᵀ·ω,
the particles' is Σ m (x − c) × v.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.crumble import box_hull, lattice_in_hull, sample_chunk
from alloy.topology.prefracture import prefracture
from alloy.topology.types import TopologyBudget

H = (0.03, 0.02, 0.01)
V = (0.3, -0.1, 0.2)
W = (2.0, -5.0, 3.0)


def _rotation(seed: int = 0) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _quat_xyzw(r: np.ndarray) -> tuple[float, float, float, float]:
    q = wp.quat_from_matrix(wp.mat33(*r.ravel().tolist()))
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _scene(device: str, shape: str, n_min: int = 8, with_mpm: bool = True):
    from alloy.topology.crumble import add_crumble_source

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverImplicitMPM.register_custom_attributes(b)
    rot = _rotation()
    body = b.add_body(xform=wp.transform(wp.vec3(0.1, 0.0, 0.3), wp.quat(*_quat_xyzw(rot))))
    if shape == "box":
        b.add_shape_box(body, hx=H[0], hy=H[1], hz=H[2], cfg=newton.ModelBuilder.ShapeConfig(density=800.0))
        hull = box_hull(*H)
    else:
        chunk = prefracture(newton.Mesh(box_hull(0.04, 0.03, 0.02).astype(np.float32),
                                        _box_faces()), n_chunks=6, seed=3)[2]
        b.add_shape_convex_hull(body, mesh=chunk, cfg=newton.ModelBuilder.ShapeConfig(density=800.0))
        hull = np.asarray(chunk.vertices, dtype=np.float64)
    rec = add_crumble_source(b, body, hull, n_min=n_min, trigger="manual", budget=TopologyBudget(spare_particles=20000))
    m = b.finalize(device=device)
    return m, body, rec


def _box_faces() -> np.ndarray:
    from scipy.spatial import ConvexHull

    return ConvexHull(box_hull(0.04, 0.03, 0.02)).simplices.astype(np.int32).reshape(-1)


def _set_motion(m: newton.Model, state: newton.State, body: int) -> None:
    qd = state.body_qd.numpy()
    qd[body] = (*V, *W)
    state.body_qd.assign(qd)


def _body_momentum(m: newton.Model, state: newton.State, body: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, qd = state.body_q.numpy()[body].astype(np.float64), state.body_qd.numpy()[body].astype(np.float64)
    r = np.array(wp.quat_to_matrix(wp.quat(*q[3:]))).reshape(3, 3)
    com = q[:3] + r @ m.body_com.numpy()[body]
    mass = float(m.body_mass.numpy()[body])
    i_w = r @ m.body_inertia.numpy()[body].astype(np.float64) @ r.T
    return mass * qd[:3], i_w @ qd[3:], com


def _particle_momentum(m: newton.Model, state: newton.State, ids: range, com: np.ndarray):
    x = state.particle_q.numpy()[ids].astype(np.float64)
    v = state.particle_qd.numpy()[ids].astype(np.float64)
    mass = m.particle_mass.numpy()[ids].astype(np.float64)
    return (mass[:, None] * v).sum(0), (mass[:, None] * np.cross(x - com, v)).sum(0)


@pytest.mark.parametrize("shape", ["box", "voronoi"])
def test_crumble_conserves_momentum(device, shape):
    m, body, rec = _scene(device, shape)
    state = m.state()
    _set_motion(m, state, body)
    p_body, l_body, com = _body_momentum(m, state, body)
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(spare_particles=20000), entries={})
    ctl.request_crumble([body])
    ctl.apply(state)
    assert ctl.step_events.numpy()[0] == 1
    p, l_rel = _particle_momentum(m, state, rec.particles, com)
    lin = np.linalg.norm(p - p_body) / np.linalg.norm(p_body)
    ang = np.linalg.norm(l_rel - l_body) / np.linalg.norm(l_body)
    print(f"TOP-5 {shape}: N={len(rec.particles)} spacing={rec.spacing:.4g} lin_err={lin:.3e} ang_err={ang:.3e}")
    assert lin <= 1e-6
    assert ang <= 0.02
    flags = m.particle_flags.numpy()[rec.particles]
    assert np.all(flags & int(newton.ParticleFlags.ACTIVE))
    led = ctl.ledger()
    assert abs(led.mass_after[0] / led.mass_before[0] - 1.0) <= 1e-6 and led.reset_required[0]


def test_raw_lattice_deficit_matches_theory():
    """Without moment matching a cube lattice of n per side holds I(1 − 1/n²) (the §3.8 figure: 2 % low at n = 7)."""
    g, a = lattice_in_hull(box_hull(0.01, 0.01, 0.01), n_min=7)
    r = g - g.mean(0)
    i_lat = (r[:, 1] ** 2 + r[:, 2] ** 2).sum() / len(r)
    i_cont = (2 * 0.02 ** 2 / 12.0)
    assert i_lat / i_cont == pytest.approx(1.0 - 1.0 / 49.0, rel=1e-9)
    r_m, _, eig = sample_chunk(box_hull(0.01, 0.01, 0.01), 1.0, np.eye(3) * (2 * 0.02 ** 2 / 12.0), n_min=7)
    assert ((r_m[:, 1] ** 2 + r_m[:, 2] ** 2).mean()) == pytest.approx(i_cont, rel=1e-12)
    assert 1.0 < eig.min() <= eig.max() < 1.02                      # √(1/(1 − 1/49)) = 1.0102


def test_crumble_parks_body_and_mpm_runs(device):
    m, body, rec = _scene(device, "box")
    mpm_cfg = newton.solvers.SolverImplicitMPM.Config()
    mpm_cfg.voxel_size = 2.0 * rec.spacing
    mpm = newton.solvers.SolverImplicitMPM(m, config=mpm_cfg)
    s0, s1 = m.state(), m.state()
    _set_motion(m, s0, body)
    x_before = s0.body_q.numpy()[body].copy()
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(spare_particles=20000), entries={"mpm": mpm})
    ctl.request_crumble([body])
    ctl.apply(s0)
    assert ctl.body_active.numpy()[body] == 0
    np.testing.assert_allclose(s0.body_q.numpy()[body][:3], x_before[:3] + np.array([100.0, 0.0, 0.0]), atol=1e-4)
    assert np.all(s0.body_qd.numpy()[body] == 0.0)
    assert bool(m.mujoco.equality_constraint_enabled.numpy()[rec.park_eq])
    assert np.all(mpm._mpm_model.particle_flags.numpy()[rec.particles] & int(newton.ParticleFlags.ACTIVE))
    for _ in range(3):
        mpm.step(s0, s1, None, None, 1.0e-3)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()[rec.particles]
    assert np.isfinite(q).all() and np.isfinite(s0.particle_qd.numpy()[rec.particles]).all()


def test_crumble_respects_event_cap_and_never_fails(device):
    m, body, _ = _scene(device, "box")
    state = m.state()
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(spare_particles=20000, max_events_per_step=0))
    ctl.request_crumble([body])
    ctl.apply(state)
    assert ctl.step_events.numpy()[0] == 0
    assert ctl.ledger().overflow[0] == 1
    assert ctl.body_active.numpy()[body] == 1
