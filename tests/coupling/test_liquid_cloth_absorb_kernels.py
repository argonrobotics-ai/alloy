# SPDX-License-Identifier: Apache-2.0
"""Cloth takes up liquid (opt-in), kernel laws with no simulation: alloy/solvers/sph/cloth_absorb.py,
alloy/coupling/liquid_cloth_absorb.py, driven on small synthetic grids (CPU).

- A4: wicking spreads a spot at 4 D dt;
- A6: the exchange merge keeps momentum and adds no kinetic energy;
- a particle is taken one step after it touches a face and hands its momentum to the face's vertices.
The simulation laws are in test_liquid_cloth_absorb_law.py.
"""

from __future__ import annotations

import numpy as np


def _grid(n: int, h: float):
    """A flat n x n vertex grid of spacing h, split into right triangles (the add_cloth pattern)."""
    xs = np.arange(n) * h
    x = np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    x = np.concatenate([x, np.zeros((len(x), 1))], 1).astype(np.float32)
    tri = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            tri += [(a, b, c), (a, c, d)]
    return x, np.asarray(tri, np.int32)


def test_wicking_spreads_a_spot_at_four_d_t():
    """A4 (the discretisation): water put on one vertex of a regular grid spreads with <r^2> growing by exactly
    4 D dt per exchange (the 2-D diffusion solution; on the 5-point stencil the second moment grows by 2 D dt per axis
    each explicit step), the total stays exact in fixed point and every vertex stays in [0, capacity]. Bound: 1 % of
    the growth, for fixed-point rounding; 20 exchanges at D dt / h^2 = 0.05, far from the border."""
    from types import SimpleNamespace as NS

    import warp as wp

    from alloy.solvers.sph.cloth_absorb import ClothAbsorb, W_SCALE_HOST

    n, h, D, dt = 21, 0.01, 1e-4, 0.05 * 0.01 ** 2 / 1e-4
    x, tri = _grid(n, h)
    V = len(x)
    soft = NS(vid=wp.array(np.arange(V, dtype=np.int32), dtype=wp.int32), count=len(tri), device=wp.get_device("cpu"),
              arrays=NS(tri=wp.array(tri, dtype=wp.vec3i)))
    model = NS(particle_count=V, particle_flags=wp.zeros(V, dtype=wp.int32), particle_world=wp.zeros(V, dtype=wp.int32),
               particle_q=wp.array(x, dtype=wp.vec3))
    cap = np.full(V, 1.0)                                    # 1 kg per vertex: never full here
    ca = ClothAbsorb(soft, np.ones(1, np.int32), cap, np.full(V, 1.0), 0.002, model, wp.zeros(V, dtype=wp.int8), 1,
                     1e-4, np.full(V, D))
    centre = (n // 2) * n + n // 2
    w = np.zeros(V, np.int64)
    w[centre] = int(1e-3 * W_SCALE_HOST)
    ca.water.assign(w)
    kg, qd = wp.full(V, -1, dtype=wp.int32), wp.zeros(V, dtype=wp.vec3)
    r = x[:, :2] - x[centre, :2]
    r2 = []
    for _ in range(20):
        ca.elapsed.assign(np.full(1, dt, np.float32))
        ca.wick(kg, qd)
        now = ca.water.numpy()
        assert now.sum() == w.sum() and now.min() >= 0
        r2.append(float((now * (r ** 2).sum(1)).sum() / now.sum()))
    growth = np.diff([0.0, *r2])
    np.testing.assert_allclose(growth, 4 * D * dt, rtol=0.01)


def test_the_exchange_merge_keeps_momentum_and_adds_no_energy():
    """A6 (the merge kernel): a vertex of mass m at velocity v that took up M with momentum P in the exchange ends at
    (m v + P) / (m + M): momentum exact (float32), and the kinetic energy is at most the parts' before (an inelastic
    merge). Bound: 1e-6 relative."""
    import warp as wp

    from alloy.coupling import liquid_cloth_absorb as lca

    N = 4
    rng = np.random.default_rng(0)
    m0 = rng.uniform(1e-4, 5e-4, N).astype(np.float32)
    v0 = rng.normal(0, 0.3, (N, 3)).astype(np.float32)
    dm = np.array([0.0, 2e-5, 6e-5, 1e-4])
    vj = rng.normal(0, 1.0, (N, 3))
    W, IMP = lca.W_SCALE_HOST, lca.IMP_SCALE_HOST
    f32 = lambda a: wp.array(np.asarray(a, np.float32), dtype=wp.float32)    # noqa: E731
    ids = wp.array(np.arange(N, dtype=np.int32), dtype=int)
    dry = lca.DryMasses()
    dry.m, dry.inv, dry.pinv = f32(m0), f32(1 / m0), f32(1 / m0)
    dry.sph, dry.sinv, dry.meff = f32(m0), f32(1 / m0), wp.array(m0, dtype=float)
    acc_m = wp.array(np.round(dm * W).astype(np.int64), dtype=wp.int64)
    acc_p = wp.array(np.round(dm[:, None] * vj * IMP).astype(np.int64), dtype=wp.int64)
    water = wp.array(np.round(dm * W).astype(np.int64), dtype=wp.int64)
    qd = wp.array(v0, dtype=wp.vec3)
    par_m, par_inv = f32(m0), f32(1 / m0)
    arrs = [f32(m0), f32(1 / m0), par_m, par_inv, f32(m0), f32(1 / m0), wp.array(m0, dtype=float)]
    wp.launch(lca.cloth_mass_kernel, dim=N, inputs=[wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32), water,
                                                    acc_m, acc_p, 1, dry, ids, ids, ids, qd, arrs[0], arrs[1],
                                                    par_m, par_inv, arrs[4], arrs[5], arrs[6]])
    v1 = qd.numpy().astype(np.float64)
    p_before = m0[:, None] * v0 + dm[:, None] * vj
    np.testing.assert_allclose((m0 + dm)[:, None] * v1, p_before, rtol=1e-6, atol=1e-12)
    ke_before = 0.5 * (m0 * (v0 ** 2).sum(1) + dm * (vj ** 2).sum(1))
    assert ((0.5 * (m0 + dm) * (v1 ** 2).sum(1)) <= ke_before * (1 + 1e-6)).all()
    assert (acc_m.numpy() == 0).all() and (acc_p.numpy() == 0).all()
    np.testing.assert_allclose(par_m.numpy(), m0 + dm, rtol=1e-6)


def _one_particle_over_a_face(height: float):
    """A synthetic soft wall (a 3 x 3 vertex grid, 2 cm cells, 1 mm half thickness) with one liquid particle of 64 mg
    moving at (0.1, 0, -0.2) m/s above the centroid of face 0, its first slot naming that face; its ClothAbsorb."""
    from types import SimpleNamespace as NS

    import warp as wp

    from alloy._vendor.newton import ParticleFlags
    from alloy.solvers.sph import pressure as pr
    from alloy.solvers.sph.boundary import NB
    from alloy.solvers.sph.cloth_absorb import ClothAbsorb
    from alloy.solvers.sph.soft_walls import SoftArrays

    x, tri = _grid(3, 0.02)
    V, P = len(x), len(x) + 1
    c = x[tri[0]].mean(0)
    q = np.concatenate([x, (c + np.array([0, 0, height], np.float32))[None]]).astype(np.float32)
    soft = SoftArrays()
    soft.base, soft.nface, soft.bvh, soft.half = 100, len(tri), wp.uint64(0), 0.001
    soft.tri = wp.array(tri, dtype=wp.vec3i)
    soft.face_comp = wp.zeros(len(tri), dtype=wp.int32)
    soft.face_world = wp.zeros(len(tri), dtype=wp.int32)
    soft.world_has = wp.ones(1, dtype=wp.int32)
    soft.two_sided = wp.ones(1, dtype=wp.int32)
    soft.x, soft.v = wp.array(x, dtype=wp.vec3), wp.zeros(V, dtype=wp.vec3)
    soft.adj = wp.full(len(tri), wp.vec3i(-1, -1, -1), dtype=wp.vec3i)
    soft.vstart = wp.zeros(V + 1, dtype=wp.int32)
    soft.vface = wp.zeros(1, dtype=wp.int32)
    soft.vopen = wp.zeros(V, dtype=wp.int32)
    flags = np.zeros(P, np.int32)
    flags[-1] = int(ParticleFlags.ACTIVE)
    phase = np.full(P, -1, np.int8)
    phase[-1] = 1
    p = pr.Particles()
    p.q = wp.array(q, dtype=wp.vec3)
    p.qd = wp.array(np.r_[np.zeros((V, 3)), [[0.1, 0.0, -0.2]]].astype(np.float32), dtype=wp.vec3)
    p.mass = wp.array(np.full(P, 6.4e-5, np.float32), dtype=wp.float32)
    p.rho0 = wp.array(np.full(P, 1000.0, np.float32), dtype=wp.float32)
    p.flags, p.phase = wp.array(flags, dtype=wp.int32), wp.array(phase, dtype=wp.int8)
    p.world = wp.zeros(P, dtype=wp.int32)
    p.nbr, p.nbr_hi, p.nbr_wide = wp.zeros((P, 1), dtype=wp.uint16), wp.zeros((1, 1), dtype=wp.uint8), 0
    p.wstart, p.count = wp.zeros(1, dtype=wp.int32), wp.array([P], dtype=wp.int32)
    shape, face = np.full((P, NB), -1, np.int32), np.full((P, NB), -1, np.int32)
    shape[-1, 0], face[-1, 0] = 100, 0
    slots = NS(shape=wp.array(shape, dtype=wp.int32), face=wp.array(face, dtype=wp.int32))
    sw = NS(vid=wp.array(np.arange(V, dtype=np.int32), dtype=wp.int32), count=len(tri), device=wp.get_device("cpu"),
            arrays=soft)
    model = NS(particle_count=P, particle_flags=p.flags, particle_world=p.world, particle_q=p.q)
    ca = ClothAbsorb(sw, np.ones(1, np.int32), np.full(V, 1.0), np.full(V, 1e3), 0.002, model, p.phase, 1, 6.4e-5)
    return ca, p, slots, P


def _step_with_face0_open(ca, p, slots, P, k: int) -> None:
    """One uptake step in which face 0's colour takes water (the test sets the per-world step count)."""
    ca.tick.assign(np.array([ca.ncolor * k - 1], np.int32))
    ca.step(p, slots, P, 1e-3)


def test_uptake_waits_one_step_in_contact_and_hands_over_the_momentum():
    """The one-step gate and the uptake's books, exact, on a synthetic wall: a particle 0.5 mm above a face (within
    reach) is not taken in its first step in contact and is taken in the second. Its whole mass goes to the face's
    vertices (water and M, fixed point) and its momentum m v to P (fixed point); it is retired: ACTIVE cleared, phase
    -4, velocity 0. A particle out of reach (5 mm) is never taken."""
    from alloy.solvers.sph.bodies import IMP_SCALE_HOST
    from alloy.solvers.sph.cloth_absorb import ABSORBED, W_SCALE_HOST

    ca, p, slots, P = _one_particle_over_a_face(0.0015)
    _step_with_face0_open(ca, p, slots, P, 1)
    assert p.phase.numpy()[-1] == 1 and int(ca.touch.numpy()[-1]) == 1 and ca.water.numpy().sum() == 0
    _step_with_face0_open(ca, p, slots, P, 2)
    assert p.phase.numpy()[-1] == ABSORBED and (p.flags.numpy()[-1] & 1) == 0
    assert np.all(p.qd.numpy()[-1] == 0.0)
    m = np.float64(np.float32(6.4e-5))
    assert ca.water.numpy().sum() == ca.acc_m.numpy().sum() == int(np.round(m * W_SCALE_HOST))
    np.testing.assert_allclose(ca.acc_p.numpy().sum(0) / IMP_SCALE_HOST, m * np.array([0.1, 0.0, -0.2]),
                               rtol=1e-6, atol=1e-15)
    assert int(ca.soaked_count.numpy()[0]) == 1

    far, pf, sf, Pf = _one_particle_over_a_face(0.006)
    for k in (1, 2, 3):
        _step_with_face0_open(far, pf, sf, Pf, k)
    assert pf.phase.numpy()[-1] == 1 and far.water.numpy().sum() == 0 and int(far.touch.numpy()[-1]) == 0
