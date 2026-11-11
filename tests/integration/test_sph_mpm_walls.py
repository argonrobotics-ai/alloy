# SPDX-License-Identifier: Apache-2.0
"""SPH particle walls (``alloy.solvers.sph.particle_walls``, SCR-WP12-MIX): another entry's particles as a liquid's
boundary. The local plane of a lattice slab, the colour inversion, the reaction split that conserves momentum
exactly (in sum and per particle), and one standalone SolverDFSPH run over hand-flagged PROXY particles (the whole
chain: wall set, shared grid, slots, accumulate, harvest). CPU, no coupling."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import ParticleFlags
from alloy.solvers.sph import particle_walls as P
from alloy.solvers.sph.bodies import IMP_SCALE_HOST
from alloy.solvers.sph.build import add_fluid_points, lattice_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.kernels import halfspace_volume, kernel_w_np
from alloy.solvers.sph.solver import SolverDFSPH

H, KIND, A = 0.008, 0, 0.0025        # SPH support at r = 2 mm, cubic kernel, MPM spacing at a 5 mm voxel
DEV = "cpu"


def _walls(pts: np.ndarray, world: np.ndarray | None = None):
    """ParticleWallArrays over ``pts`` (all walls, volume a^3 each), in a grid grouped as SolverDFSPH's (walls in
    group -2 - world); the grid must outlive them."""
    pw = P.empty_walls(0, H, KIND, DEV)
    pw.count = len(pts)
    pw.x = wp.array(pts, dtype=wp.vec3, device=DEV)
    pw.v = wp.array(np.tile([0.1, 0.0, -0.2], (len(pts), 1)), dtype=wp.vec3, device=DEV)
    pw.vol = wp.full(len(pts), A**3, dtype=wp.float32, device=DEV)
    w = np.zeros(len(pts), np.int32) if world is None else world
    grid = wp.HashGrid(64, 64, 64, device=DEV)
    grid.reserve(len(pts), with_groups=True)
    grid.build(pw.x, H, groups=wp.array(-2 - w, dtype=wp.int32, device=DEV))
    pw.grid = grid.id
    return pw, grid


def _slab(layers: int = 8) -> np.ndarray:
    """A lattice slab whose face is the plane z = 0 (half a spacing above the top particle centres)."""
    g = np.arange(-12, 13) * A
    x, y, z = np.meshgrid(g, g, -np.arange(layers) * A - A / 2, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], 1)


@wp.kernel
def _plane_kernel(pw: P.ParticleWallArrays, xs: wp.array[wp.vec3], w: int, out: wp.array[wp.vec4],
                  vb: wp.array[wp.vec3]):
    i = wp.tid()
    phi, n, _xb, v = P.pw_plane(pw, xs[i], w)
    out[i] = wp.vec4(phi, n[0], n[1], n[2])
    vb[i] = v


def _plane(pw, xs: np.ndarray, world: int = 0):
    out = wp.zeros(len(xs), dtype=wp.vec4, device=DEV)
    vb = wp.zeros(len(xs), dtype=wp.vec3, device=DEV)
    wp.launch(_plane_kernel, dim=len(xs), inputs=[pw, wp.array(xs, dtype=wp.vec3, device=DEV), world, out, vb],
              device=DEV)
    return out.numpy(), vb.numpy()


@wp.kernel
def _invert_kernel(d: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = P.colour_distance(halfspace_volume(d[i], H, 0.0, KIND)[0], H, KIND)


def test_the_colour_inverts_to_the_half_space_distance() -> None:
    d = np.linspace(-0.006, 0.006, 25)            # the slot reach; Psi is flat in float32 near +-h
    out = wp.zeros(len(d), dtype=float, device=DEV)
    wp.launch(_invert_kernel, dim=len(d), inputs=[wp.array(d, dtype=float, device=DEV), out], device=DEV)
    np.testing.assert_allclose(out.numpy(), d, atol=2e-6)


def test_a_slab_reads_as_a_plane_at_its_face() -> None:
    """Within 4 mm of the face phi is within 0.3 mm of the true distance (the lattice sum falls short of the
    half-space integral away from the face: 0.977 at 4 mm deep); the normal points out of the slab, the wall
    velocity is the particles' own."""
    pw, _grid = _walls(_slab())
    z = np.linspace(-0.004, 0.004, 17)
    out, vb = _plane(pw, np.stack([np.full_like(z, 3e-4), np.zeros_like(z), z], 1))
    assert np.abs(out[:, 0] - z).max() < 3.0e-4
    np.testing.assert_allclose(out[:, 3], 1.0, atol=2e-3)
    np.testing.assert_allclose(vb, np.tile([0.1, 0.0, -0.2], (len(z), 1)), atol=1e-6)


def test_another_worlds_particles_are_not_a_wall() -> None:
    pts = _slab()
    pw, _grid = _walls(pts, world=np.ones(len(pts), np.int32))
    out, _vb = _plane(pw, np.array([[0.0, 0.0, 0.001]]), world=0)
    assert out[0, 0] >= 0.99 * H                 # nothing covers the support: no wall within reach


@pytest.mark.parametrize("n_fluid", [1, 7])
def test_the_reaction_split_conserves_the_impulse_exactly(n_fluid: int) -> None:
    """Each slot's impulse goes to the wall particles by V_j W_ij / c_i in int64: their sum is the slot's impulse to
    the fixed-point resolution, per component, and each particle's share is its weight's (float64 mirror)."""
    pts = _slab()
    pw, _grid = _walls(pts)
    rng = np.random.default_rng(0)
    q = np.c_[rng.uniform(-0.01, 0.01, (n_fluid, 2)), rng.uniform(0.0005, 0.004, n_fluid)]
    J = np.zeros((n_fluid, 4, 3))
    J[:, 1] = rng.normal(0.0, 1e-4, (n_fluid, 3))
    shape = np.full((n_fluid, 4), -1, np.int32)
    shape[:, 1] = 0                               # pw.base
    exch = wp.zeros((pw.count, 3), dtype=wp.int64, device=DEV)
    work = wp.zeros(1, dtype=wp.int64, device=DEV)
    wp.launch(P._accumulate_kernel, dim=(n_fluid, 4), device=DEV,
              inputs=[wp.array(q, dtype=wp.vec3, device=DEV), wp.zeros(n_fluid, dtype=wp.int32, device=DEV),
                      wp.array(shape, dtype=wp.int32, device=DEV), wp.zeros((n_fluid, 4), dtype=wp.vec3, device=DEV),
                      wp.array(J, dtype=wp.vec3, device=DEV), pw, wp.array(np.arange(pw.count, dtype=np.int32),
                      dtype=wp.int32, device=DEV), 0, exch, work])
    want = np.round(J[:, 1].astype(np.float32).astype(np.float64) * IMP_SCALE_HOST).astype(np.int64).sum(axis=0)
    np.testing.assert_array_equal(exch.numpy().sum(axis=0), want)       # exact, in fixed point
    w = A**3 * kernel_w_np(np.linalg.norm(q[:, None, :] - pts[None, :, :], axis=2), H)      # (fluid, wall)
    share = np.einsum("ij,id->jd", w / w.sum(axis=1, keepdims=True), J[:, 1])
    np.testing.assert_allclose(exch.numpy() / IMP_SCALE_HOST, share, rtol=0.0, atol=1e-5 * np.abs(J).max())


def _wall_rig():
    """A 10 x 10 x 3 lattice of PROXY particles (spacing A, face z = 0) under a 6 x 6 x 3 liquid block at r = 1 mm,
    standalone SolverDFSPH: the PROXY particles are its particle walls."""
    cfg = DFSPHConfig(particle_radius=1.0e-3)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    g = (np.arange(10) - 4.5) * A
    slab = np.stack(np.meshgrid(g, g, -(np.arange(3) + 0.5) * A, indexing="ij"), -1).reshape(-1, 3)
    for p in slab:
        b.add_particle(tuple(p), (0.0, 0.0, 0.0), 1.0e-3, radius=A / 2,
                       flags=int(ParticleFlags.ACTIVE | ParticleFlags.PROXY))
    liquid = add_fluid_points(b, cfg, lattice_points((-0.006, -0.006, 0.0), (6, 6, 3), 2.0e-3))
    b.end_world()
    return b.finalize(device=DEV), cfg, slab, np.arange(liquid.start, liquid.stop)


def test_a_liquid_rests_on_particle_walls_and_their_force_is_the_liquids_momentum_change() -> None:
    """20 steps of 1 ms: every PROXY particle is a wall, the liquid stays above the face, the walls stay put, and the
    liquid's momentum changes by its weight's impulse minus the walls' impulse (to float32: the interface creates or
    loses nothing); the harvest gives the walls' impulse / dt on the wall particles and nothing on the liquid."""
    model, cfg, slab, ids = _wall_rig()
    sol = SolverDFSPH(model, cfg)
    assert sol.pw.count == len(slab)
    s0, s1 = model.state(), model.state()
    mass = model.particle_mass.numpy()[ids].astype(np.float64)
    p0 = (mass[:, None] * s0.particle_qd.numpy()[ids]).sum(axis=0)
    for _ in range(20):
        sol.step(s0, s1, None, None, 1.0e-3)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()
    assert q[ids, 2].min() > 0.0 and np.abs(q[: len(slab)] - slab).max() < 1e-8
    J = sol.pw.particle_impulse().sum(axis=0)
    assert J[2] < -0.5 * mass.sum() * 9.81 * 0.02                      # the walls carry the liquid as it lands
    dp = (mass[:, None] * s0.particle_qd.numpy()[ids]).sum(axis=0) - p0
    np.testing.assert_allclose(dp, mass.sum() * np.array([0.0, 0.0, -9.81]) * 0.02 - J, rtol=0.0,
                               atol=1e-5 * abs(J[2]))
    out = wp.zeros(model.particle_count, dtype=wp.vec3, device=DEV)
    sol.coupling_harvest_proxy_particle_forces(wp.array(np.arange(model.particle_count, dtype=np.int64),
                                                        dtype=int, device=DEV), out, dt=0.02)
    f = out.numpy().astype(np.float64)
    np.testing.assert_allclose(f[: len(slab)], sol.pw.particle_impulse() / 0.02, rtol=1e-6, atol=1e-12)
    assert not f[ids].any()
    assert sol.pw.world_work().shape == (1,)


def test_a_wall_particle_in_no_world_is_refused() -> None:
    model, cfg, slab, _ids = _wall_rig()
    world = model.particle_world.numpy()
    world[: len(slab)] = -1
    model.particle_world.assign(world)
    with pytest.raises(ValueError, match="in no world"):
        SolverDFSPH(model, cfg)
