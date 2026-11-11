# SPDX-License-Identifier: Apache-2.0
"""Engine item 3 (SPH-VBD coupler P1): VBD surfaces as SolverDFSPH boundaries (soft_walls.py, coupling_hooks.py).

- the soft faces of a tet cube: one closed component, wound outward, every edge shared;
- their signed distance equals the cube's exact SDF, including points on a face's plane beyond an edge, where one
  face's normal gives the sign by rounding (the angle-weighted pseudo-normals fix it);
- the wall path over one staggered exchange runs from end - v dt to the end pose; lagged pairs are refused;
- the harvest returns each vertex's exchange impulse / dt;
- a fixed submerged cube as soft faces matches the same cube as a box shape, with no wall projections;
- the coupled VBD <-> SPH Scheduler runs (finite, upward fluid force on the cube)."""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp
from alloy._vendor.newton import StateFlags

from alloy.solvers.sph.build import add_fluid_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.soft_scenes import add_soft_cube, cube_pool, elastic_energy, soft_pipeline, vbd_sph_scheduler
from alloy.solvers.sph.soft_walls import SoftArrays, SoftWalls, soft_candidates, soft_topology
from alloy.solvers.sph.solver import SolverDFSPH

S = 2.0e-3  # lattice spacing at r = 1 mm


def _cube_model(edge: float = 6 * S):
    b = newton.ModelBuilder()
    ids = add_soft_cube(b, (0.0, 0.0, 0.0), edge, 3, 1000.0)
    return b.finalize(device="cpu"), ids


def test_cube_surface_is_one_closed_outward_component():
    m, ids = _cube_model()
    top = soft_topology(m, ids)
    tri, x = top["tri"], m.particle_q.numpy()[top["vid"]]
    assert len(tri) == 108 and top["comp"].max() == 0 and top["two_sided"][0] == 0
    a, b, c = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    n = np.cross(b - a, c - a)
    assert (np.einsum("ij,ij->i", n, (a + b + c) / 3.0 - 3 * S) > 0.0).all()  # every face points away from the centre
    assert (top["adj"] >= 0).all()


def test_elastic_energy_zero_at_rest_and_matches_small_stretch():
    m, _ids = _cube_model()
    q = m.particle_q.numpy().astype(np.float64)
    assert abs(elastic_energy(m, q)) < 1.0e-12
    e = 1.0e-3  # uniaxial stretch along x: psi ~ (lam + 2 mu) e^2 / 2 for small e (linear limit of the energy)
    mat = m.tet_materials.numpy()[0]
    E_lin = 0.5 * (mat[1] + 2.0 * mat[0]) * e**2 * (6 * S) ** 3
    assert elastic_energy(m, q * np.array([1.0 + e, 1.0, 1.0])) == pytest.approx(E_lin, rel=0.02)


@wp.kernel
def _phi_kernel(soft: SoftArrays, x: wp.array[wp.vec3], phi: wp.array[float]):
    i = wp.tid()
    cb, _fb, pb = soft_candidates(soft, x[i], 0)
    phi[i] = 1.0e9
    if cb[0] >= 0:
        phi[i] = pb[0]


def test_signed_distance_matches_cube_sdf_on_edge_planes():
    m, ids = _cube_model()
    sw = SoftWalls(m, 1.5 * S, "cpu", soft_particles=ids)
    L = 6 * S
    g = np.arange(-2, 15) * 0.5 * S  # half-spacing grid: points on the faces' planes beyond every edge and corner
    pts = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    rng = np.random.default_rng(0)
    pts = np.concatenate([pts, rng.uniform(-1.4 * S, L + 1.4 * S, size=(4000, 3))])
    phi = wp.zeros(len(pts), dtype=float, device="cpu")
    wp.launch(_phi_kernel, dim=len(pts), inputs=[sw.arrays, wp.array(pts, dtype=wp.vec3, device="cpu"), phi],
              device="cpu")
    d = np.abs(pts - L / 2) - L / 2
    exact = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
    near = np.abs(exact) < 1.4 * S
    got = phi.numpy()[near]
    assert (got < 1.0e8).all()
    np.testing.assert_allclose(got, exact[near], atol=1.0e-6 * S)


def _wall_rig():
    """One soft cube moving at v (no liquid near it) and one liquid particle far away (the solver needs liquid)."""
    cfg = DFSPHConfig(particle_radius=1.0e-3)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    add_fluid_points(b, cfg, np.array([[0.2, 0.2, 0.2]]))
    ids = add_soft_cube(b, (0.0, 0.0, 0.0), 6 * S, 3, 1000.0)
    b.end_world()
    m = b.finalize(device="cpu")
    return m, cfg, ids


def test_wall_path_over_one_staggered_exchange():
    """Staggered sync = end pose and end velocity: the wall runs from end - v dt_x and lands on the end pose (P1's
    prescribed wall; P2's shell inertia off)."""
    m, cfg, ids = _wall_rig()
    sol = SolverDFSPH(m, cfg, soft_particles=ids, soft_inertia=False)
    v = np.array([0.0, 0.0, 0.1])
    s0, s1 = m.state(), m.state()
    qd = s0.particle_qd.numpy()
    qd[ids] = v
    s0.particle_qd.assign(qd)
    x_sync = s0.particle_q.numpy()[ids].copy()
    dtx = 2.0e-3
    sol.coupling_notify_input_state_update(s0, int(StateFlags.PARTICLE_Q | StateFlags.PARTICLE_QD), dt=dtx)
    sol.step(s0, s1, None, None, dtx / 2)
    start = sol.soft.arrays.x.numpy().copy()  # the wall at the first substep's start (numpy() is a view on CPU)
    sol.step(s1, s0, None, None, dtx / 2)
    vid = sol.soft.vid.numpy()  # the surface vertices (the cube's interior ones are not walls)
    sync_at = {int(p): k for k, p in enumerate(ids)}
    xs = x_sync[[sync_at[int(p)] for p in vid]]
    np.testing.assert_allclose(start, xs - v * dtx, atol=1e-9)
    np.testing.assert_allclose(s0.particle_q.numpy()[vid], xs, atol=1e-9)


def test_lagged_soft_walls_are_refused():
    m, cfg, ids = _wall_rig()
    sol = SolverDFSPH(m, cfg, soft_particles=ids)
    with pytest.raises(ValueError, match="staggered"):
        sol.coupling_rewind_proxy_particle(wp.zeros(3, dtype=int, device="cpu"), m.state(),
                                           wp.zeros(3, dtype=wp.vec3, device="cpu"),
                                           wp.zeros(1, dtype=wp.vec3, device="cpu"), 2.0e-3)


def test_harvest_returns_exchange_impulse_over_dt():
    m, cfg, ids = _wall_rig()
    sol = SolverDFSPH(m, cfg, soft_particles=ids)
    V = sol.soft.vid.shape[0]
    j = np.zeros((V, 3), np.int64)
    j[3] = (1 << 50, -(1 << 49), 0)  # 1 and -0.5 N s in fixed point
    sol.soft.exch.assign(j)
    l2g = np.full(m.particle_count, -1, np.int64)
    l2g[sol.soft.vid.numpy()] = np.arange(V)
    out = wp.zeros(V, dtype=wp.vec3, device="cpu")
    sol.coupling_harvest_proxy_particle_forces(wp.array(l2g, dtype=int, device="cpu"), out, dt=0.5, particle_qd_before=None,
                                               state=None, state_out=None, contacts=None)
    f = out.numpy()
    np.testing.assert_allclose(f[3], [2.0, -1.0, 0.0])
    assert np.abs(np.delete(f, 3, axis=0)).max() == 0.0


def _fz_series(mode: str, steps: int):
    pool = cube_pool(mode, device="cpu")
    sol = SolverDFSPH(pool.model, pool.config, soft_particles=pool.soft or None)
    a, b = pool.model.state(), pool.model.state()
    sol.prime(a, 1.0e-3)
    fz, proj = [], 0
    for _ in range(steps):
        if mode == "soft":
            sol.soft.new_window()
        else:
            sol.reset_shape_meters()
        sol.stats.projected.zero_()
        sol.step(a, b, None, None, 1.0e-3)
        a, b = b, a
        f = sol.soft.vertex_impulse().sum(axis=0) if mode == "soft" else sol.shape_impulse()[pool.shape]
        fz.append(f[2] / 1.0e-3)
        proj += int(sol.stats.projected.numpy()[0])
    return np.array(fz), proj, 1000.0 * 9.81 * pool.edge**3


@pytest.mark.compile_heavy
def test_submerged_soft_cube_matches_rigid_twin():
    soft, proj, ref = _fz_series("soft", 100)
    rigid, _p, _r = _fz_series("rigid", 100)
    assert proj == 0  # the edge-sign bug pulled particles into the cube from step ~60
    assert abs(soft[10:40].mean() - rigid[10:40].mean()) < 0.005 * ref  # the runs agree before chaos separates them
    assert abs(soft[40:].mean() / ref - 1.0) < 0.25 and abs(rigid[40:].mean() / ref - 1.0) < 0.25


@pytest.mark.compile_heavy
def test_coupled_vbd_sph_floating_cube_runs():
    draft = 0.6 * 6
    pool = cube_pool("soft", depth=12, cube_at=(5, 5, 12 - draft), rho_s=600.0, device="cpu")
    sched = vbd_sph_scheduler(pool, added_mass=1000.0 * draft * S * (6 * S) ** 2)
    st, nx = pool.model.state(), pool.model.state()
    pipe = soft_pipeline(pool.model)
    con = pipe.contacts()
    for _ in range(10):
        st.clear_forces()
        pipe.collide(st, con)
        sched.step(st, nx, None, con, 2.0e-3)
        st, nx = nx, st
    q = st.particle_q.numpy()
    assert np.isfinite(q).all()
    fz = sched._proxy_particle_mappings[0].coupling_forces.numpy()[:, 2].sum()
    weight = float(pool.model.particle_mass.numpy()[pool.soft].sum()) * 9.81
    assert 0.3 * weight < fz < 3.0 * weight
    assert sched._entries["sph"].solver.soft.count == 108


@pytest.mark.gpu
def test_soft_walls_on_cuda():
    """The grouped BVH (LBVH on CUDA), the closest-face kernels and the coupled run on a GPU."""
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    b = newton.ModelBuilder()
    ids = add_soft_cube(b, (0.0, 0.0, 0.0), 6 * S, 3, 1000.0)
    m = b.finalize(device="cuda:0")
    sw = SoftWalls(m, 1.5 * S, "cuda:0", soft_particles=ids)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.4 * S, 7.4 * S, size=(4000, 3))
    phi = wp.zeros(len(pts), dtype=float, device="cuda:0")
    wp.launch(_phi_kernel, dim=len(pts), inputs=[sw.arrays, wp.array(pts, dtype=wp.vec3, device="cuda:0"), phi],
              device="cuda:0")
    d = np.abs(pts - 3 * S) - 3 * S
    exact = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
    near = np.abs(exact) < 1.4 * S
    np.testing.assert_allclose(phi.numpy()[near], exact[near], atol=1.0e-6 * S)
    pool = cube_pool("soft", depth=12, cube_at=(5, 5, 12 - 3.6), rho_s=600.0, device="cuda:0")
    sched = vbd_sph_scheduler(pool, added_mass=1000.0 * 3.6 * S * (6 * S) ** 2)
    st, nx = pool.model.state(), pool.model.state()
    pipe = soft_pipeline(pool.model)
    con = pipe.contacts()
    for _ in range(10):
        st.clear_forces()
        pipe.collide(st, con)
        sched.step(st, nx, None, con, 2.0e-3)
        st, nx = nx, st
    assert np.isfinite(st.particle_q.numpy()).all()
    fz = sched._proxy_particle_mappings[0].coupling_forces.numpy()[:, 2].sum()
    weight = float(pool.model.particle_mass.numpy()[pool.soft].sum()) * 9.81
    assert 0.3 * weight < fz < 3.0 * weight
