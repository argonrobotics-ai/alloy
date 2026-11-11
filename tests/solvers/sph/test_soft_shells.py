# SPDX-License-Identifier: Apache-2.0
"""Engine item 3 (SPH-VBD coupler P2): shell inertia inside SolverDFSPH (soft_inertia.py) and the light-shell laws
(soft_shells.py).

- the CG operator with the shell's interlinked term stays symmetric (y . A x = x . A y on random vectors), and the
  term is there (A changes);
- the proxy force array: the harvest writes the coupler's array, the same object every pass, and F_used is read from
  it; a different array raises;
- a standalone soft run (no coupler) keeps P1's prescribed walls; closed tet bodies keep them in a coupled run too
  (bitwise with the switch on or off);
- the cloth side test: a point that crossed a two-sided face within the step goes back to its start side; at the
  cloth's free border it is left alone; a closed face projects as in P1;
- L6: the dry twin oscillates at sqrt(k / (m + m_a)) (the rig itself); the wet sheet at m_a / m = 10 does too, with
  P2's interlinked solve, in few passes;
- on CUDA (gpu): the interlinked CG product stays symmetric and the wet L6 sheet runs (int64 scatters, device pinned).
"""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph import soft_shells as sh
from alloy.solvers.sph.build import add_fluid_points, lattice_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.krylov import SolveSpec
from alloy.solvers.sph.pressure import MODE_DENSITY, MODE_DIVERGENCE
from alloy.solvers.sph.soft_inertia import SoftInertia
from alloy.solvers.sph.soft_inertia import soft_project
from alloy.solvers.sph.soft_scenes import cube_pool, soft_pipeline, vbd_sph_scheduler
from alloy.solvers.sph.soft_walls import SoftArrays, SoftWalls
from alloy.solvers.sph.solver import SolverDFSPH

S = 2.0e-3  # lattice spacing at r = 1 mm


def _sheet_rig(sigma: float = 0.2):
    """A free 4 x 4 cloth sheet under a 6 x 6 x 3 liquid block (standalone SolverDFSPH, soft_particles = sheet)."""
    cfg = DFSPHConfig(particle_radius=1.0e-3)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4,
                     dim_y=4, cell_x=3 * S, cell_y=3 * S, mass=sigma * 9 * S * S, tri_ke=1.0, tri_ka=1.0, tri_kd=0.0)
    sheet = list(range(b.particle_count))
    add_fluid_points(b, cfg, lattice_points((0.0, 0.0, 0.0), (6, 6, 3), S))
    b.end_world()
    return b.finalize(device="cpu"), cfg, sheet


def _operator(sol: SolverDFSPH, state, mode: int, vec: np.ndarray, dt: float = 1.0e-3) -> np.ndarray:
    """out = A vec from the solver's CG product (inertia term included when active)."""
    cg = sol._cg
    n = sol.layout.n
    sol._ctl.converged.zero_()
    cg.restart.zero_()
    v = wp.array(vec.astype(np.float32), dtype=wp.float32, device="cpu")
    out = wp.zeros(n, dtype=wp.float32, device="cpu")
    spec = SolveSpec(mode, dt, 1.0e-3, 1.0e-2, 1, 10, 0.0, sol._tol_one)
    cg._product(sol._particles(state), sol._slot_struct(), sol._h, sol._kind, spec, v, out, 0, 0, 0)
    return out.numpy().astype(np.float64)


@pytest.mark.parametrize("mode", [MODE_DIVERGENCE, MODE_DENSITY])
def test_interlinked_operator_is_symmetric(mode):
    m, cfg, sheet = _sheet_rig()
    sol = SolverDFSPH(m, cfg, soft_particles=sheet)
    assert sol.inertia is not None and not sol.inertia.active  # standalone: prescribed until a coupler syncs
    s0, s1 = m.state(), m.state()
    sol.step(s0, s1, None, None, 1.0e-3)  # neighbours and boundary slots of this configuration
    rng = np.random.default_rng(7)
    fluid = np.zeros(sol.layout.n, bool)
    fluid[len(sheet):] = True
    x, y = (np.where(fluid, rng.standard_normal(sol.layout.n), 0.0) for _ in range(2))
    ax0 = _operator(sol, s1, mode, x)
    sol.inertia.pass_start(sol.model, 1.0e-3)
    ax, ay = _operator(sol, s1, mode, x), _operator(sol, s1, mode, y)
    assert np.abs(ax - ax0).max() > 0.1 * np.abs(ax0).max()  # the shell term is there (a 0.2 kg/m^2 sheet)
    assert abs(y @ ax - x @ ay) < 1.0e-5 * np.sqrt((y @ ay) * (x @ ax))
    assert x @ ax > x @ ax0 > 0.0  # positive semidefinite part added


def test_feedback_array_must_stay_the_same():
    m, cfg, sheet = _sheet_rig()
    sol = SolverDFSPH(m, cfg, soft_particles=sheet)
    inert: SoftInertia = sol.inertia
    l2g = wp.zeros(m.particle_count, dtype=int, device="cpu")
    f1, f2 = (wp.zeros(m.particle_count, dtype=wp.vec3, device="cpu") for _ in range(2))
    inert.capture_feedback(l2g, f1)
    inert.capture_feedback(l2g, f1)
    with pytest.raises(RuntimeError, match="different proxy force array"):
        inert.capture_feedback(l2g, f2)


def test_standalone_soft_walls_stay_prescribed():
    """No coupler sync: the soft vertices keep their own velocity through the step (P1's prescribed wall)."""
    m, cfg, sheet = _sheet_rig()
    sol = SolverDFSPH(m, cfg, soft_particles=sheet)
    s0, s1 = m.state(), m.state()
    qd = s0.particle_qd.numpy()
    qd[sheet] = (0.0, 0.0, -0.05)
    s0.particle_qd.assign(qd)
    sol.step(s0, s1, None, None, 1.0e-3)
    np.testing.assert_allclose(s1.particle_qd.numpy()[sheet], qd[sheet])
    assert not sol.inertia.active


def test_l6_dry_twin_is_the_reference_oscillator():
    rig = sh.spring_plate(10, wet=False, device="cpu")
    T = 1.5 * 2.0 * np.pi / rig.info["omega_ref"]
    d = sh.run_rig(rig, T=T)
    om, peak = sh.oscillation(d["t"], d["dz"] + rig.info["amp_ref"], rig.info["omega_ref"])
    assert om == pytest.approx(rig.info["omega_ref"], rel=0.01)
    assert peak == pytest.approx(rig.info["amp_ref"], rel=0.02)


@pytest.mark.compile_heavy
def test_coupled_harvest_writes_the_array_vbd_reads():
    """Newton 4f3b41ef (solver_coupled_proxy.py:1423-1432, 1582-1590): VBD reads proxy.coupling_forces as particle_f
    and the harvest writes into that same array every pass; F_used is taken from it."""
    rig = sh.spring_plate(10, wet=True, n=6, L=4, device="cpu")
    sched = sh._stepper(rig, True, True, rig.info["m_added"], 0.0, 20)
    sol = sched._entries["sph"].solver
    coupler = sched._proxy_particle_mappings[0].coupling_forces
    seen = []
    harvest = sol.coupling_harvest_proxy_particle_forces

    def spy(l2g, out, **kw):
        seen.append(out.ptr)
        harvest(l2g, out, **kw)

    sol.coupling_harvest_proxy_particle_forces = spy
    st, nx = rig.model.state(), rig.model.state()
    for _ in range(3):
        st.clear_forces()
        st.particle_f.assign(rig.external(st.particle_q.numpy()))
        sched.step(st, nx, None, None, 1.0e-3)
        st, nx = nx, st
    assert len(seen) >= 3 and set(seen) == {coupler.ptr}
    assert sol.inertia.feedback[1].ptr == coupler.ptr
    f_used = sol.inertia.f_used.numpy().sum(axis=0)
    assert f_used[2] < 0.0  # the liquid pushes the sheet down in the last pass's force


@pytest.mark.compile_heavy
def test_l6_wet_sheet_oscillates_with_its_added_mass():
    out = sh.l6_added_mass(10, periods=1.5, dry=False, n=6, L=6, device="cpu")
    assert out["finite"] and out["below"] == 0
    assert out["omega_ratio"] == pytest.approx(1.0, abs=0.05)
    assert out["passes_mean"] <= 4.0


@wp.kernel
def _project_kernel(soft: SoftArrays, face: wp.array[wp.int32], x: wp.array[wp.vec3], n0: wp.array[wp.vec3],
                    out: wp.array[wp.vec3], hit: wp.array[wp.int32]):
    i = wp.tid()
    xi, _n, _vb, h = soft_project(soft, face[i], x[i], 0.0, n0[i], 1.0e-6)
    out[i] = xi
    hit[i] = h


def _nearest_face(walls: SoftWalls, p: np.ndarray) -> int:
    tri, xv = walls.arrays.tri.numpy(), walls.arrays.x.numpy()
    return int(np.argmin(np.linalg.norm(xv[tri].mean(axis=1) - p, axis=1)))


def test_cloth_side_test_puts_crossers_back():
    m, _cfg, sheet = _sheet_rig()
    walls = SoftWalls(m, 4.0e-3, "cpu", soft_particles=sheet, half_thickness=0.1e-3)
    mid = np.array([6 * S, 6 * S, 0.0])  # interior of the 12 x 12 mm sheet at z = 0
    edge = np.array([12 * S, 6 * S, 0.0])  # on its free border (x = 12 mm)
    pts = np.array([mid + (0.0, 0.0, -0.4e-3), mid + (0.0, 0.0, 0.5e-3), edge + (0.3e-3, 0.0, -0.4e-3)])
    up = np.tile([0.0, 0.0, 1.0], (3, 1))
    faces = np.array([_nearest_face(walls, mid), _nearest_face(walls, mid), _nearest_face(walls, edge)], np.int32)
    out, hit = wp.zeros(3, dtype=wp.vec3, device="cpu"), wp.zeros(3, dtype=wp.int32, device="cpu")
    wp.launch(_project_kernel, dim=3, device="cpu",
              inputs=[walls.arrays, wp.array(faces, dtype=wp.int32, device="cpu"),
                      wp.array(pts, dtype=wp.vec3, device="cpu"), wp.array(up, dtype=wp.vec3, device="cpu"), out, hit])
    o, h = out.numpy(), hit.numpy()
    assert h[0] == 1 and o[0, 2] == pytest.approx(0.1e-3 + 1.0e-6, abs=1e-9)  # crossed: back on top, half + eps up
    np.testing.assert_allclose(o[0, :2], pts[0, :2], atol=1e-9)
    assert h[1] == 0 and np.allclose(o[1], pts[1])  # stayed on its side, clear of the face
    assert h[2] == 0 and np.allclose(o[2], pts[2])  # beyond the free border: liquid may pass around it


def _floating_cube_q(inertia: bool) -> np.ndarray:
    pool = cube_pool("soft", depth=12, cube_at=(5, 5, 8.4), rho_s=600.0, device="cpu")
    sched = vbd_sph_scheduler(pool, added_mass=1000.0 * 3.6 * S * (6 * S) ** 2,
                              sph_kwargs={"soft_inertia": inertia})
    assert sched._entries["sph"].solver.inertia is None  # a closed body: no shell inertia
    st, nx = pool.model.state(), pool.model.state()
    pipe = soft_pipeline(pool.model)
    con = pipe.contacts()
    for _ in range(4):
        st.clear_forces()
        pipe.collide(st, con)
        sched.step(st, nx, None, con, 2.0e-3)
        st, nx = nx, st
    return st.particle_q.numpy().copy()


@pytest.mark.compile_heavy
def test_closed_bodies_keep_p1_walls():
    np.testing.assert_array_equal(_floating_cube_q(True), _floating_cube_q(False))


@pytest.mark.gpu
def test_shell_inertia_on_cuda():
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    dev = "cuda:0"
    rig = sh.spring_plate(10, wet=True, n=6, L=6, device=dev)
    sched = sh._stepper(rig, True, True, rig.info["m_added"], 0.0, 20)
    sol = sched._entries["sph"].solver
    st, nx = rig.model.state(), rig.model.state()
    z = []
    for _ in range(20):
        st.clear_forces()
        st.particle_f.assign(rig.external(st.particle_q.numpy()))
        sched.step(st, nx, None, None, 1.0e-3)
        st, nx = nx, st
        z.append(float(st.particle_q.numpy()[rig.probe, 2].mean()) - rig.info["z0"])
    assert sol.inertia.active and np.isfinite(st.particle_q.numpy()).all()
    assert z[-1] < 0.0 and abs(z[-1]) < 3.0 * rig.info["amp_ref"]  # the sheet sinks under the column, bounded
    n = sol.layout.n
    rng = np.random.default_rng(3)
    fluid = np.zeros(n, bool)
    fluid[rig.liquid.start:rig.liquid.stop] = True
    x, y = (np.where(fluid, rng.standard_normal(n), 0.0) for _ in range(2))

    def op(vec):
        sol._ctl.converged.zero_()
        sol._cg.restart.zero_()
        out = wp.zeros(n, dtype=wp.float32, device=dev)
        spec = SolveSpec(MODE_DENSITY, 1.0e-3, 1.0e-3, 1.0e-2, 1, 10, 0.0, sol._tol_one)
        sol._cg._product(sol._particles(st), sol._slot_struct(), sol._h, sol._kind, spec,
                         wp.array(vec.astype(np.float32), dtype=wp.float32, device=dev), out, 0, 0, 0)
        return out.numpy().astype(np.float64)

    ax, ay = op(x), op(y)
    assert abs(y @ ax - x @ ay) < 1.0e-5 * np.sqrt((y @ ay) * (x @ ax))
