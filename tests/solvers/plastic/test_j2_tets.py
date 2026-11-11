# SPDX-License-Identifier: Apache-2.0
"""J2 on VBD tets (alloy/solvers/plastic/tets.py, SCR-WP7-15 P1): the self-weight column's onset (MuJoCo 4.0's row,
RERUN.md's hook), a stretched bar against the 1-D law, off is bitwise, reset, and the emitter marking a j2 soft body's
tets (CPU, light)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.cards.types import Plastic
from alloy.scene import analytic_plastic as A
from alloy.solvers.plastic.tets import NAMESPACE, TetPlastic, mark_tets, register_attributes
from alloy.solvers.vbd.base import SolverVBDAlloy
from alloy.testing import route as T

pytestmark = pytest.mark.law
DEV = "cpu"


def test_column_stays_elastic_at_76_mm_and_yields_at_77_mm():
    """RERUN.md: the onset bracket on a 1 mm grid holds the analytic σ_y / (ρ g) = 76.45 mm (MuJoCo 4.0: 76 mm).
    F32 (SCR-WP11-F32) puts the discrete onset just under 76.0 mm: plastic_change 4.3e-8 there (9e-6 of the yield strain
    0.005), 1.66e-5 at 76.2 mm; analytic 76.45 (WP7, 2026-11-09)."""
    below, above = A.column_case(0.076, None, DEV), A.column_case(0.077, None, DEV)
    assert below["ok"] and above["ok"]
    assert below["plastic_change"] < 1.0e-6 and above["plastic_change"] > 0.0
    static = 1000.0 * 9.81 * 0.076 ** 2 / (2 * 1.0e5)                  # ρ g H² / (2E), elastic
    assert below["top_drop_m"] == pytest.approx(static, rel=2e-3)


def _bar(device: str, E: float, sy0: float, hmod: float):
    """A 10 x 10 x 40 mm bar (ν 0) on the column mesh, bottom fixed, top driven; all tets J2."""
    verts, tets = A.column_mesh(0.01, 0.04)
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.add_soft_mesh(pos=wp.vec3(), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(),
                    vertices=[wp.vec3(*v) for v in verts], indices=tets.flatten().tolist(), density=1000.0,
                    k_mu=E / 2, k_lambda=0.0, k_damp=A.BETA * E / 2, add_surface_mesh_edges=False)
    ends = np.nonzero((verts[:, 2] == 0.0) | np.isclose(verts[:, 2], 0.04))[0]
    for i in ends:
        b.particle_mass[int(i)] = 0.0
    mark_tets(b, 0, len(tets), sy0, hmod)
    b.color()
    model = b.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model, verts, np.isclose(verts[:, 2], 0.04)


def test_stretched_bar_follows_the_uniaxial_law():
    E, sy0, hmod, strain = 1.0e5, 750.0, 5.0e3, 0.03
    model, verts, top = _bar(DEV, E, sy0, hmod)
    solver = SolverVBDAlloy(model, iterations=A.ITERATIONS, alloy_initial_guess="adaptive")
    s0, s1, control = model.state(), model.state(), model.control()
    q0 = s0.particle_q.numpy().copy()
    for k in range(300):                                                # 1 s ramp, 0.5 s hold
        q = s0.particle_q.numpy()                                       # only the driven end moves by hand
        q[top, 2] = q0[top, 2] + 0.04 * strain * min(1.0, (k + 1) / 200)
        s0.particle_q.assign(q)
        solver.step(s0, s1, control, None, A.DT)
        s0, s1 = s1, s0
    ebar = solver.plastic.ebar.numpy()
    centroid = verts[model.tet_indices.numpy()].mean(axis=1)[:, 2]
    mid = (centroid > 0.012) & (centroid < 0.028)                       # away from the driven ends
    z0, z1 = q0[:, 2], s0.particle_q.numpy()[:, 2]                     # the mid section's own stretch: the clamped
    lo, hi = np.isclose(z0, z0[(z0 > 0.011)].min()), np.isclose(z0, z0[(z0 < 0.029)].max())   # ends neck less
    eps = np.log((z1[hi].mean() - z1[lo].mean()) / (z0[hi].mean() - z0[lo].mean()))
    assert eps > np.log1p(strain)                                       # the middle carries more than the mean
    want = (E * eps - sy0) / (E + hmod)                                 # σ = E(ε - ε_p) = σ_y0 + H ε̄p
    assert ebar[mid].mean() == pytest.approx(want, rel=0.01)            # CPU 11-08: +0.3 %, spread 0.8 %
    fp = np.linalg.inv(solver.plastic.fp_inv.numpy()[mid])
    assert np.linalg.det(fp) == pytest.approx(1.0, abs=1e-4)            # isochoric
    assert np.log(fp[:, 2, 2]).mean() == pytest.approx(want, rel=0.03)  # the permanent stretch is along the bar


def _soft_grid(attrs: bool, device: str = DEV):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    if attrs:
        register_attributes(b)
    b.add_soft_grid(pos=wp.vec3(0.0, 0.0, 0.1), rot=wp.quat_identity(), vel=wp.vec3(), dim_x=3, dim_y=3, dim_z=3,
                    cell_x=0.01, cell_y=0.01, cell_z=0.01, density=500.0, k_mu=1e4, k_lambda=1e4, k_damp=1.0)
    b.color()
    return b.finalize(device=device)


def _run(model, cls, steps=20):
    solver = cls(model, iterations=5)
    s0, s1, control = model.state(), model.state(), model.control()
    for _ in range(steps):
        solver.step(s0, s1, control, None, 1e-3)
        s0, s1 = s1, s0
    return s0.particle_q.numpy(), solver


def test_off_is_bitwise_the_stock_solver():
    stock, _ = _run(_soft_grid(False), newton.solvers.SolverVBD)
    plain, s_plain = _run(_soft_grid(False), SolverVBDAlloy)
    zeros, s_zero = _run(_soft_grid(True), SolverVBDAlloy)              # attributes declared, every yield 0
    assert s_plain.plastic is None and s_zero.plastic is None
    assert np.array_equal(stock, plain) and np.array_equal(stock, zeros)


def test_reset_restores_the_authored_rest_shape():
    model, solver, _, _ = A.build_column(0.085, A.COLUMN, DEV)
    s0, s1, control = model.state(), model.state(), model.control()
    for _ in range(100):
        model.set_gravity((0.0, 0.0, -9.81))
        solver.step(s0, s1, control, None, A.DT)
        s0, s1 = s1, s0
    plastic: TetPlastic = solver.plastic
    assert plastic.ebar.numpy().max() > 0.0
    assert not np.array_equal(model.tet_poses.numpy(), plastic.dm0.numpy())
    solver.reset(s0)
    assert plastic.ebar.numpy().max() == 0.0 and np.array_equal(model.tet_poses.numpy(), plastic.dm0.numpy())


def test_the_emitter_marks_a_j2_soft_bodys_tets():
    from dataclasses import replace

    from alloy.scene.emitters.deformable import emit_soft

    foam = T.material("foam", **T.PU_FOAM)
    j2 = replace(foam, plastic=Plastic(yield_stress=T.q(750.0), hardening_modulus=T.q(2e3), model="j2"))
    for mat, want in ((foam, None), (j2, (750.0, 2e3))):
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        emit_soft(b, T.box("pad", mat, 0.02, 0.02, 0.02), None)
        attr = b.custom_attributes.get(f"{NAMESPACE}:yield_stress")
        if want is None:
            assert attr is None
            continue
        hard = b.custom_attributes[f"{NAMESPACE}:hardening"]
        assert sorted(attr.values) == list(range(b.tet_count)) and set(attr.values.values()) == {want[0]}
        assert set(hard.values.values()) == {want[1]}


def test_rollback_mid_yield_replays_the_plastic_state_bitwise():
    """The J2 tet state is in the VBD snapshot list (coupling/adapters.py VBD_ARRAYS): restoring a mid-yield snapshot
    and re-running gives the reference bit for bit (coordinator 08:29Z)."""
    from alloy.coupling.adapters import vbd_state_arrays, vbd_world_maps

    model, solver, _, _ = A.build_column(0.085, A.COLUMN, DEV)
    model.set_gravity((0.0, 0.0, -9.81))
    s0, s1, control = model.state(), model.state(), model.control()

    def run(n, a, b):
        for _ in range(n):
            solver.step(a, b, control, None, A.DT)
            a, b = b, a
        return a, b

    s0, s1 = run(40, s0, s1)
    arrays = vbd_state_arrays(solver)
    assert {"vbd._plastic_tet_fp_inv", "vbd._plastic_tet_ebar", "vbd._plastic_tet_poses"} <= set(arrays)
    maps = vbd_world_maps(solver)
    assert all(maps[k].shape[0] == model.tet_count for k in arrays if "plastic" in k)
    states = (s0.particle_q, s0.particle_qd, s0.particle_q_lo, s1.particle_q, s1.particle_qd, s1.particle_q_lo)
    snap = {k: a.numpy().copy() for k, a in arrays.items()}
    snap_states = [a.numpy().copy() for a in states]
    ebar_mid = solver.plastic.ebar.numpy().max()
    a, _ = run(30, s0, s1)
    ref = {k: v.numpy().copy() for k, v in arrays.items()}
    ref_q = a.particle_q.numpy().copy()
    assert 0.0 < ebar_mid < ref["vbd._plastic_tet_ebar"].max()             # snapshot taken while still yielding
    for k, v in snap.items():
        arrays[k].assign(v)
    for arr, v in zip(states, snap_states):
        arr.assign(v)
    a, _ = run(30, s0, s1)
    for k, v in ref.items():
        assert np.array_equal(arrays[k].numpy(), v), k
    assert np.array_equal(a.particle_q.numpy(), ref_q)


@pytest.mark.slow
@pytest.mark.timeout(300)                                                   # ~110 s on one CPU core
def test_a_bent_bar_stays_bent():
    """V3 bend and stay (analytic_plastic.bend_case): a tet bar under a uniform moment of 1.3 M_y keeps a bend after
    release. At 4 elements through the depth (linear tets bend ~11 % stiff) it reads loaded 0.86 and residual 0.75 of
    the section's closed form. At 8 elements: 1.04 and 1.26, of which 0.06 1/m is unfinished elastic spring-back.
    docs/notes/plasticity-design.md §2."""
    r = A.bend_case(1.3)
    assert r["ebar_grips"] == 0.0 and r["ebar_max"] > 0.0                   # only the gauge yields
    assert 0.8 < r["loaded"] / r["loaded_cf"] < 1.1
    assert 0.6 < r["residual"] / r["residual_cf"] < 1.3 and r["residual"] > 0.0


@pytest.mark.parametrize("ratio, warns", [(0.01, True), (0.2, False)])
def test_a_small_yield_strain_warns(caplog, ratio, warns):
    """docs/known_gaps/plasticity-soft-tets-low-yield.md: a j2 soft body with σ_y / E under LOW_YIELD_STRAIN (10 %)
    can yield spuriously at rest, so the hook warns once, naming the page; the ratio is read back from the tets'
    Lamé pair."""
    from dataclasses import replace

    from alloy.scene.emitters.deformable import emit_soft
    from alloy.solvers.plastic import tets as TP

    foam = T.material("foam", **T.PU_FOAM)                                  # E 1e5 Pa, ν 0.3
    j2 = replace(foam, plastic=Plastic(yield_stress=T.q(ratio * 1e5), hardening_modulus=T.q(0.0), model="j2"))
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    emit_soft(b, T.box("pad", j2, 0.02, 0.02, 0.02), None)
    m = b.finalize(device=DEV)
    assert TP.lowest_yield_strain(m, m.plastic.yield_stress.numpy()) == pytest.approx(ratio, rel=1e-4)
    with caplog.at_level("WARNING", logger="alloy.solvers"):
        assert TetPlastic.for_model(m) is not None
    assert any(TP.GAP in r.getMessage() for r in caplog.records) == warns
