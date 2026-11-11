# SPDX-License-Identifier: Apache-2.0
"""Film kernels on CPU: strain clamp (Wang 2010 rules), momentum-conserving Jacobi, plastic hinge, air force,
per-hinge bending stiffness, topology CSR and long-range pairs."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.film import FilmConfig, SolverFilm, aero, hinge, strain
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.longrange import build_pair_limits
from alloy.solvers.film.materials import HDPE_15UM
from alloy.solvers.film.rigs import FilmRig, grid_mesh, pin
from alloy.solvers.film.topology import bending_ke

pytestmark = pytest.mark.compile_heavy
DEV = "cpu"


def _arr(x, dtype):
    return wp.array(np.asarray(x), dtype=dtype, device=DEV)


def _clamp_one(x, inv_mass, cap=0.03):
    rest = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    dm = np.stack([rest[1, :2] - rest[0, :2], rest[2, :2] - rest[0, :2]], axis=1)
    delta = wp.zeros((1, 3), dtype=wp.vec3, device=DEV)
    active = wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.launch(strain.tri_corrections, dim=1,
              inputs=[_arr(x, wp.vec3), _arr(inv_mass, float), _arr([[0, 1, 2]], wp.int32),
                      _arr([np.linalg.inv(dm)], wp.mat22), _arr([cap], float), _arr([1.0], float), 0.0, delta,
                      active], device=DEV)
    return np.asarray(x) + delta.numpy()[0], int(active.numpy()[0]), dm


def _stretch(x, dm):
    F = np.stack([x[1] - x[0], x[2] - x[0]], axis=1) @ np.linalg.inv(dm)
    return np.linalg.svd(F, compute_uv=False)


def test_clamp_caps_principal_stretch_and_keeps_mass_centre() -> None:
    x = np.array([[0.0, 0.0, 0.0], [0.012, 0.0, 0.0], [0.0, 0.0105, 0.001]])
    out, active, dm = _clamp_one(x, [1.0, 2.0, 1.0])
    s = _stretch(out, dm)
    m = np.array([1.0, 0.5, 1.0])
    assert active == 1 and s.max() <= 1.03 + 1e-5
    assert np.allclose((m[:, None] * out).sum(0), (m[:, None] * x).sum(0), atol=1e-9)


def test_clamp_leaves_triangles_inside_the_cap_and_pinned_vertices() -> None:
    x = np.array([[0.0, 0.0, 0.0], [0.0101, 0.0, 0.0], [0.0, 0.01, 0.0]])
    out, active, _ = _clamp_one(x, [1.0, 1.0, 1.0])
    assert active == 0 and np.array_equal(out, x)
    x = np.array([[0.0, 0.0, 0.0], [0.013, 0.0, 0.0], [0.0, 0.01, 0.0]])
    out, active, dm = _clamp_one(x, [0.0, 1.0, 1.0])            # vertex 0 pinned: scale about it
    assert active == 1 and np.array_equal(out[0], x[0]) and _stretch(out, dm).max() <= 1.03 + 1e-5


def _sheet_rig(**film_kw) -> FilmRig:
    v, t = grid_mesh(8, 8, 5e-3)

    def build(b):
        p0, _ = add_film_mesh(b, v, t, HDPE_15UM, pos=(0.0, 0.0, 0.5))
        return {"all": p0 + np.arange(len(v))}

    return FilmRig(build, device=DEV, film=FilmConfig(**film_kw), gravity=0.0)


def test_projection_conserves_momentum_and_meets_the_cap() -> None:
    rig = _sheet_rig(strain_iterations=40)
    rng = np.random.default_rng(3)
    q = rig.q()
    q = q + rng.normal(0.0, 4e-4, q.shape).astype(np.float32)   # random stretch, up to ~20 %
    rig.s0.particle_q.assign(q)
    rig.s0.particle_qd.zero_()
    before = np.percentile(rig.solver.strain_ratio(rig.s0), 99)
    rig.solver._project(rig.s0, 1.0e-3)
    m = rig.model.particle_mass.numpy()
    p = (m[:, None] * rig.qd()).sum(0)
    assert np.abs(p).max() <= 1e-5 * (m[:, None] * np.abs(rig.qd())).sum()   # momentum stays 0
    after = np.percentile(rig.solver.strain_ratio(rig.s0), 99)
    assert after < 1.5 and after < before / 3.0


def test_plastic_hinge_moves_rest_past_yield_and_wraps() -> None:
    def run(theta_deg: float, rest_deg: float, yield_deg: float) -> float:
        th = math.radians(theta_deg)
        # x2→x3 the edge along y; x0 opposite at -x; x1 at angle θ about the edge (same convention as VBD)
        x = np.array([[-0.01, 0.005, 0.0], [0.01 * math.cos(th), 0.005, 0.01 * math.sin(th)],
                      [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float32)
        ang = wp.zeros(1, dtype=float, device=DEV)
        wp.launch(hinge.hinge_angles, dim=1, inputs=[_arr(x, wp.vec3), _arr([[0, 1, 2, 3]], wp.int32), ang], device=DEV)
        theta = float(ang.numpy()[0])
        rest = _arr([math.copysign(math.radians(rest_deg), theta) if rest_deg else 0.0], float)
        work = wp.zeros(1, dtype=float, device=DEV)
        wp.launch(hinge.plastic_hinge, dim=1, inputs=[_arr(x, wp.vec3), _arr([[0, 1, 2, 3]], wp.int32),
                                                     _arr([math.radians(yield_deg)], float), rest, work], device=DEV)
        return math.degrees(abs(theta)), math.degrees(abs(float(rest.numpy()[0])))

    theta, rest = run(30.0, 0.0, 60.0)       # inside the elastic range: no change
    assert rest == 0.0
    theta, rest = run(150.0, 0.0, 60.0)      # past it: rest follows to θ - θ_y
    assert abs(theta - 150.0) < 1e-3 and abs(rest - 90.0) < 1e-3


def test_air_force_is_the_formula_and_never_does_positive_work() -> None:
    v, t = grid_mesh(2, 2, 0.01)
    x = _arr(v.astype(np.float32), wp.vec3)
    qd = np.zeros_like(v, dtype=np.float32)
    qd[:, 2] = -1.0
    qd[:, 0] = 0.5
    p = aero.AirParams()
    p.rho, p.cd, p.ct, p.wind, p.dt, p.cap = 1.2, 1.28, 0.027, wp.vec3(0.0, 0.0, 0.0), 1e-3, 0
    f = wp.zeros(len(t), dtype=wp.vec3, device=DEV)
    wp.launch(aero.tri_air_force, dim=len(t), inputs=[p, x, _arr(qd, wp.vec3), _arr(t, wp.int32),
                                                      _arr(np.ones(len(t)), float), f], device=DEV)
    F = f.numpy().sum(0)
    A = 4e-4
    assert abs(F[2] - 0.5 * 1.2 * 1.28 * A * 1.0) < 1e-9          # normal: ½ρ Cd A v²
    assert abs(F[0] + 0.5 * 1.2 * 0.027 * A * 0.25) < 1e-9         # tangential: ½ρ Ct A v_t²
    assert float(F @ qd[0]) < 0.0


def test_bending_ke_matches_discrete_shell_weight() -> None:
    v = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
    ke = bending_ke(v, np.array([[2, 3, 0, 1]]), 1.0)               # two right triangles of 5e-5 m² on |e| = 1 cm
    assert abs(ke[0] - 3.0 * 0.01 / (2.0 * 1e-4)) < 1e-9


def test_long_range_pairs_flat_patch_only_and_capped() -> None:
    v, t = grid_mesh(16, 16, 5e-3)
    lim = build_pair_limits(v, t, np.full(len(v), 0.03, np.float32))
    assert len(lim.pairs) > 0 and np.all(lim.cap == np.float32(0.03))
    d = np.linalg.norm(v[lim.pairs[:, 0]] - v[lim.pairs[:, 1]], axis=1)
    assert np.allclose(d, lim.rest, rtol=1e-6)
    folded = v.copy()                                               # fold the sheet in half along x = 4 cm
    m = folded[:, 0] > 0.04
    folded[m, 0] = 0.08 - folded[m, 0]
    folded[m, 2] = 1e-3
    lim2 = build_pair_limits(folded, t, np.full(len(v), 0.03, np.float32))
    across = (folded[lim2.pairs[:, 0], 2] != folded[lim2.pairs[:, 1], 2])
    d2 = np.linalg.norm(folded[lim2.pairs[:, 0]] - folded[lim2.pairs[:, 1]], axis=1)
    assert np.all(d2[across] > 0.0) and np.all(lim2.rest[across] >= d2[across] * 0.999)


def test_pinned_strip_cantilever_runs_and_keeps_the_cap() -> None:
    v, t = grid_mesh(6, 2, 4e-3)

    def build(b):
        p0, _ = add_film_mesh(b, v, t, HDPE_15UM, pos=(0.0, 0.0, 0.5))
        pin(b, p0 + np.flatnonzero(v[:, 0] < 1e-9))
        return {}

    rig = FilmRig(build, device=DEV, dt=1e-3)
    rig.run(0.05)
    assert np.isfinite(rig.q()).all() and np.percentile(rig.solver.strain_ratio(rig.s0), 99) < 1.1


def test_newton_attribute_namespace() -> None:
    b = newton.ModelBuilder()
    v, t = grid_mesh(2, 2, 0.01)
    add_film_mesh(b, v, t, HDPE_15UM)
    b.color()
    m = b.finalize(device=DEV)
    assert np.allclose(m.film.strain_cap.numpy(), HDPE_15UM.strain_cap)
    assert np.allclose(m.film.yield_angle.numpy(), HDPE_15UM.extras.yield_angle)
    solver = SolverFilm(m, iterations=1)
    assert solver.film_topology.limited == len(t) and solver.film_topology.plastic > 0
