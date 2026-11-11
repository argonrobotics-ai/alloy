# SPDX-License-Identifier: Apache-2.0
"""SolverVBDAlloy's opt-in accelerators (alloy/solvers/vbd/accel.py): off = the stock loop bit for bit; the adaptive
start keeps a resting sheet still at 1 iteration, keeps free fall and pinned pieces on Newton's start bit for bit, is
re-step safe and clears on reset; Chebyshev leaves a converged step unchanged and stays penetration-free (WP11's
review conditions, 2026-11-08)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import SolverVBDAlloy
from alloy.solvers.vbd import scenes as S
from alloy.solvers.vbd.accel import AccelConfig, chebyshev_weight

ADAPTIVE = {"alloy_initial_guess": "adaptive"}
BOTH = {"alloy_initial_guess": "adaptive", "alloy_chebyshev_rho": 0.9}


def _solver(scene, iterations=10, **kw):
    return SolverVBDAlloy(scene.model, **S.solver_kwargs(scene, iterations), **kw)


def _states(scene):
    m = scene.model
    return m.state(), m.state(), m.control(), (scene.pipeline.contacts() if scene.pipeline is not None else None)


def _step(scene, solver, s0, s1, ctrl, contacts, dt=1.0e-3):
    s0.clear_forces()
    if scene.pipeline is not None:
        scene.pipeline.collide(s0, contacts)
    solver.step(s0, s1, ctrl, contacts, dt)


def _run(scene, solver, steps):
    s0, s1, ctrl, contacts = _states(scene)
    for _ in range(steps):
        _step(scene, solver, s0, s1, ctrl, contacts)
        s0, s1 = s1, s0
    return s0


def _q(state):
    return state.particle_q.numpy().copy()


def test_config_validates():
    with pytest.raises(ValueError, match="alloy_initial_guess"):
        AccelConfig("guess")
    with pytest.raises(ValueError, match="alloy_chebyshev_rho"):
        AccelConfig("inertial", 1.0)
    assert [round(chebyshev_weight(n, 0.9, w), 6) for n, w in ((1, 1.0), (2, 1.0))] == [1.0, round(2 / (2 - 0.81), 6)]


@pytest.mark.parametrize("kind", ["drop", "pinned"])
def test_flags_off_is_the_stock_solver_bit_for_bit(kind):
    scene = S.build(kind)
    stock = _run(scene, newton.solvers.SolverVBD(scene.model, **S.solver_kwargs(scene)), 60)
    off = _run(S.build(kind), _solver(S.build(kind), alloy_initial_guess="inertial", alloy_chebyshev_rho=None), 60)
    np.testing.assert_array_equal(_q(stock), _q(off))


@pytest.mark.timeout(300)      # 48 s on an idle host (600 steps at 100 iterations): over 60 s on a loaded gate host
def test_adaptive_start_keeps_a_resting_sheet_still_at_one_iteration():
    """A sheet settled on the ground, then 1 iteration per step: Newton's start x + h v + h^2 g is not undone and the
    sheet sinks; the adaptive start (no h^2 g at rest) keeps the settled sheet where it is."""
    moved = {}
    for name, kw in (("inertial", {}), ("adaptive", ADAPTIVE)):
        scene = S.build("sheet_rest")
        solver = _solver(scene, 100, **kw)
        s0, s1, ctrl, contacts = _states(scene)
        for k in range(300):
            if k == 200:
                solver.iterations, start = 1, _q(s0)
            _step(scene, solver, s0, s1, ctrl, contacts)
            s0, s1 = s1, s0
        moved[name] = float(np.abs(_q(s0) - start).max())
    assert moved["adaptive"] < 0.05 * moved["inertial"], moved


def test_adaptive_start_in_free_fall_is_newtons_start_bit_for_bit():
    scene = S.build("drop")                          # 2 cm above the ground: 30 steps of 1 ms are all free fall
    stock = _run(scene, _solver(scene), 30)
    adaptive = _run(S.build("drop"), _solver(S.build("drop"), **ADAPTIVE), 30)
    np.testing.assert_array_equal(_q(stock), _q(adaptive))


def test_free_fall_stays_on_newtons_start_when_dt_changes():
    """A coupled clock changes dt between steps (WP11's review): the realised acceleration is divided by the dt of the
    step that produced it, so free fall keeps s = 1 and Newton's start, bit for bit."""
    dts = [1.0e-3, 2.0e-3, 1.0e-3, 0.5e-3, 2.0e-3, 1.0e-3] * 3
    out = []
    for kw in ({}, ADAPTIVE):
        scene = S.build("drop")
        solver = _solver(scene, **kw)
        s0, s1, ctrl, contacts = _states(scene)
        for dt in dts:
            _step(scene, solver, s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0
        out.append(_q(s0))
    np.testing.assert_array_equal(out[0], out[1])


def test_adaptive_start_leaves_pinned_pieces_on_x_plus_hv():
    """Newton's particle_pinned_piece_start already starts a pinned piece at x + h v: the adaptive start must not
    apply on top (every piece of this scene is pinned, so the run equals the stock one bit for bit)."""
    kw = {"particle_pinned_piece_start": True}
    stock = _run(S.build("pinned"), _solver(S.build("pinned"), **kw), 80)
    scene = S.build("pinned")
    adaptive = _run(scene, _solver(scene, **kw, **ADAPTIVE), 80)
    np.testing.assert_array_equal(_q(stock), _q(adaptive))


def _copy_state(model, src):
    dst = model.state()
    for name in ("particle_q", "particle_qd", "particle_f", "particle_q_lo"):   # the remainder rides the State
        wp.copy(getattr(dst, name), getattr(src, name))
    return dst


@pytest.mark.parametrize("kw", [{}, ADAPTIVE, BOTH])
def test_a_restep_from_the_same_input_is_bit_for_bit(kw):
    """Coupled entries step an entry twice per exchange from the same input: the history must not shift twice."""
    scene = S.build("drop")
    solver = _solver(scene, **kw)
    s0, s1, ctrl, contacts = _states(scene)
    for _ in range(40):                              # into contact, with a nonzero history
        _step(scene, solver, s0, s1, ctrl, contacts)
        s0, s1 = s1, s0
    saved = _copy_state(scene.model, s0)
    _step(scene, solver, s0, s1, ctrl, contacts)
    first = _q(s1)
    s0 = _copy_state(scene.model, saved)
    _step(scene, solver, s0, s1, ctrl, contacts)
    np.testing.assert_array_equal(first, _q(s1))


def _two_worlds():
    tmpl = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    v, t = S.grid(6)
    tmpl.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.05), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                        vertices=[wp.vec3(*p) for p in v], indices=t.ravel().tolist(), density=S.DENSITY,
                        tri_ke=S.MEMBRANE_KE, tri_ka=S.MEMBRANE_KE, edge_ke=S.BEND_KE, particle_radius=S.RADIUS)
    main = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    main.replicate(tmpl, copy_count=2, spacing=(0.0, 0.0, 0.0))
    main.color()
    return main.finalize(device="cpu")


def test_reset_clears_the_masked_worlds_history():
    model = _two_worlds()
    solver = SolverVBDAlloy(model, iterations=4, **ADAPTIVE)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(5):
        solver.step(s0, s1, ctrl, None, 1.0e-3)
        s0, s1 = s1, s0
    world = model.particle_world.numpy()
    assert (solver._accel_hist.numpy() == 2).all()
    solver.reset(s0, wp.array([False, True, False], dtype=wp.bool, device="cpu"))   # worlds 0, 1, global
    hist = solver._accel_hist.numpy()
    assert (hist[world == 0] == 2).all() and (hist[world == 1] == 0).all()
    solver.reset(s0)
    assert (solver._accel_hist.numpy() == 0).all()


def _film(iterations, **kw):
    scene = S.build("pinned")                        # the A17 15 um film: the stiffest membrane regime here
    return _q(_run(scene, _solver(scene, iterations, **kw), 15))


def test_chebyshev_converges_to_the_stock_step_and_faster():
    """rho 0.9 on the stiff film: the same converged step as stock (1,000 iterations), and closer than stock at 100
    iterations (measured 2.4e-6 vs 1.8e-5 m from a 4,000-iteration stock run). rho 0.97 over-relaxes this film (2.5e-3
    m off at 30 iterations, 3.4e-4 at 2,000): the flag's documented range is rho <= 0.95."""
    ref = _film(1000)
    assert np.abs(_film(1000, alloy_chebyshev_rho=0.9) - ref).max() < 2.0e-6
    err_cheb = np.abs(_film(100, alloy_chebyshev_rho=0.9) - ref).max()
    err_stock = np.abs(_film(100) - ref).max()
    assert err_cheb < 0.5 * err_stock, (err_cheb, err_stock)


def test_accelerated_stack_stays_penetration_free():
    """Self-contact between two sheets with both accelerators: the extrapolated iterate goes through the truncation."""
    scene = S.build("stack")
    state = _run(scene, _solver(scene, **BOTH), 150)
    q = _q(state)
    lo, hi = scene.sheets
    assert np.isfinite(q).all()
    assert q[hi[0]:hi[1], 2].min() > S.STACK_Z, q[hi[0]:hi[1], 2].min() - S.STACK_Z


def test_the_history_is_in_vbd_snapshots_with_its_world_map():
    """WP11's condition 2: snapshot / rollback state (alloy.coupling.adapters) carries the adaptive start's history."""
    from alloy.coupling.adapters import vbd_state_arrays, vbd_world_maps

    model = _two_worlds()
    solver = SolverVBDAlloy(model, iterations=2, **ADAPTIVE)
    arrays, maps = vbd_state_arrays(solver), vbd_world_maps(solver)
    keys = ("vbd._accel_v_cur", "vbd._accel_a_last", "vbd._accel_hist")
    assert all(k in arrays for k in keys) and all(k in maps for k in keys)
    np.testing.assert_array_equal(maps["vbd._accel_hist"].numpy(), model.particle_world.numpy())
    off = SolverVBDAlloy(model, iterations=2)
    assert not any(k in vbd_state_arrays(off) for k in keys)


def test_chebyshev_leaves_body_contact_particles_unextrapolated():
    """Particles in a body/shape contact record keep the plain iterate (accel.AccelState.mark_held): extrapolating
    lagged friction let WP4's pinch clamp slip at a tenth of mu N ([internal note] notes/wp2-vbd-accel)."""
    scene = S.build("sheet_rest")
    solver = _solver(scene, **BOTH)
    s0, s1, ctrl, contacts = _states(scene)
    _step(scene, solver, s0, s1, ctrl, contacts)
    n = int(contacts.soft_contact_count.numpy()[0])
    touching = np.unique(contacts.soft_contact_indices.numpy()[:n].ravel())
    expected = np.zeros(scene.model.particle_count, np.int32)
    expected[touching[touching >= 0]] = 1
    assert n > 0
    np.testing.assert_array_equal(solver._accel.held.numpy(), expected)
