# SPDX-License-Identifier: Apache-2.0
"""SolverVBDMomentum (SCR-WP11-2): the exact coarse translation step and its scope.

- free component under a non-uniform load: momentum gain = external impulse (stock VBD gains ~2.5x on this cap);
- every component touching the ground, another sheet (self-contact) or pinned: bitwise the stock solver;
- a sheet dropped onto the ground comes to rest like stock; deterministic; layout chunks never cross components;
- the `vbd` capability builds through `alloy.solvers.vbd.make_vbd`, `coarse_step` picks the class;
- CUDA: a captured step replays bitwise equal to eager.
docs/notes/wp11-vbd-momentum.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import COARSE_DEFAULT, COARSE_KEY, SolverVBDAlloy, SolverVBDMomentum, build_layout, make_vbd
from alloy.solvers.vbd import components as CM
from alloy.solvers.vbd import scenes as SC

wp.config.quiet = True
STOCK = newton.solvers.SolverVBD


def _run(kind: str, cls: type, steps: int, device: str = "cpu", worlds: int = 1):
    sc = SC.build(kind, device, worlds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)         # Newton's self-contact radius rename
        solver = cls(sc.model, **SC.solver_kwargs(sc))
    return sc, solver, SC.run(sc, solver, steps)


def test_free_cap_momentum_equals_impulse():
    steps, dt = 60, 1.0e-3
    impulse = SC.FORCE * steps * dt
    sc, _, st = _run("cap_load", SolverVBDMomentum, steps)
    p = SC.momentum(sc, st)[0]
    assert abs(p[2] / impulse - 1.0) < 1e-3 and np.abs(p[:2]).max() < 1e-3 * impulse
    sc, _, st = _run("cap_load", STOCK, steps)
    # the leak the step removes (A17): stock gains ~2.5x on Newton ac7f7830, delivers ~0.6x with the membrane
    # Hessian fix (SCR-WP11-3 pin); either way far from 1
    assert abs(SC.momentum(sc, st)[0][2] / impulse - 1.0) > 0.3


@pytest.mark.parametrize("kind", ["sheet_rest", "stack", "pinned"])
def test_contact_and_pinned_components_keep_the_stock_path(kind):
    """Ground contact, sheet-on-sheet self-contact and a pinned row block the step: bitwise stock."""
    _, solver, a = _run(kind, SolverVBDMomentum, 120)
    _, _, b = _run(kind, STOCK, 120)
    assert np.array_equal(a.particle_q.numpy(), b.particle_q.numpy())
    assert np.array_equal(a.particle_qd.numpy(), b.particle_qd.numpy())
    assert not np.abs(solver.coarse.delta.numpy()).any()


def test_resting_sheet_stays_at_rest():
    """The ruling's check (the first prototype jittered at 1 m/s): a sheet dropped 2 cm settles like stock."""
    _, _, st = _run("drop", SolverVBDMomentum, 400)
    q, qd = st.particle_q.numpy(), st.particle_qd.numpy()
    assert np.abs(qd).max() < 1e-4
    assert abs(q[:, 2].mean() - SC.RADIUS) < 1e-5


def test_deterministic():
    runs = [_run("cap_load", SolverVBDMomentum, 20, worlds=2)[2].particle_q.numpy() for _ in range(2)]
    assert np.array_equal(*runs)


def test_layout_chunks_stay_inside_components():
    sc = SC.build("stack", worlds=2)
    lay = build_layout(sc.model, chunk=50)
    assert lay.count == 4 and (lay.comp_of >= 0).all()
    for c in range(lay.count):
        members = lay.members[lay.comp_start[c]:lay.comp_start[c + 1]]
        assert (lay.comp_of[members] == c).all() and np.all(np.diff(members) > 0)
        k0, k1 = lay.comp_chunk_start[c], lay.comp_chunk_start[c + 1]
        assert lay.chunk_start[k0] == lay.comp_start[c] and lay.chunk_start[k1] == lay.comp_start[c + 1]
        assert np.diff(lay.chunk_start[k0:k1 + 1]).max() <= 50


def test_labels_skip_particles_in_no_element():
    rows = [np.array([[0, 1, 2], [2, 3, 4]]), np.array([[-1, -1, 6, 7]])]
    labels = CM.label_components(9, rows)
    assert labels.tolist() == [0, 0, 0, 0, 0, -1, 1, 1, -1]


def test_factory_and_capability():
    sc = SC.build("cap_load")
    assert type(make_vbd(sc.model, {"iterations": 2, COARSE_KEY: True})) is SolverVBDMomentum
    assert type(make_vbd(sc.model, {"iterations": 2, COARSE_KEY: False})) is SolverVBDAlloy
    assert COARSE_DEFAULT is True                     # the router's vbd default since the SCR-WP11-2 acceptance
    expected = SolverVBDMomentum
    from alloy.scene.builtin_plugins import builtin_capabilities

    cap = next(c for c in builtin_capabilities() if c.name == "vbd")
    assert type(cap.factory(sc.model, {"iterations": 2})) is expected
    assert issubclass(SolverVBDMomentum, cap.solver_cls)


@pytest.mark.gpu
def test_capture_replays_bitwise(device):
    if not device.startswith("cuda"):
        pytest.skip("needs CUDA (ALLOY_DEVICE=cuda:0)")
    eager = _run("cap_load", SolverVBDMomentum, 10, device, worlds=4)[2].particle_q.numpy()
    sc = SC.build("cap_load", device, 4)
    solver = SolverVBDMomentum(sc.model, **SC.solver_kwargs(sc))
    s0, s1, ctrl = sc.model.state(), sc.model.state(), sc.model.control()
    with wp.ScopedCapture(device=device) as cap:
        s0.clear_forces()
        wp.copy(s0.particle_f, sc.force)
        solver.step(s0, s1, ctrl, None, 1.0e-3)
        wp.copy(s0.particle_q, s1.particle_q)
        wp.copy(s0.particle_qd, s1.particle_qd)
        wp.copy(s0.particle_q_lo, s1.particle_q_lo)   # the sub-ulp remainder is State data too (SCR-WP11-F32)
    for _ in range(10):                      # capture records only; every step runs on replay
        wp.capture_launch(cap.graph)
    assert np.array_equal(s1.particle_q.numpy(), eager)
