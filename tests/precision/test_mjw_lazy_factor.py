# SPDX-License-Identifier: Apache-2.0
"""The skip of the last Newton iteration's unused factorization (alloy/precision/mjw_lazy_factor.py, SCR-WP0-RIGIDDT-2
L1): bit for bit the stock solver on the API's elliptic path, factorizations really skipped, pyramidal untouched, and
its sources pinned (CPU, MuJoCo Warp route)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.precision import mjw_lazy_factor as L
from alloy.precision import mjw_pin


@pytest.fixture
def lazy():
    """Leaves the patch installed afterwards, as `import alloy.precision` does."""
    yield L
    L.install()


def _stacks(worlds: int = 2):
    """3 stacks of 2 boxes dropped from up to 2 cm: the settle takes many solver iterations, then rests at 1."""
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground(friction=1.0)
    for i in range(3):
        for k in range(2):
            z = 0.05 + 0.1 * k + 0.01 * (k + 1)
            scene.add_box(f"c{i}{k}", size=(0.1, 0.1, 0.1), pos=(0.3 * i, 0.02 * k, z), density=500.0, friction=1.0)
    return scene


def _run(on: bool, periods: int = 15, integrator: str | None = None):
    L.install() if on else L.uninstall()
    sim = alloy.build(_stacks(), device="cpu")
    solver = sim.engine.stepper.solver
    if integrator is not None:                           # before the first step (and so any graph capture)
        from alloy._vendor.mujoco_warp import IntegratorType

        solver.mjw_model.opt.integrator = int(getattr(IntegratorType, integrator.upper()))
    niter = []
    for _ in range(periods):
        sim.step(1)
        niter.append(solver.mjw_data.solver_niter.numpy().copy())
    st = sim.engine.state
    counts = L.skipped_counts(solver.mjw_data)
    return (st.body_q.numpy().copy(), st.body_qd.numpy().copy(), np.array(niter),
            0 if counts is None else int(counts.numpy().sum()), solver)


def test_installed_on_import_with_pinned_sources():
    from alloy._vendor.mujoco_warp._src import solver as S

    assert S._solver_iteration is L._iteration and S._update_gradient is L._update_gradient
    assert mjw_pin.changed(mjw_pin.LAZY) == []
    assert mjw_pin.changed() == []                       # the route's pins still read MuJoCo Warp's own source


def test_bitwise_the_stock_solver_on_the_elliptic_path(monkeypatch, lazy):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    q0, qd0, n0, skipped0, solver = _run(False)
    assert int(solver.mjw_model.opt.cone) == 1 and skipped0 == 0          # elliptic (the API's free-body default)
    q1, qd1, n1, skipped1, _ = _run(True)
    assert n0.max() > 3                                                   # the settle iterates
    assert np.array_equal(q0, q1) and np.array_equal(qd0, qd1) and np.array_equal(n0, n1)
    assert skipped1 > 0


def test_the_discrete_integrator_keeps_one_scratch_and_skips(monkeypatch, lazy):
    """WP0's review: the discrete integrator's _solve passes the iterations a dataclasses.replace copy of the Data
    each step. The scratch follows the shared solver_niter array, so one set serves every step and its counts are
    the Data's own; bit for bit the stock solver."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    q0, qd0, n0, _, _ = _run(False, integrator="discrete")
    before = len(L._SCRATCH)
    q1, qd1, n1, skipped1, solver = _run(True, integrator="discrete")
    assert int(solver.mjw_model.opt.integrator) == 4 and n0.max() > 3     # DISCRETE; the settle iterates
    assert np.array_equal(q0, q1) and np.array_equal(qd0, qd1) and np.array_equal(n0, n1)
    assert skipped1 > 0 and len(L._SCRATCH) <= before + 1                 # one scratch for this Data, not per step


def test_pyramidal_is_left_alone(monkeypatch, lazy):
    """The incremental (pyramidal) Newton path runs MuJoCo Warp's update unchanged: nothing is skipped."""
    from alloy._vendor import newton

    b = newton.ModelBuilder()
    for k in range(2):
        body = b.add_body(xform=wp.transform((0.3 * k, 0.0, 0.06), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    b.add_ground_plane()
    model = b.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, cone="pyramidal")
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(20):
        solver.step(s0, s1, ctrl, None, 0.005)
        s0, s1 = s1, s0
    counts = L.skipped_counts(solver.mjw_data)
    assert counts is None or int(counts.numpy().sum()) == 0
