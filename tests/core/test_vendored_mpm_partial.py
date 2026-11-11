# SPDX-License-Identifier: Apache-2.0
"""SPEED-NOW S5: partial Gauss-Seidel sweeps for the MPM strain solve (alloy/_vendor/patches/0009-mpm-partial-sweeps.patch,
``alloy.integration.mpm_worlds``; opt-in ``alloy.build(..., mpm_sweeps="partial")`` or ``ALLOY_MPM_PARTIAL``).

Between full sweeps a partial sweep visits only the colour blocks with a row that moved more than tau x tolerance in
the last full sweep; the stop test reads only full sweeps. So a solve stops on the same rule, the result lands within
the solve's own convergence noise of the full sweeps, and fewer rows are swept.

Scene: WP12's perf scene (a dough ball under a kinematic press moving down at 5 cm/s), 2 worlds on the CPU, world 1's
press 1 cm higher (the per-world loop runs only for more than one world)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.api.errors import BuildError
from alloy.core.errors import SolverOptionError
from alloy._vendor.newton._src.solvers.implicit_mpm import solve_rheology as SR
from alloy._vendor.newton._src.solvers.implicit_mpm.rheology_solver_kernels import mark_active_blocks
from alloy.integration import mpm_solve, mpm_worlds

# Half the band the 3e-4 tolerance was ruled on (mpm_solve: 3e-4 moved this ball's particles mean 0.07, max 0.52 mm):
# partial sweeps stop on the same test, so they may differ from full ones by the solve's own convergence noise.
MEAN_BOUND, MAX_BOUND = 3.5e-5, 2.6e-4   # m


def _scene(lift: float = 0.01, worlds: int = 2) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_object("dough", "dough_ball", pos=(0, 0, 0.040))
    press = np.array([[0.0, 0.0, 0.079], [0.0, 0.0, 0.079 + lift]])[:worlds] if worlds > 1 else (0.0, 0.0, 0.079)
    s.add_box("press", size=(0.12, 0.12, 0.01), pos=press, kinematic=True, material="steel_carbon")
    return s


def _run(periods: int = 3, lift: float = 0.01, routes: dict | None = None, worlds: int = 2, **build) -> np.ndarray:
    sim = alloy.build(_scene(lift, worlds), device="cpu", seed=0, **build)
    try:
        for _ in range(periods):
            sim.drive("press", velocity=(0, 0, -0.05), duration=sim.dt)
            sim.step(1)
        assert sim.check()["ok"]
        if routes is not None:
            routes.update(sim.routes())
        return np.asarray(sim.state(["dough"])["dough.points"]).copy()
    finally:
        sim.close()


def _record_sweeps(monkeypatch) -> dict[str, int]:
    """Counts full and partial sweeps, and fails if the stop test ever reads a partial one."""
    seen = {"full": 0, "partial": 0, "last": None}
    skip, solve, residual = SR._GaussSeidelSolver.skip_blocks, SR._GaussSeidelSolver.solve, SR._RheologySolver.eval_residual

    def skip_blocks(self, active):
        seen["last"] = active
        return skip(self, active)

    def counted(self):
        seen["full" if seen["last"] is None else "partial"] += 1
        solve(self)

    def checked(self):
        assert seen["last"] is None, "the stop test read a partial sweep"
        return residual(self)
    monkeypatch.setattr(SR._GaussSeidelSolver, "skip_blocks", skip_blocks)
    monkeypatch.setattr(SR._GaussSeidelSolver, "solve", counted)
    monkeypatch.setattr(SR._RheologySolver, "eval_residual", checked)
    return seen


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_partial_sweeps_stop_on_the_same_test_and_land_with_full_sweeps(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    monkeypatch.delenv(mpm_solve.PARTIAL_ENV, raising=False)
    full = _run()
    seen, routes = _record_sweeps(monkeypatch), {}
    partial = _run(routes=routes, mpm_sweeps="partial")
    assert routes["dough"]["mpm_sweeps"].startswith("partial, tau 0.3")    # sim.routes() names it (WP12's review)
    assert seen["partial"] > 2 * seen["full"] > 0                  # most sweeps are partial
    err = np.linalg.norm(partial - full, axis=-1)                  # m; (worlds, particles)
    assert not np.array_equal(full[0], full[1])                    # the worlds differ
    assert err.mean() < MEAN_BOUND and err.max() < MAX_BOUND, (err.mean(), err.max())   # CPU: 0.002-0.003 / 0.02-0.06 mm


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_one_world_sweeps_partially_too(monkeypatch):
    """One world (Newton keeps no per-world tables) runs Alloy's loop with [0, n] tables when partial sweeps are on, so
    1-world laws test them; off, it keeps Newton's loop (tests/integration/test_mpm_world_independence.py)."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    monkeypatch.delenv(mpm_solve.PARTIAL_ENV, raising=False)
    full = _run(worlds=1)
    seen = _record_sweeps(monkeypatch)
    partial = _run(worlds=1, mpm_sweeps="partial")
    assert seen["partial"] > 2 * seen["full"] > 0
    err = np.linalg.norm(partial - full, axis=-1)         # full: Newton's own one-world loop
    assert err.mean() < MEAN_BOUND and err.max() < MAX_BOUND, (err.mean(), err.max())   # CPU: 0.017 / 0.12 mm


def _marks(offsets: list[int]) -> list[int]:
    blocks = wp.array(np.array([[0, 2, 5], [2, 5, 6]], dtype=np.int32), dtype=int, device="cpu")
    residual = wp.array(np.array([0.0, 0.5, 0.0, 0.0, 0.2, 3.0], dtype=np.float32), dtype=float, device="cpu")
    active = wp.full(3, 7, dtype=wp.int32, device="cpu")
    worlds = wp.array(np.array(offsets, dtype=np.int32), dtype=int, device="cpu")
    wp.launch(mark_active_blocks, dim=3, inputs=[blocks, residual, 0.25, worlds], outputs=[active], device="cpu")
    return active.numpy().tolist()


def test_mark_active_blocks_flags_a_moving_block_and_any_block_across_worlds():
    assert _marks([0]) == [1, 0, 1]                                # rows 0-1 (0.5), 2-4 (<= 0.25), 5 (3.0)
    assert _marks([0, 3, 6]) == [1, 1, 1]                          # rows 2-4 span worlds 0 and 1: always swept
    assert _marks([0, 2, 6]) == [1, 0, 1]                          # each block inside one world


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_with_partial_sweeps_each_world_solves_on_its_own(monkeypatch):
    """World 0 is bit for bit the same whether world 1's press starts 2 or 4 mm higher, still in contact with the ball
    (WP8's review: a block's flag never mixes worlds; SCR-WP13-MPMISO-1's per-world stop). A press 1 cm higher does
    not reach the ball in 3 periods, so world 1 would not differ."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    monkeypatch.setenv(mpm_solve.PARTIAL_ENV, "on")
    a, b = _run(lift=0.002), _run(lift=0.004)
    assert not np.array_equal(a[1], b[1])                          # world 1 really differs
    assert np.array_equal(a[0], b[0])


def test_mpm_sweeps_setting_is_checked_and_marks_the_model(monkeypatch):
    monkeypatch.delenv(mpm_solve.PARTIAL_ENV, raising=False)
    assert mpm_solve.sweeps_tau("full", BuildError) is None
    assert mpm_solve.sweeps_tau("partial", BuildError) == mpm_solve.PARTIAL_DEFAULT_TAU
    with pytest.raises(BuildError) as e:
        alloy.build(_scene(), device="cpu", mpm_sweeps="fastest")
    assert e.value.field == "mpm_sweeps"
    with pytest.raises(BuildError):
        alloy.build(_scene(), device="cpu", mpm_sweeps="partial", processes=2)   # a split refuses; one process builds
    from alloy.api import _build

    _build._one_process(None)                                     # "full" splits as before
    with pytest.raises(BuildError):
        _build._one_process(0.3)
    model = type("M", (), {})()
    assert mpm_solve.effective_tau(model) is None
    setattr(model, mpm_solve.MODEL_TAU, 0.3)
    assert mpm_solve.effective_tau(model) == 0.3
    monkeypatch.setenv(mpm_solve.PARTIAL_ENV, "off")               # the run-wide override wins when set
    assert mpm_solve.effective_tau(model) is None
    monkeypatch.setenv(mpm_solve.PARTIAL_ENV, "0.5")
    assert mpm_solve.effective_tau(type("M", (), {})()) == 0.5
    monkeypatch.setenv(mpm_solve.PARTIAL_ENV, "2")
    with pytest.raises(SolverOptionError):
        mpm_solve.partial_tau()


def test_only_the_plain_sweep_skips_blocks():
    assert SR._GaussSeidelSolver.skips_converged and not SR._ReorderedGaussSeidelSolver.skips_converged
    assert mpm_worlds.PARTIAL_TAU is None and mpm_worlds.PARTIAL_GROUP == mpm_worlds.GRAPH_GRANULARITY
