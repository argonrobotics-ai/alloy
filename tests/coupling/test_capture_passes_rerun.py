# SPDX-License-Identifier: Apache-2.0
"""Lever #1 through IntegratedSimulation (CPU): a period whose capped captures leave worlds short of coupling passes
re-runs those worlds with the full loop, and every world ends every period bitwise where the uncapped run leaves it.

CPU stands in for a capture by patching ``capture_passes.capturing`` (the loops unroll as under CUDA capture; graphs
stay off, so each period runs its exchanges directly). WP4's ADMM ball + cloth, 4 worlds at gamma 0: its worlds need
2..20 passes, so periods leave worlds short at k_min = 2 captured passes. Compared per period: the per-world digests of
state and every snapshotted array (``alloy.testing.integration.digests``). The re-runs use no escalation budget and
do not change the clock. (A force-jump re-run, SCR-WP11-4, and a short re-run can share one rollback: the counts do
not add.)
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.coupling import capture_passes as cpm
from alloy.testing import integration as I
from alloy.testing.coupling import admm_model, admm_scheduler

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1200)]

OFFSETS = [0.0, 0.01, 0.02, 0.03]
DT = 1.0 / 480.0
CONTROL_DT = 0.02
PERIODS = 5


def _run(capped: bool, monkeypatch) -> tuple[list[np.ndarray], list[float], object]:
    from alloy.integration import build_simulation
    from alloy.scene.builtin_plugins import register_builtin_solvers

    monkeypatch.setattr(cpm, "capturing", lambda device: capped)
    with isolated_registry(empty=True):
        register_builtin_solvers()
        model = admm_model(OFFSETS, device="cpu")
        built = I.built_from_model(model, [I.entry_spec(model, n, DT) for n in ("mjc", "vbd")], control_dt=CONTROL_DT)
        sim = build_simulation(built, "standard", stepper=admm_scheduler(model, gamma=0.0), graph="off")
        digests, clocks = [], []
        for _ in range(PERIODS):
            sim.step_period()
            digests.append(I.digests(sim).copy())
            clocks.append(float(sim.stepper.report().dt_x))
    return digests, clocks, sim


def test_short_worlds_rerun_to_the_uncapped_result(monkeypatch) -> None:
    ref, ref_clock, ref_sim = _run(False, monkeypatch)
    hit, hit_clock, sim = _run(True, monkeypatch)
    cp = sim.stepper.iteration_loop.captured_passes
    print(f"capped: {cp.reruns} short re-runs, rollbacks {sim.rollbacks} (reference {ref_sim.rollbacks}), "
          f"escalation rollbacks {sim.stepper._rollbacks} ({ref_sim.stepper._rollbacks}), passes now {cp.passes}")
    assert cp.reruns > 0, "no period left a world short: nothing was tested"
    assert sim.short_reruns == cp.reruns and sim.rollbacks == ref_sim.rollbacks       # counted apart
    assert sim.stepper._rollbacks == ref_sim.stepper._rollbacks                        # no budget used
    assert hit_clock == ref_clock
    for k in range(PERIODS):
        assert np.array_equal(hit[k], ref[k]), f"period {k}: worlds {np.flatnonzero((hit[k] != ref[k]).any(1))}"


class _ShortThenEscalated:
    """StepperLike: attempt 1 leaves world 1 short and world 2 escalated (beyond onset); attempt 2 (the short replay)
    reports nothing, with garbage watch state in world 2 (it ran from its end state there); attempt 3 is clean. Each
    attempt lifts every body by ``LIFT * attempt`` per exchange, so a world's end height names its last attempt."""

    LIFT = 1.0e-4

    def __init__(self, solver, dt_x: float) -> None:
        from alloy.testing import fakes as F

        self.inner = F.FakeStepper(solver, dt_x=dt_x)
        self.world_count, self.attempt, self.masks, self.reports = self.inner.world_count, 0, [], []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def choose_clock(self, control_dt, last):
        self.attempt += 1
        return self.inner.choose_clock(control_dt, last)

    def step(self, state_in, state_out, control, contacts, dt) -> None:
        self.inner.step(state_in, state_out, control, contacts, dt)
        q = state_out.body_q.numpy().copy()
        q[:, 2] += self.LIFT * self.attempt
        state_out.body_q.assign(q)

    def report(self):
        from alloy.coupling.types import CouplingReport, WatchState

        rep = CouplingReport.quiet(self.inner.dt_x, {self.inner.name: 1}, self.world_count)
        w = self.world_count
        short, ws = np.zeros(w, bool), np.zeros(w, np.int32)
        if self.attempt == 1:
            short[1], ws[2] = True, int(WatchState.ESCALATED)
        rep = dataclasses.replace(
            rep, iters_used=np.full(w, self.attempt, np.int32), watch_state=ws,
            notes={"short": short, "probe": np.full(w, float(self.attempt)),
                   "per_entry": {"fake": np.full(w, 10.0 * self.attempt)}, "onset_level": self.attempt})
        self.reports.append(rep)
        return rep

    def rollback_requested(self, report) -> bool:
        from alloy.coupling.capture_passes import short_worlds
        from alloy.coupling.onset import escalated_beyond_onset

        return bool(short_worlds(report).any() or escalated_beyond_onset(report).any())

    def on_rollback(self, world_mask=None) -> None:
        self.masks.append(None if world_mask is None else np.asarray(world_mask).astype(int).tolist())


def test_short_worlds_go_first_then_the_kept_escalated_worlds() -> None:
    """WP12's review: one period with world 1 short and world 2 escalated. Round 1 rolls back world 1 alone (no graph
    invalidation) and keeps world 2's escalation in the merged report; round 2 rolls back world 2 (graphs invalidated).
    Each world ends on its last attempt; the merged report's per-world fields and notes say the same."""
    from types import SimpleNamespace

    from alloy.integration.simulation import IntegratedSimulation
    from alloy.testing import fakes as F
    from alloy.testing.scenes import box_on_plane

    sc = box_on_plane(4, gap=0.3)
    built = I.built_from_model(sc.model, [I.entry_spec(sc.model, "fake", 1.25e-3)], control_dt=0.01)
    stepper = _ShortThenEscalated(F.FakeSolver(sc.model, F.FakeSolver.Config(floor_z=None)), 1.25e-3)
    sim = IntegratedSimulation(built, stepper=stepper, use_graph=False)
    invalidated: list[list] = []
    sim._graphs = SimpleNamespace(invalidate=lambda: invalidated.append(list(stepper.masks)))
    sim.eager_keys[(1.25e-3, (("fake", 1),))] = 0      # the fake cache never captures: every period runs eagerly
    z0 = sim.state.body_q.numpy()[:, 2].copy()
    stepper.attempt = 0                                 # construction asks for the clock once
    sim.step_period()

    assert stepper.attempt == 3 and stepper.masks == [[0, 1, 0, 0], [0, 0, 1, 0]]
    assert invalidated == [[[0, 1, 0, 0], [0, 0, 1, 0]]], "graphs drop once, in the escalation round"
    assert sim.short_reruns == 1 and sim.rollbacks == 1
    n = sim.clock.steps_per_period
    lift = (sim.state.body_q.numpy()[:, 2] - z0) - (sim.state.body_q.numpy()[0, 2] - z0[0])   # relative to world 0
    np.testing.assert_allclose(lift, [0.0, n * stepper.LIFT, 2 * n * stepper.LIFT, 0.0], atol=1e-6)
    rep = sim._last_coupling
    assert rep.iters_used.tolist() == [1, 2, 3, 1]
    assert rep.notes["probe"].tolist() == [1.0, 2.0, 3.0, 1.0]
    assert rep.notes["per_entry"]["fake"].tolist() == [10.0, 20.0, 30.0, 10.0]
    assert rep.notes["onset_level"] == 3 and not rep.notes["short"].any()
