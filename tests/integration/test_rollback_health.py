# SPDX-License-Identifier: Apache-2.0
"""A re-run period puts the re-run worlds' health arrays back to the period start (WP4's report, 2026-11-06).

``HealthMonitor.state_arrays`` (record, latch, per-world clock, energy reference and work books) were saved only in
the end-of-period snapshot that kept worlds restore from. A re-run world kept the discarded attempt's: its health
clock counted every attempt (0.05 s after one 0.02 s period re-run three times) and its energy books carried the
discarded work into the re-run, so ENERGY_JUMP could fire on a free-falling body right after a re-run.

Scene: WP12's box under cloth (MuJoCo + VBD under WP4's Scheduler, WP5 health), 2 worlds on CPU, as SCR-SC4-7's
tests/coupling/test_rollback_bitwise.py. World 1's period is flagged a force jump twice and re-run through the real
onset path. At the same dt_x the health arrays of both worlds must equal an unfaulted run bitwise; at the refined
dt_x every world's health clock must read the simulation time.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.coupling import onset as O
from alloy.coupling.types import WatchState
from alloy.testing import integration as I
from alloy.testing import scene as S

pytestmark = [pytest.mark.compile_heavy, pytest.mark.integration, pytest.mark.timeout(1800)]

WORLDS, ROLLED = 2, 1
CONTROL_DT = 0.005
WARM, AFTER = 2, 1          # the faulted period is the 3rd; one more period after it


def _sim():
    from alloy.integration import build_simulation

    card, lib = S.scene([S.table(), S.box(), S.cloth()], [S.place("table"), S.place("box", z=0.051),
                                                          S.place("towel", z=0.0835)], worlds=WORLDS)
    built = I.assemble(dataclasses.replace(card, control_dt=CONTROL_DT), lib, "cpu", WORLDS)
    return build_simulation(built, "standard", graph="off")


def _flag_force_jump(sim, period: int, times: int) -> None:
    """World ROLLED's report says 'force jump' on the first ``times`` attempts of ``period``."""
    orig, left = sim.stepper.report, [times]

    def report():
        rep = orig()
        if sim.periods != period or left[0] == 0:
            return rep
        left[0] -= 1
        ws = np.array(rep.watch_state).copy()
        ws[ROLLED] = int(WatchState.ESCALATED)
        notes = dict(rep.notes or {})
        onset = np.asarray(notes.get("onset", np.zeros(WORLDS, bool)), bool).copy()
        onset[ROLLED] = True
        return dataclasses.replace(rep, watch_state=ws, notes={**notes, "onset": onset})

    sim.stepper.report = report


def _health(sim) -> dict[str, np.ndarray]:
    return {k: v.numpy().copy() for k, v in sim.health.state_arrays().items()}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Bitwise per field: the records have NaN fields (never equal as numbers) and struct padding (not data)."""
    fields = a.dtype.names or (None,)
    return all((a if f is None else a[f]).tobytes() == (b if f is None else b[f]).tobytes() for f in fields)


def _run(faulted: bool) -> tuple[list[dict[str, np.ndarray]], int, float]:
    sim = _sim()
    onset = sim.stepper._onset
    onset.policy = dataclasses.replace(onset.policy, onset_gain_per_exchange=math.inf)    # only the injected jump
    for _ in range(WARM):
        sim.step_period()
    reruns = onset.reruns
    if faulted:
        _flag_force_jump(sim, sim.periods, 2)
    rows = []
    for _ in range(AFTER + 1):
        sim.step_period()
        rows.append(_health(sim))
    return rows, onset.reruns - reruns, sim.periods * CONTROL_DT


def _keep_dt(monkeypatch) -> None:
    attempt = O.OnsetRefiner.attempt

    def keep_dt(self, control_dt, j, dt_x, j_max):
        return dt_x, attempt(self, control_dt, j, dt_x, j_max)[1]

    monkeypatch.setattr(O.OnsetRefiner, "attempt", keep_dt)


def test_a_same_dt_rerun_restores_health_bitwise(monkeypatch) -> None:
    _keep_dt(monkeypatch)
    with isolated_registry(empty=True):
        ref, _, _ = _run(False)
        hit, reruns, _ = _run(True)
    assert reruns == 2
    assert "health.time" in ref[0]
    for k in range(AFTER + 1):
        for name, a in ref[k].items():
            assert _same(a, hit[k][name]), f"{name} left the reference at P+{k}: {a} vs {hit[k][name]}"


def test_a_refined_rerun_keeps_the_health_clock_on_sim_time() -> None:
    with isolated_registry(empty=True):
        hit, reruns, t = _run(True)
    assert reruns == 2
    np.testing.assert_allclose(hit[-1]["health.time"], np.full(WORLDS, t), rtol=0, atol=1e-9)
