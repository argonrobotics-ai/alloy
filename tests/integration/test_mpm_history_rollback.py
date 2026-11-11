# SPDX-License-Identifier: Apache-2.0
"""Engine item 2 follow-up (WP4's review): a rolled-back period restores the MPM entry's history too.

The coupled MPM entry keeps its material history (elastic strain, Jp, stress, the APIC velocity gradient) in its own
``state_0`` (``alloy.coupling.entry_history``), outside the parent State. ``SchedulerMixin.state_arrays`` now lists it
with per-particle world maps, so a rollback or a kept world restores it with the rest of the period start. The
pattern is SCR-SC4-7's (tests/coupling/test_rollback_bitwise.py), on 2 worlds of CPL-3's cup on sand (CPU):

- World 1's period is flagged a force jump twice and re-run through the real onset path at the same dt_x, so the
  re-run repeats the unfaulted step exactly when the whole period start is restored.
- Both worlds must stay on the unfaulted reference bitwise: bodies, particles and the MPM history.
- Without the history in the snapshot, the re-run starts from the failed attempt's history: world 1 leaves the
  reference at once (Jp off by 1.2e-2, body_qd by 0.13 at P+1; wp12-results "rollback MPM history").

Not covered: a re-run at a refined dt_x. The kept world's particles move by 6e-8 m with or without this snapshot
(1e-5 m before SCR-SOFT-2-2 gave each world its own grid); the solver's warm-start fields (``mpm.last_step.ws_*``)
have no world axis. It is not bitwise for MPM entries.
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

pytestmark = [pytest.mark.compile_heavy, pytest.mark.integration, pytest.mark.timeout(1800)]

WORLDS, ROLLED, KEPT = 2, 1, 0
WARM, AFTER = 4, 2          # the cup lands in the first period; the faulted period is the 5th
STATE = ("body_q", "body_qd", "particle_q", "particle_qd")


def _sim():
    from alloy.integration import build_simulation

    card, lib = I.cup_on_sand(worlds=WORLDS)
    return build_simulation(I.assemble(card, lib, "cpu", WORLDS), "standard", graph="off")


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


def _world_state(sim, world: int) -> dict[str, np.ndarray]:
    """One world's State arrays and its MPM history (the entry's state_0)."""
    m, s = sim.model, sim.state
    sel = {"body": m.body_world.numpy() == world, "particle": m.particle_world.numpy() == world}
    out = {n: getattr(s, n).numpy()[sel["particle" if n.startswith("particle") else "body"]] for n in STATE}
    e = sim.stepper._entries["mpm"]
    pw = e.view.particle_world.numpy() == world
    for name, arr in vars(e.state_0.mpm).items():
        if hasattr(arr, "numpy") and arr.shape[0] == pw.shape[0]:
            out[f"mpm.{name}"] = arr.numpy()[pw]
    return out


def _run(faulted: bool) -> tuple[list[list[dict[str, np.ndarray]]], int]:
    sim = _sim()
    onset = sim.stepper._onset
    onset.policy = dataclasses.replace(onset.policy, onset_gain_per_exchange=math.inf)    # only the injected jump
    for _ in range(WARM):
        sim.step_period()
    reruns = sim.stepper._onset.reruns
    if faulted:
        _flag_force_jump(sim, sim.periods, 2)
    rows = []
    for _ in range(AFTER + 1):
        sim.step_period()
        rows.append([_world_state(sim, w) for w in range(WORLDS)])
    return rows, sim.stepper._onset.reruns - reruns


def test_a_same_dt_rerun_restores_the_mpm_history(monkeypatch):
    attempt = O.OnsetRefiner.attempt

    def keep_dt(self, control_dt, j, dt_x, j_max):
        return dt_x, attempt(self, control_dt, j, dt_x, j_max)[1]

    monkeypatch.setattr(O.OnsetRefiner, "attempt", keep_dt)
    with isolated_registry(empty=True):
        ref, _ = _run(False)
        hit, reruns = _run(True)
    assert reruns == 2
    assert any(k.startswith("mpm.") for k in ref[0][ROLLED])
    for k in range(AFTER + 1):
        for w in (ROLLED, KEPT):
            for name, a in ref[k][w].items():
                assert np.array_equal(a, hit[k][w][name]), f"world {w}: {name} left the reference at P+{k}"
