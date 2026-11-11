# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-7: a force-jump re-run (SCR-WP11-4) restores its world completely, and the worlds it keeps stay on their
own trajectory. CPU, WP12's box-under-cloth scene (tests/integration/test_rollback_latch.py: MuJoCo + VBD under the
Scheduler, alloy's SyncedMuJoCo), 2 worlds, the cloth already resting on the box.

World 1's period is flagged a force jump twice (the trigger itself is off, so no other world is) and re-run through the real
onset path. At the same dt_x (the refinement patched out) both worlds must then follow the unfaulted reference
bitwise; at the refined dt_x the kept world 0 must, on the reference's clock (replayed after the faulted period).
Worlds share one exchange clock, so on its own clock the refined world moves every world's VBD substeps and the kept
world leaves the reference with it: test_a_refined_world_changes_the_shared_clock_for_every_world, known gap
docs/known_gaps/coupling-shared-exchange-clock.md. These are restore tests, not world independence.
Before the fix MuJoCo Warp's own qpos/qvel and SyncedMuJoCo's
last-written coordinates were outside the snapshot, so a restored or kept world was re-pushed through the float32
coordinate round trip and left the reference: 1e-8 m in the re-run period, mm eight periods later.
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

WORLDS, ROLLED, KEPT = 2, 1, 0
CONTROL_DT = 0.005
WARM, AFTER = 8, 3          # the towel (2.5 mm over the box) lands at ~0.02 s; the faulted period is the 9th
STATE = ("body_q", "body_qd", "particle_q", "particle_qd", "joint_q", "joint_qd")


def _sim():
    from alloy.integration import build_simulation

    card, lib = S.scene([S.table(), S.box(), S.cloth()], [S.place("table"), S.place("box", z=0.051),
                                                          S.place("towel", z=0.0835)], worlds=WORLDS)
    return build_simulation(I.assemble(dataclasses.replace(card, control_dt=CONTROL_DT), lib, "cpu", WORLDS),
                            "standard", graph="off")


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
    m, s = sim.model, sim.state
    sel = {"body": m.body_world.numpy() == world, "particle": m.particle_world.numpy() == world}
    out = {}
    for name in STATE:
        arr = getattr(s, name).numpy()
        kind = "particle" if name.startswith("particle") else "body" if name.startswith("body") else None
        out[name] = arr[sel[kind]] if kind else arr
    return out


def _clock(sim, after: int, replay: dict | None) -> dict:
    """The scene clock's decision per period; with ``replay``, periods after ``after`` take the reference's.

    The clock is one for all worlds (VBD's stable step is the minimum over worlds), so a re-run world's own
    trajectory can move it. On this towel the reference's VBD stable step at P+2 is 1.8 % above the 2-substep
    threshold, and the refined re-run world drapes ~9 % faster (C3's per-body gains, 2026-11-08): the scene then
    takes 4 VBD substeps at P+3 in every world. That is the clock coupling worlds by design, not a restore leak, so
    the refined case replays the reference's clock after the faulted period and checks the restore alone."""
    decide, seen = sim.stepper._clock.decide, {}

    def wrapped(*args, **kwargs):
        if replay is not None and sim.periods > after and sim.periods in replay:
            return replay[sim.periods]
        seen[sim.periods] = d = decide(*args, **kwargs)
        return d

    sim.stepper._clock.decide = wrapped
    return seen


def _run(faulted: bool, replay: dict | None = None) -> tuple[list, int, dict]:
    sim = _sim()
    onset = sim.stepper._onset                       # the towel draping onto the table would flag real jumps too
    onset.policy = dataclasses.replace(onset.policy, onset_gain_per_exchange=math.inf)
    clock = _clock(sim, WARM, replay)
    for _ in range(WARM):
        sim.step_period()
    reruns = sim.stepper._onset.reruns
    if faulted:
        _flag_force_jump(sim, sim.periods, 2)
    rows = []
    for _ in range(AFTER + 1):
        sim.step_period()
        rows.append((I.digests(sim).copy(), _world_state(sim, ROLLED)))
    return rows, sim.stepper._onset.reruns - reruns, clock


def _runs(monkeypatch) -> tuple[list, list, int]:
    """Reference and faulted runs with the refinement patched out (the re-run keeps dt_x)."""
    attempt = O.OnsetRefiner.attempt

    def keep_dt(self, control_dt, j, dt_x, j_max):
        return dt_x, attempt(self, control_dt, j, dt_x, j_max)[1]

    monkeypatch.setattr(O.OnsetRefiner, "attempt", keep_dt)
    with isolated_registry(empty=True):
        ref, _, _ = _run(False)
        hit, reruns, _ = _run(True)
    return ref, hit, reruns


def test_a_same_dt_rerun_leaves_every_world_on_the_reference(monkeypatch) -> None:
    ref, hit, reruns = _runs(monkeypatch)
    assert reruns == 2
    for k in range(1, AFTER + 1):                    # P+0's report differs by design (ROLLED_BACK in world 1)
        assert np.array_equal(ref[k][0][KEPT], hit[k][0][KEPT]), f"kept world left the reference at P+{k}"
        for name in STATE:
            assert np.array_equal(ref[k][1][name], hit[k][1][name]), f"rolled world: {name} differs at P+{k}"


@pytest.fixture(scope="module")
def refined():
    """The unfaulted reference and the refined re-run twice: with the reference's clock replayed after the faulted
    period, and on its own clock."""
    with isolated_registry(empty=True):
        ref, _, clock = _run(False)
        replayed, reruns, _ = _run(True, clock)
        free, _, own = _run(True)
    return dict(ref=ref, clock=clock, replayed=replayed, reruns=reruns, free=free, own=own)


def test_a_refined_rerun_leaves_the_kept_world_on_the_reference(refined) -> None:
    """The restore, on the reference's clock: the kept world takes its first-run end state back exactly."""
    ref, hit = refined["ref"], refined["replayed"]
    assert refined["reruns"] == 2
    for k in range(AFTER + 1):
        assert np.array_equal(ref[k][0][KEPT], hit[k][0][KEPT]), f"kept world left the reference at P+{k}"


@pytest.mark.xfail(strict=False, reason="a 1.8 % margin: the reference's VBD stable step at P+2 sits just above the "
                   "2-substep threshold, and sub-ulp changes to VBD velocities (SCR-WP11-F32) move it. The gap stands "
                   "(known gap coupling-shared-exchange-clock); the demonstration needs a scenario with margin")
def test_a_refined_world_changes_the_shared_clock_for_every_world(refined) -> None:
    """Worlds are NOT independent across a refined re-run (known gap coupling-shared-exchange-clock): the scene has
    one exchange clock, and VBD's stable step is the minimum over worlds. The refined re-run world's towel drapes
    faster than the reference's, so at P+3 the scene takes 4 VBD substeps instead of 2 (C3's per-body gains,
    2026-11-08; the reference sits 1.8 % above the 2-substep threshold). The kept world follows the reference
    exactly until then and leaves it in that period, on the clock its neighbour chose."""
    ref, free, clock, own = refined["ref"], refined["free"], refined["clock"], refined["own"]
    changed = [p for p in sorted(clock) if p > WARM and p in own and own[p].substeps != clock[p].substeps]
    assert changed, "the re-run world no longer moves the shared clock: update this test and the known gap"
    first = changed[0] - WARM                        # P+k of the first period on a different clock
    assert own[changed[0]].substeps["vbd"] > clock[changed[0]].substeps["vbd"]
    for k in range(min(first, AFTER + 1)):
        assert np.array_equal(ref[k][0][KEPT], free[k][0][KEPT]), f"kept world left the reference at P+{k}"
    if first <= AFTER:
        assert not np.array_equal(ref[first][0][KEPT], free[first][0][KEPT])
