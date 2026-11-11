# SPDX-License-Identifier: Apache-2.0
"""WP12 acceptance: a watchdog rollback and an INVALID latch in the same period leave every other world bitwise
unchanged (SPEC §7 WP12, D9 CPU population: all scenes).

Scene: a box resting on a cloth over a table, 4 worlds, MuJoCo + VBD under WP4's Scheduler, WP5 health. In the
faulted run, world 1's watchdog ledger gets 1 kJ (ESCALATED -> per-world rollback and re-run at half dt_x) and
world 2's box is kicked above SPEED_FAIL in the same period (INVALID -> latched, frozen). Worlds 0 and 3 must
equal a fault-free reference run bitwise at the end of that period.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.coupling.types import WatchState
from alloy.precision.types import HealthFlags, Severity
from alloy.testing import integration as I
from alloy.testing import scene as S

pytestmark = [pytest.mark.compile_heavy, pytest.mark.integration, pytest.mark.timeout(900)]

WORLDS, ROLLED, LATCHED = 4, 1, 2
OTHERS = [w for w in range(WORLDS) if w not in (ROLLED, LATCHED)]
CMD = "PYTHONPATH=. python -m pytest tests/integration/test_rollback_latch.py -q"


CONTROL_DT = 0.005   # 4x fewer steps per period than the 0.02 s default (CPU budget); the logic is per period


def _card():
    card, lib = S.scene([S.table(), S.box(), S.cloth()], [S.place("table"), S.place("box", z=0.051),
                                                          S.place("towel", z=0.12)], worlds=WORLDS)
    return dataclasses.replace(card, control_dt=CONTROL_DT), lib


def _sim():
    from alloy.integration import build_simulation

    card, lib = _card()
    return build_simulation(I.assemble(card, lib, "cpu", WORLDS), "standard", graph="off")


def _box_local(built) -> int:
    """Local body index of the box in its world template."""
    return int(built.handles["box"].template.body.start)


def _run(faults: bool, warm: int = 1):
    """``warm`` clean periods, then the (faulted) period; returns (sim, digests after warm, digests after)."""
    sim = _sim()
    for _ in range(warm):
        sim.step_period()
    before = I.digests(sim)
    if faults:
        n = sim.clock.steps_per_period
        box = I.world_body(sim.model, LATCHED, local=_box_local(sim.built))
        sim.add_step_observer(I.EscalateOnce(sim, ROLLED, at=sim.steps + n // 2))
        sim.add_step_observer(I.KickOnce(sim, box, at=sim.steps + n // 4))
    sim.step_period()
    return sim, before, I.digests(sim)


@pytest.fixture(scope="module")
def runs():
    with isolated_registry(empty=True):
        ref = _run(faults=False)
        hit = _run(faults=True)
    return ref, hit


def test_scene_is_coupled_under_the_scheduler(runs):
    (ref, _, _), _ = runs
    assert type(ref.stepper).__name__ in ("Scheduler", "SchedulerADMM"), type(ref.stepper).__name__
    assert ref.health is not None


def test_rollback_and_latch_happened(runs):
    _, (sim, _, _) = runs
    rep = sim.report()
    assert sim.rollbacks >= 1
    assert rep["coupling"].rollbacks >= 1
    health = rep["health"]
    assert LATCHED in health.invalid_worlds
    assert int(health.flags[LATCHED]) & int(HealthFlags.LATCHED)
    assert int(health.flags[LATCHED]) & int(HealthFlags.SPEED_FAIL)
    assert int(health.severity[ROLLED]) < int(Severity.INVALID)
    assert sim.latch_log is not None and sim.latch_log.latched == {LATCHED}     # SCR-JOB2-4: warned, once
    first = health.first_failure.world(LATCHED)                   # WP5 9e19d854, with the sim time from update(dt=)
    assert first is not None and "SPEED_FAIL" in first["flags"] and np.isfinite(first["time_s"]), first
    assert int(np.asarray(sim.stepper.report().watch_state)[ROLLED]) != int(WatchState.ESCALATED)


def test_other_worlds_bitwise_unchanged(runs):
    (ref, ref_before, ref_after), (sim, hit_before, hit_after) = runs
    assert np.array_equal(ref_before, hit_before), "runs differ before the faulted period"
    same = [bool(np.array_equal(ref_after[w], hit_after[w])) for w in OTHERS]
    changed = [not np.array_equal(ref_after[w], hit_after[w]) for w in (ROLLED, LATCHED)]
    I.record("WP12-ROLLBACK-LATCH", float(all(same)), "pass" if all(same) else "fail", CMD,
             others_equal=same, faulted_changed=changed, rollbacks=sim.rollbacks, worlds=WORLDS)
    assert all(same), f"other worlds changed: {dict(zip(OTHERS, same))}"
    assert all(changed), "the faulted worlds should differ from the reference"
