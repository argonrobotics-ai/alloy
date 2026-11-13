# SPDX-License-Identifier: Apache-2.0
"""Gap #10b hooks: host-side per-world data that steps nothing. The contact gate's per-world need, the ladder's
per-world sticky-trip row (carried through the fine re-run), and both as optional inputs of ``per_world_rungs``."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.coupling import fine_rerun as FR
from alloy.coupling.ladder import PeriodLadder, escalate
from alloy.coupling.world_clock_need import AttemptLevels, ClockGrid, per_world_rungs
from alloy.coupling.contact_gate import ContactGate
from alloy.core.registry import isolated_registry
from alloy.coupling.types import ClockPolicy, SchedulerPolicy
from alloy.testing.coupling import ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth

SPEC = ClothSceneSpec(box_mass=0.2, cells=8)
CONTROL_DT = 0.01
GRID = ClockGrid(control_dt=0.01, dt_x_max=math.inf, j_max=10)


# ---------------------------------------------------------------------------- hook 1: the gate
def _gate(drops: list[float]):
    from alloy.scene.builtin_plugins import register_builtin_solvers

    with isolated_registry(empty=True):
        register_builtin_solvers()
        model = coupling_box_on_cloth(len(drops), SPEC, drop_offsets=drops, device="cpu")
        clock = ClockPolicy(contact_gate=True, gate_margin=0.01)
        sched = cloth_scheduler(model, SPEC, policy=SchedulerPolicy(clock=clock), vbd_adaptive=False)
    return sched._gate, model


def _reference_bound(gate: ContactGate) -> float:
    """``bound()`` recomputed from the public arrays alone."""
    own = gate.ends.bound_s[gate.armed_host]
    return float(min(gate.ends.fixed_bound_s, float(own.min()) if own.size else math.inf))


@pytest.mark.integration
def test_world_need_reads_only_and_keeps_the_gate_bound() -> None:
    """Only world 1's box is near: both worlds' proxies arm (a pair arms whole), the bound is today's, and
    ``world_need`` leaves ``armed_host``, ``hold`` and ``bound()`` bitwise as they were."""
    gate, model = _gate([0.5, 0.0])
    state = model.state()
    for _ in range(gate.hold_periods + 2):
        gate.evaluate(state, CONTROL_DT)
    armed, hold, bound = gate.armed_host.copy(), gate.hold.copy(), gate.bound()
    need = gate.world_need()
    need2 = gate.world_need()
    assert np.array_equal(gate.armed_host, armed) and np.array_equal(gate.hold, hold)
    assert gate.bound() == bound == _reference_bound(gate) and np.array_equal(need, need2)
    assert armed.all()                                            # route-level arming: the far world is armed too
    assert need.shape == (2,) and bool(np.isfinite(need[1]))
    assert np.isinf(need[0]), "the far world's own proxy is not near"
    assert float(need.min()) == bound


@pytest.mark.integration
def test_world_need_is_infinite_where_a_world_has_nothing_armed() -> None:
    gate, model = _gate([0.5, 0.5])
    state = model.state()
    for _ in range(gate.hold_periods + 2):
        gate.evaluate(state, CONTROL_DT)
    assert np.isinf(gate.world_need()).all() and math.isinf(gate.bound())


@pytest.mark.integration
def test_touched_unarmed_arms_the_touching_world_only_in_the_need() -> None:
    gate, model = _gate([0.5, 0.5])
    state = model.state()
    for _ in range(gate.hold_periods + 2):
        gate.evaluate(state, CONTROL_DT)
    idx = np.flatnonzero(np.asarray(gate.ends.world) == 0 & np.isfinite(gate.ends.bound_s))
    gate.touched.assign(np.eye(1, max(gate.count, 1), int(idx[0]), dtype=np.int32)[0])
    assert gate.accept()
    need = gate.world_need()
    assert np.isfinite(need[0]) and np.isinf(need[1]) and gate.armed_host.any()
    assert float(need.min()) == gate.bound()


@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_calling_world_need_never_changes_a_stepping_digest(monkeypatch) -> None:
    from tests.coupling.test_contact_gate import FAR, _run

    base = _run(FAR, 8, gate=True)
    orig = ContactGate.evaluate

    def evaluate(self, state, control_dt):
        out = orig(self, state, control_dt)
        self.world_need()
        return out

    monkeypatch.setattr(ContactGate, "evaluate", evaluate)
    got = _run(FAR, 8, gate=True)
    for k, (a, b) in enumerate(zip(base, got)):
        assert a["exch"] == b["exch"] and a["armed"] == b["armed"] and np.array_equal(a["digest"], b["digest"]), k


# ---------------------------------------------------------------------------- hook 2: the sticky floor row
def _sched(escalations: int = 99):
    clock = SimpleNamespace(j_floor=0)

    def clock_escalate() -> bool:
        if clock.j_floor >= escalations:
            return False
        clock.j_floor += 1
        return True

    clock.escalate = clock_escalate
    return SimpleNamespace(_ladder=PeriodLadder(), _clock=clock, _alloy_pairs=[], _strong=set(),
                           _escalate_mode=lambda: False, _watchdog=SimpleNamespace(mark_failed=lambda w: None))


def test_a_trip_counts_in_its_world_only_and_the_global_state_is_as_before() -> None:
    sched = _sched()
    escalate(sched, np.array([False, True, False]))
    assert sched._ladder._floor_world.tolist() == [0, 1, 0]
    assert sched._trips == 1 and sched._clock.j_floor == 1
    escalate(sched, np.array([True, True, False]))
    assert sched._ladder._floor_world.tolist() == [1, 2, 0] and sched._trips == 2 and sched._clock.j_floor == 2


def test_escalate_without_a_ladder_is_unchanged() -> None:
    sched = _sched()
    del sched._ladder
    escalate(sched, np.array([True]))
    assert sched._trips == 1 and sched._clock.j_floor == 1


def test_clear_worlds_clears_the_masked_rows_only() -> None:
    ladder = PeriodLadder()
    ladder.clear_worlds(np.array([True, False]))               # nothing recorded yet: no-op
    ladder.note_floor_trip(np.array([True, True, True, False]))
    ladder.note_floor_trip(np.array([True, False, True, True]))
    assert ladder._floor_world.tolist() == [2, 1, 2, 1]
    ladder.clear_worlds(np.array([True, False, False, True]))
    assert ladder._floor_world.tolist() == [0, 1, 2, 0]
    ladder.clear_worlds(None)
    assert ladder._floor_world.tolist() == [0, 0, 0, 0]


def test_the_floor_row_rides_with_its_world_through_a_fine_rerun() -> None:
    rows = FR.ROWS["_ladder"]
    assert "_floor_world" in rows
    coarse, fine = PeriodLadder(), PeriodLadder()
    coarse.note_floor_trip(np.array([False, True, False, False]))
    coarse.note_floor_trip(np.array([False, True, False, True]))              # rows: [0, 2, 0, 1]
    r = FR._Rows(np.array([1, 3]), np.array([0, 1]), 4, 3)
    FR._cut(coarse, fine, rows, r)
    assert fine._floor_world.tolist() == [2, 1, 2]                            # idle slot takes the first world's
    fine.note_floor_trip(np.array([False, True, False]))                      # a trip inside the fine re-run
    FR._paste(coarse, fine, rows, r)
    assert coarse._floor_world.tolist() == [0, 2, 0, 2]


# ---------------------------------------------------------------------------- the formula takes both
def test_per_world_rungs_without_the_inputs_is_unchanged_and_with_them_names_them() -> None:
    interface, att = np.array([0.01 / 16, 0.01 / 4, math.inf]), AttemptLevels(floor=3)
    base = per_world_rungs(interface, math.inf, GRID, att)
    none = per_world_rungs(interface, math.inf, GRID, att, gate=None, floor_world=None)
    assert none.j.tolist() == base.j.tolist() == [4, 3, 3] and none.terms() == base.terms()

    gate = np.array([math.inf, 0.01 / 32, 0.01 / 8])
    got = per_world_rungs(interface, math.inf, GRID, AttemptLevels(), gate=gate)
    assert got.j.tolist() == [4, 5, 3] and got.terms() == ["interface", "gate", "gate"]

    row = np.array([0, 1, 0])                                                 # only world 1 tripped
    fl = per_world_rungs(interface, math.inf, GRID, att, floor_world=row)
    assert fl.j.tolist() == [4, 3, 0] and fl.terms() == ["interface", "floor_world", "none"]
