# SPDX-License-Identifier: Apache-2.0
"""The escalation ladder handled within its period first (alloy/coupling/ladder.py, SCR-WP3-C3RAND fix 2), CPU.

XB4-9's randomized C3 box-on-pad (64 worlds, L4, main 377349c8e): ladder trips in 1-11 different worlds in each of
periods 1-5 raised the sticky clock floor (j 4 -> 7) and switched the pair to staggered, for all 64 worlds, and 0.5 s
never finished. Now a world's first trip re-runs its period one grid level finer per re-run, that period only; the
floor and the coupling mode stay. A world that trips again within the watchdog window, or a period that ends still
escalated, takes the sticky ladder as before (``test_watchdog``'s light box: a resting instability). Here a real
Scheduler (fake solvers, 4 worlds) is driven through choose_clock / rollback_requested / on_rollback with reports
whose watch states are set by hand."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.coupling.ladder import PeriodLadder
from alloy.coupling.types import SchedulerPolicy, WatchdogPolicy, WatchState
from alloy.testing.coupling import fake_drop_model, fake_scheduler, fixed_clock_policy

CONTROL_DT = 0.01
DT0 = CONTROL_DT / 16          # j = 4: the clock the scene's bounds pick (fixed policy)


def test_the_ladder_decides_within_the_period_or_sticky():
    """First trip on a re-run: one more level; the same period's next re-run: one more; a later trip of the same
    world within the window: sticky; another world: within; off, or a new period's escalation: sticky."""
    lad = PeriodLadder(True, window_s=0.05)                      # 5 periods at 10 ms
    w1, w2 = np.array([False, True, False]), np.array([False, False, True])
    lad.new_period()
    assert lad.within_period(w1, True, CONTROL_DT) and lad.j(4, 10) == 5
    assert lad.within_period(w1, True, CONTROL_DT) and lad.j(4, 10) == 6 and lad.j(9, 10) == 10   # capped at j_max
    lad.new_period()
    notes = lad.notes()
    assert lad.j(4, 10) == 4 and (notes["ladder_level"], notes["ladder_refines"]) == (0, 2)
    assert notes["ladder_refines_per_world"].tolist() == [0, 2, 0] and lad.counts(3) == [0, 2, 0]
    assert lad.within_period(w2, True, CONTROL_DT)               # another world: its own first trip
    lad.new_period()
    assert not lad.within_period(w1, True, CONTROL_DT)           # world 1 again, 2 periods later: sticky
    for _ in range(6):
        lad.new_period()
    assert lad.within_period(w1, True, CONTROL_DT)               # past the window: a first trip again
    assert not lad.within_period(w2, False, CONTROL_DT)          # a period accepted escalated: sticky
    off = PeriodLadder(False)
    off.new_period()
    assert not off.within_period(w1, True, CONTROL_DT)


def _sched(**watchdog):
    model = fake_drop_model([0.5] * 4, device="cpu")
    policy = fixed_clock_policy(DT0)
    if watchdog:
        policy = dataclasses.replace(policy, watchdog=dataclasses.replace(policy.watchdog, **watchdog))
    return fake_scheduler(model, mode="lagged", policy=policy)


def _report(sched, escalated=()):
    rep = sched.report()
    ws = np.zeros(4, dtype=np.int32)
    ws[list(escalated)] = int(WatchState.ESCALATED)
    return dataclasses.replace(rep, watch_state=ws)


def _period(sched, last, trips=(), reruns=1):
    """One period: an attempt whose report escalates ``trips`` (if any), then up to ``reruns`` re-runs that come back
    clean. Returns (each attempt's dt_x, the period's final report)."""
    dts = [sched.choose_clock(CONTROL_DT, last)[0]]
    rep = _report(sched, trips)
    for _ in range(reruns if trips else 0):
        assert sched.rollback_requested(rep)
        sched.on_rollback(None)
        dts.append(sched.choose_clock(CONTROL_DT, rep)[0])
        rep = _report(sched)
    assert not sched.rollback_requested(rep)
    return dts, rep


def _sticky(sched):
    return sched._clock.j_floor, getattr(sched, "_trips", 0), [es.staggered for es in sched._sets]


def test_one_worlds_trip_refines_its_period_and_no_other():
    """World 1 trips in period 1: the re-run takes dt_x / 2, the floor, trips and mode stay, and period 2 is back on
    the scene's clock. World 2's first trip (period 3) is the same. World 1 again within the 1 s window (period 4)
    takes the sticky ladder: the floor rises and stays."""
    sched = _sched()
    before = _sticky(sched)
    dts, rep = _period(sched, None, trips=[1])
    assert dts == pytest.approx([DT0, DT0 / 2], rel=1e-12) and _sticky(sched) == before
    assert rep.notes["ladder_refines"] == 1
    dts, rep = _period(sched, rep)
    assert dts == pytest.approx([DT0], rel=1e-12)
    dts, rep = _period(sched, rep, trips=[2])
    assert dts == pytest.approx([DT0, DT0 / 2], rel=1e-12) and _sticky(sched) == before
    dts, rep = _period(sched, rep, trips=[1])
    assert dts == pytest.approx([DT0, DT0 / 2], rel=1e-12)
    assert sched._clock.j_floor == 5 and sched._trips == 1                # sticky: trip 1 raised the floor
    dts, rep = _period(sched, rep)
    assert dts == pytest.approx([DT0 / 2], rel=1e-12)                    # and it stays
    dts, rep = _period(sched, rep, trips=[1])                            # trip 2: the strong mode, dt_x unchanged
    assert dts == pytest.approx([DT0 / 2] * 2, rel=1e-12) and all(_sticky(sched)[2])
    assert sched.graph_key()[-2:] == ("ladder", 2)                       # a new graph key: ω0 / staggered are baked


def test_a_period_that_stays_escalated_takes_the_sticky_ladder():
    """Both re-runs of the period escalate again (the budget, 2, is spent): the period is accepted escalated and the
    next period's clock takes the sticky ladder."""
    sched = _sched()
    sched.choose_clock(CONTROL_DT, None)
    key = sched.graph_key()
    rep = _report(sched, [3])
    for k in range(2):
        assert sched.rollback_requested(rep)
        sched.on_rollback(None)
        assert sched.choose_clock(CONTROL_DT, rep)[0] == pytest.approx(DT0 / 2 ** (k + 1), rel=1e-12)
        rep = _report(sched, [3])
    assert not sched.rollback_requested(rep)                     # budget spent: accepted escalated
    assert sched._clock.j_floor == 0 and sched.graph_key() == key
    assert sched.choose_clock(CONTROL_DT, rep)[0] == pytest.approx(DT0 / 2, rel=1e-12)
    assert sched._clock.j_floor == 5
    # no rollback (so no graph invalidation by the driver) before this sticky step: the key itself must change, since a
    # later step's mode switch (staggered, ω0_strong: kernel inputs baked into a capture) keeps dt_x
    assert sched.graph_key() == (*key, "ladder", 1)


def test_switched_off_every_trip_is_sticky():
    sched = _sched(per_period_ladder=False)
    assert WatchdogPolicy().per_period_ladder and not sched._alloy_policy.watchdog.per_period_ladder
    dts, _ = _period(sched, None, trips=[1])
    assert dts == pytest.approx([DT0, DT0 / 2], rel=1e-12) and sched._clock.j_floor == 5
    assert isinstance(SchedulerPolicy().watchdog, WatchdogPolicy)


def _three_boxes(trip_period: int | None, periods: int = 6):
    """Three 5 kg boxes resting on 4 mm sheets (alloy.Scene, CPU, the production route). With ``trip_period``, the
    Scheduler's report of that period's first attempt says world 1 escalated (a synthetic trip: the real trigger is
    the watchdog's). Returns each period's (dt_x, substeps) and every world's box pose and sheet points."""
    import alloy

    side = (5.0 / 8000.0) ** (1.0 / 3.0)
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.102), resolution=(9, 9), mass=0.03, thickness=0.004,
                    self_contact=False)
    scene.add_box("box", size=(side,) * 3, pos=(0, 0, 0.104 + 0.001 + side / 2), mass=5.0)
    sim = alloy.build(scene, device="cpu")
    st, eng = sim.engine.stepper, sim.engine
    calls = {"n": 0}
    report = st.report

    def tripped():
        rep = report()
        calls["n"] += 1
        if trip_period is not None and eng.periods == trip_period and calls.get("tripped") is None:
            calls["tripped"] = True
            ws = np.asarray(rep.watch_state).copy()
            ws[1] = int(WatchState.ESCALATED)
            rep = dataclasses.replace(rep, watch_state=ws)
        return rep

    st.report = tripped
    clocks, states = [], []
    for _ in range(periods):
        sim.step(1)
        clocks.append((eng._last_coupling.dt_x, dict(eng._last_coupling.substeps)))
        s = sim.state(["box", "cloth"])
        states.append({k: np.array(v, copy=True) for k, v in s.items() if k != "t"})
    return clocks, states, sim


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_kept_worlds_are_bitwise_a_run_without_the_trip():
    """Law (CPU): world 1 trips once in period 2. Only world 1 re-runs that period, at dt_x / 2; the clock of every
    period is the no-trip run's; worlds 0 and 2 are bitwise the no-trip run's in every period; world 1 differs from
    period 2 on; the health report counts the refine for world 1 alone; the floor and mode stay."""
    base_clocks, base, _ = _three_boxes(None)
    clocks, run, sim = _three_boxes(2)
    trip = 2                                                     # the third period (periods counted from 0)
    assert clocks[:trip] + clocks[trip + 1:] == base_clocks[:trip] + base_clocks[trip + 1:]   # the scene's clock
    assert clocks[trip] == (pytest.approx(base_clocks[trip][0] / 2, rel=1e-12), base_clocks[trip][1])   # the re-run
    for p, (a, b) in enumerate(zip(run, base, strict=True)):
        for key in a:
            for w in (0, 2):
                np.testing.assert_array_equal(a[key][w], b[key][w], err_msg=f"period {p} {key} world {w}")
    assert any(not np.array_equal(run[-1][k][1], base[-1][k][1]) for k in run[-1])
    st = sim.engine.stepper
    assert st.period_refines() == [0, 1, 0] and sim.health()["refined_periods"] == [0, 1, 0]
    assert st._clock.j_floor == 0 and getattr(st, "_trips", 0) == 0 and sim.check()["ok"]
