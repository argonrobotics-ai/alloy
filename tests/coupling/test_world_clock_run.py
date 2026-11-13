# SPDX-License-Identifier: Apache-2.0
"""Gap #10b P2 slice 2a (alloy/coupling/world_clock_run.py): the two-pack driver behind
``alloy.build(scene, clock_worlds=n)``, white box. CPU, MuJoCo Warp + VBD.

Rig (test_world_clock_laws.py's): 4 worlds, a 3 cm box on a sheet on a table, the box's density drawn per world, so
one world's own clock need is one level finer than every other's. The black-box laws are in that file."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import alloy
from alloy.testing.coupling_world_clock import NAMES, SEED, SLOTS, W
from alloy.testing.coupling_world_clock import escalated as _escalated
from alloy.testing.coupling_world_clock import level as _level
from alloy.testing.coupling_world_clock import scene as _scene
from alloy.testing.coupling_world_clock import state as _state
from alloy.testing.coupling_world_clock import tripping as _tripping


@pytest.fixture(scope="module")
def run() -> Any:
    """One 5-period run with the option: per period the states, the residents, both packs' levels and the books."""
    sim = alloy.build(_scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)
    start = _state(sim)
    rows = []
    for _ in range(5):
        sim.step(1)
        clock = sim.engine.world_clock
        rows.append({"state": _state(sim), "resident": clock.resident.copy(), "need": clock.need.copy(),
                     "level": _level(sim), "book": sim.report()["world_clock"], "ok": sim.check()["ok"],
                     "notes": sim.engine._last_coupling.notes})
    yield sim, start, rows
    sim.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_the_finest_world_resides_from_the_first_period_and_the_books_add_up(run: Any) -> None:
    """The one world whose need is finest takes the slot before the first period; each period takes
    W 2^Jc + n 2^Jf exchanges, no fewer than the worlds' own needs sum to, against W 2^Jf for one shared clock."""
    sim, _, rows = run
    need = rows[0]["need"]
    fine = int(np.argmax(need))
    assert np.count_nonzero(need == need.max()) == 1
    taken = shared = 0
    for p, row in enumerate(rows):
        assert row["resident"].tolist() == [w == fine for w in range(W)], p
        book = row["book"]
        j_c, j_f = book["j_coarse"], book["j_fine"]
        assert j_f == need[fine] and j_c == int(np.delete(row["need"], fine).max()), (p, j_c, j_f, row["need"])
        taken += W * 2 ** j_c + SLOTS * 2 ** j_f
        shared += W * 2 ** j_f
        assert book["exchanges"] == taken and book["exchanges_shared"] == shared, (p, book)
        assert taken >= (p + 1) * int(np.sum(2 ** row["need"]))             # the census's floor: no counting bug
    assert taken * 1.25 <= shared


@pytest.mark.law
@pytest.mark.timeout(900)
def test_the_coarse_clock_is_the_unparked_worlds_own_need(run: Any) -> None:
    """Every period the coarse pack's level is the finest need among the worlds it still steps (it does not remember
    the resident), and the report gives each world the step it ran at."""
    sim, _, rows = run
    for p, row in enumerate(rows):
        rest = ~row["resident"]
        assert row["level"] == int(row["need"][rest].max()), (p, row["level"], row["need"])
        dt = np.asarray(row["notes"]["dt_x_world"])
        assert np.all(dt[rest] == sim.dt / 2 ** row["level"]) and np.all(dt[~rest] == sim.dt / 2 ** row["book"]["j_fine"])


@pytest.mark.law
@pytest.mark.timeout(900)
def test_reads_are_the_fine_slots_and_health_stays_ok(run: Any) -> None:
    """A resident's read is its fine slot's state bit for bit, the coarse pack's health raises nothing for its parked
    slot, and the packs agree on the time."""
    sim, _, rows = run
    clock = sim.engine.world_clock
    assert all(row["ok"] for row in rows)
    for e in np.flatnonzero(clock.resident):
        fine = clock.fine.state(NAMES, world=int(clock.slot_of[e]))
        for k, v in _state(sim).items():
            np.testing.assert_array_equal(v[e], fine[k], err_msg=k)
    assert sim.t == clock.fine.t
    health = sim.health()
    assert not health["invalid_worlds"] and not health.get("degraded_worlds"), health


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_the_coarse_worlds_are_a_run_with_that_slot_parked(run: Any) -> None:
    """F1: the worlds left in the coarse pack are bit for bit a run without the option in which the resident's slot
    was parked in the same periods (here: from the first), and so is the coarse clock."""
    _, _, rows = run
    fine = int(np.flatnonzero(rows[0]["resident"])[0])
    ref = alloy.build(_scene(), device="cpu", seed=SEED)
    try:
        ref.engine.stepper.park_worlds([fine], ref.engine.state)
        rest = [w for w in range(W) if w != fine]
        for p, row in enumerate(rows):
            ref.step(1)
            assert _level(ref) == row["level"], p
            for k, v in _state(ref).items():
                np.testing.assert_array_equal(v[rest], row["state"][k][rest], err_msg=f"period {p}: {k}")
    finally:
        ref.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_reset_leaves_nobody_resident_and_every_world_at_its_start(run: Any) -> None:
    """``sim.reset()``: nobody resides, every fine slot is parked again, every world reads its start state, and the
    finest world enters again at the next period. (Whether the run after a reset repeats the first bit for bit is
    the reset law's, with or without the option.)"""
    sim, start, rows = run
    sim.reset()
    clock = sim.engine.world_clock
    assert not clock.resident.any()
    assert clock.fine.engine.stepper.parked.all() and not sim.engine.stepper.parked.any()
    for k, v in _state(sim).items():
        np.testing.assert_array_equal(v, start[k], err_msg=k)
    sim.step(1)
    assert clock.resident.tolist() == rows[0]["resident"].tolist()
    assert sim.check()["ok"]


@pytest.mark.law
@pytest.mark.parametrize("kw", [{"clock_worlds": True}, {"clock_worlds": W}, {"clock_worlds": -1},
                                {"clock_worlds": 1, "rerun_worlds": 1}, {"clock_worlds": 1, "processes": 2}])
def test_bad_values_are_build_errors(kw: dict[str, Any]) -> None:
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(_scene(), device="cpu", seed=SEED, **kw)
    assert err.value.field == "clock_worlds"


# --- S2b: a re-run ask is an entry ----------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_rerun_ask_enters_the_fine_pack_and_the_others_do_not_rerun() -> None:
    """Without the option one world's trip re-runs the period for all W worlds, one level finer. With it the world
    goes back to the period start alone and steps the period in the fine slot one level finer; the coarse pack does
    not re-run (no rollback, W 2^J exchanges that period), the other worlds end the period bit for bit as in a run
    with no trip (today's kept worlds keep their first run too), and the run takes fewer exchanges than today's. Once quiet, the world goes back to the coarse pack."""
    world, period = 1, 2
    off, s0 = _tripping(0, world, period)
    on, s1 = _tripping(SLOTS, world, period)
    try:
        rest = [w for w in range(W) if w != world]
        assert off[period]["rollbacks"] == 1 and on[period]["rollbacks"] == 0
        j = on[period]["level"]
        assert on[period]["taken"] - on[period - 1]["taken"] == W * 2 ** j           # the coarse pack ran it once
        assert on[period]["resident"].tolist() == [w == world for w in range(W)]
        assert on[period]["dt"][world] == s1.dt / 2 ** (j + 1) and np.all(on[period]["dt"][rest] == s1.dt / 2 ** j)
        book = on[period]["book"]
        assert book["rerun_entries"] == 1 and book["refused"] == 0 and s1.report()["fine_reruns"] == 1
        for p in range(period + 1):                              # kept worlds: the same bits either way
            for k, v in on[p]["state"].items():
                np.testing.assert_array_equal(v[rest], off[p]["state"][k][rest], err_msg=f"period {p}: {k}")
        assert on[-1]["book"]["exchanges"] < off[-1]["taken"]
        assert not on[-1]["resident"].any()                      # quiet again: back in the coarse pack
        assert all(r["ok"] for r in on)
        z_on, z_off = on[-1]["state"]["box.pos"][world, 2], off[-1]["state"]["box.pos"][world, 2]
        assert abs(z_on - z_off) < 1e-3                          # the tripped world rests where today's leaves it
    finally:
        s0.close()
        s1.close()


# --- WP10's review of S2a ([internal note] scr/GAP-10B.md): M1-M3 and Q2, each planted on the unfixed driver first --------

def _fine_stepping(sim: Any) -> tuple[Any, list[bool]]:
    """(the fine pack's stepper, a flag list that is [True] while the fine pack steps its period)."""
    fine = sim.engine.world_clock.fine_eng
    step, inside = fine.step_period, []

    def stepped() -> None:
        inside[:] = [True]
        try:
            step()
        finally:
            inside.clear()

    fine.step_period = stepped
    return fine.stepper, inside


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_sticky_escalation_inside_a_rerun_entrys_period_survives_the_period() -> None:
    """M1: world 1's re-run ask in period 1 enters the fine pack one level finer, and the fine pack's own first
    attempt of that period trips too, under today's ladder (``per_period_ladder`` off: every trip sticky), so its
    floor rises for good. The period's end puts back only the entry's one-period level, not that rise."""
    seen: dict[str, Any] = {}

    def setup(sim: Any) -> None:
        clock = sim.engine.world_clock
        fs, inside = _fine_stepping(sim)
        fs._ladder.on = False
        report, escalate = fs.report, fs._clock.escalate

        def tripped() -> Any:
            rep = report()
            if inside and "tripped" not in seen:
                seen["tripped"] = True
                rep = _escalated(rep, int(clock.slot_of[1]))
            return rep

        def escalated() -> bool:
            out = escalate()
            seen["floor"] = int(fs._clock.j_floor)
            return out

        fs.report, fs._clock.escalate = tripped, escalated

    on, sim = _tripping(SLOTS, 1, 1, 2, setup=setup)
    try:
        clock = sim.engine.world_clock
        fs = clock.fine_eng.stepper
        assert on[1]["book"]["rerun_entries"] == 1 and "floor" in seen, (on[1]["book"], seen)
        assert seen["floor"] == on[1]["level"] + 2, seen          # the entry ran at j + 1; the sticky rise is j + 2
        assert (int(fs._clock.j_floor), int(fs._trips)) == (seen["floor"], 1), (fs._clock.j_floor, seen)
        assert clock._floor is None
    finally:
        sim.close()


REPEAT = "a repeat trip"


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_repeat_trip_is_refused_and_the_coarse_floor_rises_as_without_the_option() -> None:
    """M2: world 1 trips in periods 1 and 2. Without the option the second trip, within the watchdog window of the
    first, takes the sticky ladder (the coarse floor rises for good). With it the first trip is an entry, the world
    leaves at the next period, and its second ask is refused as a repeat trip (``rerun_fallbacks`` counts it): the
    coarse pack re-runs and its floor and trips go as without the option."""
    off, s0 = _tripping(0, 1, (1, 2), 3)
    on, s1 = _tripping(SLOTS, 1, (1, 2), 3)
    try:
        assert off[2]["floor"] > off[1]["floor"] and off[2]["trips"] == 1          # premise: today's 2nd trip sticks
        assert on[1]["book"]["rerun_entries"] == 1 and on[1]["resident"][1]
        fallbacks = on[2]["book"]["rerun_fallbacks"]
        assert sum(n for why, n in fallbacks.items() if why.startswith(REPEAT)) == 1, fallbacks
        assert on[2]["book"]["rerun_entries"] == 1 and not on[2]["resident"].any() and on[2]["rollbacks"] == 1
        assert [(r["floor"], r["trips"]) for r in on] == [(r["floor"], r["trips"]) for r in off]
        assert all(r["ok"] for r in on)
    finally:
        s0.close()
        s1.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_an_entrant_runs_no_entry_at_an_own_step_coarser_than_in_the_pack_it_left() -> None:
    """M3: in the entrant's first fine period every estimating entry's own step (dt_x / substeps) is no coarser than
    in the coarse pack's period it left. Rig: MuJoCo's stable-step estimate is pinned (both packs) at a third of the
    coarse dt_x over its safety, so the coarse pack runs it at 4 substeps; world 1 trips in period 1 and enters."""
    from types import SimpleNamespace

    def setup(sim: Any) -> None:
        st = sim.engine.stepper
        safety = st._clock.safety_for.get("mujoco", st._clock.policy.safety)
        value = sim.dt / 2**6 / (3.0 * safety)                  # the rig's coarse level is j 6
        pinned = SimpleNamespace(alloy_stable_dt=lambda state, contacts, out: out.fill_(value))
        for stepper in (st, sim.engine.world_clock.fine_eng.stepper):
            stepper._estimators["mujoco"] = pinned

    on, sim = _tripping(SLOTS, 1, 1, 2, setup=setup)
    try:
        coarse, fine = sim.engine._last_coupling, sim.engine.world_clock.fine_eng._last_coupling
        assert on[1]["book"]["rerun_entries"] == 1 and coarse.substeps["mujoco"] == 4, (on[1]["book"], coarse.substeps)
        for name, n in fine.substeps.items():
            own, there = fine.dt_x / n, coarse.dt_x / coarse.substeps[name]
            assert own <= there * (1.0 + 1.0e-9), (name, fine.dt_x, dict(fine.substeps), coarse.dt_x,
                                                   dict(coarse.substeps))
    finally:
        sim.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_failed_fine_period_leaves_the_residents_mirrored_and_no_floor_set() -> None:
    """Q2: the fine pack's period raises (here after it stepped, as an INVALID world under the precision profile
    raises): the error reaches the caller, the re-run entry's one-period floor is put back and the residents are
    mirrored (a read is the fine slot's state, an INVALID latch included), and a reset then starts clean."""
    held: dict[str, Any] = {}

    def setup(sim: Any) -> None:
        fine = sim.engine.world_clock.fine_eng
        step = fine.step_period
        held["sim"], held["start"] = sim, _state(sim)

        def failing() -> None:
            step()
            raise RuntimeError("injected fine-pack failure")

        fine.step_period = failing

    with pytest.raises(Exception, match="injected fine-pack failure"):
        _tripping(SLOTS, 1, 1, 2, setup=setup)
    sim = held["sim"]
    try:
        clock = sim.engine.world_clock
        assert clock.resident.tolist() == [w == 1 for w in range(W)]
        assert clock._floor is None and int(clock.fine_eng.stepper._clock.j_floor) == 0
        fine = clock.fine.state(NAMES, world=int(clock.slot_of[1]))
        for k, v in _state(sim).items():
            np.testing.assert_array_equal(v[1], fine[k], err_msg=k)
        sim.reset()
        assert not clock.resident.any()
        for k, v in _state(sim).items():
            np.testing.assert_array_equal(v, held["start"][k], err_msg=k)
    finally:
        sim.close()
