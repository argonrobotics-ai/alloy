# SPDX-License-Identifier: Apache-2.0
"""Gap #10b P2 S2a, the reviews' laws for the two-pack driver (alloy/coupling/world_clock_run.py; [internal note]
scr/GAP-10B.md: WP4 items 1, 5(b) and 8, WP12's W2 and W3). Each was run on the driver without its fix first (the
answers section of that file gives the fail-before evidence). CPU, MuJoCo Warp + VBD; the rig is
alloy/testing/coupling_world_clock.py's (test_world_clock_run.py's)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

import alloy
from alloy.coupling.world_clock import Moves
from alloy.testing.coupling_world_clock import SEED, SLOTS, escalated, level, scene, state, tripping

PERIODS = 3


def _fine_clock(clock: Any) -> dict[str, Any]:
    """What a world entering the fine pack finds there: the fine clock's host attributes, the slot rows of the
    ladder and the force-jump trigger, and the parked mask the trigger reads."""
    fs = clock.fine_eng.stepper
    out = {f"clock.{k}": v for k, v in vars(fs._clock).items() if isinstance(v, (int, float, bool, type(None)))}
    for name in ("last", "per_world", "_floor_world"):
        out[f"ladder.{name}"] = np.array(getattr(fs._ladder, name, ()), copy=True).tolist()
    out["onset.last"] = np.array(fs._onset.last, copy=True).tolist()
    out["onset.parked"] = None if fs._onset.parked is None else fs._onset.parked.tolist()
    return out


def _run(sim: Any) -> list[dict[str, Any]]:
    clock, rows = sim.engine.world_clock, []
    for _ in range(PERIODS):
        sim.step(1)
        rows.append({"state": state(sim), "resident": clock.resident.tolist(), "level": level(sim),
                     "j_fine": sim.report()["world_clock"]["j_fine"]})
    return rows


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_reset_repeats_the_run_with_a_world_resident_from_the_first_period() -> None:
    """WP4 1 and 8: a world resides from period 0; N periods, ``sim.reset()``, N periods: the two runs are equal bit
    for bit (states, residents, both levels); after the reset the force-jump trigger's parked mask is the parked
    set in both packs (nobody in the main one, every idle slot in the fine one), the second run's first entry finds
    the fine clock as the first run's did, and the reset copies no resident back (it drops them)."""
    sim = alloy.build(scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)
    try:
        clock, found, copies = sim.engine.world_clock, [], []
        enter, leave = clock._enter, clock._leave

        def entered(world: int, slot: int) -> None:
            enter(world, slot)
            found.append(_fine_clock(clock))

        def left(slot: int, world: int, *, copy: bool = True) -> None:
            copies.append(copy)
            leave(slot, world, copy=copy)

        clock._enter, clock._leave = entered, left
        first = _run(sim)
        assert first[0]["resident"].count(True) == 1, first[0]["resident"]     # premise: resident from period 0
        n_first = len(found)
        copies.clear()
        sim.reset()
        assert copies and not any(copies), copies                       # item 8: dropped, not copied back
        st, fs = sim.engine.stepper, clock.fine_eng.stepper
        assert st._onset.parked is None and st._parked_mask() is None, st._onset.parked
        assert fs._onset.parked is not None and fs._onset.parked.tolist() == fs.parked.tolist() == [True] * SLOTS
        second = _run(sim)
        assert len(found) > n_first and found[n_first] == found[0], (found[0], found[n_first])
        for p, (a, b) in enumerate(zip(first, second, strict=True)):
            assert (a["resident"], a["level"], a["j_fine"]) == (b["resident"], b["level"], b["j_fine"]), p
            for key, v in a["state"].items():
                np.testing.assert_array_equal(v, b["state"][key], err_msg=f"period {p}: {key}")
    finally:
        sim.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_rerun_ask_that_not_every_world_can_enter_stays_todays_whole_rerun() -> None:
    """WP12 W2: when the fine pack cannot take every world of a re-run ask, the driver restores nobody: the coarse
    pack re-runs the period as without the option (one rollback; every world's state bit for bit today's) and the
    books count the fallback."""
    def setup(sim: Any) -> None:
        sim.engine.world_clock.clock.enter_now = lambda worlds: Moves(refused=tuple(int(w) for w in worlds))

    off, s0 = tripping(0, 1, 1, 2)
    on, s1 = tripping(SLOTS, 1, 1, 2, setup=setup)
    try:
        assert off[1]["rollbacks"] == 1 and on[1]["rollbacks"] == 1, (off[1]["rollbacks"], on[1]["rollbacks"])
        book = on[1]["book"]
        assert book["rerun_entries"] == 0 and not on[1]["resident"].any(), book
        assert sum(n for why, n in book["rerun_fallbacks"].items() if "no fine slot" in why) == 1, book
        for p in range(2):
            assert on[p]["level"] == off[p]["level"], p
            for k, v in on[p]["state"].items():
                np.testing.assert_array_equal(v, off[p]["state"][k], err_msg=f"period {p}: {k}")
    finally:
        s0.close()
        s1.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_transfer_refusal_turns_the_option_off_at_build(monkeypatch: Any, caplog: Any, capsys: Any) -> None:
    """WP12 W3: a world that cannot move between the two simulations is found at build: the build warns why, as for
    a scene that cannot take the option, and the simulation steps without it."""
    from alloy.coupling import world_clock_run
    from alloy.coupling.fine_pack import TransferError

    def refuse(*_: Any, **__: Any) -> Any:
        raise TransferError("planted: arrays without a per-world axis")

    monkeypatch.setattr(world_clock_run, "WorldTransfer", refuse)
    with caplog.at_level(logging.WARNING):
        sim = alloy.build(scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)
    try:
        said = caplog.text + capsys.readouterr().err
        assert "clock_worlds is off: transfer: planted" in said, said
        assert getattr(sim.engine, "world_clock", None) is None and getattr(sim.engine, "fine_rerun", None) is None
        sim.step(1)
        assert sim.check()["ok"] and "world_clock" not in sim.report()
    finally:
        sim.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_health_latched_world_never_enters() -> None:
    """WP4 5(b): the world that would reside from period 0 is latched by health (frozen, INVALID) before it: it stays
    in the main simulation, where its latch holds it."""
    sim = alloy.build(scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)
    try:
        clock = sim.engine.world_clock
        need = clock._need()
        world = int(np.argmax(need))
        assert np.count_nonzero(need == need.max()) == 1, need
        latched = sim.engine.health._latched.numpy().copy()
        latched[world] = 1
        sim.engine.health._latched.assign(latched)
        sim.step(1)
        assert not clock.resident[world] and clock.clock.enters == 0, (clock.resident, clock.clock.enters)
    finally:
        sim.close()


# --- WP10's should-fixes S1-S4 (the draft 744ce0f6f changed each; planted on the driver before it) ---------------------

def _resident(periods: int = 1) -> tuple[Any, Any, int, int]:
    """(sim, driver, world, slot): the rig's finest world resides from period 0, ``periods`` periods run."""
    sim = alloy.build(scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)
    sim.step(periods)
    clock = sim.engine.world_clock
    world = int(np.flatnonzero(clock.resident)[0])
    return sim, clock, world, int(clock.slot_of[world])


@pytest.mark.law
@pytest.mark.timeout(900)
def test_an_entrant_does_not_take_the_last_occupants_escalation() -> None:
    """S1: the fine pack's last report ended escalated in a slot (its last occupant's re-runs spent); the next world
    to enter that slot does not send the fine pack to the sticky ladder for it."""
    sim, clock, world, slot = _resident()
    try:
        clock.evict([world])
        fine = clock.fine_eng
        fine._last_coupling = escalated(fine._last_coupling, slot)
        fs = fine.stepper
        before = (int(getattr(fs, "_trips", 0)), int(fs._clock.j_floor))
        sim.step(1)
        assert clock.resident[world], clock.resident                     # premise: it entered again
        assert (int(getattr(fs, "_trips", 0)), int(fs._clock.j_floor)) == before
    finally:
        sim.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_residents_need_is_held_to_the_fine_floor() -> None:
    """S2: a resident whose slot's floor the fine pack raised is judged no coarser than that floor (it does not leave
    to a coarser clock than it was given)."""
    sim, clock, world, slot = _resident()
    try:
        fs = clock.fine_eng.stepper
        base = int(clock._need()[world])
        target = min(base + 1, int(fs._alloy_policy.clock.j_max))
        assert target > base, (base, target)
        rows = np.zeros(SLOTS, dtype=np.int64)
        rows[slot] = 1
        fs._ladder._floor_world, fs._clock.j_floor = rows, target
        assert int(clock._need()[world]) == target
    finally:
        sim.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_the_packs_watchdog_heads_are_checked_after_the_fine_period() -> None:
    """S3: the two packs end a period at the same watchdog window head (a resident's ring column reads at it), or the
    driver raises, as for the ticks."""
    sim = alloy.build(scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)      # a world resides from period 0
    try:
        fine = sim.engine.world_clock.fine_eng
        step = fine.step_period

        def skewed() -> None:
            step()
            fine.stepper._watchdog.head += 1

        fine.step_period = skewed
        with pytest.raises(Exception, match="watchdog heads"):
            sim.step(1)
    finally:
        sim.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_leaving_world_takes_its_watchdog_window_back() -> None:
    """S4: a resident's column of the watchdog's window ring as its own pack left it (here: refilled, as a fine
    rebase does) goes back to the main simulation with it."""
    sim, clock, world, slot = _resident(2)
    try:
        wf = clock.fine_eng.stepper._watchdog
        ring = wf.ring.numpy().copy()
        ring[:, slot] = 7.0 + np.arange(ring.shape[0])
        wf.ring.assign(ring)
        clock.evict([world])
        np.testing.assert_array_equal(sim.engine.stepper._watchdog.ring.numpy()[:, world], ring[:, slot])
    finally:
        sim.close()
