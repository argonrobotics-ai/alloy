# SPDX-License-Identifier: Apache-2.0
"""Per-world exchange clock, the two-pack driver (known gap coupling-shared-exchange-clock; opt-in
``alloy.build(scene, clock_worlds=n)``; [internal note] notes/wp4-perworld-clock/DESIGN.md sections 17-24).

Two simulations of one scene: the coarse pack of all W worlds and a fine pack of n slots. A world whose own clock need
is finer than the rest's (``world_clock_need``) *resides* in a fine slot for as long as that lasts
(``world_clock.WorldClock`` decides who). Its coarse slot is parked (``parked.py``: held, out of every clock bound), so
the coarse pack runs at the rung of the worlds left in it, and the fine pack runs its own Scheduler clock over its
residents (its idle slots are parked too).

One control period (``Simulation.step_period`` calls ``begin_period`` first and ``end_period`` last):
1. need: each world's own level from its interface bound and its band rows (a resident's band rows are its slot's);
2. ``WorldClock.update`` -> moves. A leaver goes fine -> coarse (whole world: state, stepper rows, health; unpark),
   then an enterer coarse -> fine (with its randomized parameters and watchdog reference; park);
3. the residents' Control rows go coarse -> fine;
4. the coarse pack steps its period (its own loop and re-runs), then the fine pack steps the same period on its own
   clock, from the same tick;
5. mirror: every resident's world goes fine -> coarse and the parked slot's held copy is re-taken
   (``refresh_parked``), so every read (state, health, check, render) sees it;
6. the period's report is the coarse pack's with the residents' rows from their slots; ``notes["world_clock"]`` and
   ``sim.report()["world_clock"]`` carry the books.

A write to a resident (``set_pose``, ``settle``, a held wrench) evicts it first (whole world back, unpark): the write
then lands on the coarse pack as without the option, and the world may enter again at the next period. ``reset``
evicts without a copy and starts the fine pack over.

A re-run ask is an entry (``try_entry``, called where the coarse pack would re-run a period for a few worlds E): E goes
back to the period start alone and into free fine slots, the other worlds keep their first run, and the fine pack
steps the period one level finer than the coarse pack ran it (its own trigger takes it finer still if needed). With
no free slot, for a re-run of every world, or for a world's repeat trip within the watchdog window (the entrant's trip
is kept in its ladder rows), the coarse pack re-runs as without the option and its sticky ladder acts.

A re-run ask that not every world of E can enter, and a world latched by health, stay where they are (WP12 W2, WP4
5(b)); a moving of worlds that the two packs cannot do is found at build, which then turns the option off (W3).
Not here yet: ``world_clock_parts.LIMITS`` (``report()["limits"]``).
Laws: tests/coupling/test_world_clock_run.py, test_world_clock_review.py, test_world_clock_laws.py.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import warp as wp

from alloy.core.errors import CouplingError
from alloy.coupling.capture_passes import rerun_mask
from alloy.coupling.fine_pack import TransferError, WorldTransfer
from alloy.coupling.fine_rerun import ROWS, _refusal, _sized, _wrench_held
from alloy.coupling.parked import quiet, seed_stable
from alloy.coupling.types import WatchState
from alloy.coupling.world_clock import WorldClock, WorldClockConfig
from alloy.coupling.world_clock_params import move_watch_reference
from alloy.coupling.world_clock_need import AttemptLevels, ClockGrid, per_world_rungs
from alloy.coupling.world_clock_parts import LIMITS, Books, merge_packs, scene_refusal

log = logging.getLogger("alloy.coupling")

REPEAT = "a repeat trip within the watchdog window: the sticky ladder, as without the option"


class WorldClockRun:
    """The driver on the coarse simulation (``engine.world_clock``). ``fine``: the fine pack's API ``Sim``."""

    def __init__(self, eng: Any, fine: Any, config: WorldClockConfig) -> None:
        self.eng, self.fine, self.fine_eng = eng, fine, fine.engine
        self.W, self.slots = int(eng.model.world_count), int(config.slots)
        self.clock = WorldClock(self.W, config)
        self.books = Books()
        self.why: str | None = None             # why nobody may enter right now (a held wrench, observers, ...)
        self._transfers: tuple[Any, WorldTransfer, WorldTransfer] | None = None
        self._start: tuple[int, int, int] = (0, 0, 0)
        self._census = (0, 0)
        self.need = np.zeros(self.W, dtype=np.int64)
        self.reruns = 0                         # periods a re-run ask became an entry (``fine_reruns`` in the report)
        self.fallbacks: dict[str, int] = {}     # why a re-run stayed whole -> count
        self._floor: tuple[int, int] | None = None   # a re-run entry's period: the fine floor before it and its level
        self._watch_start: tuple[int, np.ndarray] | None = None
        eng.fine_rerun = self                   # Simulation.step_period asks here before it re-runs a period
        self.fine_eng.latch_log = None          # the coarse pack polls, after the mirror
        for st in (eng.stepper, self.fine_eng.stepper):
            st.keep_world_rows(True)            # each world's own band need (off without the option)
        self._park_idle()

    # ------------------------------------------------------------------------------------------------ reads
    @property
    def resident(self) -> np.ndarray:
        return self.clock.resident

    @property
    def slot_of(self) -> np.ndarray:
        return self.clock.slot_of

    def report(self) -> dict[str, Any]:
        b = self.books
        return {"slots": self.slots, "residents": np.flatnonzero(self.resident).tolist(),
                "exchanges": b.exchanges, "exchanges_shared": b.shared, "periods": b.periods,
                "resident_periods": b.resident_periods, "enters": self.clock.enters, "leaves": self.clock.leaves,
                "evictions": b.evictions, "refused": b.refused, "rerun_entries": b.rerun_entries,
                "rerun_fallbacks": dict(self.fallbacks), "j_coarse": b.j_coarse, "j_fine": b.j_fine,
                "need": self.need.tolist(), "paused": self.why, "limits": list(LIMITS)}

    # ------------------------------------------------------------------------------------------------ the period
    def begin_period(self) -> None:
        """Before the coarse pack's period: decide, move, and hand the residents their controls."""
        eng = self.eng
        self.why = self._pause_reason()
        if self.why is not None:
            self.evict()
        else:
            self.need = self._need()
            moves = self.clock.update(self.need, self._eligible())
            for slot, world in moves.leave:
                self._leave(slot, world)
            for world, slot in moves.enter:
                self._enter(world, slot)
        self._start = (eng.clock.ticks, eng.steps, eng.periods)
        watch = eng.stepper._watchdog
        self._watch_start = (watch.head, watch.elapsed_ns.numpy().copy())
        self._census = (self._taken(eng), self._taken(self.fine_eng))
        if self.resident.any():
            there = self._pair()[1]
            for world in np.flatnonzero(self.resident):
                there.copy_controls(int(world), int(self.slot_of[world]))
            self._align_watch()

    def end_period(self) -> None:
        """After the coarse pack's period: the fine pack steps it, the residents are mirrored, the report merged."""
        eng, fine = self.eng, self.fine_eng
        coarse = eng._last_coupling
        j_c = self._level(coarse)
        worlds = np.flatnonzero(self.resident)
        j_f = j_c
        if worlds.size:
            fine.clock.ticks, fine.steps, fine.periods = self._start
            try:
                fine.step_period()
            except Exception:
                self._mirror(worlds, self.slot_of[worlds])   # Q2: reads see where they got to (a health latch too)
                raise
            finally:
                self._end_floor()
            ends = (fine.clock.ticks, eng.clock.ticks, fine.stepper._watchdog.head, eng.stepper._watchdog.head)
            if ends[0] != ends[1] or ends[2] != ends[3]:      # S3: the ring heads too (a resident's column reads)
                raise CouplingError(f"per-world clock: the packs ended the period at different ticks or watchdog "
                                    f"heads (fine, coarse: ticks {ends[:2]}, heads {ends[2:]})")
            slots = self.slot_of[worlds]
            self._mirror(worlds, slots)
            j_f = self._level(fine._last_coupling)
            eng._last_coupling = self._merged(coarse, fine._last_coupling, worlds, slots, j_f)
        self._book(j_c, j_f, bool(worlds.size))
        notes = getattr(eng._last_coupling, "notes", None)
        if notes is not None:
            notes["world_clock"] = {"resident": self.resident.copy(), "j_coarse": j_c, "j_fine": j_f}

    # ------------------------------------------------------------------------------------------------ re-run asks
    def __call__(self, eng: Any, report: Any, start: tuple[int, int]) -> Any:
        return self.try_entry(report)

    def try_entry(self, report: Any) -> Any:
        """The coarse pack asks to re-run this period for the worlds E of ``report``. When E can take free fine
        slots: E alone goes back to the period start and enters them, and ``report`` comes back (the period stands
        for every other world; ``end_period`` steps E in the fine pack). Else None: today's re-run, untouched."""
        eng, mask = self.eng, rerun_mask(report)
        if mask is None:
            return None
        E = np.asarray(mask).astype(bool)
        free = int((self.clock.world_in < 0).sum())
        why = self._pause_reason() or self._repeat(E) or _refusal(eng, eng.stepper, report, E, free)
        if why is None:
            try:
                self._pair()
            except TransferError as exc:
                why = f"transfer: {exc}"
        if why is not None:
            return self._fallback(why, E)
        idle = not self.resident.any()
        moves = self.clock.enter_now(np.flatnonzero(E))
        if len(moves.enter) != int(E.sum()):             # W2: every world of E enters or none does, before a restore
            for world, _ in moves.enter:
                self.clock.release(world)
            return self._fallback("a world of the re-run found no fine slot", E)
        back = eng._mask(E.astype(np.int32))
        eng._good.restore(eng.state, back)
        if eng._health_snap is not None:
            eng._health_snap.restore(eng.state, back)
        fs = self.fine_eng.stepper
        if idle and self._watch_start is not None:       # the fine pack's window as the period began
            fs._watchdog.head = self._watch_start[0]
            fs._watchdog.elapsed_ns.assign(self._watch_start[1])
        for world, slot in moves.enter:
            self._enter(world, slot)
            last = _sized(fs._onset, "last", ROWS["_onset"]["last"], self.slots)
            last[slot] = ROWS["_onset"]["last"]          # armed: the fine pack's own trigger sees the jump
            fs._onset.last = last
            self._note_trip(slot)
            self._pair()[1].copy_controls(world, slot)
        clock, j_max = fs._clock, int(fs._alloy_policy.clock.j_max)
        saved = int(clock.j_floor) if self._floor is None else self._floor[0]
        clock.j_floor = max(int(clock.j_floor), min(self._level(report) + 1, j_max))
        self._floor = (saved, int(clock.j_floor))
        self.reruns += 1
        self.books.rerun_entries += len(moves.enter)
        log.info("per-world clock: worlds %s re-run this period in the fine pack", np.flatnonzero(E).tolist())
        return report

    # ------------------------------------------------------------------------------------------------ writes
    def evict(self, worlds: Any = None, *, copy: bool = True) -> None:
        """Residents (``worlds``: ids or a bool mask; None: all) go back to the coarse pack now, before a write to
        them. ``copy=False`` drops their fine state (a reset writes the start state over it)."""
        mask = self.resident.copy()
        if worlds is not None:
            pick = np.zeros(self.W, dtype=bool)
            pick[np.asarray(worlds).astype(bool) if np.asarray(worlds).dtype == bool
                 else np.asarray(worlds, dtype=np.int64).reshape(-1)] = True
            mask &= pick
        for world in np.flatnonzero(mask):
            slot, _ = self.clock.release(int(world))
            self._leave(slot, int(world), copy=copy)
            self.books.evictions += 1

    def on_reset(self, world_mask: Any = None) -> None:
        """``engine.reset(world_mask)``, before it runs: the reset worlds leave with no copy; a reset of every world
        also starts the fine pack and the residency rule over. A re-run entry's one-period floor is put back first."""
        self._end_floor()
        ladder = getattr(self.eng.stepper, "_ladder", None)
        if hasattr(ladder, "clear_worlds"):
            ladder.clear_worlds(None if world_mask is None else (
                world_mask.numpy() if isinstance(world_mask, wp.array) else np.asarray(world_mask)).astype(bool))
        if world_mask is None:
            self.evict(copy=False)
            self.clock.reset()
            self.fine_eng.reset()
            self._park_idle()
            return
        mask = world_mask.numpy() if isinstance(world_mask, wp.array) else np.asarray(world_mask)
        self.evict(mask[: self.W].astype(bool), copy=False)

    def close(self) -> None:
        self.fine.close()

    # ------------------------------------------------------------------------------------------------ parts
    def _fallback(self, why: str, E: np.ndarray) -> None:
        """A re-run ask stays today's whole re-run (``report()["world_clock"]["rerun_fallbacks"]`` counts why)."""
        self.fallbacks[why] = self.fallbacks.get(why, 0) + 1
        self.books.refused += int(E.sum())

    def transfer_refusal(self) -> str | None:
        """Why a world cannot move between the two packs (tried at build: WP12's W3), or None."""
        try:
            self._pair()
        except TransferError as exc:
            return f"transfer: {exc}"
        return None

    def _pause_reason(self) -> str | None:
        eng = self.eng
        if getattr(eng, "_observers", None) or getattr(self.fine_eng, "_observers", None):
            return "step observers (a drive): their clocks do not move with a world"
        if _wrench_held(eng):
            return "a held wrench: its work is booked on the coarse simulation"
        return None

    def _repeat(self, E: np.ndarray) -> str | None:
        """A world of E tripped within the watchdog window of an earlier period's trip, or the ladder is today's (every
        trip sticky): the sticky ladder must act, so the coarse pack re-runs as without the option (WP10's M2)."""
        ladder = self.eng.stepper._ladder
        return REPEAT if not ladder.on or ladder.repeats(E, float(self.eng.clock.control_dt)) else None

    def _note_trip(self, slot: int) -> None:
        """An entrant's trip in its slot's ladder rows, as ``PeriodLadder.within_period`` writes a re-run's (M2): dated
        the fine period about to run (the coarse ladder has begun this period, the fine one not yet)."""
        ladder = self.fine_eng.stepper._ladder
        last, runs = (_sized(ladder, n, ROWS["_ladder"][n], self.slots) for n in ("last", "per_world"))
        last[slot], runs[slot] = ladder.period + 1, runs[slot] + 1
        ladder.last, ladder.per_world = last, runs

    def _eligible(self) -> np.ndarray:
        """Worlds that may reside: not FAILED, not latched by health (WP4 5(b)): such a world is read where it is."""
        ok = np.ones(self.W, dtype=bool)
        latched = getattr(getattr(self.eng, "health", None), "_latched", None)
        if latched is not None:
            ok &= latched.numpy()[: self.W] == 0
        last = self.eng._last_coupling
        ws = np.asarray(getattr(last, "watch_state", ()))
        if ws.shape == (self.W,):
            ok &= ws != int(WatchState.FAILED)
        try:
            self._pair()
        except TransferError as exc:
            if self.resident.any():
                raise
            self.why = f"transfer: {exc}"
            ok[:] = False
        return ok

    def _need(self) -> np.ndarray:
        """Each world's own level (no attempt extras): its interface bound and its band rows, at the coarse pack's
        sticky floor; a resident's band rows are its slot's, and it is held to the fine pack's floor too."""
        st, fs = self.eng.stepper, self.fine_eng.stepper
        policy = st._alloy_policy.clock
        grid = ClockGrid(float(self.eng.clock.control_dt), policy.dt_x_max, policy.j_max)
        band = np.full(self.W, math.inf)
        rows = st._band.world_bounds(need=True)
        if rows is not None and rows.shape == (self.W,):
            band = rows.copy()
        worlds = np.flatnonzero(self.resident)
        theirs = fs._band.world_bounds(need=True)
        if worlds.size and theirs is not None:
            band[worlds] = theirs[self.slot_of[worlds]]
        floor = int(getattr(st._clock, "j_floor", 0))
        row = getattr(getattr(st, "_ladder", None), "_floor_world", None)
        row = row if row is not None and row.shape == (self.W,) else None
        need = per_world_rungs(st._world_bounds(self.W, None), band, grid, AttemptLevels(floor), floor_world=row).j
        mine = _sized(fs._ladder, "_floor_world", 1, self.slots)[self.slot_of[worlds]] > 0   # S2: and the fine floor
        need[worlds] = np.maximum(need[worlds], np.where(mine, int(fs._clock.j_floor), 0))
        return need

    def _pair(self) -> tuple[Any, WorldTransfer, WorldTransfer]:
        """(key, coarse -> fine, fine -> coarse) transfers, rebuilt when either pack's array set changed (a solver
        allocates some lazily)."""
        key = (tuple(sorted(self.eng.state_arrays())), tuple(sorted(self.fine_eng.state_arrays())))
        if self._transfers is None or self._transfers[0] != key:
            self._transfers = (key, WorldTransfer(self.eng, self.fine_eng),
                               WorldTransfer(self.fine_eng, self.eng, params="same"))
        return self._transfers

    def _park_idle(self) -> None:
        """Every idle fine slot is parked: it bounds no clock and asks for nothing."""
        idle = self.clock.world_in < 0
        if idle.any():
            self.fine_eng.stepper.park_worlds(idle, self.fine_eng.state)

    def _host_rows(self, src: Any, dst: Any, a: int, b: int) -> None:
        """The ladder's and the force-jump trigger's per-world rows of ``src``'s world ``a`` -> ``dst``'s ``b``; a
        period index moves onto ``dst``'s own period count."""
        for name, rows in ROWS.items():
            so, do = getattr(src, name), getattr(dst, name)
            shift = int(do.period) - int(so.period)
            for row, default in rows.items():
                out = _sized(do, row, default, int(dst.model.world_count))
                value = int(_sized(so, row, default, int(src.model.world_count))[a])
                out[b] = value + shift if row == "last" and value > default // 2 else value
                setattr(do, row, out)

    def _enter(self, world: int, slot: int) -> None:
        eng, fine = self.eng, self.fine_eng
        st, fs = eng.stepper, fine.stepper
        there = self._pair()[1]
        fs.unpark_worlds([slot])
        there.copy(world, slot)
        seed_stable(fs, slot, st._stable_host)           # its first fine period: no entry's step coarser (M3)
        fine._last_coupling = quiet(fine._last_coupling, np.arange(self.slots) == slot)   # S1: not the last occupant's
        self._host_rows(st, fs, world, slot)
        fs._census_mass = None                           # the slot's proxies carry the world's masses now
        if getattr(fine, "_graphs", None) is not None:   # a parameter move: the captures may bake the old values
            fine._graphs.invalidate()
            fine.exchange_nodes.clear()
        st.park_worlds([world], eng.state)
        log.info("per-world clock: world %d resides in fine slot %d (need %d)", world, slot, int(self.need[world]))

    def _leave(self, slot: int, world: int, *, copy: bool = True) -> None:
        eng, fine = self.eng, self.fine_eng
        st, fs = eng.stepper, fine.stepper
        st.unpark_worlds([world])
        if copy:
            self._pair()[2].copy(slot, world)
            move_watch_reference(fine, eng, slot, world)   # S4: its ring column as its own pack left it (rebases)
            self._host_rows(fs, st, slot, world)
        fs.park_worlds([slot], fine.state)
        log.info("per-world clock: world %d left fine slot %d", world, slot)

    def _end_floor(self) -> None:
        """A re-run entry's level was for its period only; a sticky rise above it that the fine pack's own ladder made
        in that period stays (WP10's M1)."""
        if self._floor is None:
            return
        (saved, level), self._floor = self._floor, None
        clock = self.fine_eng.stepper._clock
        if int(clock.j_floor) <= level:
            clock.j_floor = saved
            clock.jump_next()

    def _align_watch(self) -> None:
        """The watchdog's window head and elapsed time are a pack's: the fine pack takes the coarse pack's before both
        step the period, so a resident's ring column reads as in its own pack."""
        wc, wf = self.eng.stepper._watchdog, self.fine_eng.stepper._watchdog
        wf.head = wc.head
        wp.copy(wf.elapsed_ns, wc.elapsed_ns)

    def _mirror(self, worlds: np.ndarray, slots: np.ndarray) -> None:
        back = self._pair()[2]
        for world, slot in zip(worlds, slots, strict=True):
            back.copy(int(slot), int(world))
        self.eng.stepper.refresh_parked(worlds, self.eng.state)

    def _merged(self, coarse: Any, fine: Any, worlds: np.ndarray, slots: np.ndarray, j_f: int) -> Any:
        merged = merge_packs(coarse, fine, worlds, slots, (self.W, self.slots))
        notes = merged.notes if merged.notes is not None else {}
        census = notes.get("census")
        if isinstance(census, dict) and len(census.get("j_world", ())) == self.W:
            j_world = list(census["j_world"])
            for world in worlds:
                j_world[int(world)] = j_f                  # as it ran: the fine pack's level
            notes["census"] = {**census, "j_world": j_world}
        dt_world = np.full(self.W, float(coarse.dt_x))
        dt_world[worlds] = float(fine.dt_x)
        notes["dt_x_world"] = dt_world
        return merged

    def _level(self, report: Any) -> int:
        dt_x = float(getattr(report, "dt_x", 0.0) or 0.0)
        return int(round(math.log2(float(self.eng.clock.control_dt) / dt_x))) if dt_x > 0.0 else 0

    @staticmethod
    def _taken(eng: Any) -> int:
        census = getattr(eng.stepper, "clock_census", None)
        return int((census() if census is not None else {}).get("world_exchanges", 0))

    def _book(self, j_c: int, j_f: int, resident: bool) -> None:
        """Count the period: both packs' exchanges as their censuses counted them (re-runs included), and a shared
        clock's: every world at the finer of the two levels, plus the coarse pack's re-run exchanges."""
        b = self.books
        coarse = self._taken(self.eng) - self._census[0]
        fine = self._taken(self.fine_eng) - self._census[1]
        b.exchanges += coarse + fine
        b.shared += self.W * 2 ** max(j_c, j_f) + max(coarse - self.W * 2 ** j_c, 0)
        b.periods += 1
        b.resident_periods += int(resident)
        b.j_coarse, b.j_fine = j_c, j_f


def install(eng: Any, fine: Any, slots: int) -> WorldClockRun:
    """Put the driver on ``eng`` (``engine.world_clock``); ``fine``: the fine pack's API ``Sim`` of ``slots`` worlds."""
    eng.world_clock = WorldClockRun(eng, fine, WorldClockConfig(slots=int(slots)))
    return eng.world_clock


def uninstall(eng: Any) -> None:
    """Take the driver off ``eng`` before any period (a refusal found at build): the option is off, the second
    simulation freed."""
    run, eng.world_clock, eng.fine_rerun = eng.world_clock, None, None
    eng.stepper.keep_world_rows(False)
    run.close()


__all__ = ["LIMITS", "WorldClockRun", "install", "merge_packs", "scene_refusal", "uninstall"]
