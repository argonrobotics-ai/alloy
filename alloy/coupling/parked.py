# SPDX-License-Identifier: Apache-2.0
"""Parked worlds: slots of a Scheduler that are held still and kept out of its shared clock (gap #10b P2, slice 1;
[internal note] notes/wp4-perworld-clock/DESIGN.md section 16).

A world whose own need holds the exchange clock fine is to live in a second, finer simulation (``fine_pack.py``) for
as long as that lasts. Its slot in this pack is then *parked*. This module is the parked slot alone: no second pack
and no move between packs yet. A parked world (``Scheduler.park_worlds``; none by default):

- **holds no clock bound.**
  - The interface bound takes each pair's lightest proxy among the other worlds (``_clock_mass``).
  - The contact gate's arming and bound skip its proxies (``ContactGate.park``).
  - The contact band skips its particles (``ContactBandBound.park``).
  - Each entry's stable step is the minimum over the other worlds (``stable_host``), so the substep rungs and the
    graph-budget cap bound follow them alone.
- **asks for no re-run.**
  - The force-jump trigger never arms it (``OnsetRefiner.parked``).
  - The gate ignores a touch of its unarmed proxies.
  - The clock and the rollback rule read its report rows quiet (``quiet``: watch state OK, not short of passes), so it
    trips no ladder, takes no rollback budget and sends no world back.
- **is held.** After every exchange its State arrays and the stepper's ``state_arrays()`` (the coupling carry, the
  watchdog, the solvers' adapters) go back to what they were when it was parked. That is how the health latch holds
  an INVALID world, here from inside the stepper. The solvers still step it from the held state every exchange: the
  compute is spent, and nothing it does in one exchange lasts.
- **is marked.** The report says which worlds are parked (``notes["parked"]``, bool [W]), their stable-step rows read
  inf, and the census gives them ``clock_census.PARKED_RUNG`` in place of a rung.

Order inside an exchange (``step``): the solvers and the coupling passes, the watchdog's update (it may write any
watch state for a parked world), the estimators and the band, then the hold, which puts the parked rows back. The
report is read after the period's last exchange, so it reads held rows; ``quiet`` covers the host writes in between
(a whole re-run marks every world rolled back until its first exchange). A parked world therefore never reads
escalated, and reads FAILED only if it was FAILED when parked.

``refresh_parked`` re-takes the held copy after a driver wrote a parked slot's rows from outside (the world's real
state, mirrored back from the pack it lives in, so reads see it); without it the next exchange's hold undoes the
write.

``unpark_worlds`` returns a world to normal stepping from its held state. Its own clock stood still: its force-jump
cooldown and its last ladder trip move forward by the periods it was parked, so neither ran out while it was held, and
the watchdog's energy reference and work sums are the held ones, which match the held state.

Parking or unparking changes which worlds the bounds cover, so the next period's clock goes straight to the level
they ask for (``ClockController.jump_next``, as after a gate change) instead of waiting out the quiet periods. A park
by ``park_latched`` (ALLOY_PARK_LATCHED, ``follow=True``) also re-takes the band bound over the worlds left and takes
the solvers' substep rungs straight to what their estimators want (a rung a frozen world had raised drops at once; a
dt_x made finer still keeps each entry's own step).

With no world parked nothing here runs: every mask point takes its old branch on a ``None`` mask, the hold launches
nothing and no host read is added. A capture bakes the hold in, so periods with a parked world are graphs of their own
(``graph_key``); changing which worlds are parked needs no new capture (the mask is a device array).

What a driver must not expect: writers that run after ``stepper.step`` (a topology controller, step observers such as
``sim.drive``) are outside this hold.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import numpy as np
import warp as wp

from alloy.core.errors import CouplingError
from alloy.core.snapshot import WorldSnapshot
from alloy.coupling.match_reset import world_bools
from alloy.coupling.types import WatchState

log = logging.getLogger("alloy.coupling")

__all__ = ["ParkedWorlds", "ParkedWorldsMixin", "band_over", "quiet", "seed_stable", "stable_host"]

QUIET_NOTES = ("short",)      # per-world report notes that ask for a re-run (lever #1; a force jump is never armed)


def stable_host(sched: Any, stable: np.ndarray) -> dict[str, float]:
    """Each estimating entry's stable step [s] for the clock from ``stable`` ([entries, W]): the minimum over the
    worlds, without the parked ones (inf when every world is parked: the entry then keeps its lowest rung)."""
    parked = getattr(sched, "_parked_mask", lambda: None)()
    names = [(e, n) for e, n in enumerate(sched._entry_order) if sched._estimators.get(n) is not None]
    if parked is None:
        return {n: float(np.min(stable[e])) for e, n in names}
    return {n: float(np.min(stable[e][~parked], initial=math.inf)) for e, n in names}


def band_over(band: Any, parked: np.ndarray | None, bound: float) -> float:
    """The contact-band bound [s] for the next clock over the worlds not ``parked`` (F2, WP11): the last read's
    per-world rows (``ContactBandBound.world_bounds``, the shared rule world by world) without the parked ones. The
    last read was taken with the old parked set, so a world parked now (a health-latched one, ``park_latched``) would
    otherwise hold the next period's clock at its band need. ``bound`` (the shared value) without per-world rows."""
    rows = band.world_bounds(need=False) if getattr(band, "worlds", 0) else None
    if rows is None or not np.size(rows):
        return float(bound)
    keep = np.ones(rows.shape, dtype=bool) if parked is None else ~np.asarray(parked, dtype=bool)[: rows.size]
    return float(np.min(rows[keep], initial=math.inf))


def seed_stable(sched: Any, slot: int, host: dict[str, float]) -> None:
    """World ``slot``'s stable-step rows take ``host`` (another pack's minimum per entry: ``Scheduler._stable_host``)
    and the clock's minimum is taken again, so a world moved in from that pack runs its first period at no entry step
    coarser than it had there. The pack's own estimate replaces the rows at its first exchange (gap #10b, WP10's M3)."""
    if not host:
        return
    rows = sched._stable_dt.numpy().copy()
    for e, name in enumerate(sched._entry_order):
        if name in host:
            rows[e, slot] = host[name]
    sched._stable_dt.assign(rows)
    sched._stable_host = stable_host(sched, rows)


def quiet(report: Any, parked: np.ndarray | None) -> Any:
    """``report`` as the clock, the rollback rule and the driver may read it: the parked worlds' rows ask for nothing
    (watch state OK, FAILED kept; not short of passes; no stable-step estimate, so the precision rung pin does not
    flag them) and ``notes["parked"]`` names them. ``report`` itself when no world is parked or it has no per-world
    rows."""
    if report is None or parked is None:
        return report
    ws = np.asarray(report.watch_state)
    if ws.shape != parked.shape:
        return report
    notes = dict(report.notes or {})
    for key in QUIET_NOTES:
        rows = np.asarray(notes.get(key, ()))
        if rows.shape == parked.shape:
            notes[key] = rows.astype(bool) & ~parked
    if isinstance(notes.get("stable_dt"), dict):          # what a held world did in one exchange is no estimate
        notes["stable_dt"] = {n: np.where(parked, np.inf, v) if np.shape(v) == parked.shape else v
                              for n, v in notes["stable_dt"].items()}
    notes["parked"] = parked.copy()
    calm = np.where(parked & (ws != int(WatchState.FAILED)), int(WatchState.OK), ws).astype(ws.dtype)
    return dataclasses.replace(report, watch_state=calm, notes=notes)


def _as_mask(worlds: Any, world_count: int) -> np.ndarray:
    """bool [W] from a bool mask of that length or a list of world ids."""
    arr = np.asarray(worlds)
    if arr.dtype == bool and arr.shape == (world_count,):
        return arr.copy()
    if arr.dtype == bool or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise CouplingError(f"worlds must be a bool mask of length {world_count} or world ids, got {arr.dtype} "
                            f"of shape {arr.shape}")
    mask = np.zeros(world_count, dtype=bool)
    mask[arr.astype(np.int64).reshape(-1)] = True
    return mask


class ParkedWorlds:
    """One Scheduler's parked worlds: the host and device masks, the held copy and when each world was parked."""

    def __init__(self, world_count: int, device: Any) -> None:
        self.mask = np.zeros(world_count, dtype=bool)
        self.since = np.zeros(world_count, dtype=np.int64)       # the period count when each world was parked
        self.device_mask = wp.zeros(world_count, dtype=wp.int32, device=device)
        self.held: WorldSnapshot | None = None                   # allocated by the first park

    @property
    def any(self) -> bool:
        return bool(self.mask.any())

    def save(self, state: Any, worlds: np.ndarray) -> None:
        """Take ``worlds`` (bool [W]) of ``state`` and the stepper's arrays as their held copy."""
        self.held.save(state, wp.array(worlds.astype(np.int32), dtype=wp.int32, device=self.device_mask.device))

    def hold(self, state_out: Any) -> None:
        """Device, after an exchange: the parked worlds go back to their held copy."""
        self.held.restore(state_out, self.device_mask)


class ParkedWorldsMixin:
    """For the Scheduler (before ``ClockCensusMixin`` in its bases): the parked worlds of the module docstring."""

    _parked: ParkedWorlds | None = None          # created by the first ``park_worlds``

    @property
    def parked(self) -> np.ndarray:
        """bool [W]: the parked worlds (a copy)."""
        p = self._parked
        return np.zeros(int(self.model.world_count), dtype=bool) if p is None else p.mask.copy()

    def _parked_mask(self) -> np.ndarray | None:
        """The parked worlds (bool [W]), or None when there are none: every mask point's old branch."""
        p = self._parked
        return p.mask if p is not None and p.any else None

    # ------------------------------------------------------------------ park / unpark (host, between periods)
    def park_worlds(self, worlds: Any, state: Any, *, follow: bool = False) -> None:
        """Park ``worlds`` (a bool mask [W] or world ids) as they are in ``state``, the State the next period starts
        from. Call between periods. Worlds already parked keep their held copy. ``follow`` (only
        ``park_latched``, ALLOY_PARK_LATCHED): the next clock also takes the band bound and the substep rungs over the
        worlds left (``_parked_changed``); every other caller keeps the clock rule as before. Example:
        ``sched.park_worlds([3], sim.state)``."""
        count = int(self.model.world_count)
        p = self._parked if self._parked is not None else ParkedWorlds(count, self.model.device)
        new = _as_mask(worlds, count) & ~p.mask
        if not new.any():
            return
        if p.held is None:
            p.held = self._held_snapshot(state)
        self._calm_watch(new)
        p.save(state, new)
        p.mask |= new
        p.since[new] = self._onset.period
        self._parked = p
        self._parked_changed(follow=follow)
        log.info("coupling: parked worlds %s (%d of %d parked)", np.flatnonzero(new).tolist(), int(p.mask.sum()), count)

    def refresh_parked(self, worlds: Any, state: Any) -> None:
        """Re-take the held copy of the parked ``worlds`` (a bool mask [W] or world ids) from ``state`` and the
        stepper's arrays, after a driver wrote their rows. Call between periods, before the driver's period-start
        snapshot, so a rollback restores the same rows.

        The whole copy is re-taken, not the State arrays alone. Between periods a slot's stepper rows are the held
        ones (the hold was their last write), so a driver that wrote only State rows re-takes only those, and a
        driver that moved the whole world in (``fine_pack.WorldTransfer`` writes stepper rows too) has all of it held.
        After a State-only write the slot's watchdog, carry and solver rows are stale against the new State. That is
        safe: what they produce in an exchange is put back by the hold or masked (module docstring)."""
        p, mask = self._parked, _as_mask(worlds, int(self.model.world_count))
        if p is None or (mask & ~p.mask).any():
            raise CouplingError(f"refresh_parked: worlds {np.flatnonzero(mask).tolist()} are not all parked "
                                f"(parked: {np.flatnonzero(self.parked).tolist()})")
        if mask.any():
            self._calm_watch(mask)
            p.save(state, mask)

    def unpark_worlds(self, worlds: Any = None) -> None:
        """Return ``worlds`` (a bool mask [W] or world ids; None: every parked world) to normal stepping from their
        held state, their cooldowns as old as when they were parked (module docstring). Call between periods."""
        p = self._parked
        if p is None or not p.any:
            return
        out = p.mask.copy() if worlds is None else _as_mask(worlds, p.mask.size) & p.mask
        if not out.any():
            return
        self._resume_cooldowns(out, self._onset.period - p.since)
        p.mask &= ~out
        self._parked_changed()
        log.info("coupling: unparked worlds %s (%d still parked)", np.flatnonzero(out).tolist(), int(p.mask.sum()))

    def _held_snapshot(self, state: Any) -> WorldSnapshot:
        """The held copy's buffers: every State array and the stepper's ``state_arrays()``, as a rollback restores."""
        snap = WorldSnapshot(self.model, state, extra=dict(self.state_arrays()),
                             extra_world=dict(self.state_world_maps()))
        if snap.whole_arrays:
            raise CouplingError(f"a world cannot be parked: these arrays have no per-world axis to hold it by: "
                                f"{list(snap.whole_arrays)}")
        return snap

    def _calm_watch(self, worlds: np.ndarray) -> None:
        """The held watch state of ``worlds`` starts OK (FAILED stays): an escalation of the period before the park
        would otherwise be put back, and reported, after every exchange."""
        ws = self._watchdog.watch_state.numpy().copy()
        ws[worlds & (ws != int(WatchState.FAILED))] = int(WatchState.OK)
        self._watchdog.watch_state.assign(ws)

    def _resume_cooldowns(self, worlds: np.ndarray, periods: np.ndarray) -> None:
        """``worlds`` were held for ``periods`` ([W]): their last force jump and last ladder trip are that much later,
        so a cooldown or a repeat window that was running when they were parked is still running."""
        for rows in (self._onset.last, self._ladder.last):
            if rows.shape == worlds.shape:
                rows[worlds] += periods[worlds]

    def _parked_changed(self, follow: bool = False) -> None:
        """Hand the parked set to every mask point that keeps its own copy, let the next clock go straight to the
        level the new set asks for, and take the stable steps over that set (a device read, here only). ``follow``
        (``park_latched``): the band bound over that set too (``band_over``), and the substep rungs straight to what
        its estimators want (``jump_next(rungs=True)``); off, the rungs keep their hysteresis, as for
        ``clock_worlds=``'s parks."""
        p, mask = self._parked, self._parked_mask()
        p.device_mask.assign(p.mask.astype(np.int32))
        self._onset.parked = None if mask is None else mask.copy()
        gate = getattr(self, "_gate", None)
        if gate is not None:
            gate.park(mask)
        self._band.park(mask)
        self._clock.jump_next(rungs=follow)          # no quiet periods: the bound's worlds changed, not the scene
        if self._stable_host:                        # a report was read: its minimum had the old parked set
            self._stable_host = stable_host(self, self._stable_dt.numpy())
        if follow:
            self._band_bound = band_over(self._band, mask, self._band_bound)

    # ------------------------------------------------------------------ the hold (device, every exchange)
    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float, *,
             observables: Any = None) -> None:
        super().step(state_in, state_out, control, contacts, dt, observables=observables)
        p = self._parked
        if p is not None and p.any:
            p.hold(state_out)

    def graph_key(self) -> tuple:
        """A capture bakes the hold in or out: periods with a parked world are graphs of their own."""
        return super().graph_key() + (("parked",) if self._parked_mask() is not None else ())

    def reset_worlds(self, world_mask: Any = None) -> None:
        """``sim.reset(world_mask)``: a reset world starts over, so it is no longer held."""
        worlds = world_bools(world_mask)
        if self._parked_mask() is not None:
            self.unpark_worlds(None if worlds is None else worlds[: self._parked.mask.size])
        super().reset_worlds(world_mask)

    # ------------------------------------------------------------------ the clock and the re-run rules (host)
    def _clock_mass(self, key: tuple[str, str]) -> float:
        """Pair ``key``'s lightest proxy mass [kg] for the clock bound: over every world as built
        (``Scheduler._pair_mass``), or with parked worlds over the others."""
        parked = self._parked_mask()
        if parked is None:
            return float(self._pair_mass.get(key, np.inf))
        per_world = self._census_masses(parked.size).get(key)
        return math.inf if per_world is None else float(np.min(per_world[~parked], initial=math.inf))

    def choose_clock(self, control_dt: float, last: Any) -> tuple[float, dict[str, int]]:
        return super().choose_clock(control_dt, quiet(last, self._parked_mask()))

    def rollback_requested(self, report: Any) -> bool:
        return super().rollback_requested(quiet(report, self._parked_mask()))

    def report(self) -> Any:
        report = super().report()
        parked = self._parked_mask()
        if parked is None:
            return report
        self._last_report = quiet(report, parked)
        return self._last_report
