# SPDX-License-Identifier: Apache-2.0
"""#10b P1 S2, the fine pack in the re-run ([internal note] notes/wp4-perworld-clock/DESIGN.md sections 13-15).

Today an escalated world re-runs its period with every other world (``IntegratedSimulation._roll_back``). With a
``FineRerun`` on the simulation (``sim.fine_rerun``; off unless set), the escalated worlds E re-run alone in a fine
pack of W_f slots, a second simulation of the same scene, and the report merges as today's keep mask does:
1. Every refusal comes first, with no side effect (``FineRerun.candidates``): the kind of re-run, |E| <= W_f, no
   contact gate, no observers or held wrench, no SPH / MPM / MuJoCo C entry, no whole arrays, the fine pack built
   (a failed build is cached), and ``WorldTransfer.check`` for every world both ways.
2. The coarse pack as today: E back to the period start and ``on_rollback(E)`` (``_roll_back``), then its
   ``choose_clock`` decides the re-run attempt with all its side effects.
3. E moves into the slots (idle slots take E[0]); the stepper's host state is mirrored (``_pin``: the clock, the
   ladder's and onset's rows cut to the slots, the captured passes, the sticky ladder's trips, mode and strong pairs,
   the attempt's dt_x, substeps and limits) and the
   fine pack's first attempt replays it through lever #1's path (``_rerun = _short_rerun = True``). The counters
   start where the coarse pack's are (steps: the period start, the check cadence; ``health._step``: as it stands).
4. The fine pack steps its period with its own loop: a nested escalation takes its own decisions.
5. E moves back, the host state with it (``_mirror``), the counters and clock take the fine pack's, and the coarse
   stepper adopts the merged report (``_adopt``: stable steps, the band at the larger of the first attempt's and the
   fine pack's speeds, the band re-run judge). After step 2 a failure raises; it never falls back (WP12).
Not yet: the contact gate's rows (refused), the census of the fine attempt (S3).
Laws: tests/coupling/test_fine_rerun.py.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy.core.errors import CouplingError
from alloy.coupling.capture_passes import PER_WORLD_REPORT_FIELDS, merge_reports, rerun_mask, short_worlds
from alloy.coupling.contact_band import read_band
from alloy.coupling.fine_pack import TransferError, WorldTransfer

log = logging.getLogger("alloy.coupling")

ROWS = {"_ladder": {"last": -(2**62), "per_world": 0}, "_onset": {"last": -(2**62)}}   # per-world host rows: default
HOST = ("_clock", "_band_rerun")                 # pack-wide host objects: their attributes are deep-copied
SCALARS = ("_dt_x", "_attempt_limit", "_steps_expected", "_control_dt", "_rollbacks", "_rollbacks_period",
           "_rung_changes", "_substeps_now", "_band_bound", "_stable_host")
ADOPTED = ("_band_bound", "_stable_host")        # back from the fine pack through ``_adopt``, not copied
NEVER = ("SPH", "MPM", "MjcCore")    # entry solvers (class names) a fine pack never takes: WP10, WP12; MuJoCo C (CPU
                                     # default route): the moved world parts from today's after its re-run (S2b)


@dataclasses.dataclass(frozen=True)
class _Rows:
    worlds: np.ndarray        # coarse world ids
    slots: np.ndarray         # their fine slots
    W: int                    # coarse world count
    W_f: int                  # fine slots


def _widen(first: Any, fine: Any, r: _Rows, unmerged: list[str], key: str) -> Any:
    """``first``'s [W] array with the ``worlds`` rows from ``fine``'s [W_f] ``slots``; a scalar or non-array from
    ``fine`` (the re-run's, as today); any other array keeps ``first``'s and is named in ``unmerged``."""
    if not isinstance(fine, np.ndarray) and not isinstance(first, np.ndarray):
        return fine
    a, b = np.asarray(first) if first is not None else None, np.asarray(fine) if fine is not None else None
    if a is not None and b is not None and a.shape == (r.W,) and b.shape == (r.W_f,):
        out = a.copy()
        out[r.worlds] = b[r.slots]
        return out
    unmerged.append(key)
    return first


def widen_report(first: Any, fine: Any, worlds: Sequence[int], slots: Sequence[int]) -> Any:
    """The fine pack's report as the coarse pack's: per-world fields and notes (one dict level deep, as
    ``merge_notes``) are ``first``'s with each world in ``worlds`` taken from its fine ``slots`` row; scalars are the
    fine pack's (the re-run's, as today's merge takes them). ``notes["fine_pack"]["unmerged"]`` names per-world
    values of another shape, kept from ``first``. ``merge_reports(first, widened, keep)`` then gives today's merge."""
    w, s = np.asarray(worlds, np.int64), np.asarray(slots, np.int64)
    r = _Rows(w, s, len(first.watch_state), len(fine.watch_state))
    unmerged: list[str] = []
    fields = {n: _widen(getattr(first, n), getattr(fine, n), r, unmerged, n) for n in PER_WORLD_REPORT_FIELDS}
    a, b = dict(getattr(first, "notes", None) or {}), dict(getattr(fine, "notes", None) or {})
    notes: dict[str, Any] = {}
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(a.get(k), Mapping):
            notes[k] = {kk: _widen(a[k].get(kk), vv, r, unmerged, f"{k}.{kk}") for kk, vv in v.items()}
        else:
            notes[k] = _widen(a.get(k), v, r, unmerged, k)
    notes["fine_pack"] = {"worlds": w.copy(), "slots": s.copy(), "unmerged": sorted(unmerged)}
    return dataclasses.replace(fine, notes=notes, **fields)


# ---------------------------------------------------------------------------------------------- host state mirror
def _assign(dst: Any, src: Any, skip: Sequence[str] = ()) -> None:
    """Deep copies of ``src``'s host attributes onto ``dst`` in place (device arrays and ``skip`` stay ``dst``'s)."""
    for k, v in vars(src).items():
        if k not in skip and not isinstance(v, wp.array):
            setattr(dst, k, copy.deepcopy(v))


def _sized(obj: Any, name: str, default: Any, n: int) -> np.ndarray:
    """A per-world row array of ``obj`` at length ``n`` (a lazily sized one at its default)."""
    a = np.asarray(getattr(obj, name))
    return a.copy() if a.shape == (n,) else np.full(n, default, dtype=np.int64)


def _cut(coarse: Any, fine: Any, rows: Mapping[str, Any], r: _Rows) -> None:
    """Fine's host object = coarse's, its per-world rows cut to the slots (idle slots take the first world's)."""
    _assign(fine, coarse, skip=tuple(rows))
    take = np.concatenate([r.worlds, np.full(r.W_f - r.worlds.size, r.worlds[0])])
    for name, default in rows.items():
        setattr(fine, name, _sized(coarse, name, default, r.W)[take])


def _paste(coarse: Any, fine: Any, rows: Mapping[str, Any], r: _Rows) -> None:
    """Coarse's host object = fine's, its per-world rows the coarse ones with the worlds' rows from their slots."""
    keep = {name: _sized(coarse, name, default, r.W) for name, default in rows.items()}
    _assign(coarse, fine, skip=tuple(rows))
    for name, default in rows.items():
        keep[name][r.worlds] = _sized(fine, name, default, r.W_f)[r.slots]
        setattr(coarse, name, keep[name])


def _mirror(dst: Any, src: Any, r: _Rows, *, into_fine: bool) -> None:
    """The stepper's host state, coarse -> fine (``into_fine``) or back: ``HOST`` objects, ``ROWS``, ``SCALARS`` and
    the captured passes' host fields."""
    for name in HOST:
        _assign(getattr(dst, name), getattr(src, name))
    for name, rows in ROWS.items():
        if into_fine:
            _cut(getattr(src, name), getattr(dst, name), rows, r)
        else:
            _paste(getattr(dst, name), getattr(src, name), rows, r)
    for name in SCALARS:
        if hasattr(src, name) and (into_fine or name not in ADOPTED):
            setattr(dst, name, copy.deepcopy(getattr(src, name)))
    cs, cd = src._captured_passes(), dst._captured_passes()
    if cs is not None and cd is not None:
        _assign(cd[0], cs[0], skip=("world_count", "device"))
    _sticky_state(dst, src)


def _sticky_state(dst: Any, src: Any) -> None:
    """The sticky ladder's pack-wide state (``ladder.escalate``; the clock's floor goes with ``HOST``): its trips,
    the strong pairs, the proxy sets' staggered mode and the loop's passes, ADMM's rho raises and rho."""
    for name in ("_trips", "_rho_raises"):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))
    if hasattr(src, "_strong"):
        dst._strong = set(src._strong)
    for a, b in zip(getattr(dst, "_sets", None) or (), getattr(src, "_sets", None) or (), strict=True):
        a.staggered, a.strong, a.mapping.mode = b.staggered, b.strong, b.mapping.mode
    if getattr(src, "_loop", None) is not None and getattr(dst, "_loop", None) is not None:
        dst._loop.cfg = src._loop.cfg
    if hasattr(src, "_rho_raises") and hasattr(src, "_coupling"):
        dst._coupling = copy.deepcopy(src._coupling)


def _pin(fine: Any, coarse: Any, r: _Rows) -> None:
    """The fine stepper replays the coarse pack's re-run attempt (DESIGN 13.2, WP10 (a) 1-3)."""
    _mirror(fine, coarse, r, into_fine=True)
    cp = fine._captured_passes()
    if cp is not None:
        cp[0]._full_next = cp[0].full            # begin_attempt(rerun=True) takes the attempt's full loop from it
    fine._apply_substeps(coarse._substeps_now)
    fine._rerun = fine._short_rerun = True        # choose_clock -> _replay_attempt
    fine._gate_rerun = False
    if (cp is not None and cp[0].active) or getattr(fine, "_gate", None) is not None or fine._band_rerun.on:
        fine._onset_last = fine._onset.last.copy()
        fine._save_attempt_watch()


def _adopt(coarse: Any, fine: Any, merged: Any, vmax0: np.ndarray, stable0: Mapping[str, float]) -> None:
    """What the coarse stepper's ``report`` would have done after today's re-run, from the merged rows: the stable
    steps (the smaller of the first attempt's and the fine pack's), the band at the larger of their speeds (per route:
    conservative, DESIGN 15 (b) 1), the band re-run's judge, and the merged report as its last."""
    fs = getattr(fine, "_stable_host", None) or {}
    coarse._stable_host = {k: min(v, fs.get(k, v)) for k, v in stable0.items()}
    band = coarse._band
    n = len(band.routes)
    if n:
        host = band.vmax.numpy()
        host[:n] = np.maximum(vmax0[:n], np.asarray(fine._band.last_speed)[:n])
        band.vmax.assign(host)
    coarse._band_bound = read_band(coarse)       # the re-run attempt's: the first attempt's report read its own
    coarse._last_report = merged
    if hasattr(coarse, "_census_kept"):
        coarse._census_kept = 0                   # S3 counts the fine attempt; kept worlds re-ran nothing


WATCH_ROWS = ("e_scale", "z_ref", "m_tot", "l_char")   # the watchdog's per-world reference, from the world's start


def _watch_reference(fine: Any, coarse: Any, r: _Rows) -> None:
    """The energy watchdog's rows outside ``state_arrays`` go into the slots with their worlds: E_scale, the datum,
    mass and extent (set from each world's start state: a slot holds the start it was built with), the 1 s window
    ring's columns and head, and the pack's elapsed time (idle slots take the first world's)."""
    wf, wc = fine._watchdog, coarse._watchdog
    take = np.concatenate([r.worlds, np.full(r.W_f - r.worlds.size, r.worlds[0])])
    for name in WATCH_ROWS:
        getattr(wf, name).assign(getattr(wc, name).numpy()[take])
    wp.copy(wf.elapsed_ns, wc.elapsed_ns)
    wf.head = wc.head
    wf.ring = _ring_into(wf, wc.ring.numpy()[:, take] if wc.ring is not None else None)
    wf.has_reference = True


def _ring_into(wf: Any, columns: np.ndarray | None) -> Any:
    """The fine watchdog's ring holding ``columns`` ([periods, W_f]): written in place when it already has that shape,
    else a new array; None when the coarse pack has no ring yet."""
    if columns is None:
        return None
    if wf.ring is not None and tuple(wf.ring.shape) == columns.shape:
        wf.ring.assign(columns)
        return wf.ring
    return wp.array(columns, dtype=wp.float64, device=wf.device)


def _watch_back(coarse: Any, fine: Any, r: _Rows) -> None:
    """Back from the slots: the worlds' window-ring columns (a nested escalation rebased them), their last energy and
    passive gain (the ledger's read; the kept worlds keep their first run's) and the elapsed time (as after today's
    re-run). The reference rows never change in a period."""
    wf, wc = fine._watchdog, coarse._watchdog
    wp.copy(wc.elapsed_ns, wf.elapsed_ns)
    for name in ("energy", "gain"):
        row = getattr(wc, name).numpy().copy()
        row[r.worlds] = getattr(wf, name).numpy()[r.slots]
        getattr(wc, name).assign(row)
    if wc.ring is not None and wf.ring is not None:
        ring = wc.ring.numpy()
        ring[:, r.worlds] = wf.ring.numpy()[:, r.slots]
        wc.ring.assign(ring)


# ------------------------------------------------------------------------------------------------ the fine re-run
class FineRerun:
    """The fine pack of ``slots`` worlds for one simulation, built at its first use by ``factory(slots)`` (an API
    ``Sim``, kept alive: its workdir holds the cards); ``reruns`` and ``fallbacks`` (reason -> count) for the rows."""

    def __init__(self, factory: Callable[[int], Any], slots: int) -> None:
        self.factory, self.slots = factory, max(int(slots), 1)
        self.sim: Any = None
        self.failed: str | None = None
        self.reruns = 0
        self.fallbacks: dict[str, int] = {}

    def __call__(self, eng: Any, report: Any, start: tuple[int, int]) -> Any:
        return self.try_rerun(eng, report, start)

    def pack(self) -> Any:
        """The fine pack's engine (built once; a failed build is cached as none, with its reason)."""
        if self.sim is None and self.failed is None:
            try:
                self.sim = self.factory(self.slots)
            except Exception as exc:  # noqa: BLE001 - any build failure (OOM included) leaves today's re-run
                self.failed = f"{type(exc).__name__}: {exc}"
                log.warning("fine pack: build failed, re-runs stay whole (%s)", self.failed)
                return None
            eng = self.sim.engine
            eng.latch_log = None                  # the coarse pack polls after the merge (WP12 14.5)
            parts = getattr(eng, "parts", None)
            if getattr(parts, "rung_pin", None) is not None:
                parts.rung_pin = None
        return None if self.sim is None else self.sim.engine

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def candidates(self, eng: Any, report: Any) -> tuple[np.ndarray | None, Any]:
        """(E, (there, back) transfers) when this re-run can take the fine pack, else (None, the reason); no side
        effects."""
        st, mask = eng.stepper, rerun_mask(report)
        if mask is None:
            return None, "no per-world watch state"
        E = np.asarray(mask).astype(bool)
        why = _refusal(eng, st, report, E, self.slots)
        if why is not None:
            return None, why
        fine = self.pack()
        if fine is None:
            return None, f"no fine pack ({self.failed})"
        try:
            there, back = WorldTransfer(eng, fine), WorldTransfer(fine, eng)
            for s, w in enumerate(np.flatnonzero(E)):
                there.check(int(w), s)
                back.check(s, int(w))
        except TransferError as exc:
            return None, f"transfer: {exc}"
        return E, (there, back)

    def try_rerun(self, eng: Any, report: Any, start: tuple[int, int]) -> Any:
        """The merged report after E re-ran in the fine pack, or None (refused: today's whole re-run, untouched)."""
        E, got = self.candidates(eng, report)
        if E is None:
            self.fallbacks[got] = self.fallbacks.get(got, 0) + 1
            log.info("fine pack: whole re-run (%s)", got)
            return None
        merged = self._rerun(eng, report, start, E, got)
        self.reruns += 1
        if eng.stepper.rollback_requested(merged):
            log.warning("fine pack: the merged report still asks for a rollback (taken as is)")
        return merged

    def _rerun(self, eng: Any, report: Any, start: tuple[int, int], E: np.ndarray, transfers: Any) -> Any:
        there, back = transfers
        fine, st = self.sim.engine, eng.stepper
        worlds = np.flatnonzero(E)
        r = _Rows(worlds, np.arange(worlds.size), int(E.size), self.slots)
        # the first attempt's per-route maxima: its report() already moved them to last_speed and zeroed vmax (WP10)
        vmax0 = np.asarray(getattr(st._band, "last_speed", None) if st._band.routes else [], dtype=np.float64).copy()
        stable0 = dict(getattr(st, "_stable_host", None) or {})
        keep = eng._roll_back(report, start)       # as today: E to the period start, on_rollback(E), the tail
        if keep is None:
            raise CouplingError("fine pack: the coarse rollback was not per world")
        st.choose_clock(eng.clock.control_dt, eng._last_coupling)    # the re-run attempt, its side effects as today
        for s in range(self.slots):
            there.copy(int(worlds[min(s, worlds.size - 1)]), s)
        _pin(fine.stepper, st, r)
        _watch_reference(fine.stepper, st, r)
        fine.clock.ticks, fine.steps, fine.periods = eng.clock.ticks, eng.steps, eng.periods
        if getattr(fine, "health", None) is not None:
            wp.copy(fine.health._step, eng.health._step)
        fine.step_period()
        for s, w in zip(r.slots, worlds, strict=True):
            back.copy(int(s), int(w))
        _mirror(st, fine.stepper, r, into_fine=False)
        _watch_back(st, fine.stepper, r)
        st._apply_substeps(fine.stepper._substeps_now)
        eng.clock.ticks, eng.clock.j, eng.steps = fine.clock.ticks, fine.clock.j, fine.steps
        eng.substeps = dict(fine.substeps)
        if getattr(eng, "health", None) is not None:
            wp.copy(eng.health._step, fine.health._step)
        merged = merge_reports(report, widen_report(report, fine._last_coupling, worlds, r.slots), keep)
        _adopt(st, fine.stepper, merged, vmax0, stable0)
        return merged


def _refusal(eng: Any, st: Any, report: Any, E: np.ndarray, slots: int) -> str | None:
    """Why this re-run stays whole (DESIGN 13, 14.1), or None."""
    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415  (integration imports coupling)

    checks = (
        ((report.notes or {}).get("rerun_all"), "a whole re-run (lever #2)"),
        (short_worlds(report).any(), "short worlds (lever #1)"),
        (not E.any() or E.all(), "every world or none"),
        (int(E.sum()) > slots, f"{int(E.sum())} worlds over {slots} slots"),
        (getattr(getattr(st, "_band_rerun", None), "wanted", False), "a band re-run"),
        (getattr(st, "_gate", None) is not None, "the contact gate (its rows: S2b)"),
        (bool(eng._good.whole_arrays), "arrays without a per-world axis"),
        (bool(getattr(eng, "_observers", None)), "step observers"),
        (_wrench_held(eng), "a held wrench (sim.apply_wrench: its work is booked on this simulation's stepper)"),
        (next((f"entry {type(s).__name__}" for s in entry_solvers(st).values() for n in NEVER
               if n in type(s).__name__), None), None),
        (not hasattr(st, "_ladder"), "a stepper without the period ladder"),
    )
    return next((why or bad for bad, why in checks if bad), None)


def _wrench_held(eng: Any) -> bool:
    """Whether ``Sim.apply_wrench`` holds a wrench (health_work.WrenchWork wraps only this simulation's stepper, so
    a world re-run in the fine pack would carry its body_f without booking its work; OSS-3)."""
    ww = getattr(eng, "_alloy_wrench_work", None)                 # health_work.WRENCH_ATTR
    return ww is not None and bool(np.any(getattr(ww, "_host", 0.0)))


__all__ = ["FineRerun", "widen_report"]
