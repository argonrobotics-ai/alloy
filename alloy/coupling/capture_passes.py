# SPDX-License-Identifier: Apache-2.0
"""Captured coupling passes (mixed-speed lever #1, [internal note] notes/mixed-speed-1107.md).

Under CUDA graph capture a residual loop whose entry solvers allocate inside a step cannot be a conditional node, so
``ProxyIterationLoop`` and ``AdmmIterationLoop`` unrolled it to k_max passes. An ADMM exchange (k 2..20) then captured
20 passes while its worlds converged after 2: rigid + paste + cloth had 55k graph nodes per exchange and took 3.28 s per
10 ms, against 0.43 s with 2 passes. ``CapturedPasses`` caps the captured passes and keeps the loop's results:

- a capture unrolls ``passes`` passes (k_min at first). After them a kernel flags each world still unconverged in
  ``short`` (int32 [W], over one period attempt, cleared by ``begin_attempt``);
- the coupler's report carries the flags (``notes["short"]``). A period that leaves a world short is rolled back for
  those worlds and run again with the full loop (``full``: k_max passes in a capture, the residual-gated loop eagerly).
  So every world ends the period where the k_max loop leaves it: a converged world is frozen at its converging pass
  in both, and the other worlds keep their first run (``IntegratedSimulation``'s per-world rollback);
- ``graph_key`` carries the captured pass count, so capped and full exchanges are separate graphs;
- ``passes`` doubles (to k_max at most) after ``raise_after`` periods with a short within ``window`` periods, and halves
  (to k_min at least) after ``decay_after`` periods in a row without one. A scene whose worlds need more passes stops
  paying a re-run per period.

Only a driver that re-runs short worlds turns this on (``driver``: ``IntegratedSimulation`` calls the stepper's
``allow_capped_captures``). A stepper captured by anything else unrolls k_max passes as before, so its results do not
change. Eager periods and conditional-node loops (``capture_while``) run the residual loop as before: nothing is
capped. A loop with k_min == k_max (lagged: exactly 2) always runs its passes.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from alloy.coupling import residuals as rs

if TYPE_CHECKING:
    from alloy.coupling.types import CouplingReport

log = logging.getLogger("alloy.coupling")

__all__ = ["CapturedPasses", "capturing", "merge_notes", "merge_reports", "pick_worlds", "rerun_mask", "short_worlds"]


@wp.kernel(enable_backward=False)
def mark_short_kernel(world_converged: wp.array[wp.int32], short_flags: wp.array[wp.int32]):
    """After the captured passes of one exchange: a world still iterating (ACTIVE) is short for this attempt.
    (``short`` is a C++ keyword: no kernel argument may take that name.)"""
    w = wp.tid()
    if world_converged[w] == rs.ACTIVE:
        short_flags[w] = 1


def capturing(device: Any) -> bool:
    """A CUDA graph capture is recording on ``device`` (the loops unroll their passes then)."""
    dev = wp.get_device(device)
    return bool(dev.is_cuda and dev.is_capturing)


def short_worlds(report: "CouplingReport | None") -> np.ndarray:
    """bool [W]: worlds a capped capture left unconverged in this report's attempt (empty without flags)."""
    notes = getattr(report, "notes", None) or {}
    return np.asarray(notes.get("short", np.zeros(0, dtype=bool)), dtype=bool)


def pick_worlds(sel: np.ndarray, a: Any, b: Any) -> Any:
    """``a`` where ``sel``, else ``b``, when both are per-world arrays of ``sel``'s shape; else ``b``."""
    if a is None or b is None:
        return b
    a, b_arr = np.asarray(a), np.asarray(b)
    return np.where(sel, a, b_arr) if a.shape == b_arr.shape == sel.shape else b


def merge_notes(sel: np.ndarray, first: Any, rerun: Any) -> dict[str, Any]:
    """A re-run report's notes with ``first``'s per-world arrays (also inside dicts, one level deep) where ``sel``.
    A kept world ran the re-run too, from its end state, so its re-run values mean nothing; a short re-run keeps
    escalated worlds, whose force-jump flags must stay the first attempt's."""
    a, b = first or {}, rerun or {}
    out: dict[str, Any] = {}
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            out[k] = {kk: pick_worlds(sel, a[k].get(kk), vv) for kk, vv in v.items()}
        else:
            out[k] = pick_worlds(sel, a.get(k), v)
    return out


PER_WORLD_REPORT_FIELDS = ("res_vel", "res_pos", "res_force", "iters_used", "watch_state", "interface_work", "energy")


def rerun_mask(report: "CouplingReport") -> np.ndarray | None:
    """int32 [W] worlds to re-run: all when the stepper asks for the whole period (lever #2's gate), else those a capped
    capture left short of coupling passes (lever #1: that re-run replays the attempt), else the escalated ones; None
    when the report has no per-world watch state."""
    from alloy.coupling.types import WatchState  # noqa: PLC0415 - kept lazy like the CouplingReport hint

    ws = getattr(report, "watch_state", None)
    if ws is None or len(ws) == 0:
        return None
    escalated = np.asarray(ws) == int(WatchState.ESCALATED)
    if (getattr(report, "notes", None) or {}).get("rerun_all"):      # lever #2: the whole period at another clock
        return np.ones_like(escalated, dtype=np.int32)
    short = short_worlds(report)
    if short.shape == escalated.shape and short.any():     # short worlds re-run first, alone (SchedulerMixin)
        return short.astype(np.int32)
    return escalated.astype(np.int32)


def merge_reports(first: "CouplingReport", rerun: "CouplingReport", keep: np.ndarray) -> "CouplingReport":
    """Per-world fields (and per-world notes, ``merge_notes``) from ``first`` where ``keep`` is set, else from
    ``rerun``; scalars from ``rerun``."""
    sel = np.asarray(keep).astype(bool)
    fields = {name: pick_worlds(sel, getattr(first, name), getattr(rerun, name)) for name in PER_WORLD_REPORT_FIELDS}
    notes = merge_notes(sel, getattr(first, "notes", None), getattr(rerun, "notes", None))
    return dataclasses.replace(rerun, notes=notes, **fields)

@dataclass
class CapturedPasses:
    """Host state of one iteration loop's capped captures. ``enabled``: k_min < k_max and no conditional-node loop;
    ``driver``: the driver re-runs short worlds (nothing is capped until it says so)."""

    world_count: int
    device: Any
    enabled: bool
    driver: bool = False
    raise_after: int = 2              # periods with a short within ``window`` that double ``passes``
    window: int = 16                  # [periods]
    decay_after: int = 64             # periods in a row without a short that halve ``passes``
    passes: int = 1                   # captured passes now (the loop starts it at k_min), within [k_min, k_max]
    full: bool = False                # this attempt runs the full loop (a short re-run)
    reruns: int = 0                   # short re-runs so far
    _full_next: bool = False
    _period: int = 0
    _quiet: int = 0                   # periods in a row without a short
    _short_periods: list[int] = field(default_factory=list)
    short: wp.array = field(init=False)

    def __post_init__(self) -> None:
        self.short = wp.zeros(max(self.world_count, 1), dtype=wp.int32, device=self.device)

    # ------------------------------------------------------------------ capture (Python body of an exchange)
    def captured(self, k_min: int, k_max: int) -> int:
        """Passes a capture runs now: ``passes`` within [k_min, k_max]; k_max when nothing is capped."""
        if not self.active or self.full or k_min >= k_max:
            return k_max
        return min(max(self.passes, k_min), k_max)

    @property
    def active(self) -> bool:
        return self.enabled and self.driver

    def capped(self, k_min: int, k_max: int) -> bool:
        return self.captured(k_min, k_max) < k_max

    def mark(self, world_converged: wp.array) -> None:
        wp.launch(mark_short_kernel, dim=self.world_count, inputs=[world_converged, self.short], device=self.device)

    # ------------------------------------------------------------------ host, once per period attempt
    def begin_attempt(self, rerun: bool, k_min: int) -> None:
        """Before an attempt's first exchange: clear the flags; a re-run accepted by ``accept`` runs the full loop."""
        self.full, self._full_next = self._full_next, False
        if not rerun:
            self._period += 1
            self._quiet += 1
            if self._quiet >= self.decay_after and self.passes > k_min:
                self._set(max(self.passes // 2, k_min), f"halved after {self._quiet} quiet periods")
        if self.active:
            self.short.zero_()

    def wants_rerun(self, report: "CouplingReport") -> bool:
        return bool(self.active and not self.full and short_worlds(report).any())

    def accept(self, report: "CouplingReport", k_min: int, k_max: int) -> bool:
        """on_rollback: True (the re-run runs the full loop) when the attempt left worlds short."""
        if not self.wants_rerun(report):
            return False
        self._full_next, self.reruns, self._quiet = True, self.reruns + 1, 0
        recent = [p for p in self._short_periods if p > self._period - self.window]
        self._short_periods = recent if self._period in recent else [*recent, self._period]   # periods, not attempts
        now = self.captured(k_min, k_max)
        log.info("coupling: %d world(s) short of passes at %d captured; re-run with the full loop",
                 int(short_worlds(report).sum()), now)
        if len(self._short_periods) >= self.raise_after and now < k_max:
            self._short_periods.clear()
            self._set(min(2 * now, k_max), f"{self.raise_after} short periods within {self.window}")
        return True

    def _set(self, passes: int, why: str) -> None:
        log.info("coupling: captured passes %d -> %d (%s)", self.passes, passes, why)
        self.passes, self._quiet = passes, 0

    def graph_key(self, k_min: int, k_max: int) -> tuple:
        """What a capture bakes in: the unrolled pass count (empty when nothing can be capped)."""
        if not self.active or k_min >= k_max:
            return ()
        return ("passes", self.captured(k_min, k_max))

    def notes(self) -> dict[str, Any]:
        if not self.active:
            return {}
        return {"short": self.short.numpy()[: self.world_count].astype(bool), "captured_passes": self.passes,
                "short_reruns": self.reruns}
