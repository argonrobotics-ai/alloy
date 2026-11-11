# SPDX-License-Identifier: Apache-2.0
"""Grow VBD self-contact rows between periods, and re-run a period that dropped pairs (GAP-13B).

``alloy.solvers.vbd.self_contact_rows`` sizes each self-contact row to the scene and, inside the step, only records
what each row needed. ``regrow_rows`` is the host side, called once per attempt of a period, outside any captured
graph (as ``grid_growth.regrow``, whose restore it shares):

- no row over its fill mark: nothing happens (one read of four ints per solver);
- rows over the mark, none over its size: they grow, the captured graphs are dropped (they hold the old arrays) and
  the period stands, since no pair was dropped;
- a row over its size: the rows grow, every world goes back to the period start and the period re-runs, so the
  accepted period has every pair. Health's dropped-pairs count goes back to its value at the period start too
  (``save_period_start`` keeps it, after any reset between periods), so
  the abandoned attempt raises no SELF_CONTACT_OVERFLOW.

Rows at their cap, or a scene at its bytes cap, do not grow: their pairs drop and the flag stays, as before.
``sim.row_regrows`` counts the re-runs and ``RowGuard.grown`` lists every growth.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import warp as wp

from alloy.integration.grid_growth import drop_graphs, restore_period_start
from alloy.solvers.vbd.self_contact_rows import Growth, RowState

log = logging.getLogger("alloy.integration")

WATCH_ATTR = "_alloy_self_contact_watch"      # alloy.precision.self_contact's watch on the solver, when health has one

__all__ = ["RowGuard", "regrow_rows", "row_guard"]


class RowGuard:
    """The solvers of one simulation whose self-contact rows can grow."""

    def __init__(self, rows: Mapping[str, RowState]) -> None:
        self.rows = dict(rows)
        self.grown: list[tuple[str, Growth]] = []
        self._dropped_start: dict[str, wp.array] = {}

    @classmethod
    def for_solvers(cls, solvers: Mapping[str, Any]) -> RowGuard | None:
        rows = {name: s.alloy_rows for name, s in solvers.items() if getattr(s, "alloy_rows", None) is not None}
        return cls(rows) if rows else None

    def grow(self) -> list[Growth]:
        """Grow every solver's rows that ask for it (host reads); the growths made."""
        out = [(name, g) for name, state in self.rows.items() if (g := state.grow()) is not None]
        self.grown += out
        return [g for _, g in out]

    def keep_dropped(self) -> None:
        """Remember health's dropped-pairs counts as they stand (a device copy). ``save_period_start`` calls it, so a
        reset between periods is in the baseline."""
        for name, watch in self._watches():
            saved = self._dropped_start.get(name)
            if saved is None:
                saved = self._dropped_start[name] = wp.zeros_like(watch.dropped)
            wp.copy(saved, watch.dropped)
            if hasattr(watch, "keep_period"):            # its pass-through marks and crossings too
                watch.keep_period()

    def restore_dropped(self) -> None:
        """Put health's dropped-pairs counts back to the period start (the attempt that dropped them is gone)."""
        for name, watch in self._watches():
            saved = self._dropped_start.get(name)
            if saved is None:
                watch.dropped.zero_()
            else:
                wp.copy(watch.dropped, saved)
            if hasattr(watch, "restore_period"):
                watch.restore_period()

    def _watches(self) -> list[tuple[str, Any]]:
        found = ((name, getattr(state.solver, WATCH_ATTR, None)) for name, state in self.rows.items())
        return [(name, w) for name, w in found if w is not None]


def row_guard(sim: Any) -> RowGuard | None:
    """``sim``'s guard, made on first use from its solvers; None when no solver has rows that can grow."""
    if not hasattr(sim, "_alloy_row_guard"):
        solvers = getattr(sim, "solvers", None) or ()
        named = solvers if isinstance(solvers, Mapping) else {str(i): s for i, s in enumerate(solvers)}
        sim._alloy_row_guard = RowGuard.for_solvers(named)
    return sim._alloy_row_guard


def regrow_rows(sim: Any, start: tuple[int, int]) -> bool:
    """Grow the self-contact rows of ``sim`` (an IntegratedSimulation) that ask for it. True when pairs had dropped
    this period: the period start is restored (``start``: clock ticks and steps) and the caller re-runs the period."""
    guard = row_guard(sim)
    if guard is None:
        return False
    grown = guard.grow()
    if not grown:
        return False
    drop_graphs(sim)
    if not any(g.dropped for g in grown):
        return False
    restore_period_start(sim, start)             # also puts health's dropped counts back (the guard is on sim)
    sim.row_regrows = getattr(sim, "row_regrows", 0) + 1
    log.info("period %d re-runs on the grown self-contact rows (re-run %d)", sim.periods, sim.row_regrows)
    return True
