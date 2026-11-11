# SPDX-License-Identifier: Apache-2.0
"""Helpers the surface-flow tasks share, paint and rinse alike (SC9's clear-coat and flush-floor): DC5's supply cross
ledger read from ``SurfaceFlow.read()``, the per-world index of a labelled body (the tasks' kinematic tools), and
MuJoCo Warp's capacity overflows (it drops the contacts or constraints past ``nconmax`` / ``njmax`` with only a stdout
line; WP8's finding via WP15), every substep's through WP8's ``rinse_hand.OverflowWatch`` plug-in (DC5's find)."""

from __future__ import annotations

from typing import Any

import numpy as np

SUPPLY_REL = 1e-9               # DC5's cross ledger closes to float64 rounding, relative to what was sent
SUPPLY_SINKS = ("to_tank", "lost", "overspray", "mist", "caught", "lumped")    # where sent water went, off the film


def supply_drift(read: dict, world: int = 0) -> float:
    """DC5's cross ledger, sent + swept + withdrawn = supplied + pending + to_tank + lost + overspray + mist + caught
    + lumped (``host_api.SurfaceFlow.read``), relative to what was sent (``read``: ``SurfaceFlow.read()``)."""
    def g(key: str) -> float:
        return float(np.asarray(read[key]).reshape(-1)[world])

    took = g("supply_sent") + g("supply_swept") + g("supply_withdrawn")
    rest = g("supplied_volume") + g("pending_volume") + sum(g("supply_" + k) for k in SUPPLY_SINKS)
    return abs(took - rest) / max(g("supply_sent"), 1e-30)


def bodies(model: Any, label: str) -> np.ndarray:
    """The body labelled ``label`` (or ``.../label``) in every world, ordered by world."""
    labels, where = list(model.body_label), model.body_world.numpy()
    idx = [i for i, s in enumerate(labels) if s == label or s.endswith("/" + label)]
    return np.array(sorted(idx, key=lambda i: int(where[i])), dtype=np.int64)


class OverflowWatch:
    """MuJoCo Warp's capacity overflows per world, ORed over every period it is read (``read`` after each
    ``step_period``): every substep's, through WP8's ``rinse_hand.OverflowWatch`` plug-in when the simulation was
    built with one (Newton zeroes the overflow word before each MuJoCo step, so a period-end read sees only the last
    substep); else the period's last substep (``rinse_hand.period_overflow``). No MuJoCo Warp solver: never."""

    def __init__(self, sim: Any) -> None:
        from alloy.tasks.rinse_hand import mjw_solver, overflow_watch  # noqa: PLC0415

        has = getattr(sim, "stepper", None) is not None                # a harness simulation (others: no watch)
        self.solver, self.plugin = (mjw_solver(sim), overflow_watch(sim)) if has else (None, None)
        self.seen: np.ndarray | None = None

    def read(self) -> None:
        from alloy.tasks.rinse_hand import period_overflow  # noqa: PLC0415

        bits = period_overflow(self.plugin, self.solver)
        if bits is not None:
            now = np.asarray(bits, bool)
            self.seen = now if self.seen is None else self.seen | now

    def worlds(self, n: int) -> np.ndarray:
        """1 where a world dropped contacts or rows in any substep read so far, per world [n] (0: none)."""
        return np.zeros(n, np.int64) if self.seen is None else self.seen.astype(np.int64)


class Watched:
    """A simulation whose every ``step_period`` also reads its :class:`OverflowWatch`; all else is the simulation's.
    Build it with ``rinse_hand.OverflowWatch()`` among its plug-ins so every substep counts."""

    def __init__(self, sim: Any) -> None:
        self.sim, self.watch = sim, OverflowWatch(sim)

    def step_period(self) -> None:
        self.sim.step_period()
        self.watch.read()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sim, name)


__all__ = ["OverflowWatch", "SUPPLY_REL", "SUPPLY_SINKS", "Watched", "bodies", "supply_drift"]
