# SPDX-License-Identifier: Apache-2.0
"""Each world's own exchange level: the per-world need behind gap #10b's per-world clock.

The shared clock (``ClockController.decide``) picks one grid level j from the tightest bound over every world. This
module applies the same formula world by world, to bounds that are known per world, and says which term set each
world's level. It only reads; the shared clock and every step are unchanged.

Terms, in the order ``TERMS`` names them:
- ``interface``: the world's own pair bound (stiffness and damping at its lightest proxy; with the contact gate, its
  own armed proxies);
- ``band``: the world's own contact-band bound (``ContactBandBound.world_bounds``: its records' relative speed and,
  when it holds a record, its fastest particle);
- ``floor``: the sticky ladder floor. It is pack-wide today (one world's trip raises it for all), so a world bound by
  it may need less;
- ``attempt``: the attempt's own extra halvings (an onset or ladder re-run), capped at the attempt's level;
- ``gate``: the contact gate's own bound for the world (``ContactGate.world_need``), when given and tighter than
  the pair bound;
- ``floor_world``: the floor, applied only to worlds whose own sticky-trip row (``PeriodLadder._floor_world``) is
  non-zero, when that row is given;
- ``none``: nothing binds; the coarsest level.

The maximum over worlds equals the shared level whenever no pack-wide-only term binds (the floor, or a band speed
from a world that holds no record of its own).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.coupling.clocks import grid_level

TERMS = ("none", "interface", "band", "floor", "attempt", "gate", "floor_world")


@dataclass(frozen=True)
class ClockGrid:
    """The grid the levels live on: control period [s], the coarsest allowed dt_x [s] and the finest level."""

    control_dt: float
    dt_x_max: float
    j_max: int


@dataclass(frozen=True)
class AttemptLevels:
    """What an attempt adds for every world: the sticky floor, the attempt's extra halvings and its own level."""

    floor: int = 0
    extra: int = 0
    j_attempt: int | None = None


@dataclass(frozen=True)
class WorldNeed:
    """Per world: its level ``j`` and the index into ``TERMS`` of the term that set it."""

    j: np.ndarray
    term: np.ndarray

    def terms(self) -> list[str]:
        return [TERMS[int(t)] for t in self.term]


def _levels(bound: np.ndarray, grid: ClockGrid) -> np.ndarray:
    """``grid_level`` of each bound (once per distinct value: batches hold few)."""
    vals, inv = np.unique(np.minimum(bound, grid.dt_x_max), return_inverse=True)
    levels = np.array([grid_level(float(v), grid.control_dt, grid.j_max)[0] for v in vals], dtype=np.int64)
    return levels[inv.reshape(-1)]


def per_world_rungs(interface: np.ndarray, band: np.ndarray | float, grid: ClockGrid,
                    attempt: AttemptLevels | None = None, gate: np.ndarray | None = None,
                    floor_world: np.ndarray | None = None) -> WorldNeed:
    """Each world's level from its own ``interface`` and ``band`` bounds [s] (``band`` a scalar: the shared bound for
    every world), by the shared clock's formula: the level of the tighter bound, no lower than the floor, plus the
    attempt's halvings, no higher than the attempt's level. ``gate`` (s per world) tightens the interface bound and
    names it ``gate`` where it binds; ``floor_world`` (trips per world) restricts the floor to the worlds that
    tripped (term ``floor_world``). Both absent: the result without them."""
    attempt = attempt or AttemptLevels()
    interface = np.asarray(interface, dtype=np.float64)
    gated = np.zeros(interface.shape, dtype=bool)
    if gate is not None:
        gate = np.asarray(gate, dtype=np.float64)
        gated = gate < interface
        interface = np.minimum(interface, gate)
    band = np.broadcast_to(np.asarray(band, dtype=np.float64), interface.shape)
    j_if, j_band = _levels(interface, grid), _levels(band, grid)
    j = np.maximum(j_if, j_band)
    term = np.where(j == 0, 0, np.where(j_band > j_if, 2, 1))
    term = np.where(np.isinf(np.minimum(interface, band)) & (j == 0), 0, term)
    term = np.where((term == 1) & gated, 5, term)
    floor = np.full(j.shape, attempt.floor, dtype=np.int64)
    if floor_world is not None:
        floor = np.where(np.asarray(floor_world) > 0, floor, 0)
    below = j < floor
    j, term = np.where(below, floor, j), np.where(below, 6 if floor_world is not None else 3, term)
    if attempt.extra > 0:
        j, term = j + attempt.extra, np.full_like(term, 4)
    if attempt.j_attempt is not None:
        j = np.minimum(j, int(attempt.j_attempt))
    return WorldNeed(j=j.astype(np.int64), term=term.astype(np.int64))


def shared_level(need: WorldNeed) -> int:
    """The level a clock shared by these worlds takes: the finest of theirs."""
    return int(need.j.max()) if need.j.size else 0


def saving(need: WorldNeed) -> float:
    """sum_w 2^j_w / (W 2^J): the share of the shared clock's exchanges the worlds' own levels would run."""
    if not need.j.size:
        return 1.0
    return float(np.sum(2.0 ** need.j) / (need.j.size * 2.0 ** shared_level(need)))


__all__ = ["TERMS", "AttemptLevels", "ClockGrid", "WorldNeed", "per_world_rungs", "saving", "shared_level"]
