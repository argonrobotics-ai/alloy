# SPDX-License-Identifier: Apache-2.0
"""Escalation ladder (coupling-time §4.4.5), first handled within its period (SCR-WP3-C3RAND fix 2, WP4 2026-11-08).

A ladder escalation (a world's energy or residual rule tripped, not a force jump) used to go straight to the sticky
ladder, ``escalate``. Trip 1 raised ``ClockController.j_floor``, which never relaxes; trip 2 switched the pairs to
their strong mode; both for every world. With many worlds that ratchets: XB4-9's randomized C3 box-on-pad (64 worlds,
L4, main 377349c8e) tripped in 1-11 different worlds in each of periods 1-5. The floor went from j 4 to 7 (dt_x 0.156
to 0.0195 ms) and the pair went staggered, for all 64 worlds, and 0.5 s never finished.

``PeriodLadder`` handles a first trip within its period. The driver's per-world rollback re-runs the escalated worlds
(the others keep their first run bitwise), one grid level finer per re-run, up to ``max_rollbacks_per_period``. The
floor, the coupling mode and the trips counter stay as they were, and the next period starts on the clock the scene's
bounds pick. The sticky ladder still takes over:
- when a world trips again within the watchdog's window (``WatchdogPolicy.window``) of an earlier period's trip: a
  resting instability (a light box on cloth at too coarse a step, ``test_watchdog``) or a dt-independent added mass
  needs the lasting remedy;
- when a period ends still escalated (its re-runs spent): the next period takes it;
- when ``WatchdogPolicy.per_period_ladder`` is False (today's behaviour).
The re-run itself still steps every world at the finer level for that period. That is the known gap
coupling-shared-exchange-clock (WP3); what this removes is the lasting part.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger("alloy.coupling")

__all__ = ["PeriodLadder", "escalate"]


class PeriodLadder:
    """Host state: this period's extra grid levels and each world's last ladder trip (period index)."""

    def __init__(self, on: bool = True, window_s: float = 1.0) -> None:
        self.on, self.window_s = bool(on), float(window_s)
        self.period = 0
        self.level = 0              # this period's extra halvings
        self.refines = 0            # within-period re-runs over the run (report notes)
        self.last = np.zeros(0, dtype=np.int64)
        self.per_world = np.zeros(0, dtype=np.int64)    # per world: its within-period re-runs (sim.health())

    def new_period(self) -> None:
        self.period, self.level = self.period + 1, 0

    def within_period(self, worlds: np.ndarray, rerun: bool, control_dt: float) -> bool:
        """For the escalated ``worlds`` (bool [W]) of the attempt just reported: True when this period's re-run takes
        one more level (no sticky change), False when the sticky ladder must act (module docstring)."""
        worlds = np.asarray(worlds, dtype=bool)
        if self.last.shape != worlds.shape:
            self.last = np.full(worlds.shape, -(2**62), dtype=np.int64)
            self.per_world = np.zeros(worlds.shape, dtype=np.int64)
        recent = max(math.ceil(self.window_s / control_dt - 1e-9), 1) if control_dt > 0 else 1
        age = self.period - self.last
        repeat = bool(np.any(worlds & (age > 0) & (age <= recent)))
        self.last[worlds] = self.period
        if not (self.on and rerun) or repeat:
            return False
        self.level += 1
        self.refines += 1
        self.per_world[worlds] += 1
        log.info("coupling escalation in %d world(s): period re-run at %d more level(s), for this period only (the "
                 "clock floor and the coupling mode stay)", int(worlds.sum()), self.level)
        return True

    def j(self, j: int, j_max: int) -> int:
        """The attempt's grid level: the clock's, plus this period's extra levels."""
        return min(int(j) + self.level, int(j_max))

    def notes(self) -> dict[str, Any]:
        return {"ladder_level": self.level, "ladder_refines": self.refines,
                "ladder_refines_per_world": self.per_world.copy()}

    def counts(self, world_count: int) -> list[int]:
        """Per world: the periods re-run finer for its escalation (0 for every world before any)."""
        return self.per_world.tolist() if self.per_world.shape == (world_count,) else [0] * int(world_count)


def escalate(sched: Any, worlds: np.ndarray) -> None:
    """The sticky ladder (coupling-time §4.4.5): trip 1 halves dt_x for good (the clock's floor); trip 2 the strong
    mode (staggered + Aitken ω0_strong / ADMM ρ×4: a dt-independent added mass); later trips halve dt_x; then the pairs
    are strong and the worlds FAILED."""
    sched._trips = getattr(sched, "_trips", 0) + 1
    if sched._trips == 2 and sched._escalate_mode():
        return
    if sched._clock.escalate() or sched._escalate_mode():
        return
    for p in sched._alloy_pairs:
        sched._strong.add((p.source, p.destination))
    sched._watchdog.mark_failed(worlds)
    log.error("coupling escalation exhausted: pairs marked strong %s, %d worlds FAILED", sorted(sched._strong),
              int(np.sum(worlds)))
