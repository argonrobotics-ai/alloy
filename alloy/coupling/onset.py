# SPDX-License-Identifier: Apache-2.0
"""Force-jump re-runs (SCR-WP11-4): a period whose exchange injected a jump of energy runs again at a finer dt_x.

At a force jump (first contact) the lagged proxy books ``R^2 dt_x^2 / m``: the destination's proxy runs a force-free
step while the source decelerates under the harvested R (WP11's diagnosis, [internal note]). The
watchdog flags such an exchange (``EnergyWatchdog``: bond > ``onset_gain_per_exchange`` x E_scale in a world this
refiner armed, with no other escalation in the attempt). ``OnsetRefiner`` arms the worlds and decides, on the host
once per attempt, whether to re-run the period at dt_x / 2**level:

- a world is armed at most once per ``onset_cooldown_s`` (and through the levels of its own re-run): a first contact
  is one event; a bond that stays over the limit under a held load (WP4's clamped towel: contacts breaking and
  re-forming every few periods) is not chased period after period, and it is not flagged either, so the world's
  watch state is the ladder's alone;
- only when every escalated world is a force jump (an escalation for another reason takes the ladder);
- at most ``onset_refine_levels`` halvings, and the last level runs with the trigger off. The levels are not cut
  short when one does not help: the onset injection falls about as dt_x^4 overall but not monotonically per level
  (the contact phase changes with dt_x; CPL-4 at 3.5 mm: 5.0e-3, 2.4e-4, 2.6e-4, 3e-5 E_scale at /1, /2, /4, /8);
- for that period only: the clock, the ladder and the rollback budget are untouched.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from alloy.core.timebase import grid_dt
from alloy.coupling.types import WatchState

if TYPE_CHECKING:
    from alloy.coupling.types import CouplingReport, WatchdogPolicy

log = logging.getLogger("alloy.coupling")

__all__ = ["OnsetRefiner", "escalated_beyond_onset"]


def _escalated(report: "CouplingReport") -> np.ndarray:
    return np.asarray(report.watch_state) == int(WatchState.ESCALATED)


def _onset(report: "CouplingReport", like: np.ndarray) -> np.ndarray:
    onset = np.asarray((report.notes or {}).get("onset", np.zeros_like(like)), dtype=bool)
    return onset if onset.shape == like.shape else np.zeros_like(like)


def escalated_beyond_onset(report: "CouplingReport | None") -> np.ndarray:
    """Worlds ESCALATED for a reason other than a force jump (bool [W]; empty without a report)."""
    if report is None:
        return np.zeros(0, dtype=bool)
    escalated = _escalated(report)
    return escalated & ~_onset(report, escalated)


@dataclass
class OnsetRefiner:
    """Host state of one Scheduler's force-jump re-runs."""

    policy: "WatchdogPolicy"
    level: int = 0                  # this period's extra dt_x halvings
    reruns: int = 0                 # total force-jump re-runs
    pending: bool = False           # on_rollback accepted a force-jump re-run; the next choose_clock refines
    period: int = 0                 # periods begun (new_period)
    control_dt: float = 0.0         # [s], from attempt()
    last: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # per world: period of its re-run
    parked: np.ndarray | None = None    # bool [W]: worlds the trigger never arms (held still, ``parked.py``)

    def new_period(self) -> None:
        self.level, self.pending, self.period = 0, False, self.period + 1

    def armed(self, world_count: int) -> np.ndarray:
        """int32 [W]: 1 where the trigger may flag a force jump this attempt (outside the world's cooldown, or inside
        its own re-run this period)."""
        if self.last.shape != (world_count,):
            self.last = np.full(world_count, -(2**62), dtype=np.int64)
        quiet = math.ceil(self.policy.onset_cooldown_s / self.control_dt - 1e-9) if self.control_dt > 0 else 0
        armed = ((self.last == self.period) | (self.period - self.last > quiet)).astype(np.int32)
        if self.parked is not None and self.parked.shape == armed.shape:
            armed[self.parked] = 0                 # a parked world is held: what it does in an exchange is no event
        return armed

    def wants_rerun(self, report: "CouplingReport") -> bool:
        """Some world escalated and every escalated world is a force jump (no side effects; only armed worlds are
        flagged, and the last level runs with the trigger off)."""
        escalated = _escalated(report)
        return bool(escalated.any() and not (escalated & ~_onset(report, escalated)).any())

    def accept(self, report: "CouplingReport") -> bool:
        """on_rollback: True (and the re-run is armed) when the rolled-back attempt was a force jump."""
        if not self.wants_rerun(report):
            return False
        self.pending = True
        self.reruns += 1
        return True

    def attempt(self, control_dt: float, j: int, dt_x: float, j_max: int) -> tuple[float, float]:
        """(dt_x, single-exchange limit) for the next attempt: halved once more after an accepted re-run; the
        trigger is off (inf) when no finer level is left."""
        self.control_dt = float(control_dt)
        if self.pending:
            self.pending, self.level = False, self.level + 1
        level_j = min(j + self.level, j_max)
        last = self.level >= self.policy.onset_refine_levels or level_j >= j_max
        limit = math.inf if last else self.policy.onset_gain_per_exchange
        if not self.level:
            return dt_x, limit
        fine = grid_dt(control_dt, level_j)
        log.info("coupling force jump: period re-run %d at dt_x %.4g s (clock %.4g s)", self.level, fine, dt_x)
        return fine, limit

    def notes(self, onset_only: Any, exchange_peak: Any) -> dict[str, Any]:
        """Report notes; a flagged world's cooldown starts in the period it was flagged (also without a rollback)."""
        onset = np.asarray(onset_only, dtype=bool)
        if self.last.shape == onset.shape:
            self.last[onset] = self.period
        return {"onset": onset, "exchange_peak": np.asarray(exchange_peak, dtype=float),
                "onset_level": self.level, "onset_reruns": self.reruns}
