# SPDX-License-Identifier: Apache-2.0
"""MuJoCo at one step per exchange in coupled scenes with precision pairs (SCR-JOB2-1 item 2, ruling 2026-11-04).

``check_isolated`` refuses a MuJoCo entry that substeps: its later substeps would get precision rows posed at the
exchange start. So ``make_entries`` pins that entry at rung 1 (``max_substeps`` 1) with the pairs' L2 dt as its
target, and here:

- ``pin_exchange`` caps the Scheduler's exchange interval (``ClockPolicy.dt_x_max``) at the pinned entry's target dt,
  so its one step per exchange is never longer than that target;
- ``RungPinWatch`` compares, once per period, the entry's per-world stability estimate (WP4's
  ``CouplingReport.notes["stable_dt"]``) with the exchange interval. In a world whose estimate asks for a smaller
  step than one step per exchange gives, the health monitor raises the named DEGRADED flag ``HealthFlags.RUNG_PINNED``
  ("precision rung pinned", WP5) through its ``capacity_flags``. The rung never changes: the card lowers the pair dt.

Scenes without precision pairs, or with one dynamic entry, are not touched.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy.coupling.types import ClockPolicy, SchedulerPolicy
from alloy.scene.types import BuiltScene

log = logging.getLogger("alloy.integration")

__all__ = ["RungPinWatch", "pin_exchange", "pinned_entries"]


def pinned_entries(built: BuiltScene) -> dict[str, float]:
    """Target dt [s] of each MuJoCo entry that steps once per exchange in a coupled scene with precision pairs."""
    dyn = [e for e in built.entries if e.name]
    if len(dyn) < 2 or not built.precision_pairs:
        return {}
    return {e.name: float(e.target_dt) for e in dyn
            if e.capability == "mujoco" and max(int(e.substeps), int(e.max_substeps)) == 1}


def pin_exchange(policy: SchedulerPolicy, built: BuiltScene) -> SchedulerPolicy:
    """``policy`` with ``clock.dt_x_max`` capped at the pinned entries' target dt (unchanged without one)."""
    pinned = pinned_entries(built)
    if not pinned:
        return policy
    cap = min(pinned.values())
    if policy.clock.dt_x_max <= cap:
        return policy
    log.info("precision rung pinned: exchange interval capped at %.6g s (MuJoCo entries %s at one step per exchange)",
             cap, sorted(pinned))
    return dataclasses.replace(policy, clock=dataclasses.replace(policy.clock, dt_x_max=cap))


class RungPinWatch:
    """Per-world int32 flag array for ``HealthMonitor(capacity_flags=[(flags, RUNG_PINNED)])``, refreshed once per
    period from the stepper's report (module docstring)."""

    def __init__(self, pinned: Mapping[str, float], clock: ClockPolicy, world_count: int, device: Any) -> None:
        self.pinned, self.clock = dict(pinned), clock
        self.flags = wp.zeros(int(world_count), dtype=wp.int32, device=device)
        self.flagged = np.zeros(int(world_count), dtype=bool)        # host copy of the last update
        self._said: set[str] = set()

    @classmethod
    def for_built(cls, built: BuiltScene, policy: SchedulerPolicy) -> RungPinWatch | None:
        pinned = pinned_entries(built)
        if not pinned:
            return None
        return cls(pinned, policy.clock, built.world_count, built.model.device)

    def health_source(self) -> tuple[wp.array, int] | None:
        """``(flags, HealthFlags.RUNG_PINNED)`` for the health monitor; None (logged) until WP5's flag exists."""
        from alloy.precision.types import HealthFlags  # noqa: PLC0415 - WP5

        bit = getattr(HealthFlags, "RUNG_PINNED", None)
        if bit is None:
            log.warning("precision rung pinned: HealthFlags.RUNG_PINNED is not defined yet (WP5); a pinned MuJoCo "
                        "entry whose stability estimate asks for a smaller dt is logged, not flagged")
            return None
        return self.flags, int(bit)

    def update(self, report: Any) -> np.ndarray:
        """Flag the worlds whose pinned entry needs a smaller step than the exchange interval; returns the mask."""
        stable = (getattr(report, "notes", None) or {}).get("stable_dt")
        if stable is None:
            self._once("missing", "precision rung pinned: the stepper reports no per-world stability estimate "
                                  "(CouplingReport.notes['stable_dt'], WP4); nothing is flagged")
            return self.flagged
        dt_x, limit = float(report.dt_x), float(self.clock.safety)
        mask = np.zeros_like(self.flagged)
        for name in self.pinned:
            est = stable.get(name)
            if est is None:
                self._once(f"no-estimator:{name}", f"precision rung pinned: entry {name!r} has no stability estimate "
                           "(no estimator registered for its solver); it is not flagged")
                continue
            mask |= dt_x > limit * np.asarray(est, dtype=np.float64)
        if mask.any():
            self._once("unmet", f"precision rung pinned: {int(mask.sum())} world(s) ask for a step below the "
                                f"exchange interval {dt_x:.6g} s at one MuJoCo step per exchange; lower the pair "
                                "dt on the card (flag RUNG_PINNED, DEGRADED)")
        if not np.array_equal(mask, self.flagged):
            self.flags.assign(mask.astype(np.int32))
            self.flagged = mask
        return mask

    def _once(self, key: str, msg: str) -> None:
        if key not in self._said:
            self._said.add(key)
            log.warning(msg)
