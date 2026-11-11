# SPDX-License-Identifier: Apache-2.0
"""A world reset clears its health (SCR-WP5-19, coordinator GO 2026-11-08 21:17Z).

Before: ``sim.reset()`` restored the state but none of HealthMonitor's per-world arrays, so a world that once went
INVALID stayed latched and frozen after the reset (CPU: a blown sheet reset and stepped 0.1 s moved 0.0 m), and the
sticky peaks (CLOTH_STRETCH, SPIN) and fallback counters (FALLBACK_USED) carried over into the new run. The Gym envs
(``alloy.envs.health.WorldHealth``) and the bully (``alloy.bully.scene_sim``) each restored them themselves.

``WorldReset(monitor)`` keeps build-time copies of the monitor's per-world arrays: ``state_arrays()`` (record, inbox,
latch, time, energy reference) and the run peaks of ``RangeChecks`` and ``StartChecks``. ``monitor.reset_worlds(mask)``
puts them back for the masked worlds (all worlds if None) and calls every capacity source's ``reset`` (the MuJoCo
Warp pivot-clamp counter, WP1's self-contact watch). The caller then takes the energy reference again
(``reset_reference``), as ``IntegratedSimulation.reset`` does. Worlds outside the mask keep every array, bit for bit.
Start overlaps (START_PEN) are judged once per build: a reset world starts from the same placement and keeps them.
"""

from __future__ import annotations

from typing import Any, Callable

import warp as wp

from alloy._vendor import newton
from alloy.core.snapshot import WorldSnapshot


@wp.kernel
def _zero_masked(values: wp.array[wp.int32], mask: wp.array[wp.int32], all_worlds: int):
    w = wp.tid()
    if all_worlds != 0 or mask[w] != 0:
        values[w] = 0


def zero_worlds(values: wp.array) -> Callable[[wp.array | None], None]:
    """A ``CapacitySource.reset`` for a per-world int32 counter: zero the masked worlds (all if None)."""
    def reset(world_mask: wp.array | None) -> None:
        mask = world_mask if world_mask is not None else values
        wp.launch(_zero_masked, dim=values.shape[0], inputs=[values, mask, int(world_mask is None)],
                  device=values.device)
    return reset


class WorldReset:
    """Build-time copies of a HealthMonitor's per-world arrays (made at construction, before any step)."""

    def __init__(self, monitor: Any):
        self.monitor = monitor
        arrays = {f"health/{k}": v for k, v in monitor.state_arrays().items()}
        for name, arr in (("cloth_peak", monitor.ranges.peak), ("cloth_tri", monitor.ranges.peak_tri),
                          ("spin_peak", monitor.start.peak), ("spin_body", monitor.start.peak_body)):
            arrays[f"health/{name}"] = arr
        self._snap = WorldSnapshot(monitor.model, newton.State(), arrays)
        self._state = newton.State()
        self._snap.save(self._state)

    def __call__(self, world_mask: wp.array | None = None) -> None:
        self._snap.restore(self._state, world_mask)
        for src in getattr(self.monitor, "_sources", ()):
            if getattr(src, "reset", None) is not None:
                src.reset(world_mask)


def reset_worlds(monitor: Any, world_mask: wp.array | None = None) -> None:
    """``HealthMonitor.reset_worlds``: the masked worlds' health as built (module docstring). Example:
    ``sim.engine.health.reset_worlds(mask)`` before ``reset_reference(state, mask)``."""
    monitor._world_reset(world_mask)


__all__ = ["WorldReset", "reset_worlds", "zero_worlds"]
