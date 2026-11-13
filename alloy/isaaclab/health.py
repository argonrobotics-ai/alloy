# SPDX-License-Identifier: Apache-2.0
"""Alloy's per-world health checks on Isaac Lab's worlds: bad worlds named, counted and reported, not kept silently.

Every Isaac Lab physics step, Alloy's ``HealthMonitor`` (``alloy.precision.health``) checks each world on the device:
non-finite state, penetration past 1 cm (PEN_FAIL), speeds past 20 m/s (SPEED_FAIL), more contacts or constraint rows
than MuJoCo Warp's buffers hold (CONTACT_OVERFLOW / ROW_OVERFLOW: contacts silently dropped), joints pulled apart
(JOINT_FAIL), and more. Kernels only, so it rides inside Isaac Lab's CUDA graph.

Isaac Lab resets worlds itself, so a world can go bad and be reset before anyone looks. When Isaac Lab resets a world,
:meth:`IsaacLabHealth.on_reset` first adds that world's verdict to running per-world counts (episodes that ended
INVALID / DEGRADED and the flags seen), then clears its record. :meth:`report` reads both on the host;
:meth:`invalid_now` gives a torch-ready mask for a termination term. Nothing here changes the physics: worlds are
flagged, not frozen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import warp as wp

from alloy.precision.health import HealthMonitor
from alloy.precision.health_reset import reset_worlds
from alloy.precision.types import CheckPolicy, HealthFlags, Severity, StepHealth

log = logging.getLogger("alloy.isaaclab")

# every check every step: Isaac Lab replays one captured graph per step, so a host step counter cannot pick which
# checks run (Alloy's own graphs key on the check pattern instead)
POLICY = CheckPolicy(level="full", every=1, full_every=1)


@wp.kernel
def _fold_and_mask(record: wp.array[StepHealth], reset_mask: wp.array[wp.bool], world_count: int,
                   n_invalid: wp.array[wp.int32], n_degraded: wp.array[wp.int32], flags_seen: wp.array[wp.uint32],
                   mask_i32: wp.array[wp.int32]):
    """For each world Isaac Lab resets: count how its episode ended, keep its flags, mark it for the record reset."""
    w = wp.tid()
    on = reset_mask[w] or reset_mask[world_count]     # Isaac Lab's last slot: the global entities, i.e. all worlds
    mask_i32[w] = wp.where(on, 1, 0)
    if not on:
        return
    sev = record[w].severity
    if sev >= 2:
        n_invalid[w] = n_invalid[w] + 1
    elif sev == 1:
        n_degraded[w] = n_degraded[w] + 1
    flags_seen[w] = flags_seen[w] | record[w].flags


@dataclass
class HealthSummary:
    """Host view: worlds INVALID / DEGRADED now, episodes that ended so, flags by name, the worst world's story."""

    worlds: int
    invalid_now: list[int]
    degraded_now: int
    episodes_invalid: int
    episodes_degraded: int
    flags: dict[str, int] = field(default_factory=dict)   # flag -> worlds that have shown it (now or in a past episode)
    first_failure: str = ""

    def line(self) -> str:
        flags = ", ".join(f"{k} {v}" for k, v in sorted(self.flags.items(), key=lambda kv: -kv[1])) or "none"
        return (f"[INFO] [alloy] health: {len(self.invalid_now)} of {self.worlds} worlds INVALID now, "
                f"{self.degraded_now} DEGRADED; episodes ended INVALID {self.episodes_invalid}, DEGRADED "
                f"{self.episodes_degraded}; worlds per flag: {flags}" + (f"; first: {self.first_failure}"
                                                                         if self.first_failure else ""))


class IsaacLabHealth:
    """One ``HealthMonitor`` over the bridged model, fed after every Isaac Lab physics step."""

    QUIET = int(HealthFlags.DET_DOWNGRADED | HealthFlags.FAR_FROM_ORIGIN)   # notices; never a reason to report

    def __init__(self, solver: Any, speed_limit_mps: float | None = None):
        model = solver.model
        policy = POLICY if speed_limit_mps is None else replace(POLICY, speed_fail_mps=float(speed_limit_mps))
        self.monitor = HealthMonitor(model, policy, solvers=[solver])
        self.world_count, self.device = int(model.world_count), model.device
        W, dev = self.world_count, self.device
        self.n_invalid = wp.zeros(W, dtype=wp.int32, device=dev)
        self.n_degraded = wp.zeros(W, dtype=wp.int32, device=dev)
        self.flags_seen = wp.zeros(W, dtype=wp.uint32, device=dev)
        self._mask_i32 = wp.zeros(W, dtype=wp.int32, device=dev)
        self._step = 0

    def update(self, state_in: Any, state_out: Any, contacts: Any, dt: float) -> None:
        self.monitor.update(self._step, state_in, state_out, contacts, dt=dt)

    def on_reset(self, world_mask: wp.array) -> None:
        """Isaac Lab reset the masked worlds (bool, ``world_count + 1``): fold their verdicts, then clear them."""
        wp.launch(_fold_and_mask, dim=self.world_count,
                  inputs=[self.monitor.record, world_mask, self.world_count, self.n_invalid, self.n_degraded,
                          self.flags_seen, self._mask_i32], device=self.device)
        reset_worlds(self.monitor, self._mask_i32)

    def invalid_now(self) -> wp.array:
        """int32 per world, 1 where the world is INVALID now (``wp.to_torch`` for a termination term)."""
        out = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        wp.launch(_invalid_mask, dim=self.world_count, inputs=[self.monitor.record, out], device=self.device)
        return out

    def report(self) -> HealthSummary:
        rec = self.monitor.record.numpy()
        sev, flags = rec["severity"].astype(np.int32), rec["flags"].astype(np.uint32)
        seen = flags | self.flags_seen.numpy()
        per_flag = {f.name: int(np.count_nonzero(seen & int(f))) for f in HealthFlags
                    if f.name and not (int(f) & self.QUIET) and np.any(seen & int(f))}
        invalid = [int(w) for w in np.flatnonzero(sev >= int(Severity.INVALID))]
        first = ""
        if invalid:
            try:
                first = f"world {invalid[0]}: {self.monitor.poll().failure_text(invalid[0])}"
            except Exception as err:  # the summary must never break a training run
                first = f"world {invalid[0]} ({err})"
        return HealthSummary(self.world_count, invalid, int(np.count_nonzero(sev == int(Severity.DEGRADED))),
                             int(self.n_invalid.numpy().sum()), int(self.n_degraded.numpy().sum()), per_flag, first)


@wp.kernel
def _invalid_mask(record: wp.array[StepHealth], out: wp.array[wp.int32]):
    w = wp.tid()
    out[w] = wp.where(record[w].severity >= 2, 1, 0)
