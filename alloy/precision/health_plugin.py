# SPDX-License-Identifier: Apache-2.0
"""Plug-in depth checks in the HealthMonitor (coordinator ruling 2026-11-04 18:25Z, SCR-SC11-1 item 4).

Some plug-ins measure overlap that is not a Newton contact, so the contact checks cannot see it. First user: SC11's
puncture plug-in, whose ``untracked_depth`` is the needle or thread crossing the pad surface away from open holes. Such
a plug-in registers, under a name, a per-world depth array (float32 [world_count], metres; 0 or NaN = nothing
crossed) and a limit:
- explicitly, with ``HealthMonitor.register_depth_check(name, depth, limit_m)``;
- or by exposing ``health_depth_checks()`` on its solver (an iterable of ``DepthCheck``), read when the monitor is
  built.

Each health check launches one kernel per registered check, in registration order. The first check whose depth
exceeds its limit raises PEN_FAIL (INVALID) for that world. The record keeps that check's index and depth
(``plugin_check``, ``plugin_depth_m``), and the first-failure snapshot keeps them as for every flag, so
``failure_text`` names the plug-in. When no check fails, ``plugin_depth_m`` is the deepest registered depth of that
check.

Register before any graph capture: the launches are part of the captured update. With nothing registered, nothing
launches and the record is as before.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from typing import Any

import warp as wp

from alloy.precision.health_kernels import Accum
from alloy.precision.types import DepthCheck

log = logging.getLogger("alloy.precision")


@wp.kernel
def check_plugin_depth(depth: wp.array[wp.float32], limit: wp.float32, index: wp.int32, acc: Accum):
    """One registered check, one thread per world. A world already failed by an earlier check keeps that one."""
    w = wp.tid()
    d = depth[w]
    if not (d > 0.0) or acc.plug_check[w] >= 0:
        return
    if d > limit:
        acc.plug_check[w] = index
        acc.plug_depth[w] = d
    elif d > acc.plug_depth[w]:
        acc.plug_depth[w] = d


class PluginChecks:
    """The registered depth checks of one HealthMonitor."""

    def __init__(self, world_count: int, device: Any) -> None:
        self.world_count = int(world_count)
        self.device = wp.get_device(device)
        self._checks: list[DepthCheck] = []

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._checks)

    def register(self, check: DepthCheck) -> int:
        """Validate and add one check; returns its index (the value of ``plugin_check`` when it trips)."""
        _validate(check, self.world_count, self.device, self.names)
        self._checks.append(check)
        log.info("health: plug-in depth check %r registered (limit %.3g m)", check.name, check.limit_m)
        return len(self._checks) - 1

    def discover(self, solvers: Sequence[Any]) -> None:
        """Register the checks every solver that exposes ``health_depth_checks()`` returns."""
        for solver in solvers:
            fn = getattr(solver, "health_depth_checks", None)
            if callable(fn):
                for check in _as_checks(fn(), type(solver).__name__):
                    self.register(check)

    def launch(self, acc: Accum) -> None:
        for i, c in enumerate(self._checks):
            wp.launch(check_plugin_depth, dim=self.world_count, inputs=[c.depth, wp.float32(c.limit_m), i, acc],
                      device=self.device)


def _as_checks(items: Iterable[Any], owner: str) -> list[DepthCheck]:
    out = list(items)
    bad = [x for x in out if not isinstance(x, DepthCheck)]
    if bad:
        raise TypeError(f"{owner}.health_depth_checks() must yield DepthCheck, got {type(bad[0]).__name__}")
    return out


def _validate(check: DepthCheck, world_count: int, device: Any, taken: tuple[str, ...]) -> None:
    if not isinstance(check.name, str) or not check.name:
        raise ValueError("a plug-in depth check needs a non-empty name")
    if check.name in taken:
        raise ValueError(f"plug-in depth check {check.name!r} is already registered")
    d = check.depth
    if not isinstance(d, wp.array) or d.dtype != wp.float32 or d.ndim != 1 or d.shape[0] != world_count:
        raise ValueError(f"plug-in depth check {check.name!r}: depth must be a float32 wp.array of length {world_count}")
    if wp.get_device(d.device) != device:
        raise ValueError(f"plug-in depth check {check.name!r}: depth is on {d.device}, the monitor on {device}")
    if not (math.isfinite(check.limit_m) and check.limit_m > 0.0):
        raise ValueError(f"plug-in depth check {check.name!r}: limit_m must be finite and > 0, got {check.limit_m}")
