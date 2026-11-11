# SPDX-License-Identifier: Apache-2.0
"""Pushes: a random horizontal force on the robot's base for a short time, on a timer, in every world.

The force is held through ``sim.apply_wrench`` (world frame, at the base's centre of mass), so the engine's health
check books its work as outside work. Every world is pushed at the same moments with its own force. Example::

    Pushes(every_s=8.0, duration_s=0.1, force=(80.0, 80.0, 0.0))     # up to +-80 N along x and y for 0.1 s
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError


@dataclass(frozen=True)
class Pushes:
    """When and how hard the base is pushed. ``force``: the largest |force| per world axis (N), drawn uniformly."""

    every_s: float = 8.0
    duration_s: float = 0.1
    force: tuple[float, float, float] = (80.0, 80.0, 0.0)

    def validate(self) -> None:
        """Raise ``EnvError`` on a bad value. Example: ``Pushes(every_s=0).validate()`` raises."""
        if self.every_s <= 0 or self.duration_s <= 0 or self.duration_s >= self.every_s or min(self.force) < 0:
            raise EnvError("pushes need 0 < duration_s < every_s and force >= 0",
                           "e.g. Pushes(every_s=8, duration_s=0.1, force=(80, 80, 0))", field="pushes")


class Pusher:
    """Applies the pushes of a ``Pushes`` spec; ``force`` (W, 3) is the force held now (zeros between pushes)."""

    def __init__(self, spec: Pushes | None, sim: Any, body: str, dt: float, xp: Any) -> None:
        self.spec, self.sim, self.body, self.xp = spec, sim, body, xp
        self.worlds = int(sim.worlds)
        self.force = xp.zeros((self.worlds, 3))
        self.active = False
        if spec is None:
            return
        spec.validate()
        self.every = max(2, int(round(spec.every_s / dt)))
        self.length = min(self.every - 1, max(1, int(round(spec.duration_s / dt))))
        self._half = xp.asarray(np.asarray(spec.force, np.float32))

    def before_step(self, tick: int) -> None:
        """Start or end a push before control step number ``tick`` (0, 1, 2, ... since the env was made)."""
        if self.spec is None or tick == 0:
            return
        phase = tick % self.every
        if phase == 0:
            self.hold(self.xp.uniform((self.worlds, 3), -1.0, 1.0) * self._half)
        elif phase == self.length and self.active:
            self.clear()

    def hold(self, force: Any) -> None:
        """Hold ``force`` (W, 3) N on the base until ``clear``. Example: ``pusher.hold(f)``."""
        self.force = self.xp.asarray(force)
        self.sim.apply_wrench(self.body, force=self.xp.to_numpy(self.force))
        self.active = True

    def clear(self) -> None:
        """Remove the push. Example: ``pusher.clear()``."""
        self.force = self.xp.zeros((self.worlds, 3))
        self.sim.apply_wrench(self.body, force=(0.0, 0.0, 0.0))
        self.active = False


__all__ = ["Pusher", "Pushes"]
