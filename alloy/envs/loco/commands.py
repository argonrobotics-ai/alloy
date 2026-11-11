# SPDX-License-Identifier: Apache-2.0
"""Velocity commands: what each world's robot is asked to do, redrawn on a timer and at every reset.

A command is (forward m/s, left m/s, yaw rate rad/s) in the robot's own heading frame. Example::

    spec = VelocityCommands(lin_x=(-1.0, 1.0), lin_y=(-0.5, 0.5), yaw=(-1.0, 1.0), resample_s=10.0)
    env.commands.set((0.8, 0.0, 0.0))          # the same fixed command in every world (for a rollout or a clip)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError


@dataclass(frozen=True)
class VelocityCommands:
    """Ranges the command is drawn from (uniform), how often, and the share of draws that ask to stand still."""

    lin_x: tuple[float, float] = (-1.0, 1.0)     # m/s, forward
    lin_y: tuple[float, float] = (-0.5, 0.5)     # m/s, to the left
    yaw: tuple[float, float] = (-1.0, 1.0)       # rad/s, counter-clockwise seen from above
    resample_s: float = 10.0                     # a new draw this often within an episode
    standing: float = 0.1                        # share of draws that are (0, 0, 0)
    deadband: float = 0.1                        # a drawn planar speed below this (m/s) becomes 0

    def validate(self) -> None:
        """Raise ``EnvError`` on a bad range. Example: ``VelocityCommands(lin_x=(1, -1)).validate()`` raises."""
        for name in ("lin_x", "lin_y", "yaw"):
            lo, hi = getattr(self, name)
            if hi < lo:
                raise EnvError(f"command range {name}=({lo}, {hi})", "pass (low, high) with high >= low", field=name)
        if self.resample_s <= 0 or not 0.0 <= self.standing <= 1.0:
            raise EnvError("resample_s must be > 0 and standing within [0, 1]", "e.g. resample_s=10, standing=0.1",
                           field="resample_s")


class CommandSampler:
    """Per-world velocity commands ``value`` (W, 3). Example: ``sampler.draw(mask); sampler.value``."""

    def __init__(self, spec: VelocityCommands, worlds: int, dt: float, xp: Any) -> None:
        spec.validate()
        self.spec, self.worlds, self.xp = spec, worlds, xp
        self.every = max(1, int(round(spec.resample_s / dt)))                 # control steps between draws
        lo = np.array([spec.lin_x[0], spec.lin_y[0], spec.yaw[0]], np.float32)
        hi = np.array([spec.lin_x[1], spec.lin_y[1], spec.yaw[1]], np.float32)
        self._lo, self._span = xp.asarray(lo), xp.asarray(hi - lo)
        self.value = xp.zeros((worlds, 3))
        self.fixed = False

    def draw(self, mask: Any) -> None:
        """New commands for the worlds in the bool (W,) ``mask`` (no change after ``set``)."""
        if self.fixed:
            return
        xp, w, spec = self.xp, self.worlds, self.spec
        new = xp.uniform((w, 3), 0.0, 1.0) * self._span + self._lo
        slow = xp.norm(new[:, 0:2], -1) < spec.deadband
        planar = new[:, 0:2] * xp.float(~slow)[:, None]
        new = xp.cat([planar, new[:, 2:3]], -1)
        moving = xp.float(xp.uniform((w,), 0.0, 1.0) >= spec.standing)
        self.value = xp.where(mask[:, None], new * moving[:, None], self.value)

    def tick(self, steps: Any) -> None:
        """Redraw in worlds whose episode step count (W,) hit the resample period. Example: ``tick(s["steps"])``."""
        self.draw((steps > 0) & (steps % self.every == 0))

    def set(self, command: Any) -> None:
        """Hold one command (3,) or one per world (W, 3) until ``release``. Example: ``set((0.8, 0, 0))``."""
        value = self.xp.asarray(command)          # numbers, a NumPy array, or a tensor on any device
        if tuple(value.shape) not in ((3,), (self.worlds, 3)):
            raise EnvError(f"command has shape {tuple(value.shape)}", f"pass (vx, vy, yaw_rate) or an array of shape "
                           f"({self.worlds}, 3)", field="command")
        self.value = self.xp.zeros((self.worlds, 3)) + value
        self.fixed = True

    def release(self) -> None:
        """Go back to random commands (drawn at the next reset or resample time)."""
        self.fixed = False


__all__ = ["CommandSampler", "VelocityCommands"]
