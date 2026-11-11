# SPDX-License-Identifier: Apache-2.0
"""Action delay: a joint target reaches its servo a fixed time after it was commanded.

``DelayLine`` is the delay itself, on host arrays: push the targets commanded at each control period, get back the
ones commanded ``delay`` seconds before, linearly interpolated between control periods for a delay that is not a
whole number of them. The fit pushes each world's candidate commands through one; ``scene.add_robot(sysid=)``
installs one in front of the robot's servos (``SimDelay``), so every way of commanding the robot is delayed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class DelayLine:
    """A fractional delay of ``steps`` control periods per column. ``steps``: any shape, >= 0.

    Example: ``line = DelayLine(np.array([1.5])); line.push(np.array([1.0]))`` returns the first value pushed
    until 1.5 periods have passed."""

    def __init__(self, steps: Any) -> None:
        self.steps = np.maximum(np.asarray(steps, np.float64), 0.0)
        self.whole = np.floor(self.steps).astype(np.int64)
        self.frac = self.steps - self.whole
        self.depth = int(self.whole.max(initial=0)) + 2
        self.history: np.ndarray | None = None       # (depth, *shape): history[k] was pushed k periods ago
        self._at = np.indices(self.steps.shape)
        self._fresh = np.zeros(self.steps.shape, bool)

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Forget the history (of the columns in ``mask``): the next push fills it with its own values."""
        if mask is None or self.history is None:
            self.history = None
        else:
            self._fresh |= np.asarray(mask, bool)

    def push(self, values: np.ndarray) -> np.ndarray:
        """Take this period's commanded values, return the delayed ones (same shape)."""
        values = np.asarray(values, np.float64)
        if self.history is None:
            self.history = np.broadcast_to(values, (self.depth, *values.shape)).copy()
        else:
            self.history = np.roll(self.history, 1, axis=0)
            self.history[0] = values
        if self._fresh.any():
            self.history[:, self._fresh] = values[self._fresh]
            self._fresh[...] = False
        near = self.history[(self.whole, *self._at)]
        far = self.history[(self.whole + 1, *self._at)]
        return (1.0 - self.frac) * near + self.frac * far


class SimDelay:
    """Delays the joint targets of some control rows of a running simulation.

    Before each control period ``before()`` reads what is commanded now (whoever wrote it: ``sim.set_targets``,
    ``sim.run(targets=)``, a Gym env's action writer), and puts the delayed values in its place for the step;
    ``after()`` puts the commanded values back, so the arrays always read as what was last commanded."""

    def __init__(self, engine: Any, rows: dict[str, np.ndarray], seconds: dict[str, np.ndarray], dt: float) -> None:
        self.engine, self.rows = engine, rows
        self.lines = {k: DelayLine(np.asarray(seconds[k], np.float64) / float(dt)) for k in rows}
        self._held: dict[str, np.ndarray] = {}

    def _array(self, kind: str) -> Any:
        ctrl = self.engine.control
        return ctrl.mujoco.ctrl if kind == "ctrl" else ctrl.joint_target_q

    def before(self) -> None:
        """Swap the commanded targets for the delayed ones (call right before a control period)."""
        for kind, rows in self.rows.items():
            arr = self._array(kind)
            host = arr.numpy()
            commanded = host[rows].astype(np.float64)
            self._held[kind] = commanded
            host[rows] = self.lines[kind].push(commanded)
            arr.assign(host)

    def after(self) -> None:
        """Put the commanded targets back (call right after the control period)."""
        for kind, rows in self.rows.items():
            arr = self._array(kind)
            host = arr.numpy()
            host[rows] = self._held[kind]
            arr.assign(host)

    def reset(self, world_mask: Any = None) -> None:
        """Engine reset hook: the worlds that reset start with an empty history."""
        for line in self.lines.values():
            if world_mask is None:
                line.reset()
                continue
            mask = np.asarray(world_mask.numpy(), bool)[: line.steps.shape[0]]
            line.reset(np.broadcast_to(mask[:, None], line.steps.shape))


def period_steps(seconds: float, dt: float) -> float:
    """A delay in control periods. Example: ``period_steps(0.015, 0.01)`` -> 1.5."""
    return max(float(seconds), 0.0) / float(dt) if math.isfinite(seconds) else 0.0


__all__ = ["DelayLine", "SimDelay", "period_steps"]
