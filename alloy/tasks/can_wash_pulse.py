# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can, robot stage: the hose valve's pulses while the lathe turns the can under the FR3's jets.

Stage 1's nozzle moved its aim along every band at ``Rinse.speed`` (0.25 m/s), so each metre of band took the same
dose. On the lathe a band at radius ``r`` passes the jets at the can's turn rate times ``r``: 0.09 m/s for the floor's
innermost ring at 1 rad/s, and slower still in each stroke's cosine ramps. So the valve is on-off, open in a share
``|ω| r / v`` of the control periods (a sigma-delta count of the can's measured turn rate over the last period, at
most every period): each band takes stage 1's dose per metre whatever the can's speed, and the jets keep their full
speed (a throttled orifice would slow them). Where ``|ω| r`` is over ``v`` (the 0.26 m rim band at 1 rad/s) the
valve stays open and the band takes ``v / (|ω| r)`` of stage 1's dose, booked by ``share``.
"""

from __future__ import annotations

import math

import numpy as np

from alloy.tasks.rinse_flush_policy import CONTROL_DT


def yaw_of(q: np.ndarray) -> np.ndarray:
    """``[...]`` the yaw [rad] of poses ``[..., 7]`` (x, y, z, qx, qy, qz, qw)."""
    x, y, z, w = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Pulse:
    """The valve's openings for one band at radius ``r`` [m] (``worlds`` cans), stage 1's aim speed ``v`` [m/s]."""

    def __init__(self, r: float, v: float, worlds: int, dt: float = CONTROL_DT) -> None:
        if r <= 0.0 or v <= 0.0:
            raise ValueError(f"a band needs a positive radius and speed, got r={r}, v={v}")
        self.r, self.v, self.dt = r, v, dt
        self.acc = np.zeros(worlds)
        self.open_s = np.zeros(worlds)
        self.on_s = 0.0                          # s the jets were allowed (the UR turning the can)
        self.last: np.ndarray | None = None

    def opening(self, yaw: np.ndarray, on: bool) -> np.ndarray:
        """``[W]`` this period's opening (0 or 1) from the cans' yaws now [rad]; shut unless ``on`` (the jets may
        run: the UR is turning the can)."""
        yaw = np.asarray(yaw, np.float64)
        step = np.zeros_like(yaw) if self.last is None else (yaw - self.last + math.pi) % (2.0 * math.pi) - math.pi
        self.last = yaw.copy()
        if not on:
            return np.zeros_like(yaw)
        self.on_s += self.dt
        self.acc += np.minimum(1.0, np.abs(step) / self.dt * self.r / self.v)
        out = (self.acc >= 1.0).astype(np.float64)
        self.acc -= out
        self.open_s += out * self.dt
        return out

    def duty(self) -> np.ndarray:
        """``[W]`` the valve's open share of the periods the jets were allowed."""
        return self.open_s / max(self.on_s, 1e-12)

    def share(self, turned: float) -> np.ndarray:
        """``[W]`` each band's dose over stage 1's for the length the can ``turned`` [rad] under the jets: the valve's
        open time against stage 1's time over that length."""
        return self.open_s * self.v / max(abs(turned) * self.r, 1e-12)


__all__ = ["Pulse", "yaw_of"]
