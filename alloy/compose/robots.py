# SPDX-License-Identifier: Apache-2.0
"""Robots in a composed scene: a robot file stands at the table's edge, objects stay clear of its base and within
its reach, and the load checks after each placement include the robot's links (an object in the arm's way is moved).

The first robot stands at the middle of the table's near edge (-x) facing +x, the second at the far edge facing -x
(two arms across a table). A ``pose`` (x, y, z, yaw) puts a robot anywhere instead. Objects go in the part of the
table in front of the robots, at least ``clear`` from each base and at most ``reach`` from one.

Example::

    >>> import numpy as np
    >>> (arm,) = place_robots((RobotSpec("my_arm.xml"),), np.array([-0.58, -0.38]), np.array([0.58, 0.38]))
    >>> [round(v, 3) for v in arm.pose]               # near edge, facing +x
    [-0.43, 0.0, 0.0, 0.0]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.compose import geom
from alloy.compose.spec import ComposeError, RobotSpec

MAX_ROBOTS = 2
DISC_SIDES = 24


@dataclass(frozen=True)
class PlacedRobot:
    spec: RobotSpec
    pose: tuple[float, float, float, float]          # x, y, z, yaw

    def base_pose(self) -> tuple[float, ...]:
        x, y, z, yaw = self.pose
        return (x, y, z) + geom.quat_from_matrix(geom.yaw_matrix(yaw))

    def disc(self) -> np.ndarray:
        """The kept-free base area as a polygon (world xy)."""
        a = np.linspace(0.0, 2 * math.pi, DISC_SIDES, endpoint=False)
        r = self.spec.clear / math.cos(math.pi / DISC_SIDES)          # circumscribed: covers the true disc
        return np.stack([self.pose[0] + r * np.cos(a), self.pose[1] + r * np.sin(a)], axis=1)


def place_robots(robots: tuple[RobotSpec, ...], lo: np.ndarray, hi: np.ndarray, top: float = 0.0
                 ) -> list[PlacedRobot]:
    """Base poses: given ones as is; else the near edge (-x) then the far edge (+x) of the table box [lo, hi]."""
    if len(robots) > MAX_ROBOTS and any(r.pose is None for r in robots):
        raise ComposeError(f"{len(robots)} robots without poses", f"give each robot a pose, or use at most "
                           f"{MAX_ROBOTS} (they stand at the two ends of the table)", field="robots")
    out, free = [], 0
    for r in robots:
        if r.pose is not None:
            out.append(PlacedRobot(r, r.pose))
            continue
        cy = float(lo[1] + hi[1]) / 2
        pose = (float(lo[0]) + r.clear, cy, top, 0.0) if free == 0 else (float(hi[0]) - r.clear, cy, top, math.pi)
        out.append(PlacedRobot(r, pose))
        free += 1
    return out


def reach_area(robots: list[PlacedRobot], lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The box of the table in front of the robots and within reach of at least one of them (clipped to it)."""
    boxes = []
    for p in robots:
        x, y, _, yaw = p.pose
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        near = np.array([x, y]) + fwd * (p.spec.clear + 0.05)
        far = np.array([x, y]) + fwd * p.spec.reach
        side = np.array([-fwd[1], fwd[0]]) * 0.6 * p.spec.reach
        pts = np.array([near + side, near - side, far + side, far - side])
        boxes.append((pts.min(axis=0), pts.max(axis=0)))
    blo = np.min([b[0] for b in boxes], axis=0)
    bhi = np.max([b[1] for b in boxes], axis=0)
    out_lo, out_hi = np.maximum(blo, lo), np.minimum(bhi, hi)
    if np.any(out_hi - out_lo <= 0.05):
        raise ComposeError("the robots' reach does not cover the table", "move the robots (pose) or raise reach",
                           field="robots")
    return out_lo, out_hi


__all__ = ["PlacedRobot", "RobotSpec", "place_robots", "reach_area"]
