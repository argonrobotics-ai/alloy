# SPDX-License-Identifier: Apache-2.0
"""Rigid moves of the variants: a yaw about a vertical axis through a centre, then a horizontal offset, applied to
poses (x, y, z, qx, qy, qz, qw), body transforms and particle positions alike."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Move:
    """One rigid group's move: ``offset`` (dx, dy) [m] and ``yaw`` [rad] about the vertical through ``center``."""

    names: tuple[str, ...]
    offset: tuple[float, float]
    yaw: float
    center: tuple[float, float, float]

    def points(self, p: np.ndarray) -> np.ndarray:
        """Positions [..., 3] moved."""
        p = np.asarray(p, np.float64)
        c = np.asarray(self.center, np.float64)
        out = c + (p - c) @ yaw_matrix(self.yaw).T
        out[..., 0] += self.offset[0]
        out[..., 1] += self.offset[1]
        return out

    def quats(self, q: np.ndarray) -> np.ndarray:
        """Orientations [..., 4] (x, y, z, w) turned by the yaw."""
        return qmul(yaw_quat(self.yaw), np.asarray(q, np.float64))

    def pose(self, pose: np.ndarray) -> list[float]:
        """A pose (x, y, z, qx, qy, qz, qw) moved."""
        pose = np.asarray(pose, np.float64)
        return [float(v) for v in (*self.points(pose[:3]), *self.quats(pose[3:7]))]


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b of (x, y, z, w) quaternions (b may be [..., 4])."""
    ax, ay, az, aw = a
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectors v [N, 3] rotated by quaternions q [N, 4] (x, y, z, w)."""
    xyz, w = q[:, :3], q[:, 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def to_world(body_q: np.ndarray, bodies: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Body-frame points [N, 3] to world (body -1: already world)."""
    out = np.asarray(points, np.float64).copy()
    has = bodies >= 0
    if np.any(has):
        q = body_q[bodies[has]].astype(np.float64)
        out[has] = q[:, :3] + rotate(q[:, 3:7], out[has])
    return out


__all__ = ["Move", "qmul", "rotate", "to_world", "yaw_matrix", "yaw_quat"]
