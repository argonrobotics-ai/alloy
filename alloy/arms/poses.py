# SPDX-License-Identifier: Apache-2.0
"""Pose maths for the arm kit and the clips (host, float64, batched over leading axes).

A pose is ``[..., 7]``: position (m) then quaternion ``(x, y, z, w)`` (Newton's order). Orientations given by
hand are roll, pitch, yaw in degrees, extrinsic x-y-z (``R = Rz(yaw) Ry(pitch) Rx(roll)``). A TCP frame sits at the
pad centre: z = approach (out of the fingers), x = jaw axis; fingers straight down is roll 180.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DOWN", "angle_between", "compose", "invert", "pose", "quat_from_axis_angle", "quat_from_rpy", "quat_mul",
    "quat_rotate", "quat_to_mat", "rot_vec", "slerp",
]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of ``[..., 4]`` xyzw quaternions."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)  # fmt: skip


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``[..., 3]`` vectors by ``[..., 4]`` quaternions."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_from_axis_angle(axis, angle) -> np.ndarray:
    """``[..., 3]`` axis (normalised here) and ``[...]`` angle (rad) -> ``[..., 4]``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = 0.5 * np.asarray(angle, dtype=np.float64)[..., None]
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def quat_from_rpy(rpy_deg) -> np.ndarray:
    """Roll, pitch, yaw (deg, extrinsic xyz) ``[..., 3]`` -> ``[..., 4]``."""
    r, p, y = np.moveaxis(np.radians(np.asarray(rpy_deg, dtype=np.float64)), -1, 0)
    qx = quat_from_axis_angle(np.broadcast_to([1.0, 0.0, 0.0], np.shape(r) + (3,)), r)
    qy = quat_from_axis_angle(np.broadcast_to([0.0, 1.0, 0.0], np.shape(p) + (3,)), p)
    qz = quat_from_axis_angle(np.broadcast_to([0.0, 0.0, 1.0], np.shape(y) + (3,)), y)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)],
                    -2)  # fmt: skip


def pose(xyz, rpy_deg=(180.0, 0.0, 0.0)) -> np.ndarray:
    """``[..., 7]`` pose from a position (m) and roll, pitch, yaw (deg); default fingers straight down."""
    xyz = np.asarray(xyz, dtype=np.float64)
    q = quat_from_rpy(np.broadcast_to(np.asarray(rpy_deg, dtype=np.float64), xyz.shape))
    return np.concatenate([xyz, q], axis=-1)


DOWN = quat_from_rpy([180.0, 0.0, 0.0])  # fingers straight down, jaw along world x


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a * b`` for ``[..., 7]`` poses (b expressed in a's frame)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.concatenate([a[..., :3] + quat_rotate(a[..., 3:], b[..., :3]), quat_mul(a[..., 3:], b[..., 3:])], -1)


def invert(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    qi = quat_conj(a[..., 3:])
    return np.concatenate([-quat_rotate(qi, a[..., :3]), qi], axis=-1)


def rot_vec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad) of ``[..., 4]`` quaternions, shortest way round."""
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., 3:] < 0.0, -q, q)
    s = np.linalg.norm(q[..., :3], axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 3])
    scale = np.where(s > 1e-12, angle / np.maximum(s, 1e-300), 2.0)
    return q[..., :3] * scale[..., None]


def angle_between(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Angle (rad) of the rotation taking ``qa`` to ``qb``."""
    return np.linalg.norm(rot_vec(quat_mul(quat_conj(qa), qb)), axis=-1)


def slerp(qa: np.ndarray, qb: np.ndarray, s) -> np.ndarray:
    """Shortest-path spherical interpolation, ``s`` in [0, 1] broadcast over the leading axes."""
    qa, qb = np.asarray(qa, dtype=np.float64), np.asarray(qb, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    rel = quat_mul(quat_conj(qa), qb)
    v = rot_vec(rel)
    angle = np.linalg.norm(v, axis=-1)
    axis = np.where(angle[..., None] > 1e-12, v / np.maximum(angle, 1e-300)[..., None], [1.0, 0.0, 0.0])
    return quat_mul(qa, quat_from_axis_angle(axis, angle * s))
