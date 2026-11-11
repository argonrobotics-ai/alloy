# SPDX-License-Identifier: Apache-2.0
"""The spring-damper pull of the mouse-drag tool, as plain float64 maths (no Warp, no simulation): the force that drags
a point of a body toward a target, and the torque it makes about the body's centre of mass.

``F = k (target - attach) - c v_attach`` with ``k = m w^2``, ``c = 2 m w`` (critically damped, ``w = 2 pi f``, ``f`` =
``FREQ_HZ``), so a body hangs ``g / w^2`` (6.2 cm) below the target whatever its mass. The force is capped at
``CAP_G`` times the body's weight, and at ``ARTICULATED_CAP_N`` for a link of a robot (its subtree mass is not known
here), so a far target cannot blow the simulation up. Example::

    w = spring_wrench(pose, local_attach, target, mass=0.2, cap=force_cap(0.2, False))
    w.force, w.torque          # world frame, about the centre of mass
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81
FREQ_HZ = 2.0                   # natural frequency of the pull
OMEGA = 2.0 * math.pi * FREQ_HZ
CAP_G = 50.0                    # force cap, in body weights
ARTICULATED_CAP_N = 200.0       # force cap on a robot link


@dataclass(frozen=True)
class BodyPose:
    """One body's state, world frame of its simulation world: origin ``pos``, ``quat`` (x, y, z, w), the linear
    velocity ``lin`` of its centre of mass, ``ang`` velocity, and the centre of mass ``com`` in the body frame."""

    pos: np.ndarray
    quat: np.ndarray
    lin: np.ndarray
    ang: np.ndarray
    com: np.ndarray


@dataclass(frozen=True)
class Wrench:
    """A force (N) and a torque (N m) about the centre of mass, world frame, and the attach point they act at."""

    force: np.ndarray
    torque: np.ndarray
    attach: np.ndarray


def rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """``vec`` rotated by the unit quaternion ``quat`` (x, y, z, w)."""
    q = np.asarray(quat, dtype=np.float64)
    u, w = q[:3], q[3]
    return 2.0 * np.dot(u, vec) * u + (w * w - np.dot(u, u)) * vec + 2.0 * w * np.cross(u, vec)


def to_body_frame(pose: BodyPose, world_point: np.ndarray) -> np.ndarray:
    """A world point in the body frame (what stays fixed to the body while it moves)."""
    inverse = np.array([-pose.quat[0], -pose.quat[1], -pose.quat[2], pose.quat[3]])
    return rotate(inverse, np.asarray(world_point, dtype=np.float64) - pose.pos)


def force_cap(mass: float, articulated: bool) -> float:
    """The largest pull (N) on a body of ``mass`` kg."""
    cap = CAP_G * mass * G
    return min(cap, ARTICULATED_CAP_N) if articulated else cap


def spring_wrench(pose: BodyPose, local_attach: np.ndarray, target: np.ndarray, mass: float, cap: float) -> Wrench:
    """The pull on the point ``local_attach`` (body frame) toward ``target`` (world frame), and its torque. Example:
    a 0.2 kg box 0.3 m below its target gets ``0.2 * w^2 * 0.3`` N up, less its damping."""
    attach = pose.pos + rotate(pose.quat, np.asarray(local_attach, dtype=np.float64))
    lever = attach - (pose.pos + rotate(pose.quat, pose.com))
    velocity = pose.lin + np.cross(pose.ang, lever)
    force = mass * OMEGA**2 * (np.asarray(target, dtype=np.float64) - attach) - 2.0 * mass * OMEGA * velocity
    size = float(np.linalg.norm(force))
    if size > cap > 0.0:
        force = force * (cap / size)
    return Wrench(force, np.cross(lever, force), attach)
