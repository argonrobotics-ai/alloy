# SPDX-License-Identifier: Apache-2.0
"""Observation and reward terms for legged velocity tracking, over the keys of ``alloy.envs.loco.state``.

Observation entries are ``(label, fn)`` pairs for ``Task(obs=[...])``; reward terms are ``alloy.envs.checks.Check``
objects giving a (W,) value. Example::

    from alloy.envs.loco import terms as T
    obs = [T.obs("base.ang_vel_b"), T.obs("base.gravity_b"), T.obs("command"), T.obs("joints.pos")]
    reward = {"track": (1.5, T.track_lin_vel()), "smooth": (-0.01, T.action_rate())}
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from alloy.envs.checks import Check
from alloy.envs.loco.sensors import GRAVITY


def obs(key: str, scale: float = 1.0) -> tuple[str, Callable[[Any], Any]]:
    """An observation entry reading state ``key`` (times ``scale``). Example: ``obs("base.accel_b", 1 / 9.81)``."""
    if scale == 1.0:
        return key, lambda s: s[key]
    return key, lambda s: s[key] * scale


#: what the policy sees by default (55 numbers for the Go2): base velocity, gyroscope, gravity direction,
#: accelerometer (in g), command, joint angles and speeds, the last action, foot contacts
POLICY_OBS = ("base.lin_vel_b", "base.ang_vel_b", "base.gravity_b", ("base.accel_b", 1.0 / GRAVITY), "command",
              "joints.pos", "joints.vel", "action", "feet.contact")
#: what only the critic sees on top (training-time information a real robot does not have)
PRIVILEGED_OBS = ("base.height", "feet.air_time", "feet.height", ("joints.torque", 0.05), ("push.force", 0.01))


def entries(keys: Any) -> list[tuple[str, Callable[[Any], Any]]]:
    """Observation entries from keys or (key, scale) pairs. Example: ``entries(POLICY_OBS)``."""
    return [obs(k) if isinstance(k, str) else obs(*k) for k in keys]


def _exp(xp: Any, x: Any) -> Any:
    return xp.torch.exp(x) if xp.name == "torch" else np.exp(x)


def _sq(x: Any) -> Any:
    return (x * x).sum(-1)


def track_lin_vel(sigma: float = 0.25) -> Check:
    """exp(-|commanded - actual planar base velocity|^2 / sigma): 1 when tracking exactly."""
    return Check("track_lin_vel",
                 lambda s: _exp(s.xp, -_sq(s["command"][:, 0:2] - s["base.lin_vel_b"][:, 0:2]) / sigma))


def track_yaw_rate(sigma: float = 0.25) -> Check:
    """exp(-(commanded - actual yaw rate)^2 / sigma)."""
    return Check("track_yaw_rate",
                 lambda s: _exp(s.xp, -((s["command"][:, 2] - s["base.ang_vel_b"][:, 2]) ** 2) / sigma))


def lin_vel_z() -> Check:
    """Squared vertical base speed (bouncing)."""
    return Check("lin_vel_z", lambda s: s["base.lin_vel_b"][:, 2] ** 2)


def ang_vel_xy() -> Check:
    """Squared roll and pitch rates."""
    return Check("ang_vel_xy", lambda s: _sq(s["base.ang_vel_b"][:, 0:2]))


def flat_orientation() -> Check:
    """Squared horizontal part of the gravity direction in the base frame (0 when level)."""
    return Check("flat_orientation", lambda s: _sq(s["base.gravity_b"][:, 0:2]))


def torques() -> Check:
    """Sum of squared servo torques (N m)^2."""
    return Check("torques", lambda s: _sq(s["joints.torque"]))


def joint_acc() -> Check:
    """Sum of squared joint accelerations (rad/s^2)^2."""
    return Check("joint_acc", lambda s: _sq(s["joints.acc"]))


def action_rate() -> Check:
    """Sum of squared changes of the action since the last step."""
    return Check("action_rate", lambda s: _sq(s["action"] - s["action.prev"]))


def body_contacts() -> Check:
    """How many of the robot's non-foot collision shapes touch something (needs solver contact forces)."""
    return Check("body_contacts", lambda s: s["body.contacts"])


def feet_slide() -> Check:
    """Squared planar speed (m/s)^2 of every foot that touches the ground, summed over feet (0 for planted feet)."""
    def fn(s: Any) -> Any:
        v = s["feet.vel"]
        return ((v[:, :, 0] ** 2 + v[:, :, 1] ** 2) * s["feet.contact"]).sum(-1)
    return Check("feet_slide", fn)


def pose_deviation() -> Check:
    """Sum of squared leg joint angles (rad^2): how far the legs are from the stand."""
    return Check("pose_deviation", lambda s: _sq(s["joints.pos"]))


def hip_deviation(every: int = 3) -> Check:
    """Sum of squared hip (abduction) angles: the first joint of each leg, every ``every``-th action column."""
    return Check("hip_deviation", lambda s: _sq(s["joints.pos"][:, ::every]))


def base_height(target: float) -> Check:
    """Squared error of the base height against ``target`` (m)."""
    return Check("base_height", lambda s: (s["base.height"] - target) ** 2)


def feet_too_high(limit: float = 0.10) -> Check:
    """How far each foot's sole is above ``limit`` (m), summed over feet (0 below it)."""
    return Check("feet_too_high", lambda s: s.xp.clip(s["feet.height"] - limit, 0.0, 10.0).sum(-1))


def feet_air_time(threshold: float = 0.2, min_command: float = 0.1, cap: float = 0.5) -> Check:
    """On each landing: the flight time (at most ``cap`` s) minus ``threshold`` s, summed over feet; 0 while the
    command is to stand."""
    def fn(s: Any) -> Any:
        moving = s.xp.float(s.xp.norm(s["command"][:, 0:2], -1) > min_command)
        air = s.xp.clip(s["feet.air_time"], 0.0, cap)
        return ((air - threshold) * s["feet.touchdown"]).sum(-1) * moving
    return Check("feet_air_time", fn)


def stand_still(min_command: float = 0.1) -> Check:
    """Sum of |joint angle| (distance from the stand) while the command is to stand; 0 otherwise."""
    def fn(s: Any) -> Any:
        idle = (s.xp.norm(s["command"][:, 0:2], -1) <= min_command) & (abs(s["command"][:, 2]) <= min_command)
        return abs(s["joints.pos"]).sum(-1) * s.xp.float(idle)
    return Check("stand_still", fn)


def fell(min_height: float = 0.15, max_tilt_deg: float = 60.0) -> Check:
    """True when the base is below ``min_height`` (m) or tilted more than ``max_tilt_deg`` from upright."""
    limit = -math.cos(math.radians(max_tilt_deg))
    return Check("fell", lambda s: (s["base.height"] < min_height) | (s["base.gravity_b"][:, 2] > limit))


__all__ = ["POLICY_OBS", "PRIVILEGED_OBS", "action_rate", "ang_vel_xy", "base_height", "body_contacts",
           "entries", "feet_air_time", "feet_slide", "feet_too_high", "hip_deviation", "pose_deviation",
           "fell",
           "flat_orientation", "joint_acc", "lin_vel_z", "obs", "stand_still", "torques", "track_lin_vel",
           "track_yaw_rate"]
