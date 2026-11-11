# SPDX-License-Identifier: Apache-2.0
"""State keys of a legged robot, added to the environment's batched state (``env.state``).

Read from the engine at the moment of the call (W = worlds, J = leg joints, F = feet)::

    s["base.lin_vel_b"]   (W, 3) m/s    base velocity in the base frame (x forward, y left, z up)
    s["base.ang_vel_b"]   (W, 3) rad/s  base angular velocity in the base frame (a gyroscope)
    s["base.gravity_b"]   (W, 3)        unit gravity direction in the base frame: (0, 0, -1) when upright
    s["base.height"]      (W,) m        base origin above the ground (z = 0, or the terrain under it)
    s["joints.pos"]       (W, J) rad    leg joint angles (0 = the stand)      s["joints.vel"]  (W, J) rad/s
    s["joints.torque"]    (W, J) N m    servo torque kp (target - q) - kd qd, clipped to the motor limit (an estimate
                                        from the servo law at the end of the control step, not a solver read)
    s["feet.pos"]         (W, F, 3) m   foot sphere centres
    s["feet.height"]      (W, F) m      sole above the ground

Kept by the environment between steps (``alloy.envs.loco.sensors``): ``command``, ``action``, ``action.prev``,
``joints.target``, ``joints.acc``, ``base.accel_b``, ``feet.contact``, ``feet.force``, ``feet.air_time``,
``feet.touchdown``, ``feet.vel``, ``body.contacts``, ``push.force``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from alloy.envs.loco.robot import RobotRows
from alloy.envs.state import BatchState, Rows, rotate


def rotate_inverse(xp: Any, q_xyzw: Any, v: Any) -> Any:
    """World vectors (W, 3) in the frame of (W, 4) xyzw quaternions. Example: ``rotate_inverse(xp, q, v)``."""
    conj = xp.cat([-q_xyzw[:, 0:3], q_xyzw[:, 3:4]], -1)
    return rotate(xp, conj, v)


def _base_q(s: LocoState) -> Any:
    return s.array("body_q")[s.robot.base]


def _lin_vel_b(s: LocoState) -> Any:
    return rotate_inverse(s.xp, _base_q(s)[:, 3:7], s.array("body_qd")[s.robot.base][:, 0:3])


def _ang_vel_b(s: LocoState) -> Any:
    return rotate_inverse(s.xp, _base_q(s)[:, 3:7], s.array("body_qd")[s.robot.base][:, 3:6])


def _gravity_b(s: LocoState) -> Any:
    q = _base_q(s)
    down = s.xp.zeros((s.worlds, 3)) + s.xp.asarray([0.0, 0.0, -1.0])
    return rotate_inverse(s.xp, q[:, 3:7], down)


def _feet_pos(s: LocoState) -> Any:
    xp, rr = s.xp, s.robot
    q = s.array("body_q")[rr.feet]                                            # (W, F, 7)
    w, f = int(q.shape[0]), int(q.shape[1])
    flat = q.reshape(w * f, 7)
    offset = (xp.zeros((w, f, 3)) + rr.foot_offset).reshape(w * f, 3)
    return (flat[:, 0:3] + rotate(xp, flat[:, 3:7], offset)).reshape(w, f, 3)


def _above_ground(s: LocoState, points: Any) -> Any:
    """z of ``points`` (..., 3) above the ground under them: above z = 0, or above the terrain when the robot's
    rows carry a ground-height function (``RobotRows.ground``)."""
    ground = s.robot.ground
    return points[..., 2] if ground is None else points[..., 2] - ground(points[..., 0:2])


def _torque(s: LocoState) -> Any:
    rr = s.robot
    raw = rr.robot.kp * (s["joints.target"] - s["joints.pos"]) - rr.robot.kd * s["joints.vel"]
    return s.xp.clip(raw, -rr.torque_limit, rr.torque_limit)


DERIVED: dict[str, Callable[[Any], Any]] = {
    "base.lin_vel_b": _lin_vel_b,
    "base.ang_vel_b": _ang_vel_b,
    "base.gravity_b": _gravity_b,
    "base.height": lambda s: _above_ground(s, _base_q(s)[:, 0:3]),
    "base.quat_xyzw": lambda s: _base_q(s)[:, 3:7],
    "base.lin_vel_w": lambda s: s.array("body_qd")[s.robot.base][:, 0:3],
    "joints.pos": lambda s: s.array("joint_q")[s.robot.q],
    "joints.vel": lambda s: s.array("joint_qd")[s.robot.qd],
    "joints.torque": _torque,
    "feet.pos": _feet_pos,
    "feet.height": lambda s: _above_ground(s, s["feet.pos"]) - s.robot.foot_radius,
}


class LocoState(BatchState):
    """A ``BatchState`` that also answers the legged-robot keys of this module. Example: ``env.state["joints.pos"]``."""

    def __init__(self, arrays: Callable[[str], Any], rows: Rows, robot: RobotRows, *,
                 extra: Mapping[str, Any] | None = None, start: BatchState | None = None) -> None:
        super().__init__(arrays, rows, extra=extra, start=start)
        self.robot = robot

    def array(self, name: str) -> Any:
        """An engine array by name ("body_q", "joint_qd", ...). Example: ``s.array("body_q")``."""
        return self._arrays(name)

    def keys(self) -> list[str]:  # type: ignore[override]
        """Every key this state can give. Example: ``"base.gravity_b" in s.keys()``."""
        return super().keys() + list(DERIVED)

    def _compute(self, key: str) -> Any:
        fn = DERIVED.get(key)
        return fn(self) if fn is not None else super()._compute(key)


__all__ = ["DERIVED", "LocoState", "rotate_inverse"]
