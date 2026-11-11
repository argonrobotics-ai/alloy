# SPDX-License-Identifier: Apache-2.0
"""What a legged-robot environment needs to know about its robot: base, leg joints, feet and the servo law.

``GO2`` describes the Unitree Go2 that ships in the library (``library/robots/emb_e8_go2``, MuJoCo Menagerie,
BSD-3-Clause): 12 position servos (kp 50 N m/rad, kd 0.5 N m s/rad), joint angle 0 = the stand. Example::

    from alloy.envs.loco import GO2
    GO2.joint_names("go2")[:3]      # ['go2/FL_hip_joint', 'go2/FL_thigh_joint', 'go2/FL_calf_joint']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.layout import Layout


@dataclass(frozen=True)
class Foot:
    """A foot: a sphere of ``radius`` at ``offset`` (m) in the frame of ``body``. Example: ``GO2.feet[0].body``."""

    body: str
    offset: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class LeggedRobot:
    """Names and constants of a legged robot (bodies and joints without the "<robot>/" prefix)."""

    base: str
    joints: tuple[str, ...]
    feet: tuple[Foot, ...]
    kp: float                              # servo stiffness, N m / rad
    kd: float                              # servo damping, N m s / rad
    torque_limit: tuple[float, ...]        # per joint, N m
    stand_height: float                    # base height in the stand, m

    def joint_names(self, robot: str) -> list[str]:
        """Full joint names in action order. Example: ``GO2.joint_names("go2")``."""
        return [f"{robot}/{j}" for j in self.joints]


_LEGS = ("FL", "FR", "RL", "RR")
GO2 = LeggedRobot(
    base="base",
    joints=tuple(f"{leg}_{part}_joint" for leg in _LEGS for part in ("hip", "thigh", "calf")),
    feet=tuple(Foot(f"{leg}_calf", (-0.002, 0.0, -0.213), 0.022) for leg in _LEGS),
    kp=50.0, kd=0.5, torque_limit=(23.7, 23.7, 45.43) * 4, stand_height=0.288)


class RobotRows:
    """Index arrays of one robot's base, leg joints and feet in the engine's batched arrays (made once)."""

    def __init__(self, robot: LeggedRobot, name: str, layout: Layout, xp: Any) -> None:
        self.robot, self.name, self.xp = robot, name, xp
        joints = [layout.joint(j) for j in robot.joint_names(name)]
        self.q = xp.asarray(np.concatenate([j.q for j in joints], axis=1), "int64")            # (W, J)
        self.qd = xp.asarray(np.concatenate([j.qd for j in joints], axis=1), "int64")
        self.base = xp.asarray(layout.body(f"{name}/{robot.base}"), "int64")                    # (W,)
        self.feet = xp.asarray(np.stack([layout.body(f"{name}/{f.body}") for f in robot.feet], 1), "int64")  # (W, F)
        self.foot_offset = xp.asarray(np.array([f.offset for f in robot.feet], np.float32))     # (F, 3)
        self.foot_radius = xp.asarray(np.array([f.radius for f in robot.feet], np.float32))     # (F,)
        self.torque_limit = xp.asarray(np.array(robot.torque_limit, np.float32))
        self.ground: Any = None                 # fn(xy (..., 2)) -> ground height (...,) m; None: flat ground at z = 0
        self.lower = np.concatenate([j.lower for j in joints]).astype(np.float32)
        self.upper = np.concatenate([j.upper for j in joints]).astype(np.float32)


__all__ = ["GO2", "Foot", "LeggedRobot", "RobotRows"]
