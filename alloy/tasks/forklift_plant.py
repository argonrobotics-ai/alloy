# SPDX-License-Identifier: Apache-2.0
"""Kinematic test plant for the forklift operator: a logic test, not physics.

Bicycle kinematics (yaw rate ``v tan(steer) / L``), a first-order drive response to throttle with a brake and a grade
term ``g sin(grade)``, and rate-limited lift (0.6 m/s), tilt (3 deg/s) and sideshift (0.1 m/s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from alloy.tasks.forklift_control import G, Controls, Observation, TruckSpec, clip

Terrain = Callable[[float, float, float], tuple[bool, float]]      # (x, y, yaw) -> (on_ramp, grade along heading)

BLADE_CLEAR, BLADE_LENGTH = 0.009, 1.0   # m: lowered blades' clearance (F/params.py:87), heel to tip


def flat(x: float, y: float, yaw: float) -> tuple[bool, float]:
    return False, 0.0


def ramp_along_x(x0: float, x1: float, slope: float = 0.136) -> Terrain:
    """Ground rising toward +x by ``slope`` between x0 and x1."""
    def terrain(x: float, y: float, yaw: float) -> tuple[bool, float]:
        return (True, math.atan(slope * math.cos(yaw))) if x0 <= x <= x1 else (False, 0.0)
    return terrain


@dataclass
class KinematicPlant:
    spec: TruckSpec = TruckSpec()
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    fork_height: float = 0.0
    tilt: float = 0.0
    sideshift: float = 0.0
    t: float = 0.0
    load_on_forks: bool = False
    terrain: Terrain = flat
    restraint: Callable[[float], bool | None] = lambda t: None
    lift_rate: float = 0.6
    tilt_rate: float = math.radians(3.0)
    side_rate: float = 0.1
    steer_rate: float = 1.5
    drive_lag: float = 0.15
    _steer: float = 0.0
    _drive: float = 0.0

    def observe(self) -> Observation:
        on_ramp, grade = self.terrain(self.x, self.y, self.yaw)
        tip = max(self.fork_height + BLADE_CLEAR - BLADE_LENGTH * math.sin(self.tilt), 0.0)   # no floor: clipped
        return Observation(self.t, self.x, self.y, self.yaw, self.speed, self.fork_height, self.tilt, self.sideshift,
                           on_ramp, grade, self.load_on_forks, self.restraint(self.t), blade_pitch=-self.tilt,
                           tip_height=tip)

    def _accelerate(self, c: Controls, grade: float, dt: float) -> None:
        spec = self.spec
        can_drive = c.key_on and c.engine_on and c.direction != 0 and not c.park and c.throttle * c.direction > 0
        goal = spec.throttle_accel * c.throttle if can_drive else 0.0
        self._drive += (goal - self._drive) * min(dt / self.drive_lag, 1.0)
        v = self.speed + (self._drive - G * math.sin(grade) - 0.05 * self.speed) * dt
        decel = (spec.brake_accel * c.brake + (9.0 if c.park else 0.0)) * dt
        self.speed = 0.0 if abs(v) <= decel else v - math.copysign(decel, v)

    def step(self, c: Controls, dt: float) -> Observation:
        _, grade = self.terrain(self.x, self.y, self.yaw)
        self._accelerate(c, grade, dt)
        goal = clip(c.steer, -self.spec.max_steer, self.spec.max_steer)
        self._steer += clip(goal - self._steer, -self.steer_rate * dt, self.steer_rate * dt)
        self.yaw += self.speed * math.tan(self._steer) / self.spec.wheelbase * dt
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.fork_height = clip(self.fork_height + self.lift_rate * clip(c.lift, -1, 1) * dt, 0.0, self.spec.lift_max)
        lim = self.spec.tilt_limit
        self.tilt = clip(self.tilt + self.tilt_rate * clip(c.tilt, -1, 1) * dt, -lim, lim)
        lim = self.spec.sideshift_limit
        self.sideshift = clip(self.sideshift + self.side_rate * clip(c.sideshift, -1, 1) * dt, -lim, lim)
        self.t += dt
        return self.observe()
