# SPDX-License-Identifier: Apache-2.0
"""Forklift operator interface and low-level controllers (host side, pure Python + numpy).

The operator sees an ``Observation`` and answers a ``Controls`` each control period, so the same logic can drive a
kinematic test plant (``forklift_plant``) or Alloy's vehicle plug-in with its hydraulic mast. This module holds the
truck numbers, the two data types and the controllers that do not depend on a path: the speed controller (grade
feed-forward, throttle/brake split), the stop profile and the P servos with deadbands for lift, tilt and sideshift.

Truck numbers are Gym-Anything's Hyster H2.5A (benchmarks/roboworld/environments/forklift/params.py, GA repo febb260).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81
REST_SPEED = 0.02          # m/s: below this the truck counts as at rest (park only then)
SPEED_MARGIN = 0.87        # commanded speed is this fraction of a segment's cap (10-15 % margin)


def kmh(value: float) -> float:
    """km/h to m/s."""
    return value / 3.6


@dataclass(frozen=True)
class ServoSpec:
    """A P servo with a deadband on one axis whose travel is [lo, hi]."""

    gain: float          # command per unit of error
    deadband: float
    lo: float
    hi: float


@dataclass(frozen=True)
class TruckSpec:
    """Hyster H2.5A numbers (params.py of GA's forklift environment, line numbers below)."""

    wheelbase: float = 1.649                      # m, params.py:65
    max_steer: float = 1.1                        # rad, INFERRED (rear-steer knuckle; params.py:77 gives 80 deg inner)
    tilt_limit: float = math.radians(6.0)         # rad, forward and back, params.py:84-85
    free_lift: float = 0.96                       # m, params.py:88
    stage_stroke: float = 1.69                    # m per stage, 2 stages, params.py:86
    lift_max: float = 4.44                        # m, GA's highest carriage reach (params.py:63 comment)
    sideshift_limit: float = 0.10                 # m, params.py:89
    throttle_accel: float = 1.5                   # m/s² per unit throttle, INFERRED
    brake_accel: float = 3.0                      # m/s² at full brake, INFERRED

    @property
    def lift_servo(self) -> ServoSpec:
        return ServoSpec(gain=12.0, deadband=0.003, lo=0.0, hi=self.lift_max)

    @property
    def tilt_servo(self) -> ServoSpec:
        return ServoSpec(gain=1.0 / math.radians(1.5), deadband=math.radians(0.15),
                         lo=-self.tilt_limit, hi=self.tilt_limit)

    @property
    def sideshift_servo(self) -> ServoSpec:
        return ServoSpec(gain=50.0, deadband=0.002, lo=-self.sideshift_limit, hi=self.sideshift_limit)


@dataclass(frozen=True)
class Observation:
    """What the operator reads each control period (world frame, z up)."""

    t: float
    x: float                       # truck reference point = front (drive) axle centre
    y: float
    yaw: float
    speed: float                   # signed forward m/s
    fork_height: float             # blade bottom above the truck's ground contact plane
    tilt: float                    # rad, + = mast forward
    sideshift: float               # m, + = left
    on_ramp: bool = False
    grade: float = 0.0             # rad, signed along the heading (+ = uphill ahead)
    load_on_forks: bool = False
    restraint_green: bool | None = None   # trailer restraint light; None = no restraint system
    blade_pitch: float = 0.0       # rad, + = tips up (the blades' axis against the horizontal)
    tip_height: float = 0.0        # m, the lower blade tip's bottom above the floor under it


@dataclass(frozen=True)
class Controls:
    """What the operator commands each control period."""

    throttle: float = 0.0          # [-1, 1], + forward
    brake: float = 1.0             # [0, 1]
    steer: float = 0.0             # rad, bicycle-equivalent: yaw rate = speed * tan(steer) / wheelbase
    park: bool = False
    direction: int = 0             # -1 reverse, 0 neutral, 1 forward
    lift: float = 0.0              # [-1, 1], + pump up, - lower valve
    tilt: float = 0.0              # [-1, 1], + forward
    sideshift: float = 0.0         # [-1, 1], + left
    key_on: bool = False
    engine_on: bool = False


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def servo_reached(target: float, value: float, spec: ServoSpec) -> bool:
    return abs(target - value) <= spec.deadband


def servo_command(target: float, value: float, spec: ServoSpec) -> float:
    """P command toward ``target``; a target at a travel limit drives hard into it until within the deadband."""
    if target <= spec.lo:
        return -1.0 if value > spec.lo + spec.deadband else 0.0
    if target >= spec.hi:
        return 1.0 if value < spec.hi - spec.deadband else 0.0
    error = target - value
    return 0.0 if abs(error) <= spec.deadband else clip(error * spec.gain, -1.0, 1.0)


def speed_profile(remaining: float, cap: float, exit_speed: float = 0.0, decel: float = 0.45,
                  stop_gain: float = 1.2) -> float:
    """Speed magnitude (m/s) to hold ``remaining`` m before the segment end: cap, decelerating to ``exit_speed``."""
    ahead = max(remaining, 0.0)
    speed = min(cap * SPEED_MARGIN, math.sqrt(exit_speed ** 2 + 2.0 * decel * ahead))
    return min(speed, stop_gain * ahead) if exit_speed == 0.0 else speed


def speed_commands(v_ref: float, speed: float, direction: int, grade: float, bias: float,
                   spec: TruckSpec, kp: float = 2.5) -> tuple[float, float]:
    """(throttle, brake) holding signed speed ``direction * v_ref``; gravity along the heading is fed forward.

    ``bias`` is the integral accel (m/s²). Throttle always has the sign of ``direction``; deceleration, including
    holding back a downhill, is done by the brake.
    """
    wanted = kp * (direction * v_ref - speed) + bias
    drive = wanted + G * math.sin(grade)               # cancels the pull -g sin(grade) along the heading
    along = drive * direction
    if along >= 0.0:
        return clip(drive / spec.throttle_accel, -1.0, 1.0), 0.0
    return 0.0, clip(-along / spec.brake_accel, 0.0, 1.0)
