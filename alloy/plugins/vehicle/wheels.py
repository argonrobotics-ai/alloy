# SPDX-License-Identifier: Apache-2.0
"""The vehicle plug-in's wheels, tyres and drivetrain, from config tuples (one value per wheel, or one for all).

Wheels are virtual: each sits on a body at ``positions`` (its centre, in that body's frame), spins about the body's
y axis turned by its steering angle, and touches a flat ground at height ``ground``. The body is the chassis unless
``bodies`` names another one per wheel (an oscillating steer axle hinged to the chassis). The plug-in owns the wheel
spin and the tyre contact; the solver sees only the bodies (keep their shapes above the ground).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import warp as wp

PER_WHEEL = ("positions", "radius", "steer", "driven", "inertia", "radial_stiffness", "radial_damping", "brake_torque",
             "bodies", "arm_slope", "ackermann")


@wp.struct
class Wheel:
    center: wp.vec3              # wheel centre, its body's frame [m]
    radius: wp.float32           # [m]
    steer: wp.float32            # steering factor: angle = steer · input (0: fixed, 1: steered, -1: opposite)
    driven: wp.int32             # 1: the differential drives it
    inertia: wp.float32          # wheel spin inertia [kg·m²] (driven wheels: plus their share of the motor's)
    stiffness: wp.float32        # radial (tyre) stiffness [N/m]
    damping: wp.float32          # radial damping [N·s/m]
    brake: wp.float32            # brake torque at full brake [N·m]
    slope: wp.float32            # trailing arm: the contact moves -slope along the body's x per metre of compression
    ack: wp.float32              # Ackermann: distance to the unsteered axle [m] (0: angle = steer · input)


@wp.struct
class Tyre:
    mu: wp.float32               # peak friction coefficient (D = mu·F_z)
    bx: wp.float32               # magic formula B, C for longitudinal slip κ
    cx: wp.float32
    by: wp.float32               # B, C for slip angle α [1/rad]
    cy: wp.float32
    rolling: wp.float32          # rolling resistance coefficient c_rr (Coulomb torque c_rr·F_z·R against the spin)
    sigma_x: wp.float32          # relaxation length along the wheel [m] (tread deflection u = σ_x·κ' at steady state)
    sigma_y: wp.float32          # relaxation length across [m]
    v_low: wp.float32            # [m/s] the low-speed tread damper fades out smoothly by this speed
    zeta: wp.float32             # the low-speed damper's fraction of critical
    ground: wp.float32           # ground height [m]


@wp.struct
class Drive:
    torque: wp.float32           # motor peak torque [N·m]
    free_speed: wp.float32       # motor speed where its torque reaches 0 [rad/s]
    gear: wp.float32             # motor turns per axle turn
    driven: wp.int32             # number of driven wheels (an open differential splits the torque equally)


@dataclass(frozen=True)
class WheelSpec:
    center: tuple[float, float, float]
    radius: float
    steer: float
    driven: bool
    inertia: float
    radial_stiffness: float
    radial_damping: float
    brake_torque: float
    body: str = ""               # body label; "" = the chassis
    arm_slope: float = 0.0       # trailing-arm suspension (see Wheel.slope)
    ackermann: float = 0.0       # distance to the unsteered axle (see Wheel.ack)


def _per_wheel(c: Any, name: str, n: int) -> list[Any]:
    value = getattr(c, name)
    vector = name == "positions"
    if isinstance(value, (list, tuple)) and (not vector or isinstance(value[0], (list, tuple))):
        seq = list(value)
    else:
        seq = [value]
    if len(seq) == 1:
        return seq * n
    if len(seq) != n:
        raise ValueError(f"{name} has {len(seq)} values for {n} wheels: give one per wheel or one for all")
    return seq


def wheel_specs(c: Any) -> list[WheelSpec]:
    n = len(c.positions) if isinstance(c.positions[0], (list, tuple)) else 1
    cols = {name: _per_wheel(c, name, n) for name in PER_WHEEL}
    specs = []
    for i in range(n):
        s = WheelSpec(center=tuple(float(v) for v in cols["positions"][i]), radius=float(cols["radius"][i]),
                      steer=float(cols["steer"][i]), driven=bool(cols["driven"][i]),
                      inertia=float(cols["inertia"][i]), radial_stiffness=float(cols["radial_stiffness"][i]),
                      radial_damping=float(cols["radial_damping"][i]), brake_torque=float(cols["brake_torque"][i]),
                      body=str(cols["bodies"][i]), arm_slope=float(cols["arm_slope"][i]),
                      ackermann=float(cols["ackermann"][i]))
        bad = [k for k, v in (("radius", s.radius), ("inertia", s.inertia), ("radial_stiffness", s.radial_stiffness))
               if not (math.isfinite(v) and v > 0.0)]
        bad += [k for k, v in (("radial_damping", s.radial_damping), ("brake_torque", s.brake_torque))
                if not (math.isfinite(v) and v >= 0.0)]
        bad += ["ackermann"] if not (math.isfinite(s.ackermann) and s.ackermann >= 0.0) else []
        if bad or len(s.center) != 3 or not (math.isfinite(s.steer) and math.isfinite(s.arm_slope)):
            raise ValueError(f"wheel {i}: {bad or 'positions need 3 values'} must be finite and positive "
                             f"(radius, inertia, radial_stiffness) or >= 0")
        specs.append(s)
    if not any(s.driven for s in specs):
        raise ValueError("no driven wheel: set driven=True on at least one")
    return specs


def effective_inertia(s: WheelSpec, c: Any, driven_count: int) -> float:
    """Wheel inertia plus its share of the motor's reflected inertia (exact when the driven wheels turn together)."""
    return s.inertia + (c.motor_inertia * c.gear**2 / driven_count if s.driven else 0.0)


def wheel_array(specs: list[WheelSpec], c: Any, device: Any) -> wp.array:
    n_driven = sum(s.driven for s in specs)
    rows = []
    for s in specs:
        w = Wheel()
        w.center, w.radius, w.steer, w.driven = wp.vec3(*s.center), s.radius, s.steer, int(s.driven)
        w.inertia = effective_inertia(s, c, n_driven)
        w.stiffness, w.damping, w.brake = s.radial_stiffness, s.radial_damping, s.brake_torque
        w.slope, w.ack = s.arm_slope, s.ackermann
        rows.append(w)
    return wp.array(rows, dtype=Wheel, device=device)


def tyre_of(c: Any) -> Tyre:
    t = Tyre()
    t.mu, t.bx, t.cx, t.by, t.cy = c.mu, c.slip_b, c.slip_c, c.angle_b, c.angle_c
    t.rolling, t.ground = c.rolling_resistance, c.ground
    t.sigma_x, t.sigma_y = c.relaxation_x, c.relaxation_y
    t.v_low, t.zeta = c.low_speed, c.low_speed_damping
    return t


def drive_of(c: Any, specs: list[WheelSpec]) -> Drive:
    d = Drive()
    d.torque, d.free_speed, d.gear = c.motor_torque, c.motor_free_speed, c.gear
    d.driven = sum(s.driven for s in specs)
    return d


__all__ = ["Drive", "Tyre", "Wheel", "WheelSpec", "drive_of", "effective_inertia", "tyre_of",
           "wheel_array", "wheel_specs"]
