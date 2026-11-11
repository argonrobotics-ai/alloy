# SPDX-License-Identifier: Apache-2.0
"""Law S-1: a counterbalance forklift's lateral stability on a tilt table, with its oscillating steer axle (SCR-WP8-12
F3), against the two-regime closed form.

The truck is Gym-Anything's (Hyster H2.5A numbers, ``F/`` = gym-anything-robotics benchmarks/roboworld/environments/
forklift/): 3,990 kg, CG 0.935 m behind the front axle and 0.68 m up (F/params.py:113-114), wheelbase 1.649 m
(:65), front wheels at y = ±0.499 m (:66), rear at ±0.5685 m on a steer axle (:67) that rolls about a pin at
(-1.62, 0, 0.30) m within ±3.5° (:76-77). The axle (150 kg, centre of mass on the pin) carries the two rear wheels.

Tilted sideways (downhill -y), the truck goes through two regimes:
1. The axle rolls freely, so the rear pair acts at the pin: the support is the triangle of the two front contacts and
   the pin. The uphill front wheel lifts when gravity through the CG meets the axis from the downhill front contact to
   the pin: tan β₁ = -M_z/M_y, the moments about that axis of the vertical and lateral gravity (22.6°).
2. The chassis then turns about that axis until the axle reaches its stop (3.7°); the downhill rear wheel takes the
   load and the axis moves to the line through the downhill front and rear contacts. The truck tips when gravity
   through the turned CG meets that line (36.3°). A truck on four wheels fixed to its chassis lifts no wheel until
   38.4°.

The closed forms are for rigid tyres: the law's tyres are stiff (4e6 N/m), and their deflection lowers the angles by
under 1 % (at the forklift task's 1e6 N/m, about 2 %, toward less stable). Point wheels and the linear radial spring
overstate lateral stability slightly, by the tyre's deflection (vehicle-v2 Limits).

The check fits each lifting wheel's load against the tilt, ``F_z / cos β = a + b·tan β`` (exact for linear statics),
and compares the zero crossing with the closed form, within 3 %: the uphill front wheel for β₁ (before it lifts)
and the uphill rear wheel for β₂ (with the axle on its stop). It also checks that the axle stays near level before
β₁, and sits on its stop in regime 2.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform

TRUCK, AXLE = "forklift", "axle"
MASS, AXLE_MASS = 3990.0, 150.0                       # kg (F/params.py:91; axle INFERRED)
CG = np.array([-0.935, 0.0, 0.68])                    # whole truck, front axle at x = 0, ground z = 0 (:113-114)
PIN = np.array([-1.62, 0.0, 0.30])                    # axle pin (:76)
STOP = math.radians(3.5)                              # axle roll stop (:77)
FRONT = (0.0, 0.499, 0.323)                           # left front wheel centre and radius 0.323 m (:66, :68)
REAR = (-1.649, 0.5685, 0.2565)                       # left rear wheel centre, radius 0.2565 m (:67, :70)
HOLD, RATE, TOP = 0.4, math.radians(3.0), math.radians(40.0)
TOL = 0.03

CONFIG = {"chassis": TRUCK, "bodies": ("", "", AXLE, AXLE), "radius": (FRONT[2], FRONT[2], REAR[2], REAR[2]),
          "positions": ((FRONT[0], FRONT[1], FRONT[2]), (FRONT[0], -FRONT[1], FRONT[2]),
                        (REAR[0] - PIN[0], REAR[1], REAR[2] - PIN[2]), (REAR[0] - PIN[0], -REAR[1], REAR[2] - PIN[2])),
          "steer": (0.0, 0.0, -1.0, -1.0), "ackermann": (0.0, 0.0, FRONT[0] - REAR[0], FRONT[0] - REAR[0]),
          "driven": (True, True, False, False), "inertia": (2.0,),
          "radial_stiffness": (4.0e6,), "radial_damping": (1.5e4,), "brake_torque": (3000.0,), "mu": 1.2}


# --- closed form ------------------------------------------------------------------------------------------------------

def tip_tan(cg: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """tan β at which gravity tilted toward -y, through ``cg``, meets the axis a→b (zero moment about it)."""
    u = (b - a) / np.linalg.norm(b - a)
    r = cg - a
    m_z = float(u @ np.cross(r, (0.0, 0.0, -1.0)))
    m_y = float(u @ np.cross(r, (0.0, -1.0, 0.0)))
    return -m_z / m_y


def _rotation(u: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _axle_roll(rot: np.ndarray) -> float:
    """The axle's roll relative to the chassis that keeps both rear wheels level when the chassis is at ``rot``."""
    y, z = rot[:, 1], rot[:, 2]
    return math.atan2(-y[2], z[2])


def _turn_to_stop(u: np.ndarray, a: np.ndarray, cg_chassis: np.ndarray) -> float:
    """The turn about axis ``u`` (through ``a``) that moves the chassis CG downhill and brings the axle to its stop."""
    sign = -1.0 if (_rotation(u, 1e-3) @ (cg_chassis - a))[1] > (cg_chassis - a)[1] else 1.0
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if abs(_axle_roll(_rotation(u, sign * mid))) < STOP else (lo, mid)
    return sign * 0.5 * (lo + hi)


def closed_form() -> dict[str, float]:
    """β₁ (uphill front wheel lifts, axle swings onto its stop), β₂ (final tip), and the 4-point truck's tip [rad]."""
    a = np.array([FRONT[0], -FRONT[1], 0.0])                                   # downhill front contact
    beta1 = math.atan(tip_tan(CG, a, PIN))
    cg_chassis = (MASS * CG - AXLE_MASS * PIN) / (MASS - AXLE_MASS)
    u = (PIN - a) / np.linalg.norm(PIN - a)
    psi = _turn_to_stop(u, a, cg_chassis)
    rot = _rotation(u, psi)
    cg2 = ((MASS - AXLE_MASS) * (a + rot @ (cg_chassis - a)) + AXLE_MASS * PIN) / MASS
    phi = _axle_roll(rot)
    axle_rot = rot @ _rotation(np.array([1.0, 0.0, 0.0]), phi)
    wheel = a + rot @ (PIN - a) + axle_rot @ (np.array([REAR[0], -REAR[1], REAR[2]]) - PIN)
    b = np.array([wheel[0], wheel[1], 0.0])                                    # downhill rear contact
    beta2 = math.atan(tip_tan(cg2, a, b))
    four = math.atan(tip_tan(CG, a, np.array([REAR[0], -REAR[1], 0.0])))
    return {"beta1": beta1, "beta2": beta2, "four_point": four, "turn": abs(psi)}


# --- the rig -----------------------------------------------------------------------------------------------------

def _box(b: newton.ModelBuilder, body: int, centre: np.ndarray, half: tuple[float, float, float], mass: float) -> None:
    density = mass / (8.0 * half[0] * half[1] * half[2])
    b.add_shape_box(body, xform=body_xform(*centre), hx=half[0], hy=half[1], hz=half[2],
                    cfg=newton.ModelBuilder.ShapeConfig(density=density))


def build_truck(b: newton.ModelBuilder, pose: wp.transform | None = None) -> None:
    """The chassis (free) and the steer axle on its roll pin with ±3.5° stops (MuJoCo joint limits); ``pose`` places
    the truck's frame (front axle at ground level) in the world."""
    at = pose if pose is not None else wp.transform_identity()
    chassis = b.add_link(xform=at, label=TRUCK)
    cg_chassis = (MASS * CG - AXLE_MASS * PIN) / (MASS - AXLE_MASS)
    _box(b, chassis, cg_chassis, (1.0, 0.45, 0.35), MASS - AXLE_MASS)
    free = b.add_joint_free(chassis)
    axle = b.add_link(xform=wp.transform_multiply(at, body_xform(*PIN)), label=AXLE)
    _box(b, axle, np.zeros(3), (0.075, 0.6, 0.06), AXLE_MASS)
    pin = b.add_joint_revolute(chassis, axle, parent_xform=body_xform(*PIN), child_xform=wp.transform_identity(),
                               axis=(1.0, 0.0, 0.0), limit_lower=-STOP, limit_upper=STOP, limit_ke=1.0e7,
                               limit_kd=1.0e5, label="axle_pin")
    b.add_articulation([free, pin], label=TRUCK)


def _tilt():
    from alloy.plugins.vehicle.tilt import TiltTable  # noqa: PLC0415

    return [TiltTable(bodies=(TRUCK, AXLE), hold=HOLD, rate=RATE, stop=TOP)]


def _brake(sim: Any) -> None:
    for p in sim.plugins.plugins.values():
        if hasattr(p, "set_spin"):
            p.set_input("brake", 1.0)


def tilt_scene() -> TestScene:
    periods = int(math.ceil((HOLD + TOP / RATE) / 0.02))
    return TestScene(build=build_truck, init=_brake, ground=False, periods=periods, dt=1.25e-3, worlds=2,
                     companions=_tilt)


def axle_laws() -> list[Law]:
    return [Law("S-1_lateral_tilt", check_tilt, tilt_scene(),
                "oscillating steer axle: uphill front wheel lifts at the triangle's β₁, truck tips at β₂",
                config=CONFIG)]


# --- the check --------------------------------------------------------------------------------------------------------

def zero_crossing(beta: np.ndarray, load: np.ndarray) -> float:
    """β where the linear-statics fit ``F_z/cos β = a + b·tan β`` reaches 0."""
    x, y = np.tan(beta), load / np.cos(beta)
    slope, icept = np.polyfit(x, y, 1)
    return math.atan(-icept / slope)


def relative_roll(body_q: np.ndarray, chassis: int, axle: int) -> np.ndarray:
    """``[K]`` roll of the axle relative to the chassis [rad] (body_q quaternions are xyzw)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    rel = Rotation.from_quat(body_q[:, chassis, 3:]).inv() * Rotation.from_quat(body_q[:, axle, 3:])
    return rel.as_rotvec()[:, 0]


def check_tilt(run: Any) -> LawResult:
    cf = closed_form()
    beta = np.asarray(run.trace.plugin("tilt_table", "beta"))[:, 0].astype(np.float64)
    fz = np.asarray(run.trace.plugin(run.plugin.label, "force"))[:, 0, :, 2].astype(np.float64)
    roll = relative_roll(np.asarray(run.trace.body_q, np.float64), int(run.bodies(TRUCK)[0]),
                         int(run.bodies(AXLE)[0]))
    settled = float(fz[int(HOLD / 0.02), 0])
    pre = (beta > 0.4 * cf["beta1"]) & (fz[:, 0] > 0.05 * settled) & (beta < cf["beta1"] * 1.1)
    b1 = zero_crossing(beta[pre], fz[pre, 0])
    level = float(np.abs(roll[beta < 0.95 * b1]).max())
    on_stop = (np.abs(np.abs(roll) - STOP) < math.radians(0.3)) & (beta > b1 + math.radians(3.0))
    rear_max = float(fz[on_stop, 2].max()) if on_stop.any() else 0.0
    post = on_stop & (fz[:, 2] > 0.05 * rear_max)
    if post.sum() < 10:
        return LawResult.within(1.0, 0.0, f"regime 2 never settled on the stop ({int(post.sum())} samples; roll max "
                                f"{math.degrees(np.abs(roll).max()):.2f}°)", "axle stop as a joint limit")
    b2 = zero_crossing(beta[post], fz[post, 2])
    errs = {"beta1": b1 / cf["beta1"] - 1.0, "beta2": b2 / cf["beta2"] - 1.0}
    worst = max(abs(errs["beta1"]) / TOL, abs(errs["beta2"]) / TOL, level / math.radians(1.0))
    return LawResult.within(worst, 1.0, f"uphill front wheel lifts at {math.degrees(b1):.2f}° vs triangle "
                            f"{math.degrees(cf['beta1']):.2f}° ({errs['beta1']:+.2%}); tips at {math.degrees(b2):.2f}° "
                            f"vs {math.degrees(cf['beta2']):.2f}° on the stop ({errs['beta2']:+.2%}); axle roll before "
                            f"β₁ {math.degrees(level):.2f}°; 4-point truck: no lift until "
                            f"{math.degrees(cf['four_point']):.1f}°",
                            "wheels on the axle body, the stop as a joint limit, the axle's mobility in check_dt")


__all__ = ["CONFIG", "axle_laws", "check_tilt", "closed_form", "tilt_scene", "tip_tan", "zero_crossing"]
