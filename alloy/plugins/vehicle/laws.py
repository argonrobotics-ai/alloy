# SPDX-License-Identifier: Apache-2.0
"""Test scenes and laws of the vehicle plug-in, against closed forms (Gym-Anything forklift gap).

- ``straight_accel``: from rest at constant motor torque (no torque fade), the cart accelerates at
  ``a = (gear·T/R - c_rr·m·g) / m_eff``, with ``m_eff = m + Σ I_eff/R²`` (wheels, and the motor through the gear),
  within 2 %.
- ``coast_down`` (rolling resistance): rolling from 5 m/s with no drive, it slows at ``c_rr·m·g / m_eff``, within 3 %.
- ``cornering``: at 5 m/s with the front wheels steered by δ, the steady yaw rate is ``v·δ/L``, within 3 %. Tyres
  whose cornering stiffness is proportional to their load (D = μ·F_z) make the bicycle model neutral-steer at any
  weight split. Each axle's slip angle (mean of its pair) is ``a_y / (B·C·μ·g)`` (the linear tyre), within 5 %; the
  parallel-steered front wheels split around it (no Ackermann geometry).
- ``airborne``: a chassis held 0.5 m above the ground at full throttle gets no tyre force.
- ``S-1_lateral_tilt`` (``laws_axle``): GA's forklift with its oscillating steer axle on a tilt table: the uphill front
  wheel lifts at the stability triangle's angle and the truck tips at the second regime's, within 3 %.

The test vehicle is an 800 kg box chassis (2.0 x 1.2 x 0.4 m) on four 0.3 m wheels, 1.2 m wheelbase, rear drive,
front steer. It starts on its tyres' static deflection. There is no ground shape: the wheels are virtual, so nothing
touches the solver.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform

G = 9.81
HALF = (1.0, 0.6, 0.2)                   # chassis half extents [m]
MASS = 800.0                             # kg
WHEELBASE = 1.2                          # m (front and rear axles at x = +-0.6)
ACCEL = {"motor_free_speed": 1.0e6}      # constant motor torque (no fade) for the closed form
COAST = {"rolling_resistance": 0.03}
CORNER = {"rolling_resistance": 0.0}
THROTTLE, V0, DELTA = 0.5, 5.0, 0.05


def merged(c: Any, overrides: dict[str, Any]) -> Any:
    return dataclasses.replace(c, **overrides)


def cart(c: Any, lift: float = 0.0):
    """The box chassis resting on its tyres (wheel centres at chassis z = 0)."""
    radius = float(np.atleast_1d(c.radius)[0])
    stiffness = float(np.atleast_1d(c.radial_stiffness)[0])
    sag = MASS * G / (len(c.positions) * stiffness)

    def build(b: newton.ModelBuilder) -> None:
        body = b.add_body(xform=body_xform(0.0, 0.0, c.ground + radius - sag + lift), label=c.chassis)
        b.add_shape_box(body, hx=HALF[0], hy=HALF[1], hz=HALF[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=MASS / (8.0 * HALF[0] * HALF[1] * HALF[2])))
    return build


def start(speed: float = 0.0, throttle: float = 0.0, steer: float = 0.0):
    def init(sim: Any) -> None:
        qd = sim.state.joint_qd.numpy().reshape(-1, 6)
        qd[:, 0] = speed
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
        for p in sim.plugins.plugins.values():
            if hasattr(p, "set_spin"):
                p.set_spin(speed / float(np.atleast_1d(p.config.radius)[0]))
                p.set_input("throttle", throttle)
                p.set_input("steer", steer)
    return init


def drive_scene(c: Any) -> TestScene:
    """From rest, 30 % throttle with the front wheels steered by 0.1 rad: an accelerating arc."""
    return TestScene(build=cart(c), init=start(throttle=0.3, steer=0.1), ground=False, periods=25)


def vehicle_laws(c: Any) -> list[Law]:
    from alloy.plugins.vehicle.laws_axle import axle_laws  # noqa: PLC0415 (laws_axle imports the harness only)

    return [*_cart_laws(c), *axle_laws()]


def _cart_laws(c: Any) -> list[Law]:
    return [
        Law("straight_accel", check_accel, TestScene(build=cart(merged(c, ACCEL)), init=start(throttle=THROTTLE),
                                                      ground=False, periods=40),
            "a = (gear·T/R - c_rr·m·g) / m_eff from rest", config=ACCEL),
        Law("coast_down", check_coast, TestScene(build=cart(merged(c, COAST)), init=start(speed=V0), ground=False,
                                                  periods=50),
            "rolling resistance: a = -c_rr·m·g / m_eff", config=COAST),
        Law("cornering", check_corner, TestScene(build=cart(merged(c, CORNER)), init=start(speed=V0, steer=DELTA),
                                                  ground=False, periods=75),
            "steady yaw rate v·δ/L (neutral steer) and slip angle a_y/(B·C·μ·g)", config=CORNER),
        Law("airborne", check_airborne, TestScene(build=cart(c, lift=0.5), init=start(throttle=1.0), ground=False,
                                                   gravity=(0.0, 0.0, 0.0), periods=10),
            "a chassis 0.5 m up gets no tyre force"),
    ]


# --- checks ------------------------------------------------------------------------------------------------------

def _chassis(run: Any) -> tuple[int, float]:
    b = int(run.bodies(run.plugin.config.chassis)[0])
    return b, float(run.model.body_mass.numpy()[b])


def _fit_slope(t: np.ndarray, v: np.ndarray) -> float:
    return float(np.polyfit(t, v, 1)[0])


def check_accel(run: Any) -> LawResult:
    p, c = run.plugin, run.plugin.config
    b, m = _chassis(run)
    r = float(np.atleast_1d(c.radius)[0])
    want = (c.gear * THROTTLE * c.motor_torque / r - c.rolling_resistance * m * G) / p.effective_mass(m)
    t, v = np.asarray(run.trace.t), run.trace.body_qd[:, b, 0].astype(np.float64)
    got = _fit_slope(t[10:], v[10:])
    return LawResult.within((got - want) / want, 0.02, f"a {got:.4f} vs {want:.4f} m/s² (m_eff {p.effective_mass(m):.1f} "
                            f"kg, v {v[-1]:.2f} m/s)", "drive torque gear·T over the driven wheels, spin stepped with "
                            "the tyre force: the chassis then sees m_eff")


def check_coast(run: Any) -> LawResult:
    p, c = run.plugin, run.plugin.config
    b, m = _chassis(run)
    want = -c.rolling_resistance * m * G / p.effective_mass(m)
    t, v = np.asarray(run.trace.t), run.trace.body_qd[:, b, 0].astype(np.float64)
    got = _fit_slope(t[5:], v[5:])
    return LawResult.within((got - want) / want, 0.03, f"a {got:.4f} vs {want:.4f} m/s² from {v[0]:.2f} m/s",
                            "rolling resistance c_rr·F_z·R against each wheel's spin")


def check_corner(run: Any) -> LawResult:
    p, c = run.plugin, run.plugin.config
    b, _ = _chassis(run)
    tail = slice(-25, None)                                         # the last 0.5 s, steady
    qd = run.trace.body_qd[tail, b].astype(np.float64)
    speed = np.hypot(qd[:, 0], qd[:, 1])
    yaw = qd[:, 5]
    want_yaw = speed * DELTA / WHEELBASE
    yaw_err = float(np.abs(yaw.mean() / want_yaw.mean() - 1.0))
    wheel = np.abs(np.asarray(run.trace.plugin(p.label, "slip"))[tail, 0, :, 1]).mean(axis=0)   # [N]
    x = np.asarray([s.center[0] for s in p.specs])
    alpha = np.asarray([wheel[x > 0].mean(), wheel[x < 0].mean()])        # per axle: the bicycle model's slip angle
    want_alpha = float((speed * yaw).mean()) / (c.angle_b * c.angle_c * c.mu * G)
    slip_err = float(np.abs(alpha / want_alpha - 1.0).max())
    worst = max(yaw_err / 0.03, slip_err / 0.05)
    return LawResult.within(worst, 1.0, f"yaw rate {yaw.mean():.4f} vs v·δ/L {want_yaw.mean():.4f} rad/s "
                            f"({yaw_err:.2%}); axle slip angles {np.round(alpha, 5)} vs {want_alpha:.5f} rad ({slip_err:.2%}; "
                            f"wheels {np.round(wheel, 5)}: parallel steer splits the front pair)",
                            "lateral force -μ·F_z·sin(C·atan(B·α)) with α from the contact point's velocity")


def check_airborne(run: Any) -> LawResult:
    load = float(np.abs(np.asarray(run.trace.plugin(run.plugin.label, "force"))).max())
    b, _ = _chassis(run)
    moved = float(np.abs(run.trace.body_qd[:, b]).max())
    return LawResult.within(load + moved, 0.0, "tyre force + chassis speed with the wheels 0.5 m off the ground",
                            "apply tyre force only while the wheel's lowest point is below the ground")


__all__ = ["drive_scene", "merged", "vehicle_laws"]
