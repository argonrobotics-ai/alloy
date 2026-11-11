# SPDX-License-Identifier: Apache-2.0
"""Grade and grip laws of the vehicle plug-in's transient tyre (SCR-WP8-12 F1, WP13 review), on Gym-Anything's
H2.5A truck (``laws_axle``) on a tilt table at the forklift task's step (1.25 ms), against closed forms.

The truck faces +x; its front (drive, braked) axle is at x = 0, its CG ``a`` = 0.935 m behind it and ``h`` = 0.68 m up,
wheelbase ``L`` = 1.649 m, so ``b = L - a`` from the CG to the rear axle. On a grade θ with the front uphill the front
axle carries ``F_f = m·g·(b·cos θ - h·sin θ)/L``; with it downhill, ``+ h·sin θ``.

- **R-1 traction**: front drive, front uphill, rolling resistance ``c_rr`` on every wheel. The drive force the grade
  needs is at most the front tyres' peak ``μ·F_f``: ``tan θ* = (μ·b - c_rr·a) / (L + (μ + c_rr)·h)``. With the drive
  torque that the grade needs plus 2 %, the truck climbs at 0.96·tan θ* and spins back at 1.04·tan θ*.
- **R-1 speed**: on GA's 13.6 % ramp at full throttle, the motor line ``T = T_max·(1 - gear·ω/ω_free)`` meets the
  grade load ``R·m·g·(sin θ + c_rr·cos θ)/gear``; the truck climbs at ``ω·R/(1 + κ)``, with κ the front tyres' slip at
  their share of the drive force (inverse magic formula), within 2 %.
- **R-2 hold**: front brakes set on the 13.6 % ramp, both ways round (limits: friction ``μ·b/(L ± μ·h)``, brake torque
  ``2·T_b/(R·m·g)``): the truck creeps at most 1 mm in 10 s; also with the steer wheels at full lock (80°), where the
  steer axle's tread sees the pin's large mobility across the axle (WP13), with the tread rates settled.
- **R-2 slide**: on μ = 0.4, front downhill, at 1.15x its friction limit, the locked front tyres slide at the magic
  formula's locked value ``sin(C_x·atan(B_x))·μ`` and the truck accelerates downhill at
  ``a = (g·sin θ - c_rr·(g·cos θ - F_f/m) - μ_l·F_f/m)`` with the load transfer
  ``F_f = m·(g·b·cos θ + (g·sin θ - a)·h)/L``, within 3 %, once it has slid 4 σ_x (the tread at 98 % of locked slip).
- **G-1 slide, then stop** (WP13): on μ = 0.25 the truck slides sideways at 18° of lateral tilt; back at 8° it stops,
  its tyres carry the static share ``m·g·sin 8°`` within 3 %, and it moves uphill at most σ_y (stored tread); also at
  full lock.
- **G-2 spin, then stop** (WP13): the test cart, front brakes set, spins its rear wheels at standstill for 1 s; once
  they stop, their drive force falls to rolling resistance (no stored drive force).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable

import numpy as np

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.vehicle import laws_axle as ax

G, STEP, PERIOD, HOLD = 9.81, 1.25e-3, 0.02, 0.3
L, A_CG, H_CG = 1.649, 0.935, 0.68
B_CG = L - A_CG
GRADE = 0.136                                         # GA's yard ramp (F/facility.py:339-349)
R_FRONT = ax.FRONT[2]
TRUCK = {**ax.CONFIG, "mu": 0.865, "brake_torque": (3000.0, 3000.0, 0.0, 0.0), "rolling_resistance": 0.02,
         "relaxation_x": 0.15, "relaxation_y": 0.15, "radial_stiffness": (1.0e6,), "radial_damping": (1.5e4,),
         "gear": 16.43, "motor_torque": 400.0, "motor_free_speed": 1.0e6, "motor_inertia": 0.05}
LOCKED = math.sin(1.65 * math.atan(10.0))             # locked-wheel share of the peak (C_x 1.65, B_x 10)
FULL_LOCK = math.radians(80.0)                        # both steer wheels at GA's ~80° (parallel steering, no Ackermann)


@dataclasses.dataclass(frozen=True)
class Tilt:
    angle: float                                      # rad
    downhill: tuple[float, float]
    rate: float = math.radians(10.0)
    dwell: float = 0.0
    end: float = math.pi


def front_load(theta: float, front_uphill: bool, accel: float = 0.0) -> float:
    """Front axle load [N] of the truck on grade θ, accelerating downhill at ``accel`` (load transfer)."""
    m = ax.MASS
    along = (G * math.sin(theta) - accel) * H_CG
    return m * (G * math.cos(theta) * B_CG + (-along if front_uphill else along)) / L


def traction_limit(mu: float, c_rr: float) -> float:
    """tan θ* of the front-drive climb (front uphill)."""
    return (mu * B_CG - c_rr * A_CG) / (L + (mu + c_rr) * H_CG)


def inverse_slip(ratio: float, b: float = 10.0, c: float = 1.65) -> float:
    """κ with D·sin(C·atan(B·κ)) = ratio·D on the rising branch."""
    return math.tan(math.asin(ratio) / c) / b


def climb_speed(cfg: dict[str, Any], theta: float) -> float:
    m, r, gear = ax.MASS, R_FRONT, cfg["gear"]
    need = m * G * (math.sin(theta) + cfg["rolling_resistance"] * math.cos(theta))      # at the contacts [N]
    t_motor = need * r / gear
    omega = cfg["motor_free_speed"] * (1.0 - t_motor / cfg["motor_torque"]) / gear
    f_f = front_load(theta, True)
    per_wheel = 0.5 * (need - cfg["rolling_resistance"] * (m * G * math.cos(theta) - f_f))
    kappa = inverse_slip(per_wheel / (0.5 * cfg["mu"] * f_f))
    return omega * r / (1.0 + kappa)


def slide_accel(mu: float, theta: float, c_rr: float) -> float:
    """Downhill acceleration with the front (downhill) wheels locked, solved with the load transfer."""
    m, mu_l, s, c = ax.MASS, LOCKED * mu, math.sin(theta), math.cos(theta)
    # m·a = m·g·s - c_rr·(m·g·c - F_f) - μ_l·F_f,  F_f = m·(g·c·b + (g·s - a)·h)/L
    k = (mu_l - c_rr) * m * H_CG / L
    return (m * G * s - c_rr * m * G * c - (mu_l - c_rr) * m * (G * c * B_CG + G * s * H_CG) / L) / (m - k)


# --- running -------------------------------------------------------------------------------------------------------

def _tilt_companion(t: Tilt | None, bodies: tuple[str, ...]):
    if t is None:
        return None

    def make():
        from alloy.plugins.vehicle.tilt import TiltTable  # noqa: PLC0415

        return [TiltTable(bodies=bodies, hold=HOLD, rate=t.rate, stop=t.angle, dwell=t.dwell, end=t.end,
                          downhill=t.downhill)]
    return make


def tilt_done(t: Tilt) -> float:
    """Time [s] when the tilt first reaches its angle."""
    return HOLD + t.angle / t.rate


def run(cfg: dict[str, Any], seconds: float, inputs: dict[str, float], tilt: Tilt | None = None,
        schedule: dict[float, dict[str, float]] | None = None, build=ax.build_truck, chassis: str = ax.TRUCK,
        bodies: tuple[str, ...] = (ax.TRUCK, ax.AXLE), extra: Callable[[], list[Any]] | None = None,
        control: Callable[[float, dict[str, np.ndarray]], dict[str, float]] | None = None,
        track: tuple[str, ...] = (), device: str = "cpu", solver_kwargs: dict[str, Any] | None = None
        ) -> dict[str, np.ndarray]:
    """Step one world (on the tilt table if ``tilt``) at the task step. ``schedule`` maps a time to new inputs;
    ``control(t, last)`` returns inputs every period from the last readings; ``extra`` adds plug-ins (e.g. a second
    vehicle carrying a floor). Returns the chassis pose ``q`` [K, 7] and velocity ``qd`` [K, 6], the tyre ``force``
    [K, N, 3], ``omega`` [K, N] and tread deflection ``rate`` [K, N, 2], and ``<label>.q`` / ``.qd`` of ``track``."""
    from alloy.plugins.vehicle import Vehicle  # noqa: PLC0415

    car = Vehicle(**cfg)
    tilted = _tilt_companion(tilt, bodies)
    companions = None
    if tilted is not None or extra is not None:
        companions = lambda: [*(tilted() if tilted else []), *(extra() if extra else [])]  # noqa: E731
    scene = TestScene(build=build, ground=False, worlds=1, dt=STEP, control_dt=PERIOD, companions=companions,
                      solver_kwargs=solver_kwargs or {})
    sim = build_simulation(scene, [car], device, worlds=1)
    for k, v in inputs.items():
        car.set_input(k, v)
    pending = sorted((schedule or {}).items())
    labels = list(sim.model.body_label)
    index = {name: labels.index(name) for name in (chassis, *track)}
    out: dict[str, list[Any]] = {k: [] for k in ("t", "q", "qd", "force", "omega", "rate")}
    out.update({f"{name}.{k}": [] for name in track for k in ("q", "qd")})
    for p in range(int(round(seconds / PERIOD))):
        while pending and pending[0][0] <= p * PERIOD + 1e-9:
            for k, v in pending.pop(0)[1].items():
                car.set_input(k, v)
        if control is not None and out["t"]:
            for k, v in control(p * PERIOD, {k: v[-1] for k, v in out.items()}).items():
                car.set_input(k, v)
        sim.step_period()
        q, qd = sim.state.body_q.numpy(), sim.state.body_qd.numpy()
        out["t"].append((p + 1) * PERIOD)
        out["q"].append(q[index[chassis]].copy())
        out["qd"].append(qd[index[chassis]].copy())
        for name in track:
            out[f"{name}.q"].append(q[index[name]].copy())
            out[f"{name}.qd"].append(qd[index[name]].copy())
        out["force"].append(car.force.numpy()[0].copy())
        out["omega"].append(car.omega.numpy()[0].copy())
        out["rate"].append(car.rate.numpy()[0].copy())
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


__all__ = ["FULL_LOCK", "GRADE", "LOCKED", "TRUCK", "Tilt", "climb_speed", "front_load", "inverse_slip", "run",
           "slide_accel", "tilt_done", "traction_limit"]
