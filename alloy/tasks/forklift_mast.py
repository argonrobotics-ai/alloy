# SPDX-License-Identifier: Apache-2.0
"""Closed forms for GA's forklift mast (SCR-WP8-12 F2 laws, tests/tasks/test_forklift_mast.py), from the same part
masses and kinematics as ``forklift_truck``.

- **M-1 load moment**: on level ground the rear axle carries ``R = -M·g·x̄/L`` (x̄ the whole truck's centre of mass
  ahead of the front axle, negative behind it; L the wheelbase) and the front the rest, for any mast pose and load.
  On the tyres' radial springs the truck pitches nose down by the axles' deflection difference over L, and the
  centre of mass, 1.5 m up with a raised load, moves ahead by its height times that pitch (solved to a fixed point).
- **M-2 tip-over**: the truck tips forward about the front axle when x̄ reaches it: the load at which the rear axle's
  load is 0 (rigid: ``m_tip = -Σ m·x / x_load``, the sum over the truck without the load; on the tyres a little less).
- **M-3 two-stage lift**: one pump feeds both stages; the one needing less pressure moves first.
  - free lift (2:1 chain on the carriage): ``P = g·(m_carriage + m_side + m_forks + m_load)/A_free``, speed
    ``Q/A_free`` (Q the pump's flow compressed to P: ``Q/(1 + P/K)``);
  - then staging (the inner stage tied 1:1 to the intermediate): ``P = g·(m_int + 2·(m_inner + m_carriage + ... +
    m_load))/A_main``, the carriage rising at ``2·Q/A_main``;
  - top of the carriage 0.96 + 2 x 1.69 m above its lowest.
- **M-4 tilt**: the tilt vane holds the mast's moment about the pivot, ``ΔP = τ/D``, and turns it at ``Q/D``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import warp as wp

from alloy.plugins.vehicle import laws_axle as ax
from alloy.tasks import forklift_truck as ft

G = 9.81
WHEELBASE = -ax.REAR[0]


def fixed_masses() -> list[tuple[float, np.ndarray]]:
    """The chassis and the steer axle (they do not move with the mast)."""
    return [(ft.chassis_mass(), ft.chassis_centre()), (ax.AXLE_MASS, np.asarray(ax.PIN, np.float64))]


def centre(pose: ft.Pose, load: ft.Load | None = None) -> tuple[float, np.ndarray]:
    parts = fixed_masses() + ft.mast_masses(pose, load)
    mass = sum(m for m, _ in parts)
    return mass, sum(m * c for m, c in parts) / mass


def _ahead(point, pitch: float) -> float:
    """A truck-frame point's distance ahead of the truck's origin once the truck pitches ``pitch`` (+ nose down)."""
    return float(point[0]) * math.cos(pitch) + float(point[2]) * math.sin(pitch)


def axle_loads(pose: ft.Pose, load: ft.Load | None = None, stiffness: float = math.inf) -> tuple[float, float]:
    """(front, rear) axle loads [N] on level ground. On tyres of radial ``stiffness`` [N/m] (two per axle) the truck
    pitches toward the heavier axle and the high centre of mass moves with it."""
    mass, c = centre(pose, load)
    pitch, front, rear = 0.0, 0.0, 0.0
    for _ in range(4):
        x_c, x_f, x_r = _ahead(c, pitch), _ahead(ax.FRONT, pitch), _ahead(ax.REAR, pitch)
        rear = mass * G * (x_f - x_c) / (x_f - x_r)
        front = mass * G - rear
        pitch = 0.5 * (front - rear) / stiffness / WHEELBASE
    return front, rear


def rest_at(pose: ft.Pose, load: ft.Load | None = None, stiffness: float = 1.0e6) -> wp.transform:
    """The truck's frame at its static rest on level ground (each tyre pressed by half its axle's load), so a
    statics law starts without a drop."""
    front, rear = axle_loads(pose, load, stiffness)
    sink_front, sink_rear = 0.5 * front / stiffness, 0.5 * rear / stiffness
    pitch = (sink_front - sink_rear) / WHEELBASE                     # + about y: nose down
    return wp.transform(wp.vec3(0.0, 0.0, -sink_front), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch))


def tip_load(pose: ft.Pose, template: ft.Load, stiffness: float = math.inf) -> float:
    """The load mass [kg] (shape and place of ``template``) at which the rear axle's load reaches 0."""
    lo, hi = 0.0, 1.0e5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rear = axle_loads(pose, dataclasses.replace(template, mass=mid), stiffness)[1]
        lo, hi = (mid, hi) if rear > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def _carried(load: ft.Load | None) -> float:
    names = ("carriage", "sideshift")
    return sum(p.mass for p in ft.PARTS if p.body in names) + 2.0 * ft.FORK_MASS + (load.mass if load else 0.0)


def lift_pressures(load: ft.Load | None = None) -> tuple[float, float]:
    """Gauge pressures [Pa] that hold the free-lift and the main stages."""
    carried = _carried(load)
    m_int = next(p.mass for p in ft.PARTS if p.body == "mast_int")
    m_inner = next(p.mass for p in ft.PARTS if p.body == "mast_inner")
    return G * carried / ft.AREA_FREE, G * (m_int + 2.0 * (m_inner + carried)) / ft.AREA_MAIN


def lift_speeds(load: ft.Load | None = None, bulk: float = ft.BULK) -> tuple[float, float]:
    """Carriage speeds [m/s] in free lift and in staging, at the pump's flow. The pump meters its flow at atmospheric
    pressure, so at the lift pressure P the oil fills ``Q/(1 + P/K)`` (0.75 % less at 6 MPa)."""
    p_free, p_main = lift_pressures(load)
    return (ft.PUMP_FLOW / (1.0 + p_free / bulk) / ft.AREA_FREE,
            2.0 * ft.PUMP_FLOW / (1.0 + p_main / bulk) / ft.AREA_MAIN)


def top_height() -> float:
    return ft.FREE + 2.0 * ft.STAGE


def tilt_moment(pose: ft.Pose, load: ft.Load | None = None) -> float:
    """Gravity's moment [N·m] about the tilt pivot (+ tips the mast forward) of every part on the mast."""
    return sum(m * G * float(c[0] - ft.PIVOT[0]) for m, c in ft.mast_masses(pose, load))


def tilt_rate() -> float:
    return ft.PUMP_FLOW / ft.TILT_D


__all__ = ["axle_loads", "centre", "lift_pressures", "lift_speeds", "rest_at", "tilt_moment", "tilt_rate", "tip_load",
           "top_height"]
