# SPDX-License-Identifier: Apache-2.0
"""Waypoint paths and path tracking for the forklift operator (forward and reverse).

A ``Segment`` is a list of ``(x, y, heading, direction)`` waypoints that are all driven one way (``direction`` 1
forward, -1 reverse; ``heading`` is the truck's yaw at the point) with one speed cap. ``Path`` joins them with cubic
Hermite curves into a dense polyline and extends it past the end along the final tangent so the pursuit target never
shrinks to nothing. The truck reference point is the non-steered (front) axle, so the same pursuit law steers
forward and reverse (see ``pursue``: curvature feed-forward with heading/cross-track feedback, a Stanley-type
law that replaces pure pursuit because it also settles the final heading).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scipy.ndimage import uniform_filter1d

from alloy.tasks.forklift_control import TruckSpec, clip

Waypoint = tuple[float, float, float, int]      # x, y, heading, direction
STEP = 0.02                                     # m between path samples
EXTENSION = 2.5                                 # m the path continues past its last waypoint
OMEGA, DAMPING = 1.5, 1.0                       # 1/m, tracking error natural frequency and damping ratio


@dataclass(frozen=True)
class Segment:
    """One driving direction, one speed cap (km/h)."""

    points: tuple[Waypoint, ...]
    cap_kmh: float

    @property
    def direction(self) -> int:
        return self.points[0][3]

    @property
    def end(self) -> Waypoint:
        return self.points[-1]


def segment(points: list[Waypoint] | tuple[Waypoint, ...], cap_kmh: float) -> Segment:
    if len(points) < 2 or len({p[3] for p in points}) != 1:
        raise ValueError("a segment needs at least two waypoints, all with one direction")
    return Segment(tuple(points), cap_kmh)


def _travel_heading(point: Waypoint) -> float:
    return point[2] + (math.pi if point[3] < 0 else 0.0)


def _hermite(p0: Waypoint, p1: Waypoint) -> np.ndarray:
    a, b = np.array(p0[:2]), np.array(p1[:2])
    chord = float(np.linalg.norm(b - a))
    m0 = chord * np.array([math.cos(_travel_heading(p0)), math.sin(_travel_heading(p0))])
    m1 = chord * np.array([math.cos(_travel_heading(p1)), math.sin(_travel_heading(p1))])
    u = np.linspace(0.0, 1.0, max(2, math.ceil(chord / STEP) + 1))[:, None]
    return ((2 * u ** 3 - 3 * u ** 2 + 1) * a + (u ** 3 - 2 * u ** 2 + u) * m0
            + (-2 * u ** 3 + 3 * u ** 2) * b + (u ** 3 - u ** 2) * m1)


@dataclass(frozen=True)
class Path:
    """Dense polyline ``xy`` with cumulative length ``s``; ``length`` is the arc length to the last waypoint."""

    xy: np.ndarray
    s: np.ndarray
    length: float
    direction: int
    heading: np.ndarray        # travel heading at each sample (rad, unwrapped)
    curvature: np.ndarray      # d heading / d s (1/m, + = turning left along the travel direction)

    @staticmethod
    def from_segment(seg: Segment) -> "Path":
        pts = [_hermite(a, b) for a, b in zip(seg.points[:-1], seg.points[1:]) if a[:2] != b[:2]]
        xy = np.vstack([pts[0]] + [p[1:] for p in pts[1:]])
        tail = _travel_heading(seg.end)
        steps = np.arange(STEP, EXTENSION + STEP, STEP)[:, None]
        ext = xy[-1] + steps * np.array([math.cos(tail), math.sin(tail)])
        full = np.vstack([xy, ext])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))])
        step = np.diff(full, axis=0)
        heading = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
        heading = np.append(heading, heading[-1])
        smooth = uniform_filter1d(heading, 15, mode="nearest")
        curvature = uniform_filter1d(np.gradient(smooth, STEP), 15, mode="nearest")
        return Path(full, s, float(s[len(xy) - 1]), seg.direction, heading, curvature)

    def project(self, x: float, y: float, start: int) -> tuple[int, float]:
        """(nearest sample index at or after ``start``, arc length of the foot point)."""
        window = self.xy[start:start + 400]
        idx = start + int(np.argmin(np.hypot(window[:, 0] - x, window[:, 1] - y)))
        lo, hi = max(idx - 1, 0), min(idx + 1, len(self.xy) - 1)
        tangent = self.xy[hi] - self.xy[lo]
        tangent = tangent / (np.linalg.norm(tangent) or 1.0)
        return idx, float(self.s[idx] + (np.array([x, y]) - self.xy[idx]) @ tangent)

    def curvature_ahead(self, index: int, distance: float = 3.0) -> float:
        """Largest |curvature| (1/m) within ``distance`` m after sample ``index``."""
        return float(np.max(np.abs(self.curvature[index:index + int(distance / STEP) + 1])))

    def cross_track(self, x: float, y: float) -> float:
        """Distance from (x, y) to the real path (for checks)."""
        n = int(np.searchsorted(self.s, self.length)) + 1
        return float(np.min(np.hypot(self.xy[:n, 0] - x, self.xy[:n, 1] - y)))


@dataclass(frozen=True)
class Pursuit:
    steer: float
    index: int
    remaining: float          # m of path left to the last waypoint (negative once past it)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pursue(path: Path, x: float, y: float, yaw: float, start: int, spec: TruckSpec) -> Pursuit:
    """Steer onto the path: path curvature fed forward, heading and cross-track error fed back in arc length.

    In the travel frame (forward, or reversed with the heading turned by pi) the error obeys
    ``cte'' = kappa_cmd - kappa_path``, so ``kappa_cmd = kappa_path - 2 z w e - w² cte`` settles with natural
    frequency ``w`` per metre, whatever the speed. The body-frame curvature is ``direction * kappa_cmd`` and
    ``steer = atan(L * kappa)``. A Stanley-type law: the reference point is the non-steered axle.
    """
    idx, s_here = path.project(x, y, start)
    heading = float(path.heading[idx])
    travel_yaw = yaw + (math.pi if path.direction < 0 else 0.0)
    e_h = clip(wrap(travel_yaw - heading), -1.0, 1.0)
    cte = clip((x - path.xy[idx, 0]) * -math.sin(heading) + (y - path.xy[idx, 1]) * math.cos(heading), -1.0, 1.0)
    kappa = float(path.curvature[idx]) - 2.0 * DAMPING * OMEGA * e_h - OMEGA ** 2 * cte
    steer = math.atan(spec.wheelbase * path.direction * kappa)
    return Pursuit(clip(steer, -spec.max_steer, spec.max_steer), idx, path.length - s_here)
