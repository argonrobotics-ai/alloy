# SPDX-License-Identifier: Apache-2.0
"""Studio framing for a thumbnail: a 3/4 view fitted to the object's bounding box and a light ground plane.

Example::

    view = frame(lo, hi)            # View(eye, target, fov_deg)
    plane = ground_plane(lo, hi)    # (points, tris) under the object
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

AZIMUTH_DEG, ELEVATION_DEG, FOV_DEG = 35.0, 25.0, 30.0
FLAT_ELEVATION_DEG, FLAT_RATIO = 55.0, 0.2      # flat objects are seen from higher up
MARGIN = 1.15            # bounding-box fit: the box spans at most 1/MARGIN of the frame
GROUND_RGB = (0.78, 0.79, 0.81)


@dataclass(frozen=True)
class View:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_deg: float = FOV_DEG

    @property
    def look_at(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self.eye, self.target


def flat(lo: np.ndarray, hi: np.ndarray) -> bool:
    """Lying flat: height under FLAT_RATIO of the larger horizontal extent (cloth, labels, trays, panels)."""
    e = np.asarray(hi, float) - np.asarray(lo, float)
    return bool(e[2] < FLAT_RATIO * max(e[0], e[1]))


def frame(lo: np.ndarray, hi: np.ndarray, azimuth: float = AZIMUTH_DEG, elevation: float | None = None,
          fov: float = FOV_DEG) -> View:
    """Camera on the 3/4 diagonal (front-right, above; higher for flat objects) at the nearest distance where all
    eight bounding-box corners fit in the fov with MARGIN to spare."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = 0.5 * (lo + hi)
    e = math.radians(elevation if elevation is not None else (FLAT_ELEVATION_DEG if flat(lo, hi) else ELEVATION_DEG))
    a = math.radians(azimuth)
    u = np.array([math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), math.sin(e)])   # centre -> eye
    right = np.cross(-u, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -u)
    t = math.tan(math.radians(fov) / 2.0) / MARGIN
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - c
    d = max(float(np.max(corners @ u + np.maximum(np.abs(corners @ right), np.abs(corners @ up)) / t)), 1e-3)
    return View(tuple(map(float, c + d * u)), tuple(map(float, c)), fov)


def ground_plane(lo: np.ndarray, hi: np.ndarray, scale: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """A square under the object (top at its lowest point), ``scale`` x its largest extent wide."""
    c = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
    h = 0.5 * scale * max(float(np.max(np.asarray(hi) - np.asarray(lo))), 1e-2)
    z = float(lo[2]) - 1e-5
    pts = np.array([[c[0] - h, c[1] - h, z], [c[0] + h, c[1] - h, z], [c[0] + h, c[1] + h, z],
                    [c[0] - h, c[1] + h, z]], np.float32)
    return pts, np.array([[0, 1, 2], [0, 2, 3]], np.int32)
