# SPDX-License-Identifier: Apache-2.0
"""Marker dots printed on the gel, and how they move: FOTS-style marker motion (Zhao et al. 2024), on the GPU.

Each marker's displacement [mm] is the sum of
- dilation: the gel pushed outward by the press, ``k_dil`` x the press-weighted pull away from every pressed pixel
  within a few ``sigma_dil_mm`` (Gaussian falloff), as in FOTS's dilation term;
- shear and twist: the contact patch dragged by the object (``alloy.camera.tactile.shear``), falling off with a
  Gaussian of ``sigma_shear_mm`` / ``sigma_twist_mm`` around the patch centre, as in FOTS's shear and twist terms;
- or, instead of shear and twist, a tangential field per pixel given by the engine (``field``, mm).
FOTS's decay lengths (1.0 / 2.5 / 1.85 mm) are converted from its published constants; ``k_dil`` is set so a 1 mm
press of a 4 mm ball moves the nearest markers about 0.2 mm (an estimate, not fitted to data).

Example::

    from alloy.camera.tactile.markers import MarkerGrid
    g = MarkerGrid(pitch_mm=1.5, field_mm=(18.9, 14.2))
    g.rest_mm.shape      # (rows * cols, 2): marker centres at rest, sensor frame [mm]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..errors import CameraError


@dataclass(frozen=True)
class MarkerMotion:
    """Constants of the marker-motion model (mm, rad)."""

    k_dil: float = 1.0
    sigma_dil_mm: float = 1.0
    sigma_shear_mm: float = 2.5
    sigma_twist_mm: float = 1.85
    max_disp_frac: float = 0.9  # displacement capped at this fraction of the pitch (keeps the dot lookup local)

    def as_vec4(self) -> wp.vec4:
        return wp.vec4(self.k_dil, self.sigma_dil_mm, self.sigma_shear_mm, self.sigma_twist_mm)


class MarkerGrid:
    """A rectangular marker grid centred on the gel; ``rest_mm`` row-major (rows along +Y, columns along +X)."""

    def __init__(self, pitch_mm: float, field_mm: tuple[float, float]):
        self.pitch_x = self.pitch_y = float(pitch_mm)
        w, h = field_mm
        self.cols = max(1, int(math.floor(w / self.pitch_x)))
        self.rows = max(1, int(math.floor(h / self.pitch_y)))
        self.x0 = -0.5 * (self.cols - 1) * self.pitch_x
        self.y0 = -0.5 * (self.rows - 1) * self.pitch_y
        self._fill()

    def _fill(self) -> None:
        gx, gy = np.meshgrid(self.x0 + self.pitch_x * np.arange(self.cols), self.y0 + self.pitch_y * np.arange(self.rows))
        self.rest_mm = np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32)

    @property
    def count(self) -> int:
        return self.rows * self.cols

    @property
    def pitch(self) -> float:
        """The smaller of the two pitches [mm] (markers are kept within a fraction of it of their rest)."""
        return min(self.pitch_x, self.pitch_y)

    @staticmethod
    def from_points(points_mm: np.ndarray, tol: float = 1e-3) -> tuple[MarkerGrid, np.ndarray]:
        """The regular grid (own x and y pitch) through ``points_mm`` (M, 2) in any order, and ``order`` with
        ``grid.rest_mm[j] == points_mm[order[j]]``. Example: ``g, order = MarkerGrid.from_points(engine_rest_mm)``."""
        pts = np.asarray(points_mm, np.float64).reshape(-1, 2)
        xs, ys = np.unique(np.round(pts[:, 0], 4)), np.unique(np.round(pts[:, 1], 4))
        g = MarkerGrid.__new__(MarkerGrid)
        g.cols, g.rows, g.x0, g.y0 = len(xs), len(ys), float(xs[0]), float(ys[0])
        g.pitch_x = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0
        g.pitch_y = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.0
        g.pitch_x = g.pitch_x or g.pitch_y or 1.0
        g.pitch_y = g.pitch_y or g.pitch_x
        g._fill()
        col = np.rint((pts[:, 0] - g.x0) / g.pitch_x).astype(int)
        row = np.rint((pts[:, 1] - g.y0) / g.pitch_y).astype(int)
        slot = row * g.cols + col
        ok = len(pts) == g.count and np.unique(slot).size == len(pts) and slot.min() >= 0 and slot.max() < g.count
        if not ok or np.abs(g.rest_mm[np.clip(slot, 0, g.count - 1)] - pts).max() > tol * 10 * g.pitch:
            raise CameraError(f"{len(pts)} marker rest positions do not form a regular grid",
                              "give the markers on a rectangular grid (one x pitch, one y pitch), or no markers")
        order = np.empty(len(pts), np.int32)
        order[slot] = np.arange(len(pts), dtype=np.int32)
        return g, order


@wp.func
def px_to_mm(r: float, c: float, h: int, w: int, mm_per_px: float) -> wp.vec2:
    return wp.vec2((c + 0.5 - 0.5 * float(w)) * mm_per_px, (r + 0.5 - 0.5 * float(h)) * mm_per_px)


@wp.func
def dilation(press: wp.array3d(dtype=float), i: int, m: wp.vec2, mm_per_px: float, sigma: float, k: float) -> wp.vec2:
    """FOTS-style outward push at m [mm]: press-weighted pull away from pressed pixels within 3 sigma."""
    h = press.shape[1]
    w = press.shape[2]
    step = wp.max(1, int(0.25 * sigma / mm_per_px))
    rad = int(3.0 * sigma / mm_per_px)
    cc = int(m[0] / mm_per_px + 0.5 * float(w))
    rc = int(m[1] / mm_per_px + 0.5 * float(h))
    acc = wp.vec2(0.0)
    inv2s2 = 0.5 / (sigma * sigma)
    for dr in range(-rad, rad + 1, step):
        r = rc + dr
        if r >= 0 and r < h:
            for dc in range(-rad, rad + 1, step):
                c = cc + dc
                if c >= 0 and c < w:
                    p = press[i, r, c]
                    if p > 0.0:
                        d = m - px_to_mm(float(r), float(c), h, w, mm_per_px)
                        acc += p * d * wp.exp(-wp.dot(d, d) * inv2s2)
    area = float(step * step) * mm_per_px * mm_per_px
    return acc * (k * area / (2.0 * wp.pi * sigma * sigma * sigma))


@wp.func
def drag(m: wp.vec2, mo: wp.vec4, c0: wp.vec2, s_shear: float, s_twist: float) -> wp.vec2:
    """FOTS shear + twist of the patch (dx, dy, twist, active) around c0, Gaussian falloff."""
    if mo[3] <= 0.0:
        return wp.vec2(0.0)
    d = m - c0
    r2 = wp.dot(d, d)
    gs = wp.exp(-0.5 * r2 / (s_shear * s_shear))
    gt = wp.exp(-0.5 * r2 / (s_twist * s_twist))
    ct = wp.cos(mo[2])
    st = wp.sin(mo[2])
    rot = wp.vec2(ct * d[0] - st * d[1], st * d[0] + ct * d[1])
    return wp.vec2(mo[0], mo[1]) * gs + (rot - d) * gt


@wp.func
def sample_field(field: wp.array3d(dtype=wp.vec2), i: int, m: wp.vec2, mm_per_px: float) -> wp.vec2:
    """Engine tangential field at m [mm] (nearest pixel)."""
    h = field.shape[1]
    w = field.shape[2]
    c = wp.clamp(int(m[0] / mm_per_px + 0.5 * float(w)), 0, w - 1)
    r = wp.clamp(int(m[1] / mm_per_px + 0.5 * float(h)), 0, h - 1)
    return field[i, r, c]


@wp.kernel
def move_markers(rest: wp.array(dtype=wp.vec2), press: wp.array3d(dtype=float), motion: wp.array(dtype=wp.vec4),
                 center: wp.array(dtype=wp.vec2), field: wp.array3d(dtype=wp.vec2), use_field: int,
                 mm_per_px: float, mp: wp.vec4, max_disp: float, out: wp.array2d(dtype=wp.vec2)):
    """Displaced centre [mm] of marker j in image i; mp = (k_dil, sigma_dil, sigma_shear, sigma_twist)."""
    i, j = wp.tid()
    m = rest[j]
    u = dilation(press, i, m, mm_per_px, mp[1], mp[0])
    if use_field != 0:
        u += sample_field(field, i, m, mm_per_px)
    else:
        u += drag(m, motion[i], center[i], mp[2], mp[3])
    n = wp.length(u)
    if n > max_disp:
        u = u * (max_disp / n)
    out[i, j] = m + u


@wp.kernel
def place_markers(rest: wp.array(dtype=wp.vec2), disp: wp.array2d(dtype=wp.vec2), max_disp: float,
                  out: wp.array2d(dtype=wp.vec2)):
    """Marker centres = rest + given displacement [mm] (e.g. the engine's marker field), capped like move_markers."""
    i, j = wp.tid()
    u = disp[i, j]
    n = wp.length(u)
    if n > max_disp:
        u = u * (max_disp / n)
    out[i, j] = rest[j] + u
