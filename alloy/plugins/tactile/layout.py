# SPDX-License-Identifier: Apache-2.0
"""Taxel layouts: a planar grid (optionally foveated), a list of points, or a preset (``presets.py``).

Grid taxels are row-major, j = iy·nx + ix, centred on the pad origin. Foveation γ > 0 maps the uniform coordinate
u in (-1, 1) to sign(u)·|u|^(1 + γ), packing taxels toward the centre. It is the planar analogue of touch_grid's
``gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.plugins.tactile.noise import grid_neighbours
from alloy.plugins.tactile.presets import preset


@dataclass(frozen=True)
class Layout:
    """Resolved taxel centres ``xy`` [T, 2] (m), ``reach`` (m), grid ``neighbours`` (or None) and ``saturation``."""

    xy: np.ndarray
    reach: float
    neighbours: np.ndarray | None
    saturation: float
    grid: tuple[int, int] | None


def _axis(n: int, pitch: float, gamma: float) -> np.ndarray:
    u = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    return np.sign(u) * np.abs(u) ** (1.0 + gamma) * (n * pitch / 2.0)


def _max_gap(a: np.ndarray, pitch: float) -> float:
    return float(np.max(np.diff(a))) if len(a) > 1 else float(pitch)


def grid_layout(grid: tuple[int, int], pitch: tuple[float, float], foveation: float = 0.0) -> tuple[np.ndarray, float]:
    """Centres [nx·ny, 2] and the default reach: half the diagonal of the largest cell."""
    nx, ny = int(grid[0]), int(grid[1])
    if nx < 1 or ny < 1 or min(pitch) <= 0.0 or foveation < 0.0:
        raise ValueError(f"grid {grid} needs nx, ny >= 1, pitch > 0 and foveation >= 0 (got {pitch}, {foveation})")
    xs, ys = _axis(nx, pitch[0], foveation), _axis(ny, pitch[1], foveation)
    xy = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    return xy, 0.5 * float(np.hypot(_max_gap(xs, pitch[0]), _max_gap(ys, pitch[1])))


def points_layout(points: Any) -> tuple[np.ndarray, float]:
    """Centres from a point list, and the default reach: √2/2 times the largest nearest-neighbour spacing."""
    xy = np.asarray(points, np.float64).reshape(-1, 2)
    d =np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy)) * 1e30
    return xy, float(np.sqrt(0.5) * d.min(axis=1).max())


def build_layout(c: Any) -> Layout:
    """The layout of a ``Tactile.Config``: its preset if named, else its points, else its grid."""
    grid, pitch, points, reach, sat = c.grid, c.pitch, c.points, c.reach, c.saturation
    if c.preset:
        p = preset(c.preset)
        grid, pitch, points = p.grid, p.pitch, p.points
        reach = reach or p.reach
        sat = sat or p.saturation
    if points and len(points) == 1:
        if not reach:
            raise ValueError("a single taxel point needs an explicit reach")
        xy, default_reach, neighbours, g = np.asarray(points, np.float64).reshape(-1, 2), reach, None, None
    elif points:
        xy, default_reach = points_layout(points)
        neighbours, g = None, None
    else:
        xy, default_reach = grid_layout(grid, pitch, c.foveation)
        neighbours, g = grid_neighbours(int(grid[0]), int(grid[1])), (int(grid[0]), int(grid[1]))
    return Layout(np.asarray(xy, np.float64), float(reach or default_reach), neighbours, float(sat), g)
