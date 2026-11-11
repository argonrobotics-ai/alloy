# SPDX-License-Identifier: Apache-2.0
"""The inside of a container, from its mesh: floor, rim, capacity, the liquid level for a volume, and the largest
column of liquid or grains that fits.

Vertical lines on a grid over the container cross its surface; the solid lies between crossing pairs. A cell's
cavity runs from the top of its first solid interval (the inner floor) to the next crossing above (a shoulder or a
lid), or is open to the sky. The cavity is the cells connected to the centre whose floor lies well below the rim
(the walls stand at rim height around it), so a handle or a neighbouring part is never counted.

Example::

    >>> import numpy as np
    >>> from alloy.compose import geom
    >>> outer_v, outer_f = geom.primitive_mesh("cylinder", (0.04, 0.05))       # a solid can: no cavity
    >>> measure(outer_v, outer_f) is None
    True
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from alloy.compose import geom

CELLS = 72                         # grid cells per axis over the container's outline
MIN_DEPTH = 0.25                   # a cavity floor at least this share of the height below the rim
DEDUPE = 1e-6                      # m: crossings closer than this are one (a line through an edge)
LEVEL_ITERS = 60


@dataclass(frozen=True)
class Interior:
    """A container's cavity in its rest frame (body origin at 0, yaw 0). ``bottom``/``top`` per cavity cell (top
    +inf where open), ``cell`` the cell area (m^2), ``centre`` the opening's centre (x, y), ``floor`` the lowest
    cavity floor, ``rim`` the container's top.

    Example: ``measure(v, f).capacity_ml()``."""

    xy: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    cell: float
    centre: np.ndarray
    floor: float
    rim: float
    border: float = math.inf           # distance from ``centre`` to the nearest grid cell outside the cavity
    outside: np.ndarray | None = None  # (K, 2) grid cells outside the cavity

    @property
    def depth(self) -> float:
        return self.rim - self.floor

    def volume_below(self, level: float) -> float:
        """Cavity volume (m^3) below the rest-frame height ``level``."""
        return float(np.clip(np.minimum(level, self.top) - self.bottom, 0.0, None).sum() * self.cell)

    def capacity_ml(self, share: float = 1.0) -> float:
        """Volume (ml) up to ``share`` of the depth. Example: ``interior.capacity_ml(0.9)``."""
        return self.volume_below(self.floor + share * self.depth) * 1e6

    def level_for(self, volume_m3: float) -> float:
        """The rest-frame height a settled liquid of ``volume_m3`` reaches (bisection on ``volume_below``)."""
        lo, hi = self.floor, self.rim
        for _ in range(LEVEL_ITERS):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.volume_below(mid) < volume_m3 else (lo, mid)
        return hi

    def radius_at(self, z0: float, z1: float) -> float:
        """The largest disc around ``centre`` whose cells all hold the band [z0, z1] (floor <= z0, ceiling >= z1)."""
        bad = (self.bottom > z0) | (self.top < z1)
        d = np.linalg.norm(self.xy - self.centre[None, :], axis=1)
        half = math.sqrt(self.cell) / 2
        reach = float(np.max(d)) + half
        return float(min(d[bad].min() if bad.any() else reach, self.border)) - half

    def rect_at(self, z0: float, z1: float) -> tuple[float, float]:
        """Half sides (x, y) of the largest axis-aligned rectangle around ``centre`` whose cells all hold the band
        [z0, z1] (a box-shaped bin or tub holds more as a box than as a cylinder)."""
        bad = self.xy[(self.bottom > z0) | (self.top < z1)]
        if self.outside is not None and len(self.outside):
            bad = np.concatenate([bad, self.outside])
        if not len(bad):
            return 0.0, 0.0
        d = np.abs(bad - self.centre[None, :])
        half = math.sqrt(self.cell) / 2
        best = (0.0, 0.0, 0.0)
        for a in np.linspace(0.25, 4.0, 61):                          # aspect hy / hx
            hx = float(np.min(np.maximum(d[:, 0], d[:, 1] / a))) - half
            if hx > 0 and 4 * a * hx * hx > best[0]:
                best = (4 * a * hx * hx, hx, a * hx)
        return best[1], best[2]


def measure(vertices: np.ndarray, faces: np.ndarray, cells: int = CELLS) -> Interior | None:
    """The cavity of a rest-frame container mesh, or None when it has none (a solid, a plate, a flat tray).

    Example: see the module docstring."""
    v = np.asarray(vertices, np.float64)
    tris = v[np.asarray(faces, np.int64)]
    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    step = (hi - lo) / cells
    lo, n = lo - step, cells + 2                       # one empty cell around the outline
    gx, gy = np.meshgrid(lo[0] + (np.arange(n) + 0.5) * step[0], lo[1] + (np.arange(n) + 0.5) * step[1],
                         indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    floor_z, ceil_z = _cavity_columns(tris, gx[:, 0], gy[0, :])
    rim = float(v[:, 2].max())
    zmin = float(v[:, 2].min())
    low = np.isfinite(floor_z) & (floor_z < rim - MIN_DEPTH * (rim - zmin))
    grid = low.reshape(n, n)
    mask = _connected(grid, _start(grid, xy.reshape(n, n, 2), lo + step * n / 2))
    if mask is None or mask.sum() < 4:
        return None
    m = mask.ravel()
    centre = xy[m].mean(axis=0)
    outside = xy[~m]
    border = float(np.linalg.norm(outside - centre[None, :], axis=1).min()) if len(outside) else math.inf
    return Interior(xy[m], floor_z[m], ceil_z[m], float(step[0] * step[1]), centre, float(floor_z[m].min()), rim,
                    border, outside)


def _cavity_columns(tris: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per vertical line of the grid ``xs`` x ``ys`` (row-major): the top of the first solid interval (cavity floor)
    and the next crossing above (ceiling, +inf when open). Lines that miss the mesh get floor nan. Each grid row
    only tests the triangles whose x range covers it."""
    floor = np.full(len(xs) * len(ys), np.nan)
    ceil = np.full(len(xs) * len(ys), np.inf)
    tx0, tx1 = tris[:, :, 0].min(axis=1), tris[:, :, 0].max(axis=1)
    for r, x in enumerate(xs):
        row = tris[(tx0 <= x) & (tx1 >= x)]
        if not len(row):
            continue
        mask, z = geom._hits(row, np.stack([np.full(len(ys), x), ys], axis=1))   # noqa: SLF001 - the layout's kernel
        for k in range(len(ys)):
            zs = _dedupe(np.sort(z[k][mask[k]]))
            if len(zs) >= 2:
                floor[r * len(ys) + k] = zs[1]
                if len(zs) >= 3:
                    ceil[r * len(ys) + k] = zs[2]
    return floor, ceil


def _dedupe(zs: np.ndarray) -> np.ndarray:
    if len(zs) < 2:
        return zs
    keep = np.concatenate([[True], np.diff(zs) > DEDUPE])
    return zs[keep]


def _start(low: np.ndarray, xy: np.ndarray, middle: np.ndarray) -> tuple[int, int] | None:
    """The low cell nearest the outline's middle (where a cup's opening is)."""
    idx = np.argwhere(low)
    if not len(idx):
        return None
    d = np.linalg.norm(xy[idx[:, 0], idx[:, 1]] - middle[None, :], axis=1)
    i, j = idx[int(np.argmin(d))]
    return int(i), int(j)


def _connected(low: np.ndarray, start: tuple[int, int] | None) -> np.ndarray | None:
    if start is None:
        return None
    seen = np.zeros_like(low, dtype=bool)
    seen[start] = True
    todo = deque([start])
    n, m = low.shape
    while todo:
        i, j = todo.popleft()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < n and 0 <= b < m and low[a, b] and not seen[a, b]:
                seen[a, b] = True
                todo.append((a, b))
    return seen


__all__ = ["Interior", "measure"]
