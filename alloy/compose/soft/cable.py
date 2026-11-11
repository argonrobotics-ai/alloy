# SPDX-License-Identifier: Apache-2.0
"""Where a cable lies: a gentle random curve (or a straight line) on the table, clear of everything else, or run
from an object it is plugged into (and on to a second one).

The path is a polyline of the cable's length: heading changes by a bounded random walk (bend radius at least
``MIN_BEND``), each segment kept ``EDGE`` inside the table and clear of the other objects' outlines. A plugged end
starts at the object's side, ``PORT_HEIGHT`` up (at most half its height), points away from it and comes down to
the table within ``DROP``. Between two objects the path is a sideways bulge that uses up the extra length.

The cable's chain takes the laid shape as its rest shape: a loose cable keeps the curve it was laid in (as real
cables keep a set); the result lists this as an estimate.

Example::

    >>> pts = loose_path(layout, length=0.8, radius=0.00175)          # doctest: +SKIP
    >>> pts.shape                                                     # doctest: +SKIP
    (33, 3)
"""

from __future__ import annotations

import math

import numpy as np

from alloy.compose import geom
from alloy.compose.layout import CLEARANCE, EDGE, Layout, Placed

POINTS = 33                  # polyline vertices along a cable
MIN_BEND = 0.12              # m, smallest bend radius of a laid cable
KAPPA_STEP = 1.5             # 1/m: random-walk step of the curvature per polyline segment
PORT_HEIGHT = 0.02           # m above the table where a plugged end leaves its object
DROP = 0.06                  # m along the cable from the port to the table
LIFT = 5e-4                  # m between a lying cable and the table
TRIES = 120


def loose_path(layout: Layout, length: float, radius: float, *, straight: bool = False,
               start: np.ndarray | None = None, heading: float | None = None, skip: tuple[str, ...] = ()
               ) -> np.ndarray | None:
    """A lying cable (POINTS x 3, world) of ``length``; ``start``/``heading``: a fixed first point and direction
    (a plugged end); ``skip``: layout keys it may touch (the object it is plugged into). None when no try fits."""
    top = _table_top(layout)
    lo, hi = layout.area
    for _ in range(TRIES):
        p0 = np.asarray(start[:2], np.float64) if start is not None else layout.rng.uniform(lo, hi)
        h0 = heading if heading is not None else float(layout.rng.uniform(0.0, 2 * math.pi))
        xy = _walk(layout.rng, p0, h0, length, straight)
        z = np.full(len(xy), top + radius + LIFT)
        if start is not None:
            z = _drop(xy, float(start[2]), z[0])
        if _fits(layout, xy, radius, skip):
            return np.column_stack([xy, z])
    return None


def between(layout: Layout, a: np.ndarray, b: np.ndarray, length: float, radius: float, skip: tuple[str, ...]
            ) -> np.ndarray | None:
    """A cable from port ``a`` to port ``b`` (world, z = port heights) of ``length`` >= their distance: a sideways
    sine bulge on the table between the two drops."""
    d = np.asarray(b[:2]) - np.asarray(a[:2])
    dist = float(np.linalg.norm(d))
    if length < dist + 1e-3:
        return None
    u = d / dist
    n = np.array([-u[1], u[0]])
    s = np.linspace(0.0, 1.0, POINTS)
    top = _table_top(layout)
    for sign in (1.0, -1.0):
        amp = _amplitude(dist, length)
        xy = a[None, :2] + s[:, None] * d[None, :] + sign * amp * np.sin(math.pi * s)[:, None] * n[None, :]
        lie = top + radius + LIFT
        z = np.maximum(_drop(xy, float(a[2]), lie), _drop(xy[::-1], float(b[2]), lie)[::-1])
        if _fits(layout, xy, radius, skip):
            return np.column_stack([xy, z])
    return None


def port(target: Placed, toward: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    """(world point, outward heading) where a cable plugged into ``target`` leaves it: on its outline in the
    direction of ``toward`` (xy), ``PORT_HEIGHT`` up (at most half its height), one radius plus 1 mm out."""
    out = target.outline
    c = out.mean(axis=0)
    d = np.asarray(toward, np.float64) - c
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    reach = _ray_exit(out, c, d)
    tri = target.triangles()
    zmin, zmax = float(tri[..., 2].min()), float(tri[..., 2].max())
    z = zmin + min(PORT_HEIGHT, 0.5 * (zmax - zmin))
    p = c + d * (reach + radius + 1e-3)
    return np.array([p[0], p[1], z]), math.atan2(d[1], d[0])


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _walk(rng: np.random.Generator, p0: np.ndarray, h0: float, length: float, straight: bool) -> np.ndarray:
    ds = length / (POINTS - 1)
    xy = [np.asarray(p0, np.float64)]
    h, k = h0, 0.0
    for _ in range(POINTS - 1):
        if not straight:
            k = float(np.clip(k + rng.normal(0.0, KAPPA_STEP), -1 / MIN_BEND, 1 / MIN_BEND))
            h += k * ds
        xy.append(xy[-1] + ds * np.array([math.cos(h), math.sin(h)]))
    return np.array(xy)


def _drop(xy: np.ndarray, z_port: float, z_lie: float) -> np.ndarray:
    """Heights coming down from a port over the first ``DROP`` of arc length, then lying."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    w = np.clip(s / DROP, 0.0, 1.0)
    return z_port + (z_lie - z_port) * (3 * w ** 2 - 2 * w ** 3)


def _amplitude(dist: float, length: float) -> float:
    """Sine-bulge amplitude whose arc over ``dist`` has ``length`` (bisection)."""
    lo, hi = 0.0, length
    s = np.linspace(0.0, 1.0, 400)
    for _ in range(50):
        a = 0.5 * (lo + hi)
        y = a * np.sin(math.pi * s)
        arc = float(np.sum(np.hypot(np.diff(s) * dist, np.diff(y))))
        lo, hi = (a, hi) if arc < length else (lo, a)
    return 0.5 * (lo + hi)


def _fits(layout: Layout, xy: np.ndarray, radius: float, skip: tuple[str, ...]) -> bool:
    """Every segment inside the table and clear of the other objects and the robots' bases."""
    table = layout.support.outline
    if geom.depths(table, xy).min() < EDGE:
        return False
    gap = CLEARANCE[layout.clutter]
    others = [p.outline for k, p in layout.placed.items() if k not in skip] + list(layout.obstacles)
    for a, b in zip(xy[:-1], xy[1:]):
        seg = _segment_polygon(a, b, radius)
        if any(not geom.separated(seg, o, gap) for o in others):
            return False
    return True


def _segment_polygon(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    d = b - a
    n = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-12) * r
    return np.array([a - n, b - n, b + n, a + n])


def _ray_exit(poly: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Distance from ``c`` along ``d`` to the edge of convex polygon ``poly``."""
    lo, hi = 0.0, float(np.ptp(poly, axis=0).max()) + 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if geom.inside_depth(poly, c + d * mid) > 0 else (lo, mid)
    return hi


def _table_top(layout: Layout) -> float:
    return float(layout.support.triangles()[..., 2].max())


__all__ = ["between", "loose_path", "path_length", "port"]
