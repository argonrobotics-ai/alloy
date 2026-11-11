# SPDX-License-Identifier: Apache-2.0
"""Shapes and poses of soft items that take room on the table: cloth sheets (flat, draped over an object, hanging
over the table edge) and grain piles.

- A flat sheet and a pile go through the ordinary layout (outline, clearance, support), as a thin box and as the
  cone of the pile's angle of repose.
- A draped sheet starts flat a few millimetres above the object it covers, centred on it, and falls over it in the
  first ~0.5 s (the scene's ``settle_s``). Its whole flat outline is kept free of other objects.
- A hanging sheet starts flat on the table with 40 % of its length past an edge that has no robot, and that part
  falls and hangs.

Example::

    >>> R, H = pile_size(100e-6, math.radians(33))
    >>> round(R, 3), round(H, 3)
    (0.053, 0.034)
"""

from __future__ import annotations

import math

import numpy as np

from alloy.compose import geom
from alloy.compose.body import Body
from alloy.compose.layout import EDGE, Layout, Placed

DRAPE_GAP = 3e-3             # m above the highest point under a draped sheet
DRAPE_OFFSET = 0.15          # share of the covered object's size the drape centre may move off its centre
OVERHANG = 0.4               # share of a hanging sheet's length past the table edge
TRIES = 40
PILE_START = 0.8             # a pile starts as a box of the cone's volume, footprint = this x the cone's base
SETTLE_DRAPE_S = 0.6         # s for a draped or hanging sheet to come to rest (estimate, checked by the run)
SETTLE_PILE_S = 0.5          # s for a pile to slump to its angle of repose


def sheet_body(name: str, length: float, width: float, thickness: float) -> Body:
    """A flat sheet as a thin box (origin at its centre). Example: ``sheet_body("towel", 0.3, 0.2, 6e-4)``."""
    v, f = geom.primitive_mesh("box", (0.5 * length, 0.5 * width, 0.5 * thickness))
    return Body.from_mesh(name, v, f, np.zeros(3))


FOLD_GAP = 2.0              # the folded layers' mid-surfaces start this many sheet thicknesses apart


def folded_mesh(length: float, width: float, thickness: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """A sheet ``length`` x ``width`` folded in half across its length: a bottom layer, a vertical strip at the fold
    (the gap) and a top layer back over it, ``FOLD_GAP`` thicknesses apart, centred on the origin. One connected
    triangle mesh (vertices, faces); the folded shape is its rest shape.

    Example::

        >>> v, f = folded_mesh(0.5, 0.3, 6e-4, 0.02)
        >>> [round(float(x), 4) for x in v.max(axis=0) - v.min(axis=0)]
        [0.2494, 0.3, 0.0012]
    """
    gap = FOLD_GAP * thickness
    half = 0.5 * (length - gap)
    nu, nv = max(2, round(half / spacing)), max(2, round(width / spacing))
    u = np.linspace(0.0, half, nu + 1)
    xs = np.concatenate([u - 0.5 * half, (0.5 * half - u)])                     # out along the bottom, back on top
    zs = np.concatenate([np.full(nu + 1, -0.5 * gap), np.full(nu + 1, 0.5 * gap)])
    ys = np.linspace(-0.5 * width, 0.5 * width, nv + 1)
    verts = np.array([[x, y, z] for x, z in zip(xs, zs) for y in ys])
    cols, rows = 2 * (nu + 1), nv + 1
    faces = []
    for c in range(cols - 1):                    # column nu -> nu + 1 is the strip across the fold
        for r in range(rows - 1):
            a, b, d, e = c * rows + r, (c + 1) * rows + r, c * rows + r + 1, (c + 1) * rows + r + 1
            faces += [[a, b, e], [a, e, d]]
    return verts, np.asarray(faces, np.int64)


def pile_size(volume: float, angle: float) -> tuple[float, float]:
    """(base radius, height) of a cone of ``volume`` (m^3) with slopes at ``angle`` (rad, the angle of repose).

    Example: see the module docstring."""
    t = math.tan(angle)
    r = (3.0 * volume / (math.pi * t)) ** (1.0 / 3.0)
    return r, r * t


def pile_body(name: str, radius: float, height: float) -> Body:
    """The pile's cone, base at z = 0 (its footprint and height keep other objects clear)."""
    v, f = geom.primitive_mesh("cone", (radius, 0.5 * height))
    v = v - np.array([0.0, 0.0, v[:, 2].min()])
    return Body.from_mesh(name, v, f, np.array([0.0, 0.0, 0.25 * height]))


def pile_start(volume: float, radius: float) -> tuple[float, float]:
    """(half side, height) of the box of grains a pile starts as: the cone's volume over ``PILE_START`` of its
    base area, a square, so the grains slump outwards to the cone."""
    side = math.sqrt(math.pi) * PILE_START * radius
    return 0.5 * side, volume / (side * side)


def drape(layout: Layout, body: Body, card: str, key: str, target: str, item: int, thickness: float
          ) -> Placed | None:
    """A sheet centred over ``target`` (a layout key), ``DRAPE_GAP`` above the highest point under it."""
    t = layout.placed[target]
    lo, hi = t.outline.min(axis=0), t.outline.max(axis=0)
    for _ in range(TRIES):
        c = (lo + hi) / 2 + layout.rng.uniform(-DRAPE_OFFSET, DRAPE_OFFSET, 2) * (hi - lo)
        yaw = float(layout.rng.uniform(0.0, 2 * math.pi))
        p = Placed(key, card, body, yaw, np.array([c[0], c[1], 0.0]), target, item)
        z = _over(layout, p, [layout.support, t])
        if z is None or not _inside_table(layout, p.outline) or not layout._clear(p.outline, target):  # noqa: SLF001
            continue
        p.pos = np.array([c[0], c[1], z + DRAPE_GAP + 0.5 * thickness])
        layout.placed[key] = p
        return p
    return None


def hang(layout: Layout, body: Body, card: str, key: str, item: int, thickness: float) -> Placed | None:
    """A sheet lying on the table with ``OVERHANG`` of its length past a free table edge."""
    tlo, thi = layout.support.outline.min(axis=0), layout.support.outline.max(axis=0)
    length, width = body.size[0], body.size[1]
    edges = _free_edges(layout, tlo, thi)
    for k in range(TRIES):
        axis, sign = edges[k % len(edges)]
        other = 1 - axis
        span = (tlo[other] + EDGE + 0.5 * width, thi[other] - EDGE - 0.5 * width)
        if span[0] > span[1]:
            continue
        c = np.zeros(2)
        c[axis] = (thi[axis] if sign > 0 else tlo[axis]) - sign * (0.5 - OVERHANG) * length
        c[other] = layout.rng.uniform(*span)
        yaw = 0.0 if axis == 0 else math.pi / 2
        p = Placed(key, card, body, yaw, np.array([c[0], c[1], 0.0]), None, item)
        if not layout._clear(p.outline, None):                                                     # noqa: SLF001
            continue
        p.pos = np.array([c[0], c[1], float(layout.support.triangles()[..., 2].max()) + 5e-4 + 0.5 * thickness])
        layout.placed[key] = p
        return p
    return None


def _free_edges(layout: Layout, tlo: np.ndarray, thi: np.ndarray) -> list[tuple[int, int]]:
    """Table edges (axis, side) with no robot near them, far side (+x) first."""
    out = []
    for axis, sign in ((0, 1), (1, 1), (1, -1), (0, -1)):
        edge = thi[axis] if sign > 0 else tlo[axis]
        near = any(abs(float(np.mean(ob[:, axis])) - edge) < 0.25 for ob in layout.obstacles)
        if not near:
            out.append((axis, sign))
    return out or [(0, 1)]


def _over(layout: Layout, p: Placed, under: list[Placed]) -> float | None:
    """Highest surface z under the sheet's outline (the objects in ``under``); None if part of it is over nothing
    at all (off the table)."""
    tris = np.concatenate([u.triangles() for u in under])
    out = p.outline
    lo, hi = out.min(axis=0), out.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 15), np.linspace(lo[1], hi[1], 15), indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    xy = xy[geom.depths(out, xy) >= -1e-9]
    z = geom.ray_down(geom.cull_xy(tris, lo, hi), xy)
    return float(z.max()) if np.isfinite(z).any() else None


def _inside_table(layout: Layout, outline: np.ndarray) -> bool:
    return bool(geom.depths(layout.support.outline, outline).min() >= EDGE)


__all__ = ["DRAPE_GAP", "FOLD_GAP", "SETTLE_DRAPE_S", "SETTLE_PILE_S", "drape", "folded_mesh", "hang", "pile_body",
           "pile_size", "pile_start", "sheet_body"]
