# SPDX-License-Identifier: Apache-2.0
"""Watertight meshes for common containers and fixtures, as (vertices (V, 3), faces (F, 3)) arrays for
``Scene.add_mesh(vertices=..., faces=...)``. Faces wind counter-clockwise seen from outside; z is up and the
bottom of every shape is at z = 0, centred on x = y = 0.

Example::

    from alloy.api.meshes import open_box, open_cylinder, block_with_hole
    v, f = open_cylinder(inner_radius=0.04, depth=0.10, wall=0.004)       # a cup
    scene.add_mesh("cup", vertices=v, faces=f, collision="sdf", static=True)
"""

from __future__ import annotations

import math

import numpy as np


def box(size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """A solid box of full size (x, y, z). Example: ``v, f = box((0.1, 0.1, 0.05))``."""
    x, y, z = 0.5 * size[0], 0.5 * size[1], size[2]
    ring = _square(x, y)
    v = np.vstack([_at(ring, 0.0), _at(ring, z)])
    f = [*_cap(range(0, 4), down=True), *_cap(range(4, 8), down=False), *_wall(range(0, 4), range(4, 8), out=True)]
    return v, np.asarray(f, np.int64)


def block_with_hole(size: tuple[float, float, float], hole: tuple[float, float],
                    depth: float) -> tuple[np.ndarray, np.ndarray]:
    """A block of full ``size`` with a rectangular blind hole (x, y) of ``depth`` from its top, centred.

    Example: ``v, f = block_with_hole((0.1, 0.1, 0.05), hole=(0.0102, 0.0102), depth=0.03)``."""
    if not (0 < hole[0] < size[0] and 0 < hole[1] < size[1] and 0 < depth < size[2]):
        raise ValueError(f"hole {hole} x {depth} m must fit inside the block {size}")
    return _hollow(_square(0.5 * size[0], 0.5 * size[1]), _square(0.5 * hole[0], 0.5 * hole[1]), size[2],
                   size[2] - depth)


def open_box(inner: tuple[float, float, float], wall: float) -> tuple[np.ndarray, np.ndarray]:
    """An open-top box (bin, square cup): ``inner`` (x, y, depth) cavity, walls and floor ``wall`` thick.

    The cavity floor is at z = ``wall``. Example: ``v, f = open_box((0.08, 0.08, 0.10), wall=0.004)``."""
    ix, iy, depth = inner
    outer = _square(0.5 * ix + wall, 0.5 * iy + wall)
    return _hollow(outer, _square(0.5 * ix, 0.5 * iy), depth + wall, wall)


def open_cylinder(inner_radius: float, depth: float, wall: float, segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """A round cup: cavity radius ``inner_radius`` and ``depth``, walls and floor ``wall`` thick (floor top at
    z = ``wall``). The polygon has ``segments`` sides. Example: ``open_cylinder(0.04, 0.10, 0.004)``."""
    return _hollow(_circle(inner_radius + wall, segments), _circle(inner_radius, segments), depth + wall, wall)


# --- internals: rings are (n, 2) CCW seen from above ------------------------------------------------------------


def _square(hx: float, hy: float) -> np.ndarray:
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], float)


def _circle(r: float, n: int) -> np.ndarray:
    a = np.arange(n) * 2.0 * math.pi / n
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def _at(ring: np.ndarray, z: float) -> np.ndarray:
    return np.hstack([ring, np.full((len(ring), 1), z)])


def _hollow(outer: np.ndarray, inner: np.ndarray, height: float, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Solid between ``outer`` (z 0..height) and a cavity ``inner`` (z floor..height), open at the top."""
    n = len(outer)
    ob, ot, it, ib = (range(k * n, (k + 1) * n) for k in range(4))
    v = np.vstack([_at(outer, 0.0), _at(outer, height), _at(inner, height), _at(inner, floor)])
    faces = [*_cap(ob, down=True), *_wall(ob, ot, out=True), *_ring(ot, it), *_wall(ib, it, out=False),
             *_cap(ib, down=False)]
    return v, np.asarray(faces, np.int64)


def _cap(ring: range, down: bool) -> list[tuple[int, int, int]]:
    idx = list(ring)
    tris = [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return [(a, c, b) for a, b, c in tris] if down else tris


def _wall(lo: range, hi: range, out: bool) -> list[tuple[int, int, int]]:
    lo_l, hi_l, n, faces = list(lo), list(hi), len(lo), []
    for k in range(n):
        a, b, c, d = lo_l[k], lo_l[(k + 1) % n], hi_l[(k + 1) % n], hi_l[k]
        faces += [(a, b, c), (a, c, d)] if out else [(b, a, d), (b, d, c)]
    return faces


def _ring(outer: range, inner: range) -> list[tuple[int, int, int]]:
    o, i, n, faces = list(outer), list(inner), len(outer), []
    for k in range(n):
        a, b, c, d = o[k], o[(k + 1) % n], i[(k + 1) % n], i[k]
        faces += [(a, b, c), (a, c, d)]
    return faces
