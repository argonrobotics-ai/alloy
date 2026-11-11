# SPDX-License-Identifier: Apache-2.0
"""One object's geometry for placement: its mesh turned to a stable resting orientation, its footprint and the
height map of its underside.

The resting orientation is the authored one when that stands well (tip angle at least ``AUTHORED_TIP_DEG``, or
nearly the best the shape allows), else the convex-hull face that stands best: a pencil authored upright is laid on
its side, a mug stays upright. The tip angle is atan(distance from the centre of mass to the edge of the contact
patch / height of the centre of mass): the tilt it survives before it falls over.

Example::

    >>> import numpy as np
    >>> from alloy.compose import geom
    >>> v, f = geom.primitive_mesh("cylinder", (0.0035, 0.085))        # a pencil, authored upright
    >>> b = Body.from_mesh("pencil", v, f, np.zeros(3))
    >>> round(b.size[2], 4), b.reoriented                             # lies on its side: 7 mm tall
    (0.007, True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.compose import geom

AUTHORED_TIP_DEG = 12.0          # an authored pose that survives this tilt is kept
AUTHORED_SHARE = 0.8             # ... or one within this share of the best face's tip angle
ROUND_TIP_DEG = 10.0             # no face stands better than this: a round object, kept as authored
CONTACT_TOL = 5e-4               # m, vertices this close to the support plane touch it
MAP_CELLS = 48                   # bottom height map resolution per axis
MAX_FACES = 60                   # hull face groups examined for the resting pose


@dataclass(frozen=True)
class HeightMap:
    """Lowest z of an object on a regular xy grid of its rest frame (+inf where the object is absent)."""

    lo: np.ndarray                     # (2,) grid origin, m
    step: np.ndarray                   # (2,) cell size, m
    z: np.ndarray                      # (nx, ny)

    def at(self, xy: np.ndarray) -> np.ndarray:
        """Nearest-cell lookup at rest-frame points (K, 2); +inf outside the grid."""
        ij = np.floor((np.asarray(xy) - self.lo[None, :]) / self.step[None, :] + 0.5).astype(np.int64)
        nx, ny = self.z.shape
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        out = np.full(len(ij), math.inf)
        out[ok] = self.z[ij[ok, 0], ij[ok, 1]]
        return out


@dataclass(frozen=True)
class Body:
    """Rest-frame geometry of one object card. ``rest`` rotates the authored body frame into the rest frame (apply it
    before the yaw); vertices, ``com`` and maps are in the rest frame with the body origin at 0.

    Example: ``Body.from_mesh("cube", *geom.primitive_mesh("box", (0.01,) * 3), com=np.zeros(3))``."""

    name: str
    rest: tuple[float, float, float, float]
    vertices: np.ndarray
    faces: np.ndarray
    com: np.ndarray
    footprint: np.ndarray              # (P, 2) convex CCW outline seen from above
    bottom: HeightMap
    tip_deg: float
    reoriented: bool

    @property
    def zmin(self) -> float:
        return float(self.vertices[:, 2].min())

    @property
    def size(self) -> tuple[float, float, float]:
        """Rest-frame axis-aligned extents (x, y, z), m; x is the long side of the footprint."""
        x, y, z = (float(v) for v in self.vertices.max(axis=0) - self.vertices.min(axis=0))
        return (x, y, z)

    @property
    def area(self) -> float:
        return geom.polygon_area(self.footprint)

    def triangles(self, rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """World triangles (T, 3, 3) for rest-frame rotation ``rot`` (a yaw) and origin ``pos``."""
        return (self.vertices @ rot.T + pos[None, :])[self.faces]

    @classmethod
    def from_mesh(cls, name: str, vertices: np.ndarray, faces: np.ndarray, com: np.ndarray) -> Body:
        """Choose the resting orientation of a body-frame mesh with centre of mass ``com``."""
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        rot, tip, moved = _rest_rotation(v, np.asarray(com, dtype=np.float64))
        rot = _align_long_axis(v @ rot.T) @ rot
        rv = v @ rot.T
        return cls(name=name, rest=geom.quat_from_matrix(rot), vertices=rv, faces=f, com=rot @ com,
                   footprint=geom.hull2d(rv[:, :2]), bottom=_bottom_map(rv, f), tip_deg=tip, reoriented=moved)


def mesh_com(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Volume centroid of a closed mesh (uniform density), else the vertex mean.

    Example: ``mesh_com(*geom.primitive_mesh("box", (1, 1, 1)))`` is about 0."""
    if len(f) == 0:
        return v.mean(axis=0)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = vol.sum()
    if abs(total) < 1e-15:
        return v.mean(axis=0)
    return ((a + b + c) / 4.0 * vol[:, None]).sum(axis=0) / total


def shell_com(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area centroid of a thin shell (an open bag: its mass is in the sheet, not in the volume it bounds).

    Example: ``shell_com(*geom.primitive_mesh("box", (1, 1, 1)))`` is about 0."""
    if len(f) == 0:
        return v.mean(axis=0)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() < 1e-15:
        return v.mean(axis=0)
    return ((a + b + c) / 3.0 * area[:, None]).sum(axis=0) / area.sum()


def _tip(contact: np.ndarray, com2: np.ndarray, height: float) -> float:
    """Tip angle (deg) for a contact patch (K, 2), the centre of mass seen from above and its height."""
    depth = geom.inside_depth(geom.hull2d(contact), com2)
    return math.degrees(math.atan2(depth, max(height, 1e-9)))


def _tip_down(v: np.ndarray, com: np.ndarray, down: np.ndarray) -> float:
    """Tip angle when the object rests on its support plane with ``down`` (unit) pointing at the floor."""
    rot = geom.rotation_to(down, np.array([0.0, 0.0, -1.0]))
    rv, rc = v @ rot.T, rot @ com
    zmin = rv[:, 2].min()
    tol = CONTACT_TOL + 2e-3 * float(np.ptp(rv[:, 2]))
    return _tip(rv[rv[:, 2] <= zmin + tol, :2], rc[:2], float(rc[2] - zmin))


def _face_normals(v: np.ndarray) -> list[np.ndarray]:
    """Outward normals of the largest convex-hull face groups (coplanar facets merged)."""
    try:
        from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return []
    try:
        hull = ConvexHull(v)
    except (QhullError, ValueError):
        return []
    tri = v[hull.simplices]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    groups: dict[tuple, list] = {}
    for n, a in zip(hull.equations[:, :3], areas):
        key = tuple(np.round(n, 2))
        g = groups.setdefault(key, [np.zeros(3), 0.0])
        g[0] += n * a
        g[1] += a
    best = sorted(groups.values(), key=lambda g: -g[1])[:MAX_FACES]
    return [g[0] / np.linalg.norm(g[0]) for g in best if np.linalg.norm(g[0]) > 0]


def _rest_rotation(v: np.ndarray, com: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """(rotation, tip angle deg, whether it differs from the authored orientation)."""
    down = np.array([0.0, 0.0, -1.0])
    authored = _tip_down(v, com, down)
    options = [(t, n) for n in _face_normals(v) if (t := _tip_down(v, com, n)) > -90]
    best_tip, best_n = max(options, key=lambda o: o[0], default=(authored, down))
    keep = authored >= min(AUTHORED_TIP_DEG, AUTHORED_SHARE * best_tip)
    if keep or best_tip <= max(authored, ROUND_TIP_DEG):
        return np.eye(3), authored, False
    return geom.rotation_to(best_n, down), best_tip, True


def _align_long_axis(rv: np.ndarray) -> np.ndarray:
    """A yaw that turns the footprint's minimum-area rectangle so its long side lies along x."""
    hull = geom.hull2d(rv[:, :2])
    if len(hull) < 3:
        return np.eye(3)
    best = (math.inf, 0.0)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        r = geom.yaw_matrix(-ang)[:2, :2]
        p = hull @ r.T
        w, h = np.ptp(p[:, 0]), np.ptp(p[:, 1])
        if w * h < best[0] - 1e-12:
            best = (w * h, -ang if w >= h else -ang + math.pi / 2)
    return geom.yaw_matrix(best[1])


def _bottom_map(rv: np.ndarray, f: np.ndarray) -> HeightMap:
    lo, hi = rv[:, :2].min(axis=0), rv[:, :2].max(axis=0)
    step = np.maximum((hi - lo) / (MAP_CELLS - 1), 1e-6)
    gx, gy = np.meshgrid(np.arange(MAP_CELLS), np.arange(MAP_CELLS), indexing="ij")
    xy = lo[None, :] + np.stack([gx.ravel(), gy.ravel()], axis=1) * step[None, :]
    z = geom.ray_up(rv[f], xy) if len(f) else np.full(len(xy), math.inf)
    if len(f) == 0 or not np.isfinite(z).any():            # a point cloud: its lowest point everywhere
        z = np.full(len(xy), rv[:, 2].min())
    return HeightMap(lo=lo, step=step, z=z.reshape(MAP_CELLS, MAP_CELLS))


__all__ = ["Body", "HeightMap", "mesh_com", "shell_com"]
