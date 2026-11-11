# SPDX-License-Identifier: Apache-2.0
"""Small numpy geometry for the scene builder: quaternions (x, y, z, w), primitive meshes, 2D convex polygons and
vertical ray casts. No engine imports.

Example::

    >>> import numpy as np
    >>> v, f = primitive_mesh("box", (0.1, 0.05, 0.02))
    >>> float(v[:, 2].min()), len(f)
    (-0.02, 12)
    >>> float(ray_down(v[f], np.array([[0.0, 0.0]]))[0])     # top face of the box
    0.02
"""

from __future__ import annotations

import math

import numpy as np

SEGMENTS = 24                                  # around a cylinder, cone, capsule or sphere
RINGS = 10                                     # pole to pole on a sphere


# --- quaternions (x, y, z, w) ----------------------------------------------------------------------------------

def quat_from_matrix(r: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (x, y, z, w) of a 3x3 rotation matrix. Example: ``quat_from_matrix(np.eye(3))``."""
    m = np.asarray(r, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q4 = [0.0, 0.0, 0.0, 0.0]
        q4[i] = 0.25 * s
        q4[j] = (m[j, i] + m[i, j]) / s
        q4[k] = (m[k, i] + m[i, k]) / s
        q4[3] = (m[k, j] - m[j, k]) / s
        q = tuple(q4)
    n = math.sqrt(sum(c * c for c in q))
    out = tuple(float(c / n) for c in q)
    return out if out[3] >= 0 else tuple(-c for c in out)          # type: ignore[return-value]


def matrix_from_quat(q: tuple[float, ...]) -> np.ndarray:
    """3x3 rotation matrix of a quaternion (x, y, z, w). Example: ``matrix_from_quat((0, 0, 0, 1))``."""
    x, y, z, w = (float(c) for c in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation about +z by ``yaw`` radians. Example: ``yaw_matrix(math.pi / 2) @ [1, 0, 0]`` is +y."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` to unit vector ``b``. Example: ``rotation_to(z, -z)``."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-12:                                     # opposite: 180 deg about any axis normal to a
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


# --- primitive meshes (body frame, the engine's conventions: cylinder, cone and capsule along z) ----------------

def primitive_mesh(kind: str, size: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh (vertices, faces) of a card primitive; sizes are half extents / radius, half height.

    Example: ``primitive_mesh("cylinder", (0.01, 0.05))``."""
    s = [float(x) for x in size]
    if kind == "box":
        return _box(s[0], s[1], s[2])
    if kind == "plane":
        hx, hy = (s[0] if s and s[0] > 0 else 5.0), (s[1] if len(s) > 1 and s[1] > 0 else 5.0)
        return _box(hx, hy, 1e-4)
    if kind in ("sphere", "ellipsoid"):
        radii = (s[0], s[0], s[0]) if kind == "sphere" else (s[0], s[1], s[2])
        v, f = _sphere()
        return v * np.asarray(radii), f
    if kind == "capsule":
        v, f = _sphere()
        v = v * s[0]
        v[:, 2] += np.where(v[:, 2] >= 0, s[1], -s[1])
        return v, f
    if kind in ("cylinder", "cone"):
        return _lathe(s[0], s[1], cone=kind == "cone")
    raise ValueError(f"primitive kind {kind!r} has no mesh")


def _box(hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], dtype=np.float64)
    f = np.array([[0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]], dtype=np.int64)
    return v, f


def _sphere() -> tuple[np.ndarray, np.ndarray]:
    """Unit UV sphere with poles; even ring count so a capsule splits cleanly at z = 0."""
    verts = [[0.0, 0.0, 1.0]]
    for i in range(1, RINGS):
        th = math.pi * i / RINGS
        verts += [[math.sin(th) * math.cos(p), math.sin(th) * math.sin(p), math.cos(th)]
                  for p in np.linspace(0, 2 * math.pi, SEGMENTS, endpoint=False)]
    verts.append([0.0, 0.0, -1.0])
    faces, n, last = [], SEGMENTS, 1 + (RINGS - 1) * SEGMENTS
    for j in range(n):
        faces.append([0, 1 + j, 1 + (j + 1) % n])
        faces.append([last, 1 + (RINGS - 2) * n + (j + 1) % n, 1 + (RINGS - 2) * n + j])
    for i in range(RINGS - 2):
        a, b = 1 + i * n, 1 + (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            faces += [[a + j, b + j, b + k], [a + j, b + k, a + k]]
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def _lathe(r: float, hh: float, cone: bool) -> tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0, 2 * math.pi, SEGMENTS, endpoint=False)
    ring = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    bottom = np.column_stack([ring, np.full(SEGMENTS, -hh)])
    top = np.array([[0.0, 0.0, hh]]) if cone else np.column_stack([ring, np.full(SEGMENTS, hh)])
    v = np.vstack([bottom, top, [[0.0, 0.0, -hh]], [[0.0, 0.0, hh]]])
    cb, ct_ = len(v) - 2, len(v) - 1
    f = []
    for j in range(SEGMENTS):
        k = (j + 1) % SEGMENTS
        f.append([cb, k, j])
        if cone:
            f.append([j, k, SEGMENTS])
        else:
            f += [[j, k, SEGMENTS + k], [j, SEGMENTS + k, SEGMENTS + j], [ct_, SEGMENTS + j, SEGMENTS + k]]
    return v, np.asarray(f, dtype=np.int64)


# --- 2D convex polygons ------------------------------------------------------------------------------------------

def hull2d(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, counter-clockwise, no repeated end point (Andrew's monotone chain).

    Example: ``hull2d(np.array([[0, 0], [1, 0], [0, 1], [0.2, 0.2]]))`` has 3 corners."""
    pts = np.unique(np.round(np.asarray(points, dtype=np.float64)[:, :2], 9), axis=0)
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    upper: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def inside_depth(poly: np.ndarray, p: np.ndarray) -> float:
    """Signed distance from ``p`` to the boundary of convex CCW ``poly``: > 0 inside, < 0 outside (edge-plane
    approximation outside). A polygon with fewer than 3 corners counts as a point or segment of zero area.

    Example: ``inside_depth(square, centre)`` is half the side."""
    return float(depths(poly, np.asarray(p, dtype=np.float64).reshape(1, 2))[0])


def depths(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``inside_depth`` of many points (K, 2) at once. Example: ``depths(square, corners).min()``."""
    poly = np.asarray(poly, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        if not len(poly):
            return np.full(len(pts), -math.inf)
        return -np.linalg.norm(pts[:, None, :] - poly[None, :, :], axis=2).min(axis=1)
    a = poly
    e = np.roll(poly, -1, axis=0) - a
    n = np.stack([-e[:, 1], e[:, 0]], axis=1) / np.maximum(np.linalg.norm(e, axis=1), 1e-15)[:, None]
    return (np.einsum("kej,ej->ke", pts[:, None, :] - a[None, :, :], n)).min(axis=1)


def separated(p: np.ndarray, q: np.ndarray, gap: float) -> bool:
    """True if convex polygons ``p`` and ``q`` are at least ``gap`` apart along some edge normal (SAT).

    Example: two unit squares 2 apart are separated with gap 0.5."""
    for poly in (p, q):
        e = np.roll(poly, -1, axis=0) - poly
        axes = np.stack([e[:, 1], -e[:, 0]], axis=1)
        axes /= np.maximum(np.linalg.norm(axes, axis=1), 1e-15)[:, None]
        pp, qq = p @ axes.T, q @ axes.T
        if np.any((qq.min(axis=0) - pp.max(axis=0) >= gap) | (pp.min(axis=0) - qq.max(axis=0) >= gap)):
            return True
    return False


def polygon_area(poly: np.ndarray) -> float:
    """Area of a simple polygon (shoelace). Example: the unit square has area 1."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


# --- vertical ray casts --------------------------------------------------------------------------------------------

def _hits(tris: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(K, T) mask of which triangle each vertical line crosses and the z there."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1 = (b - a)[:, :2], (c - a)[:, :2]
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(den) > 1e-18
    den = np.where(ok, den, 1.0)
    d = xy[:, None, :] - a[None, :, :2]
    u = (d[..., 0] * v1[None, :, 1] - d[..., 1] * v1[None, :, 0]) / den[None, :]
    w = (v0[None, :, 0] * d[..., 1] - v0[None, :, 1] * d[..., 0]) / den[None, :]
    eps = 1e-9
    mask = ok[None, :] & (u >= -eps) & (w >= -eps) & (u + w <= 1 + eps)
    z = a[None, :, 2] + u * (b - a)[None, :, 2] + w * (c - a)[None, :, 2]
    return mask, z


def ray_down(tris: np.ndarray, xy: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Highest z of the triangles ``tris`` (T, 3, 3) above each point ``xy`` (K, 2); -inf where none.

    Example: see the module docstring."""
    return _ray(tris, xy, chunk, top=True)


def ray_up(tris: np.ndarray, xy: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Lowest z of the triangles over each point ``xy``; +inf where none. Example: the bottom of an object."""
    return _ray(tris, xy, chunk, top=False)


def _ray(tris: np.ndarray, xy: np.ndarray, chunk: int, top: bool) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    fill = -math.inf if top else math.inf
    out = np.full(len(xy), fill)
    if len(tris) == 0 or len(xy) == 0:
        return out
    step = max(1, chunk // max(1, len(tris)))
    for i in range(0, len(xy), step):
        mask, z = _hits(tris, xy[i:i + step])
        z = np.where(mask, z, fill)
        out[i:i + step] = z.max(axis=1) if top else z.min(axis=1)
    return out


def cull_xy(tris: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The triangles whose xy bounding box meets the box [lo, hi]. Example: ``cull_xy(t, [-1, -1], [1, 1])``."""
    if len(tris) == 0:
        return tris
    tlo, thi = tris[:, :, :2].min(axis=1), tris[:, :, :2].max(axis=1)
    keep = np.all(thi >= np.asarray(lo)[None, :], axis=1) & np.all(tlo <= np.asarray(hi)[None, :], axis=1)
    return tris[keep]


__all__ = ["cull_xy", "depths", "hull2d", "inside_depth", "matrix_from_quat", "polygon_area", "primitive_mesh",
           "quat_from_matrix", "ray_down", "ray_up", "rotation_to", "separated", "yaw_matrix"]
