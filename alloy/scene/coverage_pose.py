# SPDX-License-Identifier: Apache-2.0
"""Resting poses for the CARD-1 coverage scenes (`alloy.scene.coverage`): which way up an object is put on the table.

Each of the six axis-aligned orientations (card ±x, ±y, ±z down) is scored from surface sample points (primitive
rims and corners, mesh vertices) and a centre of mass (the mesh volume centroid; primitives: their centre):
- support polygon = convex hull of the points within SUPPORT_EPS of the lowest one; a pose is *supported* when the
  polygon has area and the centre of mass projects inside it;
- tipping angle = atan(distance from the projected centre to the nearest polygon edge / centre height).
The pose is the supported one with the largest tipping angle; when none tips beyond MIN_TIP (a pencil on its end,
2.4°), the lowest-centre pose (lying on its side, neutral for a round body) is used instead. Ties keep the authored
pose. Curves, particle regions and MJCF assemblies (articulated: a hinged box) keep the authored pose.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from alloy.cards import types as ct

UP = (0.0, 0.0, 0.0, 1.0)
_S2 = math.sqrt(0.5)
ORIENTATIONS = (UP, (1.0, 0.0, 0.0, 0.0), (0.0, _S2, 0.0, _S2), (0.0, -_S2, 0.0, _S2), (_S2, 0.0, 0.0, _S2),
                (-_S2, 0.0, 0.0, _S2))          # card -z, +z, +x, -x, -y, +y down
MIN_TIP = math.radians(10.0)
RIM = 24


def _ring(r: float, z: float, n: int = RIM) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], axis=1)


def _ellipsoid(rx: float, ry: float, rz: float) -> np.ndarray:
    th, ph = np.meshgrid(np.linspace(0.0, math.pi, 13), np.linspace(0.0, 2.0 * math.pi, RIM, endpoint=False))
    return np.stack([rx * np.sin(th) * np.cos(ph), ry * np.sin(th) * np.sin(ph), rz * np.cos(th)], -1).reshape(-1, 3)


def _primitive_points(p: ct.Primitive) -> np.ndarray:
    s = p.size
    if p.kind == "box":
        return np.array([[a, b, c] for a in (-s[0], s[0]) for b in (-s[1], s[1]) for c in (-s[2], s[2])])
    if p.kind == "cylinder":
        return np.concatenate([_ring(s[0], -s[1]), _ring(s[0], s[1])])
    if p.kind == "cone":
        return np.concatenate([_ring(s[0], -s[1]), [[0.0, 0.0, s[1]]]])
    if p.kind == "sphere":
        return _ellipsoid(s[0], s[0], s[0])
    if p.kind == "ellipsoid":
        return _ellipsoid(s[0], s[1], s[2])
    if p.kind == "capsule":
        cap = _ellipsoid(s[0], s[0], s[0])
        cap[:, 2] += np.where(cap[:, 2] >= 0.0, s[1], -s[1])      # hemispheres at ±half-height
        return cap
    raise ValueError(f"no surface points for primitive {p.kind!r}")


def _volume_centroid(v: np.ndarray, f: np.ndarray) -> np.ndarray | None:
    if len(f) == 0:
        return None
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = vol.sum()
    if abs(total) < 1e-15:
        return None
    return ((a + b + c) / 4.0 * vol[:, None]).sum(0) / total


def surface_points(card: ct.ObjectCard, lib) -> tuple[np.ndarray, np.ndarray]:
    """(points [N, 3], centre of mass [3]) in the card frame."""
    s = card.shape
    if isinstance(s, ct.Primitive):
        return _primitive_points(s), np.zeros(3)
    if isinstance(s, ct.CurveSource):
        p = np.asarray(s.points, np.float64)
        return np.concatenate([p - [0, 0, s.radius], p + [0, 0, s.radius]]), p.mean(0)
    if isinstance(s, ct.ParticleSource):
        lo, hi = (np.asarray(x, np.float64) for x in s.region)
        return np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]), \
            0.5 * (lo + hi)
    mesh = lib.mesh_for(card)
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"{card.name}: no geometry to place ({type(s).__name__})")
    v = np.asarray(mesh.vertices, np.float64)
    com = _volume_centroid(v, np.asarray(mesh.faces, np.int64))
    return v, (v.mean(0) if com is None else com)


def rotate(q: tuple[float, ...], v: np.ndarray) -> np.ndarray:
    """Rotate row vectors `v` [N, 3] by quaternion (x, y, z, w)."""
    u, w = np.asarray(q[:3], np.float64), float(q[3])
    v = np.atleast_2d(v)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def tipping_angle(points: np.ndarray, com: np.ndarray) -> float:
    """Tipping angle [rad] of a resting pose (0 when unsupported: line or point contact, or the centre outside)."""
    z0 = points[:, 2].min()
    eps = 1e-4 + 1e-3 * float(np.ptp(points, axis=0).max())
    base = points[points[:, 2] <= z0 + eps, :2]
    if len(base) < 3:
        return 0.0
    try:
        hull = ConvexHull(base)
    except QhullError:
        return 0.0
    if hull.volume < 1e-12:                                       # 2-D hull "volume" is its area
        return 0.0
    dist = -(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2])   # inside: all >= 0
    d_min = float(dist.min())
    if d_min <= 0.0:
        return 0.0
    return math.atan2(d_min, max(float(com[2] - z0), 1e-9))


def best_orientation(points: np.ndarray, com: np.ndarray) -> tuple[float, ...]:
    scores = []
    for q in ORIENTATIONS:
        p, c = rotate(q, points), rotate(q, com)[0]
        scores.append((tipping_angle(p, c), float(c[2] - p[:, 2].min())))
    tip = [s[0] for s in scores]
    k = int(np.argmax(tip))                                       # first maximum: ties keep the authored pose
    if tip[k] >= MIN_TIP:
        return ORIENTATIONS[k]
    return ORIENTATIONS[int(np.argmin([s[1] - 1e-9 * (i == 0) for i, s in enumerate(scores)]))]


def rest_pose(card: ct.ObjectCard, lib, top_z: float, gap: float, *, choose: bool = True) -> tuple[float, ...]:
    """(x, y, z, qx, qy, qz, qw): centred over the origin, lowest point `gap` above `top_z`; `choose` picks the
    orientation (`best_orientation`), else the authored one."""
    points, com = surface_points(card, lib)
    choose = choose and not isinstance(card.shape, (ct.CurveSource, ct.ParticleSource, ct.MJCFSource))
    q = best_orientation(points, com) if choose else UP
    p = rotate(q, points)
    centre = 0.5 * (p.min(0) + p.max(0))
    return (-float(centre[0]), -float(centre[1]), top_z - float(p[:, 2].min()) + gap, *q)


def top_of(card: ct.ObjectCard, lib, pose: tuple[float, ...]) -> float:
    """Highest point [m] of the object at `pose`."""
    points, _ = surface_points(card, lib)
    return pose[2] + float(rotate(pose[3:], points)[:, 2].max())


__all__ = ["rest_pose", "top_of", "best_orientation", "tipping_angle", "surface_points", "rotate", "ORIENTATIONS",
           "MIN_TIP", "UP"]
