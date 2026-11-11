# SPDX-License-Identifier: Apache-2.0
"""Stand a generated mesh on the face it would rest on. Image-to-3D models reconstruct the object as the camera saw
it, so a photo taken from slightly above gives a mesh whose base is tilted by the camera's elevation (10 to 20
degrees on a typical product shot), and a rounded base has no single flat face to read the tilt from.

``level`` drops the mesh quasi-statically: starting where a vertical line through the centre of mass leaves the
convex hull, it rolls from hull face to hull face, always over the edge the centre of mass lies beyond, until the
centre of mass is above the face it stands on. The mesh is then rotated so that face points straight down. This is
the pose the drop test would settle it in, without the bounce.

Example::

    >>> import trimesh
    >>> box = trimesh.creation.box((1.0, 1.0, 0.5))
    >>> _ = box.apply_transform(trimesh.transformations.rotation_matrix(0.3, [1, 0, 0]))
    >>> round(level(box).degrees, 1)
    17.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("alloy.create")

MIN_TILT_DEG = 0.5           # below this the mesh is already level
BIG_TILT_DEG = 35.0          # beyond a camera's elevation: the result asks you to check which way is up


@dataclass(frozen=True)
class Leveling:
    """What ``level`` did: the rotation applied (4x4, identity when none) and its angle in degrees."""

    transform: np.ndarray
    degrees: float

    @property
    def applied(self) -> bool:
        return self.degrees > 0.0


def level(mesh) -> Leveling:
    """Rotate ``mesh`` (a trimesh, in place) onto its resting face (see the module docstring)."""
    import trimesh  # noqa: PLC0415

    normal = resting_normal(mesh)
    tilt = float(np.degrees(np.arccos(np.clip(-normal[2], -1.0, 1.0))))
    if tilt < MIN_TILT_DEG:
        return Leveling(np.eye(4), 0.0)
    t = trimesh.geometry.align_vectors(normal, [0.0, 0.0, -1.0])
    mesh.apply_transform(t)
    log.info("create: generated mesh turned %.1f degrees onto its resting face", tilt)
    return Leveling(t, tilt)


def resting_normal(mesh) -> np.ndarray:
    """The outward normal of the convex-hull face the mesh comes to rest on when let go in its current pose."""
    hull = mesh.convex_hull
    com = np.asarray(mesh.center_mass if mesh.is_volume else hull.center_mass, dtype=np.float64)
    tri = hull.triangles
    face = _exit_face(hull, com, np.array([0.0, 0.0, -1.0]))
    across = _edge_neighbours(np.asarray(hull.faces))
    seen: set[int] = set()
    while face not in seen:
        seen.add(face)
        bary = _barycentric(tri[face], com)
        worst = int(np.argmin(bary))
        if bary[worst] >= -1e-9:
            break
        face = across[face][worst]
    return np.asarray(hull.face_normals[face], dtype=np.float64)


def _exit_face(hull, origin: np.ndarray, direction: np.ndarray) -> int:
    """The hull face a ray from ``origin`` (inside) along ``direction`` leaves through."""
    n = hull.face_normals
    toward = n @ direction
    dist = np.einsum("ij,ij->i", n, hull.triangles[:, 0] - origin)
    t = np.where(toward > 1e-12, dist / np.where(toward > 1e-12, toward, 1.0), np.inf)
    return int(np.argmin(t))


def _edge_neighbours(faces: np.ndarray) -> list[list[int]]:
    """For each face, the face across the edge opposite each of its three vertices."""
    owner: dict[tuple[int, int], list[int]] = {}
    for f, (a, b, c) in enumerate(faces):
        for e in ((b, c), (c, a), (a, b)):
            owner.setdefault((min(e), max(e)), []).append(f)
    out = []
    for f, (a, b, c) in enumerate(faces):
        row = []
        for e in ((b, c), (c, a), (a, b)):
            pair = owner[(min(e), max(e))]
            row.append(next((g for g in pair if g != f), f))
        out.append(row)
    return out


def _barycentric(tri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of ``p`` projected onto the triangle's plane (a negative one: outside that edge)."""
    a, b, c = tri
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if den <= 0.0:
        return np.zeros(3)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.array([1.0 - v - w, v, w])


__all__ = ["BIG_TILT_DEG", "Leveling", "level", "resting_normal"]
