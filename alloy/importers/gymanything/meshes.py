# SPDX-License-Identifier: Apache-2.0
"""Mesh preparation for imported objects: weld, orient, measure, pick the collision mode, write canonical OBJ.

Example::

    >>> from alloy.cards import geometry as geo
    >>> m = prepare(geo.read_mesh(Path("mug.obj")))          # doctest: +SKIP
    >>> facts = measure(m); facts.collision                    # doctest: +SKIP
    'sdf'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alloy.cards import geometry as geo
from alloy.importers.common import walls

log = logging.getLogger("alloy.importers.gymanything")

CONVEX_RATIO = 1.15          # hull volume / mesh volume at or below this -> collide with the convex hull
WELD_TOL = 1e-7              # m, the tolerance Newton and alloy.cards use for watertightness
MAX_FACES = 200_000          # above this the mesh is refused (decimate in the exporter)


class MeshError(ValueError):
    """A mesh cannot be imported; the message says what to change in the exporter."""


@dataclass(frozen=True)
class MeshFacts:
    """What the importer needs to know about a prepared mesh (SI units)."""

    watertight: bool
    volume: float               # m^3 (closed mesh volume, else the convex hull volume)
    hull_volume: float          # m^3
    convex_ratio: float         # hull_volume / volume (1.0 = convex)
    extents: tuple[float, float, float]   # OBB extents, largest first, m
    aabb: tuple[tuple[float, float, float], tuple[float, float, float]]
    collision: str              # "convex" | "sdf"
    min_wall: float             # m, thinnest wall to declare (``common.walls.min_wall``; inf when not measured)
    wall_probe: float = float("inf")    # m, the validator probe's reading (``geometry.wall_thickness``)


def weld(mesh: geo.MeshData, tol: float = WELD_TOL) -> geo.MeshData:
    """Merge vertices closer than ``tol``, drop degenerate and duplicate faces, drop unused vertices."""
    keys = np.round(mesh.vertices / tol).astype(np.int64)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    faces = inv.reshape(-1)[mesh.faces]
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    faces = faces[ok]
    _, keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(keep)]
    used, remap = np.unique(faces, return_inverse=True)
    return geo.MeshData(mesh.vertices[first][used], remap.reshape(-1, 3).astype(np.int64))


def prepare(mesh: geo.MeshData) -> geo.MeshData:
    """Weld and orient outward (a closed mesh with negative signed volume is flipped). Raises MeshError."""
    if len(mesh.faces) == 0:
        raise MeshError("mesh has no faces: export the object's surface triangles")
    if len(mesh.faces) > MAX_FACES:
        raise MeshError(f"{len(mesh.faces)} faces > {MAX_FACES}: decimate in the exporter")
    if not np.all(np.isfinite(mesh.vertices)):
        raise MeshError("mesh has non-finite vertices")
    m = weld(mesh)
    if geo.is_closed(m) and geo.signed_volume(m) < 0:
        m = geo.MeshData(m.vertices, m.faces[:, ::-1].copy())
    return m


def hull_volume(vertices: np.ndarray) -> float:
    """Convex hull volume [m^3] (scipy Qhull); 0 for flat or degenerate point sets."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415 - scipy only when meshes are measured

    try:
        return float(ConvexHull(vertices).volume)
    except (QhullError, ValueError):
        return 0.0


def convex_hull_mesh(mesh: geo.MeshData) -> geo.MeshData:
    """The convex hull as a closed, outward-wound mesh (Qhull simplices, each oriented away from the centroid)."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    hull = ConvexHull(mesh.vertices)
    used, remap = np.unique(hull.simplices, return_inverse=True)
    v, f = mesh.vertices[used], remap.reshape(-1, 3).astype(np.int64)
    c = v.mean(axis=0)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", n, tri.mean(axis=1) - c) < 0
    f[flip] = f[flip][:, ::-1]
    return geo.MeshData(v, f)


def measure(mesh: geo.MeshData, thin: float = float("inf")) -> MeshFacts:
    """Facts of a prepared mesh. Non-watertight meshes fall back to the convex hull volume and convex collision.
    ``min_wall`` is the validator probe's reading, re-measured by ``common.walls.min_wall`` (kept under ``thin``, the
    validator's threshold) where the probe reads a defect under 0.3 mm. Example: ``measure(mesh, THIN_LIMIT)``."""
    tight = geo.is_watertight(mesh)
    hv = hull_volume(mesh.vertices)
    vol = geo.signed_volume(mesh) if tight else hv
    ratio = hv / vol if vol > 0 else float("inf")
    collision = "convex" if (not tight or ratio <= CONVEX_RATIO) else "sdf"
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    probe = geo.wall_thickness(mesh) if tight else float("inf")     # same call (and memo) as the card validator
    wall = walls.min_wall(mesh, probe, thin) if tight else probe
    return MeshFacts(tight, float(vol), hv, float(ratio), geo.obb_extents(mesh.vertices),
                     (tuple(map(float, lo)), tuple(map(float, hi))), collision, float(wall), float(probe))


def write_canonical(path: Path, mesh: geo.MeshData, header: str) -> str:
    """Write a deterministic OBJ (``alloy.cards.geometry.write_obj``) and return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    geo.write_obj(path, mesh, header)
    return geo.sha256_file(path)
