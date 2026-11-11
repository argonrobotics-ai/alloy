# SPDX-License-Identifier: Apache-2.0
"""``features.min_wall`` of a physics mesh: Alloy's inward-ray probe, re-measured where it reads a remesh defect.

``alloy.cards.geometry.wall_thickness`` (the validator's probe) casts a ray from up to 1,024 face centroids along the
inward normal and takes the nearest exit over all rays. On scans one bad ray decides it: ``ycb_040_large_marker`` read
23 um, ``ycb_057_racquetball`` (a solid ball) 1 um. Two kinds of defect answer such rays:
- folds: a sliver face folded back onto a neighbour it shares a vertex with (the marker's 23 and 35 um);
- slivers: needle faces of the watertight remesh, a fraction of a mm^2 in all (the master chef can's 1-35 um rays
  stand for 0.3 mm^2 of its 62,000 mm^2 surface);
- inner shells: overlapping parts of an artist mesh (a stem pushed into an apple) leave surfaces inside the solid; a
  ray meets them going in (their outward normal faces it), microns under the skin, over large areas.

``min_wall`` keeps the probe's value when it is plausible (``PLAUSIBLE``, 0.3 mm) and else re-measures with ``wall``:
the same ray from every face (up to ``MAX_RAYS``), ending only where it leaves the material (a face whose outward
normal points along the ray), hits on the face's own 1-ring (faces sharing a vertex) ignored, and the thinnest wall that faces totalling ``AREA`` (10 mm^2, a 3 x 3 mm patch, about 8 faces of a 5,000-face scan) agree
on: no wall a contact can land on is smaller, and no defect above is larger. The marker reads 1.17 mm; the ball has
no thin wall, so it declares the validator's threshold (its probe still demands a value); a scan that really is thin
keeps a thin value (``ycb_029_plate``: 0.23 mm), and a wall the probe's 1,024 rays missed can read thinner. Where
the re-measure confirms a thin reading (within ``AGREE``), the mesh itself is that thin (sheets in artist meshes) and
the probe's value stays: fixing that needs the mesh, not ``min_wall``.

Example::

    from alloy.cards import geometry as geo
    wall(geo.read_mesh(Path("meshes/ycb_040_large_marker/physics.obj")))   # 1.17e-3
"""

from __future__ import annotations

import math

import numpy as np

PLAUSIBLE = 3e-4               # m: probe readings below this on a rigid scan are re-measured with ``wall``
AREA = 1e-5                    # m^2: a wall must be read by rays from faces of at least this total area
AGREE = 2.0                    # a thin re-measure within this factor of the probe confirms it: the probe's value stays
MAX_RAYS = 20_000
WITHIN = 0.05                  # m: walls at least this thick read inf (only thin walls matter to min_wall)
GROUP = 32                     # rays per near-triangle query
_COINCIDENT = 1e-7             # m: as alloy.cards.geometry (touching shells, not walls)


def _rings(f: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """For each face, the faces that share a vertex with it (itself included)."""
    flat = f.ravel()
    order = np.argsort(flat, kind="stable")
    start = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=n_vertices))])
    of_vertex = order // 3
    return [np.unique(np.concatenate([of_vertex[start[v]:start[v + 1]] for v in face])) for face in f]


def _exits(origins: np.ndarray, dirs: np.ndarray, tri: np.ndarray, ids: np.ndarray, rings: list[np.ndarray],
           faces: np.ndarray, outward: float) -> np.ndarray:
    """Nearest exit distance of each ray among triangles ``tri`` (global ids ``ids``): a face the ray leaves through
    (its outward normal, ``outward`` x the winding normal, along the ray), not in the ray's own 1-ring."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    leaves = outward * (dirs @ np.cross(e1, e2).T) > 0.0
    p = np.cross(dirs[:, None, :], e2[None, :, :])
    det = np.einsum("rtk,tk->rt", p, e1)
    ok = np.abs(det) > 1e-18
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origins[:, None, :] - tri[None, :, 0, :]
    u = np.einsum("rtk,rtk->rt", s, p) * inv
    qv = np.cross(s, e1[None, :, :])
    v = np.einsum("rk,rtk->rt", dirs, qv) * inv
    t = np.einsum("tk,rtk->rt", e2, qv) * inv
    hit = ok & leaves & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > _COINCIDENT)
    for r, face in enumerate(faces):
        hit[r] &= ~np.isin(ids, rings[face])
    return np.where(hit, t, np.inf).min(axis=1)


def ray_walls(mesh, within: float = WITHIN, max_rays: int = MAX_RAYS) -> tuple[np.ndarray, np.ndarray]:
    """``(wall [m], surface area [m^2] it stands for)`` per ray: the inward ray from a face centroid to the first face
    it leaves the material through (``_exits``), inf at ``within`` or more; rays are spread evenly over the faces (all
    of them up to ``max_rays``).

    Example::

        d, area = ray_walls(geo.read_mesh(Path("physics.obj")))
    """
    from alloy.cards import geometry as geo  # noqa: PLC0415

    v, f = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    tri = v[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    good = np.nonzero(norm > 1e-20)[0]
    if len(good) == 0:
        return np.zeros(0), np.zeros(0)
    pick = good[np.linspace(0, len(good) - 1, min(max_rays, len(good))).astype(np.int64)]
    origins = tri[pick].mean(axis=1)
    outward = 1.0 if geo.signed_volume(mesh) >= 0 else -1.0       # an inside-out mesh's normals point inward
    dirs = -outward * normals[pick] / norm[pick, None]
    rings = _rings(f, len(v))
    cells = np.floor((origins - origins.min(axis=0)) / (4.0 * within)).astype(np.int64)
    order = np.lexsort((cells[:, 0], cells[:, 1], cells[:, 2]))
    lo_t, hi_t = tri.min(axis=1), tri.max(axis=1)
    out = np.full(len(pick), np.inf)
    for k in range(0, len(order), GROUP):
        idx = order[k:k + GROUP]
        lo, hi = origins[idx].min(axis=0) - within, origins[idx].max(axis=0) + within
        near = np.nonzero(np.all(hi_t >= lo, axis=1) & np.all(lo_t <= hi, axis=1))[0]
        if len(near):
            out[idx] = _exits(origins[idx], dirs[idx], tri[near], near, rings, pick[idx], outward)
    out[out >= within] = np.inf
    return out, 0.5 * norm[pick] * (len(good) / len(pick))


def wall(mesh, area: float = AREA, within: float = WITHIN) -> float:
    """Thinnest wall [m] read over at least ``area`` of the surface, 1-ring hits ignored; inf when none is under
    ``within`` (module docstring).

    Example::

        wall(geo.read_mesh(Path("physics.obj")))
    """
    d, a = ray_walls(mesh, within)
    order = np.argsort(d)
    k = int(np.searchsorted(np.cumsum(a[order]), area))
    return float(d[order[k]]) if k < len(order) else math.inf


def min_wall(mesh, probe: float, cap: float) -> float:
    """The value to declare as ``features.min_wall``: ``probe`` (the validator's reading) when plausible, else
    ``wall``, kept below ``cap`` (the validator's threshold: its probe still demands a declaration). A ``wall`` still
    under ``PLAUSIBLE`` and within ``AGREE`` of the probe confirms it (the mesh is that thin): the probe's value stays.

    Example::

        min_wall(mesh, geo.wall_thickness(mesh), THIN)
    """
    if probe >= PLAUSIBLE:
        return probe
    w = wall(mesh)
    if w < PLAUSIBLE and probe / AGREE <= w <= probe * AGREE:
        return probe
    return min(w, math.nextafter(cap, 0.0))
