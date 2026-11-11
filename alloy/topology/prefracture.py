# SPDX-License-Identifier: Apache-2.0
"""Offline pre-fracture: a Voronoi split into convex chunks (SPEC §3.8; Müller et al., ToG 32(4) 2013).

Each chunk is a Voronoi cell clipped by the convex hull of the input mesh, so a non-convex mesh is split as its
hull, which is logged. The seeds come from a seeded RNG and lie inside the hull, so the same (mesh, n, seed) gives
the same chunks. The chunks become welded rigid bodies (``bonds.add_weld_bond``) and crumble sources
(``crumble.add_crumble_source``).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from alloy._vendor import newton
from alloy.core.errors import AssemblyError

log = logging.getLogger("alloy.topology")


def _seeds_inside(hull: ConvexHull, n: int, rng: np.random.Generator, margin: float) -> np.ndarray:
    lo, hi = hull.points.min(0), hull.points.max(0)
    out: list[np.ndarray] = []
    for _ in range(1000):
        cand = rng.uniform(lo, hi, size=(4 * n, 3))
        inside = np.all(cand @ hull.equations[:, :3].T + hull.equations[:, 3] < -margin, axis=1)
        out.extend(cand[inside])
        if len(out) >= n:
            return np.asarray(out[:n])
    raise AssemblyError(f"prefracture: could not place {n} seeds inside the hull")


def voronoi_cells(points: np.ndarray, n_chunks: int = 32, seed: int = 0) -> list[np.ndarray]:
    """Vertices of each Voronoi cell ∩ hull(points) (list of [k, 3] arrays)."""
    if not 1 <= n_chunks <= 256:
        raise AssemblyError(f"prefracture: n_chunks must be in [1, 256], got {n_chunks}")
    hull = ConvexHull(np.asarray(points, dtype=np.float64))
    size = float(np.linalg.norm(hull.points.max(0) - hull.points.min(0)))
    seeds = _seeds_inside(hull, n_chunks, np.random.default_rng(seed), 1e-3 * size)
    cells = []
    for i, s in enumerate(seeds):
        others = np.delete(seeds, i, axis=0)
        bis = np.hstack([others - s, -0.5 * ((others ** 2).sum(1) - (s ** 2).sum())[:, None]])
        halfspaces = np.vstack([hull.equations, bis]) if len(others) else hull.equations
        cells.append(HalfspaceIntersection(halfspaces, s).intersections)
    return cells


def prefracture(mesh: newton.Mesh, n_chunks: int = 32, seed: int = 0) -> list[newton.Mesh]:
    """Split ``mesh`` (its convex hull) into ``n_chunks`` convex chunk meshes in the mesh frame."""
    pts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    hull = ConvexHull(pts)
    if len(hull.vertices) < len(np.unique(pts.round(12), axis=0)):
        log.info("prefracture: mesh is not convex; splitting its convex hull (%d of %d vertices on the hull)",
                 len(hull.vertices), len(pts))
    chunks = []
    for verts in voronoi_cells(pts, n_chunks, seed):
        h = ConvexHull(verts)
        v = verts[h.vertices]
        remap = {int(j): k for k, j in enumerate(h.vertices)}
        faces = np.array([[remap[int(j)] for j in _outward(verts, tri, h)] for tri in h.simplices], dtype=np.int32)
        chunks.append(newton.Mesh(v.astype(np.float32), faces.reshape(-1)))
    vol = sum(ConvexHull(np.asarray(c.vertices)).volume for c in chunks)
    log.info("prefracture: %d chunks, volume %.6g of hull %.6g", len(chunks), vol, hull.volume)
    return chunks


def _outward(verts: np.ndarray, tri: np.ndarray, hull: ConvexHull) -> np.ndarray:
    """Order a hull triangle counter-clockwise seen from outside."""
    a, b, c = verts[tri]
    n = np.cross(b - a, c - a)
    centre = verts[hull.vertices].mean(axis=0)
    return tri if np.dot(n, a - centre) > 0 else tri[[0, 2, 1]]
