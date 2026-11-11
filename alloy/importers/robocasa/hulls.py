# SPDX-License-Identifier: Apache-2.0
"""Collision hulls: CoACD convex decomposition with at most ``max_verts`` vertices per hull.

``decompose`` runs CoACD (the ``[mesh]`` extra) on a closed or open triangle mesh and returns convex hulls with
outward faces. ``reduce_hull`` caps a hull's vertex count by keeping a farthest-point subset of its vertices (the
result lies inside the original hull, never outside). ``hull_mesh`` builds a hull from points.

Example::

    >>> import numpy as np
    >>> from alloy.importers.robocasa.hulls import hull_mesh, reduce_hull
    >>> pts = np.random.default_rng(0).normal(size=(500, 3))
    >>> v, f = hull_mesh(reduce_hull(pts, 64))
    >>> len(v) <= 64 and f.shape[1] == 3
    True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

log = logging.getLogger("alloy.importers.robocasa")

MAX_HULL_VERTS = 64
MAX_HULLS = 32           # ObjectCard Budget.collision_parts default


@dataclass(frozen=True)
class HullParams:
    """CoACD settings; ``threshold`` is CoACD's concavity bound (smaller: more, tighter hulls)."""

    threshold: float = 0.05
    max_hulls: int = MAX_HULLS
    max_verts: int = MAX_HULL_VERTS
    preprocess: str = "auto"
    seed: int = 0


def hull_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of the convex hull of ``points``, faces wound outward; ``ValueError`` if degenerate."""
    pts = np.asarray(points, dtype=np.float64)
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        raise ValueError(f"degenerate hull from {len(pts)} points (flat or too few); drop the part") from exc
    keep = hull.vertices
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    faces = remap[hull.simplices]
    verts = pts[keep]
    centre = verts.mean(axis=0)
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = np.einsum("ij,ij->i", normal, tri[:, 0] - centre) < 0
    faces[inward] = faces[inward][:, ::-1]
    return verts, faces


def reduce_hull(points: np.ndarray, max_verts: int = MAX_HULL_VERTS) -> np.ndarray:
    """At most ``max_verts`` hull vertices of ``points``: farthest-point sampling seeded at the extreme points."""
    verts, _ = hull_mesh(points)
    if len(verts) <= max_verts:
        return verts
    chosen = [int(i) for i in {*np.argmin(verts, axis=0), *np.argmax(verts, axis=0)}][:max_verts]
    dist = np.min(np.linalg.norm(verts[:, None, :] - verts[chosen][None, :, :], axis=2), axis=1)
    while len(chosen) < max_verts:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(verts - verts[nxt], axis=1))
    return verts[sorted(chosen)]


def _run_coacd(vertices: np.ndarray, faces: np.ndarray, p: HullParams) -> list[np.ndarray]:
    try:
        import coacd  # noqa: PLC0415 - optional [mesh] extra
    except ImportError as exc:
        raise ImportError("convex decomposition needs CoACD: pip install 'alloy[mesh]' (or pip install coacd)") from exc
    coacd.set_log_level("error")
    mesh = coacd.Mesh(np.asarray(vertices, np.float64), np.asarray(faces, np.int64))
    parts = coacd.run_coacd(mesh, threshold=p.threshold, max_convex_hull=p.max_hulls, preprocess_mode=p.preprocess,
                            decimate=True, max_ch_vertex=p.max_verts, seed=p.seed)
    return [np.asarray(v, np.float64) for v, _ in parts]


def decompose(vertices: np.ndarray, faces: np.ndarray, params: HullParams | None = None
              ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convex hulls covering the mesh, each with at most ``params.max_verts`` vertices (outward faces).

    Degenerate (flat) parts that CoACD returns are dropped and logged."""
    p = params or HullParams()
    out = []
    for pts in _run_coacd(vertices, faces, p):
        try:
            out.append(hull_mesh(reduce_hull(pts, p.max_verts)))
        except ValueError as exc:
            log.info("decompose: dropped a part: %s", exc)
    if not out:
        raise ValueError("CoACD returned no usable hull; check the mesh is not empty or flat")
    return out


def hull_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Volume of a closed, outward-wound triangle mesh (divergence theorem)."""
    tri = verts[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


__all__ = ["MAX_HULLS", "MAX_HULL_VERTS", "HullParams", "decompose", "hull_mesh", "hull_volume", "reduce_hull"]
