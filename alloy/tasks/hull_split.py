# SPDX-License-Identifier: Apache-2.0
"""Split a large convex hull into convex pieces with at most ``max_verts`` vertices each, within ``tol`` of its surface.

Why (SB-6, SCR-WP3-6): MuJoCo Warp's GJK/EPA cost grows with hull size. T12's finger-pad hulls have up to 1,423
vertices, and one box-vs-hull pair costs ~650k cycles, against ~60k for a box-box pair. Cutting the hulls to 64
vertices with qhull's ``maxhullvert`` is not an option: it moved the surface by an unbounded amount and triggered
BUL-0002 (docs/notes/wp13-bul0002.md).

Method, exact up to the tolerance, which is checked rather than assumed:
1. **Split.** The hull is split by planes into convex regions: each cut keeps the vertices on one side plus the
   points where the cut plane crosses hull edges. The regions' union is the hull exactly.
2. **Simplify.** Each region is replaced by the hull of a subset of its vertices, chosen farthest-point-first
   (Quickhull order) until every region vertex is within ``tol`` of it (exact point-to-hull distance).
3. **Split again** when a region needs more than ``max_verts`` vertices, across the principal axis through its
   centroid.

Bound: every point of the original surface is a convex combination of one region's vertices. Each of those vertices
lies within ``tol`` of that region's piece, and the piece is convex. So the surface is within ``tol`` of the union of
the pieces, and the bound is the largest region-vertex distance (``SplitResult.deviation_m``). The pieces are hulls
of points on or inside the original hull, so they never reach outside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from alloy.core.errors import TraceError


@dataclass(frozen=True)
class SplitConfig:
    max_verts: int = 64           # per piece (MuJoCo Warp CCD cost grows with hull size)
    tol_m: float = 2e-4           # max distance of the original hull surface to the union of the pieces
    max_pieces: int = 64          # guard against a runaway split


@dataclass(frozen=True)
class SplitResult:
    pieces: tuple[np.ndarray, ...]     # [k_i, 3] float64 vertices, k_i <= max_verts
    deviation_m: float                 # proven bound: max region-vertex distance to its piece
    hull_verts: int                    # vertices of the original hull


def hull_vertices(points: np.ndarray) -> np.ndarray:
    """The convex hull's vertices of ``points`` [n, 3]."""
    p = np.asarray(points, dtype=np.float64)
    return p[ConvexHull(p).vertices]


def outward_faces(verts: np.ndarray) -> np.ndarray:
    """Triangles [m, 3] (indices into ``verts``) of ``hull(verts)``, wound counter-clockwise seen from outside."""
    hull = ConvexHull(verts)
    tri = hull.simplices.copy()
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0.0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def _closest_on_triangles(q: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance [n, t] from points q [n, 3] to triangles (a, b, c) [t, 3] (Ericson, Real-Time Collision Detection)."""
    q = q[:, None, :]
    ab, ac, ap = b - a, c - a, q - a
    d1, d2 = np.einsum("tk,ntk->nt", ab, ap), np.einsum("tk,ntk->nt", ac, ap)
    bp, cp = q - b, q - c
    d3, d4 = np.einsum("tk,ntk->nt", ab, bp), np.einsum("tk,ntk->nt", ac, bp)
    d5, d6 = np.einsum("tk,ntk->nt", ab, cp), np.einsum("tk,ntk->nt", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        v, w = vb * denom, vc * denom
        t_ab = np.clip(d1 / (d1 - d3), 0.0, 1.0)
        t_ac = np.clip(d2 / (d2 - d6), 0.0, 1.0)
        t_bc = np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0, 1.0)
    p = a + v[..., None] * ab + w[..., None] * ac                       # interior
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    p = np.where(edge_bc[..., None], b + t_bc[..., None] * (c - b), p)
    p = np.where(edge_ac[..., None], a + t_ac[..., None] * ac, p)
    p = np.where(edge_ab[..., None], a + t_ab[..., None] * ab, p)
    p = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, p)
    p = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, p)
    p = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, p)
    return np.linalg.norm(q - p, axis=-1)


def point_hull_distance(q: np.ndarray, verts: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Euclidean distance [n] from points ``q`` to ``hull(verts)`` (0 inside)."""
    q = np.asarray(q, dtype=np.float64)
    hull = ConvexHull(verts)
    plane = q @ hull.equations[:, :3].T + hull.equations[:, 3]
    out = np.zeros(len(q))
    outside = np.nonzero(plane.max(axis=1) > 0.0)[0]
    a, b, c = (verts[hull.simplices[:, k]] for k in range(3))
    for s in range(0, len(outside), chunk):
        idx = outside[s : s + chunk]
        out[idx] = _closest_on_triangles(q[idx], a, b, c).min(axis=1)
    return out


def _initial_simplex(p: np.ndarray) -> list[int]:
    """Four points spanning a large tetrahedron: extremes along the principal axis, then farthest from line, plane."""
    axis = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)[2][0]
    i, j = int(np.argmin(p @ axis)), int(np.argmax(p @ axis))
    d = p[j] - p[i]
    k = int(np.argmax(np.linalg.norm(np.cross(p - p[i], d), axis=1)))
    n = np.cross(d, p[k] - p[i])
    m = int(np.argmax(np.abs((p - p[i]) @ n)))
    return [i, j, k, m]


def greedy_subset(p: np.ndarray, cfg: SplitConfig) -> tuple[np.ndarray, float] | None:
    """Farthest-first subset S of ``p`` with every point of ``p`` within ``cfg.tol_m`` of hull(S), and the largest
    such distance; None if S would need more than ``cfg.max_verts`` points."""
    if len(p) <= cfg.max_verts:
        return np.arange(len(p)), 0.0
    sel = _initial_simplex(p)
    while len(sel) <= cfg.max_verts:
        eq = ConvexHull(p[sel]).equations
        plane = (p @ eq[:, :3].T + eq[:, 3]).max(axis=1)
        worst = int(np.argmax(plane))
        if plane[worst] > cfg.tol_m:              # plane distance is a lower bound of the true distance
            sel.append(worst)
            continue
        dist = point_hull_distance(p, p[sel])
        worst = int(np.argmax(dist))
        if dist[worst] <= cfg.tol_m:
            return np.array(sel), float(dist[worst])
        sel.append(worst)
    return None


def clip(p: np.ndarray, normal: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact split of hull(p) by the plane ``normal·x = offset``: the vertices of both sides' convex regions."""
    hull = ConvexHull(p)
    edges = np.unique(np.sort(np.concatenate([hull.simplices[:, [0, 1]], hull.simplices[:, [1, 2]],
                                              hull.simplices[:, [0, 2]]]), axis=1), axis=0)  # fmt: skip
    side = p @ normal - offset
    cross = edges[side[edges[:, 0]] * side[edges[:, 1]] < 0.0]
    s0, s1 = side[cross[:, 0]], side[cross[:, 1]]
    t = (s0 / (s0 - s1))[:, None]
    cut = p[cross[:, 0]] + t * (p[cross[:, 1]] - p[cross[:, 0]])
    return np.concatenate([p[side <= 0.0], cut]), np.concatenate([p[side >= 0.0], cut])


def _halve(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = p.mean(axis=0)
    axis = np.linalg.svd(p - centre, full_matrices=False)[2][0]
    return clip(p, axis, float(centre @ axis))


def split_hull(points: np.ndarray, cfg: SplitConfig = SplitConfig()) -> SplitResult:
    """Convex pieces of ``hull(points)`` with at most ``cfg.max_verts`` vertices each, within ``cfg.tol_m``."""
    original = hull_vertices(points)
    stack, pieces, bound = [original], [], 0.0
    while stack:
        region = hull_vertices(stack.pop())
        found = greedy_subset(region, cfg)
        if found is not None:
            sel, dev = found
            pieces.append(region[sel])
            bound = max(bound, dev)
        else:
            stack.extend(_halve(region))
        if len(pieces) + len(stack) > cfg.max_pieces:
            raise TraceError(f"hull of {len(original)} vertices needs more than {cfg.max_pieces} pieces at "
                             f"{cfg.max_verts} vertices / {cfg.tol_m} m")  # fmt: skip
    return SplitResult(tuple(pieces), bound, len(original))


def sampled_deviation(original: np.ndarray, pieces: tuple[np.ndarray, ...], per_face: int = 6,
                      seed: int = 0) -> float:  # fmt: skip
    """Independent check of the bound: max distance from samples of hull(original)'s surface (vertices, edge
    midpoints, ``per_face`` random points per facet) to the union of ``pieces``."""
    v = hull_vertices(original)
    tri = ConvexHull(v).simplices
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(3), size=(len(tri), per_face))
    samples = [v, (v[tri[:, 0]] + v[tri[:, 1]]) / 2, np.einsum("fsk,fkd->fsd", w, v[tri]).reshape(-1, 3)]
    q = np.concatenate(samples)
    dist = np.min([point_hull_distance(q, piece) for piece in pieces], axis=0)
    return float(dist.max())
