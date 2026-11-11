# SPDX-License-Identifier: Apache-2.0
"""Long-range stretch limits: the multi-scale half of the strain-limit projection.

Per-triangle limits (strain.py) are local: a Jacobi sweep moves a correction one triangle ring, so a film carrying
its weight over 60 rows stays stretched after any affordable number of sweeps (measured: docs/notes/sc8-film-
design.md §3). Like Wang et al. 2010's multi-resolution limiting and Kim et al. 2012's long-range attachments, each
film particle also gets unilateral distance limits |x_i − x_a| ≤ (1 + cap)·d_ia to a few anchor particles per
scale (anchor spacing 4, 8, 16, ... rest edges, Poisson-disk sampled in index order), where d_ia is the rest
distance along the surface: the straight rest distance when the shortest mesh path is within ``straight_slack`` of
it (a flat patch), else the pair is skipped (folded rest shapes: a straight distance would forbid unfolding).
The limits only act on stretch (inactive in compression and folding) and move both ends mass-weighted (momentum
kept); pinned ends stay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class PairLimits:
    pairs: np.ndarray     # [C, 2] int32 particle ids
    rest: np.ndarray      # [C] float32 rest surface distance
    cap: np.ndarray       # [C] float32 strain cap of the pair (min of the two ends' triangles)
    levels: tuple[float, ...]


def _mesh_graph(x: np.ndarray, tris: np.ndarray):
    from scipy.sparse import coo_matrix

    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
    n = len(x)
    g = coo_matrix((np.concatenate([w, w]), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                   shape=(n, n)).tocsr()
    return g, float(np.median(w)) if len(w) else 0.0


def _anchors(x: np.ndarray, spacing: float) -> np.ndarray:
    """Poisson-disk anchors in index order (deterministic): a particle becomes an anchor when no anchor is within
    ``spacing`` of it."""
    pts = np.zeros((len(x), 3))
    chosen: list[int] = []
    s2 = spacing * spacing
    for i in range(len(x)):
        n = len(chosen)
        if n and float(np.min(((pts[:n] - x[i]) ** 2).sum(1))) < s2:
            continue
        pts[n] = x[i]
        chosen.append(i)
    return np.asarray(chosen, dtype=np.int64)


def build_pair_limits(x: np.ndarray, tris: np.ndarray, particle_cap: np.ndarray, *, per_level: int = 3,
                      first: int = 4, straight_slack: float = 1.10, max_levels: int = 6) -> PairLimits:
    """Pair limits on one film mesh: rest positions ``x`` [N, 3] (local ids), triangles [T, 3], per-particle cap."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    g, h = _mesh_graph(x, tris)
    if h <= 0.0:
        return PairLimits(np.zeros((0, 2), np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32), ())
    diam = float(np.linalg.norm(x.max(0) - x.min(0)))
    pairs, rest, levels = [], [], []
    spacing = first * h
    while spacing < diam and len(levels) < max_levels:
        anc = _anchors(x, spacing)
        if len(anc) < 2:
            break
        levels.append(spacing)
        tree = cKDTree(x[anc])
        k = min(per_level, len(anc))
        dist, idx = tree.query(x, k=k, distance_upper_bound=2.0 * spacing)
        dist, idx = np.atleast_2d(dist.T).T, np.atleast_2d(idx.T).T
        for c in range(0, len(anc), 256):
            block = anc[c:c + 256]
            geo = dijkstra(g, indices=block, limit=2.6 * spacing)
            for col in range(idx.shape[1]):
                sel = (idx[:, col] >= c) & (idx[:, col] < c + len(block))
                for i in np.flatnonzero(sel):
                    a = int(anc[idx[i, col]])
                    if a == i or not np.isfinite(dist[i, col]):
                        continue
                    straight = float(dist[i, col])
                    if straight < 1.5 * h or geo[idx[i, col] - c, i] > straight_slack * straight:
                        continue
                    pairs.append((min(i, a), max(i, a)))
                    rest.append(straight)
        spacing *= 2.0
    if not pairs:
        return PairLimits(np.zeros((0, 2), np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32), tuple(levels))
    P = np.asarray(pairs, np.int64)
    P, first_idx = np.unique(P, axis=0, return_index=True)
    R = np.asarray(rest, np.float64)[first_idx]
    cap = np.minimum(particle_cap[P[:, 0]], particle_cap[P[:, 1]])
    keep = cap > 0.0
    return PairLimits(P[keep].astype(np.int32), R[keep].astype(np.float32), cap[keep].astype(np.float32),
                      tuple(levels))


@wp.kernel(enable_backward=False)
def pair_corrections(
    x: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    pairs: wp.array2d[wp.int32],
    rest: wp.array[float],
    cap: wp.array[float],
    mass_ratio_cap: float,
    gate_start: wp.array[wp.int32],
    gate: wp.array[wp.int32],
    bond_intact: wp.array[wp.int8],
    delta: wp.array2d[wp.vec3],
    active: wp.array[wp.int32],
):
    c = wp.tid()
    delta[c, 0] = wp.vec3(0.0)
    delta[c, 1] = wp.vec3(0.0)
    active[c] = 0
    for g in range(gate_start[c], gate_start[c + 1]):  # a pair spanning a broken seam (seams.py) lets go
        if bond_intact[gate[g]] == wp.int8(0):
            return
    i = pairs[c, 0]
    j = pairs[c, 1]
    d = x[j] - x[i]
    L = wp.length(d)
    Lmax = rest[c] * (1.0 + cap[c])
    if L <= Lmax or L < 1.0e-12:
        return
    wi = inv_mass[i]
    wj = inv_mass[j]
    if mass_ratio_cap > 0.0 and wi > 0.0 and wj > 0.0:
        wmin = wp.max(wi, wj) / mass_ratio_cap
        wi = wp.max(wi, wmin)
        wj = wp.max(wj, wmin)
    ws = wi + wj
    if ws == 0.0:
        return
    corr = d * ((L - Lmax) / (L * ws))
    delta[c, 0] = corr * wi
    delta[c, 1] = -corr * wj
    active[c] = 1
