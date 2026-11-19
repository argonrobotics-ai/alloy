# SPDX-License-Identifier: Apache-2.0
"""Crossing meter for liquid against a cloth or film sheet: pure numpy, no simulation.

``side`` gives each particle's signed side of a triangle mesh (the sign of its offset from the closest point on the
nearest triangle, dotted with that triangle's normal; the faces of ``sim.faces(name)`` wind consistently), the
distance, and whether that closest point is on the sheet's free border (a border edge or a border vertex).
``crossings`` follows particles over frames: a particle latches its side when it first comes within ``latch``
of the sheet; a later frame on the opposite side, with the closest point NOT on the border and farther than ``tol``,
is a crossing through the sheet. A particle seen on the far side with its closest point on the border went round the
free edge ("edge"); a particle that went round the edge is tracked on its new side, so a later crossing
through the sheet's interior still counts as "through". Brute force over particles x faces, chunked: fine for a
few thousand of each.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_CHUNK = 192


@dataclass
class SideResult:
    """Per particle: side (+1 along the face normal, -1 against), distance (m), closest point on the border, face."""

    side: np.ndarray
    dist: np.ndarray
    border: np.ndarray
    face: np.ndarray


@dataclass
class CrossingCounts:
    """through / edge / kept over the tracked particles (those that came within ``latch`` of the sheet)."""

    through: int
    edge: int
    kept: int
    through_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    through_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    first_through_frame: int = -1

    @property
    def tracked(self) -> int:
        return self.through + self.edge + self.kept


def border_flags(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(edge_border (F,3) for edges v0v1, v1v2, v2v0; vertex_border (V,)): edges used by one face only."""
    f = np.asarray(faces, dtype=np.int64)
    e = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)          # (F, 3, 2)
    key = np.sort(e, axis=2) @ np.array([f.max() + 1, 1])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    edge_border = (cnt[inv.reshape(key.shape)] == 1)
    vb = np.zeros(int(f.max()) + 1, bool)
    vb[e[edge_border].ravel()] = True
    return edge_border, vb


def winding_consistent(faces: np.ndarray) -> bool:
    """True when every interior edge is traversed in opposite directions by its two faces."""
    f = np.asarray(faces, dtype=np.int64)
    d = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n = int(f.max()) + 1
    key = d[:, 0] * n + d[:, 1]
    return len(np.unique(key)) == len(key)


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...i->...", u, v)


def closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on triangles abc to p (broadcast) and the feature: 0 face, 1/2/3 edge ab/bc/ca, 4/5/6 a/b/c."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp_ = p - c
    d5, d6 = _dot(ab, cp_), _dot(ac, cp_)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    tiny = 1e-30
    den = va + vb + vc
    den = np.where(np.abs(den) < tiny, tiny, den)
    q = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]
    feat = np.zeros(q.shape[:-1], np.int8)
    tests = [   # reverse priority: later (earlier-listed) overrides
        (3, (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
         lambda: b + (c - b) * ((d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < tiny, tiny,
                                                      (d4 - d3) + (d5 - d6)))[..., None]),
        (3, (vb <= 0) & (d2 >= 0) & (d6 <= 0),
         lambda: a + ac * (d2 / np.where(d2 - d6 == 0, tiny, d2 - d6))[..., None]),
        (1, (vc <= 0) & (d1 >= 0) & (d3 <= 0),
         lambda: a + ab * (d1 / np.where(d1 - d3 == 0, tiny, d1 - d3))[..., None]),
    ]
    feats = [2, 3, 1]
    for (_, m, fn), code in zip(tests, feats):
        q = np.where(m[..., None], fn(), q)
        feat = np.where(m, code, feat)
    for m, pt, code in (((d6 >= 0) & (d5 <= d6), c, 6), ((d3 >= 0) & (d4 <= d3), b, 5), ((d1 <= 0) & (d2 <= 0), a, 4)):
        q = np.where(m[..., None], pt, q)
        feat = np.where(m, code, feat)
    return q, feat


def side(points: np.ndarray, sheet_vertices: np.ndarray, faces: np.ndarray) -> SideResult:
    """Signed side, distance and border flag of each point (N,3) against the sheet (V,3) with faces (F,3)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(sheet_vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normal = np.cross(b - a, c - a)
    edge_border, vert_border = border_flags(f)
    n = len(pts)
    out = SideResult(np.ones(n, np.int8), np.zeros(n), np.zeros(n, bool), np.zeros(n, np.int64))
    for s in range(0, n, _CHUNK):
        p = pts[s:s + _CHUNK, None, :]
        q, feat = closest_on_triangles(p, a[None], b[None], c[None])
        d = np.linalg.norm(p - q, axis=-1)
        k = np.argmin(d, axis=1)
        r = np.arange(len(k))
        off = pts[s:s + _CHUNK] - q[r, k]
        sgn = _dot(off, normal[k])
        out.side[s:s + _CHUNK] = np.where(sgn >= 0, 1, -1)
        out.dist[s:s + _CHUNK] = d[r, k]
        out.face[s:s + _CHUNK] = k
        out.border[s:s + _CHUNK] = _on_border(feat[r, k], f[k], edge_border[k], vert_border)
    return out


def _on_border(feat: np.ndarray, tri: np.ndarray, edge_border: np.ndarray, vert_border: np.ndarray) -> np.ndarray:
    r = np.arange(len(feat))
    on_edge = edge_border[r, np.clip(feat - 1, 0, 2)] & (feat >= 1) & (feat <= 3)
    on_vert = vert_border[tri[r, np.clip(feat - 4, 0, 2)]] & (feat >= 4)
    return on_edge | on_vert


def crossings(traj_points: np.ndarray, traj_sheet: np.ndarray, faces: np.ndarray, active: np.ndarray | None = None,
              *, latch: float, tol: float) -> CrossingCounts:
    """Count particles that went through the sheet. traj_points (F,N,3), traj_sheet (F,V,3) (or (V,3), fixed),
    active (F,N) bool (default all). ``latch``: distance (m) within which a particle's side is first recorded;
    ``tol``: distance (m) on the far side below which a flip is not yet called a crossing."""
    pts = np.asarray(traj_points)
    nf, n = pts.shape[:2]
    sheet = np.asarray(traj_sheet)
    act = np.ones((nf, n), bool) if active is None else np.asarray(active, bool)
    latched = np.zeros(n, np.int8)
    through = np.zeros(n, bool)
    edge = np.zeros(n, bool)
    first = np.full(n, -1, np.int64)
    for i in range(nf):
        idx = np.nonzero(act[i] & ~through)[0]
        if len(idx) == 0:
            continue
        res = side(pts[i][idx], sheet if sheet.ndim == 2 else sheet[i], faces)
        near = (latched[idx] == 0) & (res.dist < latch)
        latched[idx[near]] = res.side[near]
        far = (latched[idx] != 0) & (res.side != latched[idx]) & (res.dist > tol)
        went_edge = far & res.border
        edge[idx[went_edge]] = True
        latched[idx[went_edge]] = res.side[went_edge]      # round the border: track it on its new side from here
        crossed = far & ~res.border
        through[idx[crossed]] = True
        first[idx[crossed]] = i
    ids = np.nonzero(through)[0]
    n_through, n_edge = int(through.sum()), int((edge & ~through).sum())
    kept = int(((latched != 0) & ~through & ~edge).sum())
    return CrossingCounts(n_through, n_edge, kept, ids, first[ids], int(first[ids].min()) if len(ids) else -1)
