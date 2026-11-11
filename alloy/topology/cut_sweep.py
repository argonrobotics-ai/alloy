# SPDX-License-Identifier: Apache-2.0
"""Opened-face deviation from the blade's swept surface: SB-7 row ``cut_face_deviation`` (WP10 item 9 step 2).

A pre-split block can only open on its seams, so the cut it makes snaps to seam faces wherever the blade goes
between them. This diagnostic measures, per world, how far each twin that opened during the episode lies from the
surface the blade's cutting edge swept in the episode, and reports the max over the world's opened twins (m). Each
twin keeps its distance to the sweep so far, so a twin that opened ahead of the edge stops counting its lead once the
edge passes it. Rows are WP7's (``alloy.bully.severity``); the row has no severity while seams are pre-split, and
item 9 step 3 (virtual-node copies) is checked against it.

Geometry.
- A blade's cutting edge is its body-frame segment (``Blade.edge_a``, ``Blade.edge_b``) at the body's world pose.
- ``SweptEdges.sample`` records every blade's edge at the current state. Two consecutive samples A_k B_k and
  A_k+1 B_k+1 bound a quad, split into two triangles. The swept surface is their union. It is exact for an edge
  that translates linearly between samples, as WP7's pose writer moves it within a control period.
- The distance from a point to a triangle is exact (``point_triangle_distance``). Before a second sample exists, the
  surface is the edge segment itself.

Frame. A twin's point is its rest position (``model.particle_q``; twins coincide at rest). The class block is
clamped and gravity-free, so rest and world frames agree away from the blade. The row therefore also counts the
material's elastic displacement when the twin opened: the push ahead of the edge.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alloy.topology.blade_contact import quat_rotate
from alloy.topology.types import Blade, BondKind, BondSet

CHUNK = 1024           # points per distance batch (bounds the [points, triangles] scratch)
_TINY_AREA2 = 1.0e-30  # [m^4] squared doubled area below which a triangle is treated as its edges


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...i->...", u, v)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p to segments [a, b] (broadcast over leading axes; [..., 3] in, [...] out)."""
    ab = b - a
    t = np.clip(_dot(p - a, ab) / np.maximum(_dot(ab, ab), 1.0e-300), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact distance from points p to triangles (a, b, c), broadcast over leading axes. A degenerate triangle
    (a segment or a point) is measured by its edges."""
    n = np.cross(b - a, c - a)
    nn = _dot(n, n)
    edges = np.minimum(np.minimum(point_segment_distance(p, a, b), point_segment_distance(p, b, c)),
                       point_segment_distance(p, c, a))
    inside = ((_dot(np.cross(b - a, p - a), n) >= 0.0) & (_dot(np.cross(c - b, p - b), n) >= 0.0)
              & (_dot(np.cross(a - c, p - c), n) >= 0.0) & (nn > _TINY_AREA2))
    plane = np.abs(_dot(p - a, n)) / np.sqrt(np.maximum(nn, _TINY_AREA2))
    return np.where(inside, plane, edges)


def swept_distance(points: np.ndarray, edges: np.ndarray, first: np.ndarray) -> np.ndarray:
    """[n] distance from points [n, 3] to the surface swept by an edge sampled K times, ``edges`` [n, K, 2, 3] (each
    point with its own blade's samples), counting samples from ``first`` [n] on (its episode start)."""
    p = points[:, None, :]
    k = np.arange(edges.shape[1])[None, :]
    seg = point_segment_distance(p, edges[:, :, 0], edges[:, :, 1])
    d = np.where(k >= first[:, None], seg, np.inf).min(axis=1)
    if edges.shape[1] == 1:
        return d
    a0, b0, a1, b1 = edges[:, :-1, 0], edges[:, :-1, 1], edges[:, 1:, 0], edges[:, 1:, 1]
    quad = np.minimum(point_triangle_distance(p, a0, b0, b1), point_triangle_distance(p, a0, b1, a1))
    return np.minimum(d, np.where(k[:, :-1] >= first[:, None], quad, np.inf).min(axis=1))


def blade_slots(blade_world: np.ndarray, world_count: int) -> np.ndarray:
    """[W, m] blade indices per world, -1 padded (m = most blades in one world, at least 1)."""
    counts = np.bincount(blade_world, minlength=world_count)
    slots = np.full((world_count, max(1, int(counts.max(initial=0)))), -1, np.int64)
    fill = np.zeros(world_count, np.int64)
    for j, w in enumerate(blade_world):
        slots[w, fill[w]] = j
        fill[w] += 1
    return slots


def twin_points(bonds: BondSet, rest: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bond indices, worlds, rest points [N, 3]) of the TET_CUT twins: a twin's point is its pair's rest midpoint
    (twins coincide at rest)."""
    cut = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT))
    rest = np.asarray(rest, np.float64)
    point = 0.5 * (rest[bonds.a.numpy()[cut]] + rest[bonds.b.numpy()[cut]])
    return cut, bonds.world.numpy()[cut].astype(np.int64), point


class SweptEdges:
    """Per world, the max distance of the points opened in the episode (twins, or a split tet's cut-face corners)
    from the surface their world's blades swept in the episode so far. A point that opens ahead of the edge reads its
    lead until the edge passes it. NaN: a point opened in a world without a blade."""

    def __init__(self, blades: Sequence[Blade], body_world: np.ndarray, point_world: np.ndarray,
                 world_count: int) -> None:
        self.body = np.array([b.body for b in blades], np.int64)
        self.edge = np.array([[b.edge_a, b.edge_b] for b in blades], np.float64).reshape(-1, 2, 3)
        self.slots = blade_slots(np.asarray(body_world)[self.body].astype(np.int64), world_count)
        self.point_world = np.asarray(point_world, np.int64)
        self.point = np.zeros((len(self.point_world), 3))
        self.world_count = int(world_count)
        self.samples: list[np.ndarray] = []
        self.first = np.zeros(self.world_count, np.int64)     # per world, its episode's first sample
        self.opened = np.zeros(len(self.point_world), bool)
        self.dist = np.full(len(self.point_world), np.inf)    # per opened point, distance to the sweep so far

    def reset(self, body_q: np.ndarray, worlds: np.ndarray | None = None) -> None:
        """Start a new episode for ``worlds`` (bool [W], default all) at the current poses: clear their opened points
        and edge history."""
        sel = np.ones(self.world_count, bool) if worlds is None else np.asarray(worlds, bool)
        mine = sel[self.point_world]
        self.opened[mine] = False
        self.dist[mine] = np.inf
        self.sample(body_q)
        self.first[sel] = len(self.samples) - 1
        drop = int(self.first.min())
        if drop:
            del self.samples[:drop]
            self.first -= drop

    def edges_at(self, body_q: np.ndarray) -> np.ndarray:
        """[blades, 2, 3] world edge endpoints at body poses ``body_q`` [bodies, 7] (x, y, z, qx, qy, qz, qw)."""
        q = np.asarray(body_q, np.float64)[self.body]
        return q[:, None, :3] + quat_rotate(q[:, None, 3:7], self.edge)

    def sample(self, body_q: np.ndarray) -> None:
        self.samples.append(self.edges_at(body_q))

    def update(self, body_q: np.ndarray, opened: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Sample the edges; points newly ``opened`` [N] take ``points`` [N, 3] and are measured against the
        episode's sweep, the others against the newest quad. Returns [W] the max over each world's opened points (0
        before any opens)."""
        self.sample(body_q)
        opened = np.asarray(opened, bool)
        old = np.flatnonzero(self.opened)
        if old.size and len(self.samples) >= 2:
            self.dist[old] = np.minimum(self.dist[old], self._distance(old, len(self.samples) - 2))
        new = np.flatnonzero(opened & ~self.opened)
        if new.size:
            self.point[new] = np.asarray(points, np.float64)[new]
            self.dist[new] = self._distance(new, 0)
        self.opened |= opened
        dev = np.zeros(self.world_count)
        np.maximum.at(dev, self.point_world[self.opened], self.dist[self.opened])
        return np.where(np.isinf(dev), np.nan, dev)

    def _distance(self, idx: np.ndarray, lo: int) -> np.ndarray:
        """[n] distance of points ``idx`` to their world's sweep over samples lo, lo + 1, ... (inf: no blade)."""
        hist = np.stack(self.samples[lo:], axis=1)                    # [blades, K, 2, 3]
        world = self.point_world[idx]
        first = np.maximum(self.first[world] - lo, 0)
        out = np.full(idx.size, np.inf)
        for j in range(self.slots.shape[1]):
            blade = self.slots[world, j]
            has = np.flatnonzero(blade >= 0)
            for c in range(0, has.size, CHUNK):
                k = has[c:c + CHUNK]
                out[k] = np.minimum(out[k], swept_distance(self.point[idx[k]], hist[blade[k]], first[k]))
        return out
