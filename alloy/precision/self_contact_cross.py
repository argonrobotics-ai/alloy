# SPDX-License-Identifier: Apache-2.0
"""Did dropped self-contact pairs let two surfaces pass through each other? The meter behind SELF_CONTACT_OVERFLOW's
INVALID level (``self_contact.py`` owns the watch; known gap ``docs/known_gaps/vbd-self-contact-buffer-overflow.md``).

The law (geometric, no tolerance): a world has passed through when, after a row dropped pairs, either

- two triangles of the VBD entry that share no vertex, in the same world, intersect, and at least one of them has a
  vertex whose own vertex row, or an edge row it is an end of, dropped a pair at some detection before; or
- such a vertex is on the other side of a triangle (not its own) than at the detection before, having gone through
  the triangle's inside. Two parallel layers swap sides this way with no triangle ever intersecting another: a 1 kg
  sheet sank through a 20 g one between detections and the first clause saw nothing.

``CrossMeter`` keeps two arrays, both sticky until ``reset``:

- ``marked`` (int32 [particles]): 1 for a particle of a row that dropped a pair: the vertex of a vertex row, both ends
  of an edge row, the three corners of a triangle row.
- ``crossed`` (int32 [W]): the most crossings one detection counted in the world (0: none): intersecting pairs
  plus side changes. A crossing pair with marked vertices on both triangles counts twice.

After every detection, ``mark`` runs once per row family and ``count`` once: one kernel over triangles that returns at
once unless a corner is marked, and else asks the detector's own triangle BVH (refit to the detection's positions
just before, in both of Newton's paths) for the triangles whose boxes overlap this one's.

Intersection test: every edge of one triangle against the other triangle, both ways, with signed volumes in float64
and strict signs. ``wp.intersect_tri_tri`` is not used: it calls two triangles coplanar when a corner's unnormalised
plane distance is under 1e-6, which for 1 cm triangles is anything closer than about 1 cm, and then reports two
stacked, separate sheets as overlapping. With strict signs a corner or edge that only touches the other triangle, or
two coplanar triangles, do not count; a real crossing counts as soon as an edge is through.

Side changes: the positions of the detection before are kept (``_prev``). A marked vertex and a triangle whose
signed volume changed sign count when, at the moment the volume passes zero (positions taken as moving on straight
lines between the two detections), the vertex lies in the triangle, edges included: aligned grids put vertices
exactly over the other sheet's edges. Anything that moved more than ``pad`` between the detections is skipped: the
solver clamps a particle's move between detections to under half its self-contact query radius, so a longer move is
a reset or a teleport, not motion. ``pad`` is that radius; a vertex that went around a free edge inside one such
move would read as a crossing (not seen in the laws' scenes).

Engine rollbacks (a coupled period re-run for some worlds, a watchdog's recover) restore health's per-world arrays
from the period start, ``crossed`` among them (``state_arrays``), and with it ``seen``, a count of detections per
world. The meter keeps its own count beside it: where the two differ at the next detection, that world was put
back, so its marks are cleared and its side changes are not judged once (its particles jumped back). A pass that
was thrown away therefore leaves no crossing; what it can cost is a miss, never a false crossing.

Global particles (world -1): a triangle of theirs that crosses anything blames every world (fail closed, as
``dropped``); a world's marked triangle also tests against the global triangles.

Graph capture: the arrays are made here, at attach; ``mark`` and ``count`` only launch kernels on them, on the
detector's BVH and on the positions the detector itself just used. Nothing is read on the host and nothing in the
simulation is written.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


@wp.kernel
def _mark_dropped(count: wp.array[wp.int32], size: wp.array[wp.int32], row_particles: wp.array2d[wp.int32],
                  marked: wp.array[wp.int32]):
    i = wp.tid()
    if count[i] <= size[i]:
        return
    for k in range(3):
        p = row_particles[i, k]
        if p >= 0:
            marked[p] = 1


@wp.func
def _volume(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d) -> wp.float64:
    return wp.dot(wp.cross(b - a, c - a), d - a)


@wp.func
def _pierces(p: wp.vec3d, q: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> bool:
    """Segment p-q goes through the inside of triangle a-b-c: ends strictly on opposite sides, and the line strictly
    inside all three edges."""
    zero = wp.float64(0.0)
    sp = _volume(a, b, c, p)
    sq = _volume(a, b, c, q)
    if not ((sp > zero and sq < zero) or (sp < zero and sq > zero)):
        return False
    s0 = _volume(p, q, a, b)
    s1 = _volume(p, q, b, c)
    s2 = _volume(p, q, c, a)
    return (s0 > zero and s1 > zero and s2 > zero) or (s0 < zero and s1 < zero and s2 < zero)


@wp.func
def triangles_cross(a0: wp.vec3d, a1: wp.vec3d, a2: wp.vec3d, b0: wp.vec3d, b1: wp.vec3d, b2: wp.vec3d) -> bool:
    if _pierces(a0, a1, b0, b1, b2) or _pierces(a1, a2, b0, b1, b2) or _pierces(a2, a0, b0, b1, b2):
        return True
    return _pierces(b0, b1, a0, a1, a2) or _pierces(b1, b2, a0, a1, a2) or _pierces(b2, b0, a0, a1, a2)


@wp.func
def _crossings(bvh_id: wp.uint64, root: wp.int32, t: wp.int32, pos: wp.array[wp.vec3],
               tri: wp.array2d[wp.int32]) -> wp.int32:
    """Triangles under ``root`` (-1: the whole tree) that share no vertex with ``t`` and cross it."""
    i0, i1, i2 = tri[t, 0], tri[t, 1], tri[t, 2]
    x0, x1, x2 = pos[i0], pos[i1], pos[i2]
    a0, a1, a2 = wp.vec3d(x0), wp.vec3d(x1), wp.vec3d(x2)
    query = wp.bvh_query_aabb(bvh_id, wp.min(x0, wp.min(x1, x2)), wp.max(x0, wp.max(x1, x2)), root)
    u = wp.int32(0)
    n = wp.int32(0)
    while wp.bvh_query_next(query, u):
        j0, j1, j2 = tri[u, 0], tri[u, 1], tri[u, 2]
        if (j0 == i0 or j0 == i1 or j0 == i2 or j1 == i0 or j1 == i1 or j1 == i2
                or j2 == i0 or j2 == i1 or j2 == i2):       # the triangle itself, or a neighbour
            continue
        if triangles_cross(a0, a1, a2, wp.vec3d(pos[j0]), wp.vec3d(pos[j1]), wp.vec3d(pos[j2])):
            n += 1
    return n


@wp.kernel
def _count_crossings(bvh_id: wp.uint64, group_roots: wp.array[wp.int32], pos: wp.array[wp.vec3],
                     tri: wp.array2d[wp.int32], tri_world: wp.array[wp.int32], marked: wp.array[wp.int32],
                     hits: wp.array[wp.int32]):
    t = wp.tid()
    if marked[tri[t, 0]] == 0 and marked[tri[t, 1]] == 0 and marked[tri[t, 2]] == 0:
        return                                      # the usual case: no BVH work
    w = tri_world[t]
    worlds = hits.shape[0]
    last = group_roots.shape[0] - 1                 # the group of global triangles
    if w < 0 or w >= worlds or w >= last:           # a global triangle: any crossing blames every world
        every = _crossings(bvh_id, wp.int32(-1), t, pos, tri)
        if every > 0:
            for k in range(worlds):
                wp.atomic_add(hits, k, every)
        return
    n = wp.int32(0)
    if group_roots[w] >= 0:
        n += _crossings(bvh_id, group_roots[w], t, pos, tri)
    if group_roots[last] >= 0:
        n += _crossings(bvh_id, group_roots[last], t, pos, tri)
    if n > 0:
        wp.atomic_add(hits, w, n)


@wp.func
def _inside(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> bool:
    """The foot of ``p`` on the plane of a-b-c lies in the triangle, edges and corners included."""
    n = wp.cross(b - a, c - a)
    zero = wp.float64(0.0)
    u = wp.dot(wp.cross(b - a, p - a), n)
    v = wp.dot(wp.cross(c - b, p - b), n)
    w = wp.dot(wp.cross(a - c, p - c), n)
    return u >= zero and v >= zero and w >= zero


@wp.func
def _moved(a: wp.vec3, b: wp.vec3, pad: float) -> bool:
    return wp.length_sq(b - a) > pad * pad


@wp.func
def _flips(bvh_id: wp.uint64, root: wp.int32, v: wp.int32, pos: wp.array[wp.vec3], prev: wp.array[wp.vec3],
           tri: wp.array2d[wp.int32], pad: float) -> wp.int32:
    """Triangles under ``root`` (-1: the whole tree), not ``v``'s own, that ``v`` went through since ``prev``."""
    x0, x1 = prev[v], pos[v]
    if _moved(x0, x1, pad):
        return 0
    reach = wp.vec3(pad, pad, pad)
    query = wp.bvh_query_aabb(bvh_id, wp.min(x0, x1) - reach, wp.max(x0, x1) + reach, root)
    p0, p1 = wp.vec3d(x0), wp.vec3d(x1)
    zero = wp.float64(0.0)
    u = wp.int32(0)
    n = wp.int32(0)
    while wp.bvh_query_next(query, u):
        j0, j1, j2 = tri[u, 0], tri[u, 1], tri[u, 2]
        if j0 == v or j1 == v or j2 == v:
            continue
        if _moved(prev[j0], pos[j0], pad) or _moved(prev[j1], pos[j1], pad) or _moved(prev[j2], pos[j2], pad):
            continue
        a0, b0, c0 = wp.vec3d(prev[j0]), wp.vec3d(prev[j1]), wp.vec3d(prev[j2])
        a1, b1, c1 = wp.vec3d(pos[j0]), wp.vec3d(pos[j1]), wp.vec3d(pos[j2])
        s0 = _volume(a0, b0, c0, p0)
        s1 = _volume(a1, b1, c1, p1)
        if not ((s0 > zero and s1 < zero) or (s0 < zero and s1 > zero)):
            continue
        t = s0 / (s0 - s1)                          # where the volume passes zero, taken as linear in between
        if _inside(p0 + t * (p1 - p0), a0 + t * (a1 - a0), b0 + t * (b1 - b0), c0 + t * (c1 - c0)):
            n += 1
    return n


@wp.kernel
def _count_flips(bvh_id: wp.uint64, group_roots: wp.array[wp.int32], pos: wp.array[wp.vec3],
                 prev: wp.array[wp.vec3], tri: wp.array2d[wp.int32], particle_world: wp.array[wp.int32],
                 marked: wp.array[wp.int32], restored: wp.array[wp.int32], pad: float, hits: wp.array[wp.int32]):
    v = wp.tid()
    if marked[v] == 0:
        return                                      # the usual case: no BVH work
    w = particle_world[v]
    worlds = hits.shape[0]
    if w >= 0 and w < worlds:
        if restored[w] != 0:                        # put back by a rollback: ``prev`` is the abandoned pass's
            return
    last = group_roots.shape[0] - 1
    if w < 0 or w >= worlds or w >= last:           # a global particle: blames every world
        every = _flips(bvh_id, wp.int32(-1), v, pos, prev, tri, pad)
        if every > 0:
            for k in range(worlds):
                wp.atomic_add(hits, k, every)
        return
    n = wp.int32(0)
    if group_roots[w] >= 0:
        n += _flips(bvh_id, group_roots[w], v, pos, prev, tri, pad)
    if group_roots[last] >= 0:
        n += _flips(bvh_id, group_roots[last], v, pos, prev, tri, pad)
    if n > 0:
        wp.atomic_add(hits, w, n)


@wp.kernel
def _sync_restored(seen: wp.array[wp.int32], mine: wp.array[wp.int32], restored: wp.array[wp.int32]):
    """Per world: was ``seen`` (rolled back with health) put back since the last detection? Then count one more."""
    w = wp.tid()
    restored[w] = wp.where(seen[w] != mine[w], 1, 0)
    seen[w] = mine[w] + 1
    mine[w] = mine[w] + 1


@wp.kernel
def _clear_restored_marks(restored: wp.array[wp.int32], particle_world: wp.array[wp.int32],
                          marked: wp.array[wp.int32]):
    i = wp.tid()
    w = particle_world[i]
    if w >= 0 and w < restored.shape[0]:
        if restored[w] != 0:
            marked[i] = 0


@wp.kernel
def _fold_hits(hits: wp.array[wp.int32], crossed: wp.array[wp.int32]):
    w = wp.tid()
    crossed[w] = wp.max(crossed[w], hits[w])
    hits[w] = 0


@wp.kernel
def _clear_worlds(mask: wp.array[wp.int32], all_worlds: int, crossed: wp.array[wp.int32]):
    w = wp.tid()
    if all_worlds != 0 or mask[w] != 0:
        crossed[w] = 0


@wp.kernel
def _clear_marks(mask: wp.array[wp.int32], all_worlds: int, particle_world: wp.array[wp.int32],
                 marked: wp.array[wp.int32]):
    """A global particle's mark goes only when every world is reset (it may have dropped pairs of any world)."""
    i = wp.tid()
    if all_worlds != 0:
        marked[i] = 0
        return
    w = particle_world[i]
    if w >= 0 and w < mask.shape[0]:
        if mask[w] != 0:
            marked[i] = 0


def row_particles(rows: int, elements: np.ndarray | None, columns: tuple[int, ...]) -> np.ndarray:
    """int32 [rows, 3]: the particles a row marks when it drops a pair, padded with -1. ``elements`` None: row i is
    particle i's own row."""
    out = np.full((rows, 3), -1, dtype=np.int32)
    if elements is None:
        out[:, 0] = np.arange(rows, dtype=np.int32)
    else:
        out[:, :len(columns)] = elements[:, list(columns)]
    return out


class CrossMeter:
    """``marked`` and ``crossed`` of one triangle set (module docstring). ``tri``: int32 [T, 3] on the device;
    ``particle_world``: host int32 [particles]; ``start``: the particles' positions at attach (zeros when None)."""

    def __init__(self, tri: wp.array, particle_world: np.ndarray, world_count: int, device: Any,
                 start: wp.array | None = None) -> None:
        self.device, self.tri = device, tri
        worlds = max(int(world_count), 1)
        pw = np.ascontiguousarray(particle_world, dtype=np.int32)
        first = tri.numpy()[:, 0] if tri.shape[0] else np.zeros(0, dtype=np.int64)
        self._particle_world = wp.array(pw, dtype=wp.int32, device=device)
        self._tri_world = wp.array(pw[first].astype(np.int32), dtype=wp.int32, device=device)
        self.marked = wp.zeros(len(pw), dtype=wp.int32, device=device)
        self.crossed = wp.zeros(worlds, dtype=wp.int32, device=device)
        self._hits = wp.zeros(worlds, dtype=wp.int32, device=device)
        self._prev = wp.zeros(len(pw), dtype=wp.vec3, device=device)    # positions at the detection before
        self.seen = wp.zeros(worlds, dtype=wp.int32, device=device)      # detections per world, rolled back
        self._mine = wp.zeros(worlds, dtype=wp.int32, device=device)     # the same count, never rolled back
        self._restored = wp.zeros(worlds, dtype=wp.int32, device=device)
        if start is not None and start.shape[0] == len(pw):
            wp.copy(self._prev, start)

    def state_arrays(self) -> dict[str, wp.array]:
        """The per-world arrays an engine rollback must restore with health's own (module docstring)."""
        return {"crossed": self.crossed, "seen": self.seen}

    def begin(self) -> None:
        """At a detection, first: find the worlds a rollback put back since the last one and clear their marks."""
        wp.launch(_sync_restored, dim=self.seen.shape[0], inputs=[self.seen, self._mine],
                  outputs=[self._restored], device=self.device)
        wp.launch(_clear_restored_marks, dim=self.marked.shape[0], inputs=[self._restored, self._particle_world],
                  outputs=[self.marked], device=self.device)

    def mark(self, count: wp.array, size: wp.array, particles: wp.array) -> None:
        """Mark the particles of every row with ``count > size`` (``particles``: int32 [rows, 3], -1 = none)."""
        wp.launch(_mark_dropped, dim=count.shape[0], inputs=[count, size, particles], outputs=[self.marked],
                  device=self.device)

    def count(self, bvh_id: Any, group_roots: wp.array, pos: wp.array, pad: float = 0.0) -> None:
        """Fold this detection's crossings into ``crossed``. ``bvh_id``, ``group_roots``: the triangle BVH, refit to
        ``pos``, and its per-world group roots ([worlds + 1], the last one the global group; -1 = empty). ``pad``:
        the most a particle moves between two detections (m); 0 leaves the side changes out."""
        if not self.tri.shape[0]:
            return
        wp.launch(_count_crossings, dim=self.tri.shape[0],
                  inputs=[bvh_id, group_roots, pos, self.tri, self._tri_world, self.marked], outputs=[self._hits],
                  device=self.device)
        if pad > 0.0:
            wp.launch(_count_flips, dim=self.marked.shape[0],
                      inputs=[bvh_id, group_roots, pos, self._prev, self.tri, self._particle_world, self.marked,
                              self._restored, float(pad)], outputs=[self._hits], device=self.device)
        wp.copy(self._prev, pos)
        wp.launch(_fold_hits, dim=self.crossed.shape[0], inputs=[self._hits], outputs=[self.crossed],
                  device=self.device)

    def keep(self) -> None:
        """Remember ``crossed``, ``marked`` and the positions of the detection before as they stand (device copies),
        for ``restore``: a period that is run again must not keep what its abandoned attempt saw."""
        if not hasattr(self, "_kept"):
            self._kept = tuple(wp.empty_like(a) for a in self._own())
        for dst, src in zip(self._kept, self._own()):
            wp.copy(dst, src)

    def restore(self) -> None:
        """Put back what ``keep`` saw (nothing kept yet: nothing to undo)."""
        for dst, src in zip(self._own(), getattr(self, "_kept", ())):
            wp.copy(dst, src)

    def _own(self) -> tuple[wp.array, ...]:
        """Everything ``keep`` / ``restore`` copy: both detection counts too, so a kept-and-restored period does not
        read as an engine rollback."""
        return self.crossed, self.marked, self._prev, self.seen, self._mine

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Forget the masked worlds' crossings and their particles' marks (all when None)."""
        everything = int(world_mask is None)
        mask = world_mask if world_mask is not None else self.crossed
        wp.launch(_clear_worlds, dim=min(int(mask.shape[0]), self.crossed.shape[0]), inputs=[mask, everything],
                  outputs=[self.crossed], device=self.device)
        wp.launch(_clear_marks, dim=self.marked.shape[0], inputs=[mask, everything, self._particle_world],
                  outputs=[self.marked], device=self.device)


__all__ = ["CrossMeter", "row_particles", "triangles_cross"]
