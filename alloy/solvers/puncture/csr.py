# SPDX-License-Identifier: Apache-2.0
"""Particle -> anchor lists for the per-colour gather, and the per-needle loads after a step (device, deterministic).

Each active anchor touches up to 6 particles (a SWAGE tie only the thread's first): its tet's 4 vertices, plus for a
THREAD anchor the two particles of its segment (fixed within the step), or for a STOPPER the thread's last two.
Lists are rebuilt once per step after the events: per-world serial count and fill in anchor index order around a
device scan, so every row lists its anchors in increasing index (no atomics, bitwise repeatable).
"""

from __future__ import annotations

import warp as wp

from . import needle as N
from .events import Tracks
from .needle import Needles
from .terms import FREE, NEEDLE, STOPPER, SWAGE, THREAD, TIP, Anchors, Tissue, material_point

SLOTS = 6


@wp.func
def anchor_vertex(j: int, q: int, an: Anchors, ts: Tissue, nd: Needles) -> int:
    st = an.state[j]
    if st == FREE:
        return -1
    i = an.needle[j]
    if st == SWAGE:
        if q == 0:
            return nd.thread_start[i] + an.seg[j]
        return -1
    if q < 4:
        return ts.tet_indices[an.tet[j], q]
    first = nd.thread_start[i]
    end = first + nd.thread_count[i]
    if st == THREAD and q < 6:
        return an.seg[j] + (q - 4)
    if st == STOPPER and q < 6:
        return end - 6 + q
    return -1


@wp.kernel(enable_backward=False)
def count(an: Anchors, ts: Tissue, nd: Needles, tr: Tracks, counts: wp.array[wp.int32]):
    w = wp.tid()
    for j in range(w * tr.capacity, w * tr.capacity + tr.used[w]):
        for q in range(SLOTS):
            v = anchor_vertex(j, q, an, ts, nd)
            if v >= 0:
                counts[v] = counts[v] + 1


@wp.kernel(enable_backward=False)
def fill(an: Anchors, ts: Tissue, nd: Needles, tr: Tracks, cursor: wp.array[wp.int32], ent: wp.array[wp.int32]):
    w = wp.tid()
    for j in range(w * tr.capacity, w * tr.capacity + tr.used[w]):
        for q in range(SLOTS):
            v = anchor_vertex(j, q, an, ts, nd)
            if v >= 0:
                ent[cursor[v]] = j
                cursor[v] = cursor[v] + 1


@wp.struct
class Loads:
    force: wp.array[wp.vec3]      # per needle: force of the tissue on the needle [N]
    torque: wp.array[wp.vec3]     # per needle: torque of it about the needle body origin [N·m]
    axial: wp.array[float]        # per needle: resistance along the tip axis (> 0 opposes insertion) [N]
    thread: wp.array[wp.vec3]     # per needle: force of the tissue on the thread [N]
    anchors: wp.array[wp.int32]   # per needle: active anchors
    length: wp.array[float]       # per needle: inserted length, the largest arc length of a NEEDLE anchor [m]
    offset: wp.array[float]       # per world: largest anchor distance from its target curve [m]


@wp.kernel(enable_backward=False)
def loads(an: Anchors, ts: Tissue, nd: Needles, tr: Tracks, x: wp.array[wp.vec3], body_q: wp.array[wp.transform],
          f_anchor: wp.array[wp.vec3], off: wp.array[float], ld: Loads):
    w = wp.tid()
    for i in range(tr.needle_start[w], tr.needle_start[w + 1]):
        ld.force[i] = wp.vec3(0.0)
        ld.torque[i] = wp.vec3(0.0)
        ld.axial[i] = 0.0
        ld.thread[i] = wp.vec3(0.0)
        ld.anchors[i] = 0
        ld.length[i] = 0.0
    big = float(0.0)
    for j in range(w * tr.capacity, w * tr.capacity + tr.used[w]):
        st = an.state[j]
        if st == FREE:
            continue
        i = an.needle[j]
        ld.anchors[i] = ld.anchors[i] + 1
        big = wp.max(big, off[j])
        f = f_anchor[j]
        if st == SWAGE:
            X = N.pose(nd, i, body_q)
            ld.force[i] = ld.force[i] - f
            ld.torque[i] = ld.torque[i] + wp.cross(N.point_world(nd, i, X, nd.length[i] + an.alpha[j])
                                                   - wp.transform_get_translation(X), -f)
        elif st == TIP or st == NEEDLE:
            X = N.pose(nd, i, body_q)
            p = material_point(ts, an.tet[j], an.w[j], x)
            ld.force[i] = ld.force[i] - f
            ld.torque[i] = ld.torque[i] + wp.cross(p - wp.transform_get_translation(X), -f)
            ld.axial[i] = ld.axial[i] + wp.dot(f, N.tip_axis_world(nd, i, X))
            if st == NEEDLE:
                ld.length[i] = wp.max(ld.length[i], N.project_world(nd, i, X, p))
        else:
            ld.thread[i] = ld.thread[i] - f
    ld.offset[w] = big


@wp.kernel(enable_backward=False)
def mark_components(an: Anchors, ts: Tissue, nd: Needles, comp_of: wp.array[wp.int32], blocked: wp.array[wp.int32]):
    """Components touched by an active anchor exchange momentum with the needle: block their coarse step."""
    j = wp.tid()
    if an.state[j] == FREE:
        return
    v = ts.tet_indices[an.tet[j], 0]
    if an.state[j] == SWAGE:
        v = nd.thread_start[an.needle[j]] + an.seg[j]
    c = comp_of[v]
    if c >= 0:
        blocked[c] = 1
