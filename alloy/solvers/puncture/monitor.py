# SPDX-License-Identifier: Apache-2.0
"""Untracked needle depth (design note §4; the per-world depth WP5's plug-in health check reads, ruling item 4).

A needle inside tissue always has track anchors every h_a along its path. ``SAMPLES`` points along each needle's
centreline are located in the tissue tets (one thread per tissue tet, idempotent integer writes); a sample inside
tissue farther than ``r_track`` (h_a + the wire radius) from every active needle-borne anchor of its needle is
tunnelling. Per needle: the count of such samples and the largest distance to the nearest anchor; per world: that
distance (m), 0 when clean.
"""

from __future__ import annotations

import warp as wp

from . import needle as N
from .needle import Needles
from .terms import NEEDLE, THREAD, TIP, Anchors, Tissue, material_point
from .topology import Surface

SAMPLES = 32


@wp.func
def _bary_inside(ts: Tissue, tet: int, q: wp.vec3, x: wp.array[wp.vec3]) -> bool:
    a = x[ts.tet_indices[tet, 0]]
    m = wp.matrix_from_cols(x[ts.tet_indices[tet, 1]] - a, x[ts.tet_indices[tet, 2]] - a,
                            x[ts.tet_indices[tet, 3]] - a)
    if wp.abs(wp.determinant(m)) < 1.0e-30:
        return False
    b = wp.inverse(m) * (q - a)
    return b[0] >= 0.0 and b[1] >= 0.0 and b[2] >= 0.0 and b[0] + b[1] + b[2] <= 1.0


@wp.kernel(enable_backward=False)
def samples_inside(sf: Surface, tet_world: wp.array[wp.int32], needle_start: wp.array[wp.int32], ts: Tissue,
                   nd: Needles, x: wp.array[wp.vec3], body_q: wp.array[wp.transform], inside: wp.array[wp.int32]):
    k = wp.tid()
    tet = sf.tets[k]
    w = tet_world[k]
    for i in range(needle_start[w], needle_start[w + 1]):
        X = N.pose(nd, i, body_q)
        for c in range(SAMPLES):
            q = N.point_world(nd, i, X, nd.length[i] * float(c) / float(SAMPLES - 1))
            if _bary_inside(ts, tet, q, x):
                inside[i * SAMPLES + c] = 1


@wp.kernel(enable_backward=False)
def untracked(an: Anchors, ts: Tissue, nd: Needles, needle_start: wp.array[wp.int32], capacity: int,
              used: wp.array[wp.int32], h_a: float, tent_from: wp.array[float], x: wp.array[wp.vec3],
              body_q: wp.array[wp.transform],
              inside: wp.array[wp.int32], count: wp.array[wp.int32], depth: wp.array[float],
              world_depth: wp.array[float]):
    w = wp.tid()
    big = float(0.0)
    for i in range(needle_start[w], needle_start[w + 1]):
        X = N.pose(nd, i, body_q)
        r_track = h_a + nd.wire[i]
        n = int(0)
        dmax = float(0.0)
        for c in range(SAMPLES):
            if inside[i * SAMPLES + c] == 0:
                continue
            s = nd.length[i] * float(c) / float(SAMPLES - 1)
            if s < tent_from[i]:                      # under an exit tent: bounded by the newest anchor and the TIP
                inside[i * SAMPLES + c] = 0
                continue
            q = N.point_world(nd, i, X, s)
            dmin = float(1.0e30)
            for j in range(w * capacity, w * capacity + used[w]):
                if an.needle[j] == i and (an.state[j] == NEEDLE or an.state[j] == TIP or an.state[j] == THREAD):
                    dmin = wp.min(dmin, wp.length(material_point(ts, an.tet[j], an.w[j], x) - q))
            if dmin > r_track:
                n = n + 1
                dmax = wp.max(dmax, wp.min(dmin, 1.0))
            inside[i * SAMPLES + c] = 0
        count[i] = n
        depth[i] = dmax
        big = wp.max(big, dmax)
    world_depth[w] = big
