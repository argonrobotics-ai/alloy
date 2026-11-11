# SPDX-License-Identifier: Apache-2.0
"""Bevel early-out on flat regions (``ALLOY_CAMERA_WF_BEVEL_FLAT=0`` turns it off).

The Bevel node only hits its own object, and every ray it casts is a chord of the sphere of radius R around the
shading point P (start and end at distance R). Its output is normalize(sum_i w_i N_i) over the hit smooth normals N_i,
or the input normal N with no hits (flipped together when backfacing). So if every triangle of the object within R of
P has all three corner normals equal to the hit triangle's own (equal) corner normals N0, the output is N0 = N whatever
the rays hit, and the bevel rays can be skipped (to float rounding).

Per triangle (host, once per renderer): the "blockers" = triangles of the same object whose bounding box comes within
R of it and whose corner normals differ from N0 (at most MAX_BLOCKERS kept, more = never flat; a triangle with unequal
corner normals is never flat). Per vertex (pre-pass): flat when P is farther than R from every blocker.
Box-like objects (the table, walls, plywood) are flat except within R of their edges.

Example::

    bf, stats = build_bevel_flat(shade_data, bump_table, n_tris, "cuda:0")   # stats["no_blockers"], ...
"""
from __future__ import annotations

import numpy as np
import warp as wp

MAX_BLOCKERS = wp.constant(16)
NEVER = wp.constant(17)  # MAX_BLOCKERS + 1
SAME_N = wp.constant(1.0e-6)  # corner normals "equal": max component difference (about 1.7e-6 rad)


@wp.struct
class BevelFlat:
    nblk: wp.array(dtype=wp.int32)  # per global triangle: blockers (NEVER = never flat)
    off: wp.array(dtype=wp.int32)  # per global triangle: first blocker in lst
    lst: wp.array(dtype=wp.int32)  # blocker global triangles


@wp.func
def same_n(a: wp.vec3, b: wp.vec3) -> bool:
    d = a - b
    return wp.max(wp.max(wp.abs(d[0]), wp.abs(d[1])), wp.abs(d[2])) <= SAME_N


@wp.func
def corner(mesh: wp.uint64, f: int, k: int) -> wp.vec3:
    """Corner k of global triangle f (Warp barycentrics: (1,0) -> 0, (0,1) -> 1, (0,0) -> 2)."""
    if k == 0:
        return wp.mesh_eval_position(mesh, f, 1.0, 0.0)
    if k == 1:
        return wp.mesh_eval_position(mesh, f, 0.0, 1.0)
    return wp.mesh_eval_position(mesh, f, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def find_blockers(mesh: wp.uint64, tri_n: wp.array(dtype=wp.vec3), tri_obj: wp.array(dtype=int),
                  tri_mat: wp.array(dtype=wp.int32), bevel_on: wp.array(dtype=wp.int32),
                  radius: wp.array(dtype=float), nblk: wp.array(dtype=wp.int32), blk: wp.array(dtype=wp.int32)):
    g = wp.tid()
    nblk[g] = NEVER
    mid = tri_mat[g]
    if bevel_on[mid] == 0 or radius[mid] <= 0.0:
        return
    n0 = tri_n[g * 3]
    if not (same_n(n0, tri_n[g * 3 + 1]) and same_n(n0, tri_n[g * 3 + 2])):
        return
    a = corner(mesh, g, 0)
    b = corner(mesh, g, 1)
    c = corner(mesh, g, 2)
    r = radius[mid] * 1.01 + 1.0e-5
    lo = wp.min(wp.min(a, b), c) - wp.vec3(r)
    hi = wp.max(wp.max(a, b), c) + wp.vec3(r)
    q = wp.mesh_query_aabb(mesh, lo, hi)
    u = int(0)
    cnt = int(0)
    while wp.mesh_query_aabb_next(q, u):
        if u != g and tri_obj[u] == tri_obj[g]:
            if not (same_n(n0, tri_n[u * 3]) and same_n(n0, tri_n[u * 3 + 1]) and same_n(n0, tri_n[u * 3 + 2])):
                if cnt < MAX_BLOCKERS:
                    blk[g * MAX_BLOCKERS + cnt] = u
                cnt += 1
    nblk[g] = wp.min(cnt, NEVER)


@wp.func
def dist2_point_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    """Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.length_sq(ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.length_sq(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.length_sq(ap - ab * (d1 / (d1 - d3)))
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.length_sq(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.length_sq(ap - ac * (d2 / (d2 - d6)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.length_sq(bp - (c - b) * w)
    den = 1.0 / (va + vb + vc)
    return wp.length_sq(ap - ab * (vb * den) - ac * (vc * den))


@wp.func
def bevel_flat(bf: BevelFlat, mesh: wp.uint64, g: int, P: wp.vec3, radius: float) -> bool:
    """True when the bevel at P (frame space, on global triangle g) returns its input normal for any rays
    (triangles past the Bevel mesh, deformables, are never bevelled)."""
    if g >= bf.nblk.shape[0]:
        return True
    n = bf.nblk[g]
    if n > MAX_BLOCKERS:
        return False
    r = radius * 1.002 + 2.0e-6
    off = bf.off[g]
    for k in range(n):
        u = bf.lst[off + k]
        d2 = dist2_point_triangle(P, corner(mesh, u, 0), corner(mesh, u, 1), corner(mesh, u, 2))
        if not (d2 > r * r):  # NaN (degenerate blocker) counts as near
            return False
    return True


def build_bevel_flat(sd, bt, n_tris: int, device, enabled: bool = True) -> tuple[BevelFlat, dict]:
    """BevelFlat for the frame's shade data and Bump/Bevel table; stats: triangles by class."""
    bf = BevelFlat()
    nblk = wp.full(n_tris, NEVER, dtype=wp.int32, device=device)
    blk = wp.zeros(max(n_tris * MAX_BLOCKERS, 1) if enabled else 1, dtype=wp.int32, device=device)
    if enabled:
        wp.launch(find_blockers, dim=n_tris, inputs=[sd.bevel_mesh, sd.tri_n, sd.tri_obj, sd.tri_mat, bt.bevel_on,
                                                     bt.bevel_radius], outputs=[nblk, blk], device=device)
    nb = nblk.numpy()
    keep = np.where(nb <= MAX_BLOCKERS, nb, 0).astype(np.int64)
    off = np.zeros(n_tris, np.int64)
    off[1:] = np.cumsum(keep)[:-1]
    lst = np.zeros(max(int(keep.sum()), 1), np.int32)
    if enabled and keep.sum() > 0:
        dense = blk.numpy().reshape(n_tris, MAX_BLOCKERS)
        rows = np.repeat(np.arange(n_tris), keep)
        cols = np.arange(len(rows)) - np.repeat(off, keep)
        lst[:len(rows)] = dense[rows, cols]
    bf.nblk = nblk
    bf.off = wp.array(off.astype(np.int32), dtype=wp.int32, device=device)
    bf.lst = wp.array(lst, dtype=wp.int32, device=device)
    stats = {"tris": n_tris, "no_blockers": int((nb == 0).sum()), "some_blockers": int(((nb > 0) & (nb <= MAX_BLOCKERS)).sum()),
             "never_flat": int((nb > MAX_BLOCKERS).sum())}
    return bf, stats
