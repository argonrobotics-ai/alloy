# SPDX-License-Identifier: Apache-2.0
"""Warp functions shared by the topology kernels (no float atomics; pure per-thread math).

Strains use Newton's rest frames: triangles store ``tri_poses = inv(frameᵀ·[x1−x0, x2−x0])`` (2x2, in-plane
rest basis, newton@ac7f7830:newton/_src/geometry/rest_shape.py:22-30) and tets store ``tet_poses = inv(Dm)``,
``Dm = [x1−x0, x2−x0, x3−x0]`` (rest_shape.py:33-37). F = Ds·inv(D); principal stretches are the singular values
of F; engineering principal strain = stretch − 1.
"""

from __future__ import annotations

import warp as wp

INVERTED_STRAIN = wp.constant(1.0e6)   # strain reported for an inverted or collapsed element (always breaks)


@wp.func
def tri_stretch_range(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, inv_d: wp.mat22) -> wp.vec2:
    """(largest, smallest) principal stretch of a triangle; C = FᵀF is 2x2, eigenvalues in closed form."""
    d1 = x1 - x0
    d2 = x2 - x0
    f1 = d1 * inv_d[0, 0] + d2 * inv_d[1, 0]
    f2 = d1 * inv_d[0, 1] + d2 * inv_d[1, 1]
    a = wp.dot(f1, f1)
    b = wp.dot(f1, f2)
    c = wp.dot(f2, f2)
    disc = wp.sqrt(wp.max((a - c) * (a - c) + 4.0 * b * b, 0.0))
    lmax = 0.5 * (a + c + disc)
    lmin = wp.max(0.5 * (a + c - disc), 0.0)
    return wp.vec2(wp.sqrt(lmax), wp.sqrt(lmin))


@wp.func
def tet_stretch_range(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, inv_dm: wp.mat33) -> wp.vec2:
    """(largest, smallest) principal stretch of a tet from the SVD of F; inverted tets report (BIG, 0)."""
    ds = wp.matrix_from_cols(x1 - x0, x2 - x0, x3 - x0)
    f = ds * inv_dm
    if wp.determinant(f) <= 0.0:
        return wp.vec2(1.0 + INVERTED_STRAIN, 0.0)
    u = wp.mat33()
    s = wp.vec3()
    v = wp.mat33()
    wp.svd3(f, u, s, v)
    s0 = wp.abs(s[0])
    s1 = wp.abs(s[1])
    s2 = wp.abs(s[2])
    return wp.vec2(wp.max(s0, wp.max(s1, s2)), wp.min(s0, wp.min(s1, s2)))


@wp.func
def stretch_to_strain(st: wp.vec2) -> wp.vec2:
    """(tensile, compressive) engineering strain magnitudes from (max, min) stretch; both >= 0."""
    return wp.vec2(wp.max(st[0] - 1.0, 0.0), wp.max(1.0 - st[1], 0.0))


@wp.func
def point_segment_distance(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    ab = b - a
    t = wp.clamp(wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), 1.0e-30), 0.0, 1.0)
    return wp.length(p - (a + t * ab))


@wp.func
def to_vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def to_mat33d(m: wp.mat33) -> wp.mat33d:
    return wp.mat33d(
        wp.float64(m[0, 0]), wp.float64(m[0, 1]), wp.float64(m[0, 2]),
        wp.float64(m[1, 0]), wp.float64(m[1, 1]), wp.float64(m[1, 2]),
        wp.float64(m[2, 0]), wp.float64(m[2, 1]), wp.float64(m[2, 2]),
    )


@wp.func
def skew(w: wp.vec3) -> wp.mat33:
    """[w]x, so that [w]x·r = w × r."""
    return wp.mat33(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)


@wp.func
def segment_crosses_tet(a: wp.vec3, b: wp.vec3, x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> bool:
    """Does segment [a, b] meet the tet? (Clip the segment's parameter range by the four face half-spaces.)"""
    t0 = float(0.0)
    t1 = float(1.0)
    d = b - a
    for f in range(4):
        p = x0
        q = x1
        r = x2
        o = x3
        if f == 1:
            r = x3
            o = x2
        elif f == 2:
            q = x2
            r = x3
            o = x1
        elif f == 3:
            p = x1
            q = x2
            r = x3
            o = x0
        n = wp.cross(q - p, r - p)
        if wp.dot(n, o - p) > 0.0:
            n = -n
        num = -wp.dot(n, a - p)              # inside: n . (a + t d - p) <= 0
        den = wp.dot(n, d)
        if den == 0.0:
            if num < 0.0:
                return False
        elif den > 0.0:
            t1 = wp.min(t1, num / den)
        else:
            t0 = wp.max(t0, num / den)
        if t0 > t1:
            return False
    return True


@wp.func
def segment_segment_distance(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3) -> float:
    """Distance between segments [p1, q1] and [p2, q2] (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    s = float(0.0)
    t = float(0.0)
    if a <= 1.0e-30 and e <= 1.0e-30:
        return wp.length(r)
    if a <= 1.0e-30:
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= 1.0e-30:
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            bb = wp.dot(d1, d2)
            den = a * e - bb * bb
            if den > 0.0:
                s = wp.clamp((bb * f - c * e) / den, 0.0, 1.0)
            t = (bb * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((bb - c) / a, 0.0, 1.0)
    return wp.length((p1 + s * d1) - (p2 + t * d2))


@wp.func
def segment_near_tet(a: wp.vec3, b: wp.vec3, x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, lead: float) -> bool:
    """Is segment [a, b] within ``lead`` of the tet? It crosses the tet or passes within ``lead`` of one of its six
    edges. (Exact when both segment ends lie outside the tet's lead neighbourhood, as a blade edge spanning the
    body does; an end near a face interior is not seen.)"""
    if segment_crosses_tet(a, b, x0, x1, x2, x3):
        return True
    d = wp.min(wp.min(segment_segment_distance(a, b, x0, x1), segment_segment_distance(a, b, x0, x2)),
               wp.min(segment_segment_distance(a, b, x0, x3), segment_segment_distance(a, b, x1, x2)))
    d = wp.min(d, wp.min(segment_segment_distance(a, b, x1, x3), segment_segment_distance(a, b, x2, x3)))
    return d <= lead
