# SPDX-License-Identifier: Apache-2.0
"""Anchor energy terms on device: per-vertex force and Hessian blocks for VBD (design note §2).

An anchor is a tissue material point p = Σ w_k x_k (tet ``tet``, weights ``w``) tied to a target:
- TIP:     the needle tip, isotropic stick spring k_tip once the tip is past the point (one-sided contact);
- NEEDLE:  the closest point of the needle centreline, normal-plane spring k_n, plus tangential friction;
- THREAD:  a point of the thread polyline, as NEEDLE; its segment, fraction and direction are set at the step start
           by the events (closest point) and held through the step, so the energy is quadratic in the positions
           (re-projecting onto a light, moving thread inside the iterations made the string oscillate and blow up);
- STOPPER: the thread's last particle (a knot that cannot enter the hole), isotropic k_n;
- SWAGE:   not a tissue point: a thread particle (``seg``: 0, the first, and 1) tied to the needle's arc at the swage
           (``alpha`` past it: 0 and one thread spacing), isotropic k_tip: the swage crimps the thread coaxially (the
           thread is swaged into the needle; created at construction, never freed).
Tangential friction (NEEDLE, THREAD): smoothed Coulomb on this step's slip u along t (IPC f1, width eps_v·dt) with
limit F_c = flen·ℓ + mu·|F_n| (ℓ: the current needle length the anchor owns, ``ell`` x the tissue stretch along t) (+ fcut on the newest anchor of an inside track while the needle advances, u > 0),
plus viscous eta·ℓ·u/dt. Forces on p go to tissue vertex k with weight w_k and to thread particles with −(1−α) and
−α; Hessian blocks scale with the weight squared (Gauss-Newton, PSD).
"""

from __future__ import annotations

import warp as wp

from . import needle as N
from .needle import Needles

FREE = wp.constant(0)
TIP = wp.constant(1)
NEEDLE = wp.constant(2)
THREAD = wp.constant(3)
STOPPER = wp.constant(4)
SWAGE = wp.constant(5)


@wp.struct
class Anchors:
    state: wp.array[wp.int32]
    needle: wp.array[wp.int32]
    tet: wp.array[wp.int32]
    w: wp.array[wp.vec4]
    seg: wp.array[wp.int32]      # THREAD: global index of the segment's first particle (fixed within a step)
    alpha: wp.array[float]       # THREAD: fraction along that segment (fixed within a step)
    tdir: wp.array[wp.vec3]      # THREAD: unit segment direction at the step start (fixed within a step)
    cut: wp.array[wp.int32]      # 1: newest anchor of an inside track (cutting force)
    ell: wp.array[float]         # material length of curve the anchor owns (h_a at creation / stretch then) [m]
    trk: wp.array[wp.int32]      # track id (per needle, counts entry punctures)
    hole: wp.array[wp.int32]     # 1: a surface hole (made by a puncture), where the thread must pass the surface


@wp.struct
class Tissue:
    tet_indices: wp.array2d[wp.int32]
    tet_poses: wp.array[wp.mat33]
    fp: wp.array[float]
    fcut: wp.array[float]
    flen: wp.array[float]
    eta: wp.array[float]


@wp.struct
class Params:
    k_n: float
    k_tip: float
    h_a: float
    eps_v: float
    mu: float
    lookahead: float


@wp.func
def material_point(ts: Tissue, tet: int, w: wp.vec4, x: wp.array[wp.vec3]) -> wp.vec3:
    return (w[0] * x[ts.tet_indices[tet, 0]] + w[1] * x[ts.tet_indices[tet, 1]] + w[2] * x[ts.tet_indices[tet, 2]]
            + w[3] * x[ts.tet_indices[tet, 3]])


@wp.func
def stretch(ts: Tissue, tet: int, x: wp.array[wp.vec3], t: wp.vec3) -> float:
    """Current length per material length of the tissue fibre along the unit direction t: 1 / |F^-1 t|."""
    a = x[ts.tet_indices[tet, 0]]
    ds = wp.matrix_from_cols(x[ts.tet_indices[tet, 1]] - a, x[ts.tet_indices[tet, 2]] - a,
                             x[ts.tet_indices[tet, 3]] - a)
    f = ds * ts.tet_poses[tet]
    if wp.abs(wp.determinant(f)) < 1.0e-12:
        return 1.0
    return 1.0 / wp.max(wp.length(wp.inverse(f) * t), 1.0e-6)


@wp.func
def thread_closest(nd: Needles, i: int, k0: int, p: wp.vec3, x: wp.array[wp.vec3]):
    """Segment (global first-particle index) and fraction of the closest thread point to p, walking at most one
    segment either way from k0 (anchors move far less than a segment per step)."""
    first = nd.thread_start[i]
    last = first + nd.thread_count[i] - 2
    k = wp.clamp(k0, first, last)
    a = float(0.0)
    for _ in range(2):
        d = x[k + 1] - x[k]
        a = wp.dot(p - x[k], d) / wp.max(wp.dot(d, d), 1.0e-20)
        if a < 0.0 and k > first:
            k = k - 1
        elif a > 1.0 and k < last:
            k = k + 1
        else:
            break
    d = x[k + 1] - x[k]
    a = wp.clamp(wp.dot(p - x[k], d) / wp.max(wp.dot(d, d), 1.0e-20), 0.0, 1.0)
    return k, a


@wp.func
def tangential(t: wp.vec3, u: float, fc: float, visc: float, eps: float):
    """Smoothed Coulomb (limit fc, width eps) plus viscous (coefficient visc = eta·h_a/dt): force along t and the
    lagged Hessian scale."""
    au = wp.abs(u)
    k = float(0.0)
    if fc > 0.0:
        if au < eps:
            k = fc * (2.0 - au / eps) / eps
        else:
            k = fc / au
    k = k + visc
    return -k * u * t, k


@wp.func
def anchor_target(j: int, an: Anchors, ts: Tissue, nd: Needles, x: wp.array[wp.vec3], x0: wp.array[wp.vec3],
                  body_q: wp.array[wp.transform], body_q0: wp.array[wp.transform]):
    """(p, p0, c, dc, t, k, a): anchor point now and at the step start, target point, the target's material motion
    over the step, unit tangent (zero for the isotropic TIP / STOPPER ties), thread segment and fraction (k = -1:
    no thread particle)."""
    st = an.state[j]
    i = an.needle[j]
    tet = an.tet[j]
    X = N.pose(nd, i, body_q)
    zero = wp.vec3(0.0)
    if st == SWAGE:                                  # thread particle seg tied at arc length length + alpha
        v = nd.thread_start[i] + an.seg[j]
        s = nd.length[i] + an.alpha[j]
        c = N.point_world(nd, i, X, s)
        return x[v], x0[v], c, c - N.point_world(nd, i, N.pose(nd, i, body_q0), s), zero, int(-1), float(0.0)
    p = material_point(ts, tet, an.w[j], x)
    p0 = material_point(ts, tet, an.w[j], x0)
    if st == TIP:
        c = N.point_world(nd, i, X, 0.0)
        return p, p0, c, c - N.point_world(nd, i, N.pose(nd, i, body_q0), 0.0), zero, int(-1), float(0.0)
    if st == STOPPER:
        e = nd.thread_start[i] + nd.thread_count[i] - 1
        return p, p0, x[e], x[e] - x0[e], zero, e - 1, float(1.0)
    if st == NEEDLE:
        X0 = N.pose(nd, i, body_q0)
        s = wp.clamp(N.project_world(nd, i, X, p), 0.0, nd.length[i])
        c = N.point_world(nd, i, X, s)
        return p, p0, c, c - N.point_world(nd, i, X0, s), N.tangent_world(nd, i, X, s), int(-1), float(0.0)
    k = an.seg[j]
    a = an.alpha[j]
    c = (1.0 - a) * x[k] + a * x[k + 1]
    dc = c - ((1.0 - a) * x0[k] + a * x0[k + 1])
    return p, p0, c, dc, an.tdir[j], k, a


@wp.func
def tip_pressing(an: Anchors, nd: Needles, j: int, body_q: wp.array[wp.transform], p: wp.vec3, c: wp.vec3) -> bool:
    """A TIP anchor is a one-sided contact: it acts only once the tip is past its point along the tip axis."""
    i = an.needle[j]
    return wp.dot(c - p, N.tip_axis_world(nd, i, N.pose(nd, i, body_q))) > 0.0


@wp.func
def anchor_eval(j: int, an: Anchors, ts: Tissue, nd: Needles, pr: Params, dt: float, x: wp.array[wp.vec3],
                x0: wp.array[wp.vec3], body_q: wp.array[wp.transform], body_q0: wp.array[wp.transform]):
    """Force on the anchor point p, its 3x3 Hessian, and the thread segment / fraction it acts on (-1 if none)."""
    st = an.state[j]
    p, p0, c, dc, t, k, a = anchor_target(j, an, ts, nd, x, x0, body_q, body_q0)
    eye = wp.identity(n=3, dtype=float)
    if st == TIP:
        if not tip_pressing(an, nd, j, body_q, p, c):
            return wp.vec3(0.0), wp.mat33(0.0), k, a
        return -pr.k_tip * (p - c), pr.k_tip * eye, k, a
    if st == STOPPER:
        return -pr.k_n * (p - c), pr.k_n * eye, k, a
    if st == SWAGE:
        return -pr.k_tip * (p - c), pr.k_tip * eye, k, a
    tet = an.tet[j]
    flen = ts.flen[tet]
    eta = ts.eta[tet]
    if st == THREAD:
        i = an.needle[j]
        flen = nd.thread_flen[i]
        eta = nd.thread_eta[i]
    pn = eye - wp.outer(t, t)
    fn = -pr.k_n * (pn * (p - c))
    u = wp.dot(t, (p - p0) - dc)
    ell = an.ell[j] * stretch(ts, tet, x, t)
    fc = flen * ell + pr.mu * wp.length(fn)
    if an.cut[j] == 1 and u > 0.0:
        fc = fc + ts.fcut[tet]
    ft, kt = tangential(t, u, fc, eta * ell / dt, pr.eps_v * dt)
    return fn + ft, pr.k_n * pn + kt * wp.outer(t, t), k, a


@wp.func
def vertex_weight(j: int, v: int, an: Anchors, ts: Tissue, k: int, a: float) -> float:
    if an.state[j] == SWAGE:
        return 1.0                                   # its only listed particle is the thread's first
    tet = an.tet[j]
    w = an.w[j]
    c = float(0.0)
    for q in range(4):
        if ts.tet_indices[tet, q] == v:
            c = c + w[q]
    if k >= 0:
        if v == k:
            c = c - (1.0 - a)
        if v == k + 1:
            c = c - a
    return c


@wp.kernel(enable_backward=False)
def gather_colour(dt: float, ids: wp.array[wp.int32], an: Anchors, ts: Tissue, nd: Needles, pr: Params,
                  x: wp.array[wp.vec3], x0: wp.array[wp.vec3], body_q: wp.array[wp.transform],
                  body_q0: wp.array[wp.transform], row: wp.array[wp.int32], ent: wp.array[wp.int32],
                  forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    """One thread per particle of the colour: sum its anchors' blocks in CSR (anchor index) order."""
    v = ids[wp.tid()]
    if row[v] == row[v + 1]:
        return
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    for e in range(row[v], row[v + 1]):
        j = ent[e]
        fp, hp, k, a = anchor_eval(j, an, ts, nd, pr, dt, x, x0, body_q, body_q0)
        c = vertex_weight(j, v, an, ts, k, a)
        f = f + c * fp
        h = h + (c * c) * hp
    forces[v] = forces[v] + f
    hessians[v] = hessians[v] + h


@wp.kernel(enable_backward=False)
def anchor_forces(dt: float, an: Anchors, ts: Tissue, nd: Needles, pr: Params, x: wp.array[wp.vec3],
                  x0: wp.array[wp.vec3], body_q: wp.array[wp.transform], body_q0: wp.array[wp.transform],
                  f_out: wp.array[wp.vec3], off_out: wp.array[float]):
    """Per anchor at the end of the step: force on the tissue point (the needle/thread feels the opposite) and the
    distance of the point from its target curve (0 for free anchors)."""
    j = wp.tid()
    if an.state[j] == FREE:
        f_out[j] = wp.vec3(0.0)
        off_out[j] = 0.0
        return
    fp, hp, k, a = anchor_eval(j, an, ts, nd, pr, dt, x, x0, body_q, body_q0)
    f_out[j] = fp
    p, p0, c, dc, t, k2, a2 = anchor_target(j, an, ts, nd, x, x0, body_q, body_q0)
    d = p - c
    off_out[j] = wp.length(d - wp.dot(d, t) * t)
    if an.state[j] == TIP and not tip_pressing(an, nd, j, body_q, p, c):
        off_out[j] = 0.0
