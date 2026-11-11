# SPDX-License-Identifier: Apache-2.0
"""Exact block solve of each suture thread inside the VBD sweep (design note §2.5; PUN-6).

Why. A thread is a light, stiff chain (0.05 mg particles, EA/ℓ ~ 1e5 N/m). VBD's per-vertex Gauss-Seidel moves a
correction one particle per sweep, so under gravity a 60-particle thread settled with ~0.4 N of spurious tension at the
swage (dt 4e-4 s, 20 iterations), shrinking only with dt².

What. All particles of one thread form one colour of their own. When VBD reaches that colour, Newton has already
accumulated their contact terms (self-contact, body contact) into ``particle_forces`` / ``particle_hessians``. This
hook then takes one Newton step on the whole chain at once:
    H Δ = −∇G,  G = Σ m/(2dt²)|x − y|² + springs(EA/ℓ) + bending(EI/ℓ³) + anchors + swage + obstacles
         − contacts (linearised),
H is block-pentadiagonal (stretch couples i, i±1; bending i, i±2) with PSD blocks (projected spring Hessian), solved by
a banded block LU in one GPU thread per thread (no pivoting: H is SPD). Newton's own per-vertex update for the colour
is then steered to apply Δ exactly: with h0 = m/dt² I and f0 = m/dt² (y − x) (a thread particle has no tri/edge/tet
terms), the hook sets hessian = I − h0 and force = Δ − f0, so h⁻¹f = Δ. Stretch is therefore not a Newton spring for
the thread; its energy is booked by the ledger here (``chain_energy``).
"""

from __future__ import annotations

import warp as wp

from .needle import Needles
from .obstacles import Obstacles, contact_terms
from .terms import Anchors, Params, Tissue, anchor_eval, vertex_weight

MAXN = wp.constant(256)                      # particles per thread (scratch is [threads, MAXN, 5] blocks)


@wp.struct
class Chain:
    start: wp.array[wp.int32]       # per thread: first particle
    count: wp.array[wp.int32]
    rest: wp.array[float]           # per thread: segment rest length
    ks: wp.array[float]             # per thread: EA / rest
    kb: wp.array[float]             # per thread: EI / rest^3
    needle: wp.array[wp.int32]      # per thread: its needle
    row: wp.array[wp.int32]         # particle -> anchors CSR (the solver's)
    ent: wp.array[wp.int32]


@wp.func
def _spring_hessian(d: wp.vec3, rest: float, k: float) -> wp.mat33:
    l = wp.length(d)
    if l < 1.0e-12:
        return k * wp.identity(n=3, dtype=float)
    u = d / l
    uu = wp.outer(u, u)
    return k * uu + k * wp.max(0.0, 1.0 - rest / l) * (wp.identity(n=3, dtype=float) - uu)


@wp.func
def _blk(L: wp.array3d[wp.mat33], c: int, i: int, o: int) -> wp.mat33:
    return L[c, i, o]


@wp.kernel(enable_backward=False)
def chain_step(dt: float, ch: Chain, an: Anchors, ts: Tissue, nd: Needles, pr: Params, x: wp.array[wp.vec3],
               x0: wp.array[wp.vec3], y: wp.array[wp.vec3], mass: wp.array[float], body_q: wp.array[wp.transform],
               body_q0: wp.array[wp.transform], forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33],
               ob: Obstacles, radius: wp.array[float], A: wp.array3d[wp.mat33], g: wp.array2d[wp.vec3]):
    """One thread per suture thread. A[c, i, k] holds H[i, i+k-2] (k = 0..4, band), g[c, i] the right-hand side."""
    c = wp.tid()
    a0 = ch.start[c]
    n = ch.count[c]
    if n > MAXN:
        return
    zero = wp.mat33(0.0)
    inv_dt2 = 1.0 / (dt * dt)
    for i in range(n):
        for k in range(5):
            A[c, i, k] = zero
        v = a0 + i
        A[c, i, 2] = mass[v] * inv_dt2 * wp.identity(n=3, dtype=float) + hessians[v]
        r = mass[v] * inv_dt2 * (y[v] - x[v]) + forces[v]               # = −gradient so far (contacts from Newton)
        for e in range(ch.row[v], ch.row[v + 1]):                      # anchors and the swage tie on this particle
            j = an.needle[ch.ent[e]]
            fp, hp, k2, al = anchor_eval(ch.ent[e], an, ts, nd, pr, dt, x, x0, body_q, body_q0)
            w = vertex_weight(ch.ent[e], v, an, ts, k2, al)
            r = r + w * fp
            A[c, i, 2] = A[c, i, 2] + (w * w) * hp
        for o in range(ob.start[c], ob.start[c] + ob.count[c]):          # rigid obstacles (obstacles.py)
            if i >= ob.skip[o]:
                fo, ho = contact_terms(ob, nd, o, x[v], x0[v], radius[v], body_q, body_q0, pr.eps_v * dt)
                r = r + fo
                A[c, i, 2] = A[c, i, 2] + ho
        g[c, i] = r
    for i in range(n - 1):                                             # stretch
        v = a0 + i
        d = x[v + 1] - x[v]
        l = wp.length(d)
        f = ch.ks[c] * (l - ch.rest[c]) * d / wp.max(l, 1.0e-12)
        g[c, i] = g[c, i] + f
        g[c, i + 1] = g[c, i + 1] - f
        h = _spring_hessian(d, ch.rest[c], ch.ks[c])
        A[c, i, 2] = A[c, i, 2] + h
        A[c, i + 1, 2] = A[c, i + 1, 2] + h
        A[c, i, 3] = A[c, i, 3] - h                                   # H[i, i+1]
        A[c, i + 1, 1] = A[c, i + 1, 1] - h                           # H[i+1, i]
    kb = ch.kb[c]
    if kb > 0.0:
        for i in range(1, n - 1):                                      # bending, centre i: ½kb|x_{i-1} - 2x_i + x_{i+1}|²
            v = a0 + i
            rr = x[v - 1] - 2.0 * x[v] + x[v + 1]
            g[c, i - 1] = g[c, i - 1] - kb * rr
            g[c, i] = g[c, i] + 2.0 * kb * rr
            g[c, i + 1] = g[c, i + 1] - kb * rr
            eye = wp.identity(n=3, dtype=float)
            A[c, i - 1, 2] = A[c, i - 1, 2] + kb * eye
            A[c, i, 2] = A[c, i, 2] + 4.0 * kb * eye
            A[c, i + 1, 2] = A[c, i + 1, 2] + kb * eye
            A[c, i - 1, 3] = A[c, i - 1, 3] - 2.0 * kb * eye
            A[c, i, 1] = A[c, i, 1] - 2.0 * kb * eye
            A[c, i, 3] = A[c, i, 3] - 2.0 * kb * eye
            A[c, i + 1, 1] = A[c, i + 1, 1] - 2.0 * kb * eye
            A[c, i - 1, 4] = A[c, i - 1, 4] + kb * eye
            A[c, i + 1, 0] = A[c, i + 1, 0] + kb * eye
    # banded block LU (bandwidth 2), forward elimination
    for i in range(n):
        dinv = wp.inverse(A[c, i, 2])
        for k in range(1, 3):                                          # rows i+1, i+2
            if i + k < n:
                lk = A[c, i + k, 2 - k] * dinv                         # H[i+k, i] D_i^-1
                for m2 in range(1, 3):                                 # columns i+1, i+2 of row i
                    if i + m2 < n:
                        A[c, i + k, 2 - k + m2] = A[c, i + k, 2 - k + m2] - lk * A[c, i, 2 + m2]
                g[c, i + k] = g[c, i + k] - lk * g[c, i]
    # back substitution into g (Δ)
    for ii in range(n):
        i = n - 1 - ii
        r = g[c, i]
        for m2 in range(1, 3):
            if i + m2 < n:
                r = r - A[c, i, 2 + m2] * g[c, i + m2]
        g[c, i] = wp.inverse(A[c, i, 2]) * r
    for i in range(n):                                                 # steer Newton's per-vertex update to Δ
        v = a0 + i
        h0 = mass[v] * inv_dt2 * wp.identity(n=3, dtype=float)
        hessians[v] = wp.identity(n=3, dtype=float) - h0
        forces[v] = g[c, i] - mass[v] * inv_dt2 * (y[v] - x[v])


@wp.kernel(enable_backward=False)
def chain_energy(ch: Chain, world_of: wp.array[wp.int32], x: wp.array[wp.vec3], out: wp.array[wp.float64]):
    """Per world: stretch and bending energy of its threads (index order), added to out[w]."""
    w = wp.tid()
    e = wp.float64(0.0)
    for c in range(ch.start.shape[0]):
        if world_of[c] != w:
            continue
        a0 = ch.start[c]
        n = ch.count[c]
        for i in range(n - 1):
            dl = wp.length(x[a0 + i + 1] - x[a0 + i]) - ch.rest[c]
            e = e + wp.float64(0.5 * ch.ks[c] * dl * dl)
        for i in range(1, n - 1):
            rr = x[a0 + i - 1] - 2.0 * x[a0 + i] + x[a0 + i + 1]
            e = e + wp.float64(0.5 * ch.kb[c] * wp.dot(rr, rr))
    out[w] = out[w] + e
