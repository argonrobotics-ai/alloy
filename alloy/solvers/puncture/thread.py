# SPDX-License-Identifier: Apache-2.0
"""Suture thread bending in the per-colour hook (design note §2.5).

Each interior thread particle i carries E_b = ½ k_b |x_{i−1} − 2 x_i + x_{i+1}|² with k_b = EI / ℓ³ (ℓ: mean rest
spacing of its two segments). For a bend angle θ between equal segments |x_{i−1} − 2x_i + x_{i+1}| = 2ℓ sin(θ/2), so
small angles give ½ (EI/ℓ) θ², the discrete Kirchhoff bending energy, and the form is invariant under rigid motion.
It is quadratic in the positions, so VBD's per-vertex blocks are exact (k_b, 4 k_b, k_b along the triplet) and the
Gauss-Seidel sweep cannot overshoot. Stretch is Newton's springs (EA/ℓ). No twist (a round braided suture).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .config import NeedleSpec


def bend_stiffness(model, specs: list[NeedleSpec], device) -> wp.array:
    """Per particle: k_b of the triplet centred on it (0 for ends and non-thread particles)."""
    P = int(model.particle_count)
    kb = np.zeros(max(P, 1), np.float32)
    q = model.particle_q.numpy() if P else np.zeros((0, 3))
    for sp in specs:
        if sp.thread_count < 3 or sp.thread_ei <= 0.0:
            continue
        a, n = sp.thread_start, sp.thread_count
        seg = np.linalg.norm(np.diff(q[a:a + n], axis=0), axis=1)
        ell = 0.5 * (seg[:-1] + seg[1:])
        kb[a + 1:a + n - 1] = sp.thread_ei / ell**3
    return wp.array(kb, dtype=float, device=device)


@wp.func
def _r(x: wp.array[wp.vec3], c: int) -> wp.vec3:
    return x[c - 1] - 2.0 * x[c] + x[c + 1]


@wp.kernel(enable_backward=False)
def bend_colour(ids: wp.array[wp.int32], kb: wp.array[float], x: wp.array[wp.vec3],
                forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    """One thread per particle of the colour: the triplets centred on v−1, v and v+1 (k_b = 0 outside a thread
    interior, so the neighbours of an end contribute nothing)."""
    v = ids[wp.tid()]
    n = kb.shape[0]
    f = wp.vec3(0.0)
    h = float(0.0)
    if kb[v] > 0.0:
        f = f + 2.0 * kb[v] * _r(x, v)
        h = h + 4.0 * kb[v]
    if v > 0 and kb[v - 1] > 0.0:
        f = f - kb[v - 1] * _r(x, v - 1)
        h = h + kb[v - 1]
    if v + 1 < n and kb[v + 1] > 0.0:
        f = f - kb[v + 1] * _r(x, v + 1)
        h = h + kb[v + 1]
    if h > 0.0:
        forces[v] = forces[v] + f
        hessians[v] = hessians[v] + h * wp.identity(n=3, dtype=float)


@wp.kernel(enable_backward=False)
def bend_energy(start: wp.array[wp.int32], kb: wp.array[float], x: wp.array[wp.vec3], out: wp.array[wp.float64]):
    """Per world (particles start[w]..start[w+1]): Σ ½ k_b |r|², in index order, added to out[w]."""
    w = wp.tid()
    e = wp.float64(0.0)
    for c in range(start[w], start[w + 1]):
        if kb[c] > 0.0:
            r = _r(x, c)
            e = e + wp.float64(0.5 * kb[c] * wp.dot(r, r))
    out[w] = out[w] + e


def isolate_thread_colours(builder, threads: list[tuple[int, int]]) -> None:
    """Give each thread one colour of its own, after ``builder.color()``.

    The chain solve (chain.py) updates a whole thread at once, so its particles share one colour, and no tissue vertex
    shares it: a THREAD anchor couples thread particles with tissue vertices, which VBD would otherwise update as a
    Jacobi pair (a zig-zag mode grew from the first handover when they could share a colour).
    ``threads``: (first particle, count) each."""
    members = {int(a) + k for a, n in threads for k in range(int(n))}
    groups = [np.asarray([p for p in np.asarray(g).tolist() if p not in members], dtype=int)
              for g in builder.particle_color_groups]
    groups = [g for g in groups if g.size]
    for a, n in threads:
        groups.append(np.arange(int(a), int(a) + int(n), dtype=int))
    builder.set_coloring(groups)
