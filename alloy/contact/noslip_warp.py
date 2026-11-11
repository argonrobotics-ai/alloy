# SPDX-License-Identifier: Apache-2.0
"""Warp-level parts of the scoped noslip's block solve (:mod:`alloy.contact.noslip_block`): one warp per world.

- Shuffle primitives: ``warp_sum`` / ``warp_sum5`` (xor butterfly: the bitwise-same sum in every lane, so every lane
  takes the same branches), ``lane_prefix`` (ballot), ``lane_bcast``, and ``block_sync``. On CPU the block is one
  lane and the shuffles are identities.
- The tree phase: factor M and restart qacc on the trees the slots touch (``refresh_trees``): one lane per tree up to
  ``MAXT8`` dofs (registers), the whole warp per larger tree (left-looking Cholesky, each column's rows and each
  substitution's dot products split over the lanes). T09's 93-dof cable took ~10 ms per step as one thread.
"""

import warp as wp
from alloy._vendor.mujoco_warp._src import types as mjt

from alloy.contact import noslip_kernels as nk

LANES = 32  # one warp per world on CUDA (the shuffles below assume exactly one full warp)


@wp.func_native(snippet="WP_TILE_SYNC();")
def block_sync():
    """Block barrier (scratch written by one lane, read by another)."""
    ...


@wp.func_native(snippet="""
#if defined(__CUDA_ARCH__)
float r = v;
for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
return r;
#else
return v;
#endif
""")
def warp_sum(v: float) -> float:
    """Sum over the warp, the same bits in every lane (xor butterfly: each step adds two equal partial sums)."""
    ...


@wp.func_native(snippet="""
#if defined(__CUDA_ARCH__)
wp::vec_t<5, wp::float32> r = v;
for (int o = 16; o > 0; o >>= 1) {
  for (int k = 0; k < 5; ++k) r.c[k] += __shfl_xor_sync(0xffffffffu, r.c[k], o);
}
return r;
#else
return v;
#endif
""")
def warp_sum5(v: mjt.vec5) -> mjt.vec5:
    """Componentwise :func:`warp_sum`."""
    ...


@wp.func_native(snippet="""
#if defined(__CUDA_ARCH__)
unsigned int b = __ballot_sync(0xffffffffu, keep != 0);
return wp::vec_t<2, wp::int32>(__popc(b & ((1u << (threadIdx.x & 31)) - 1u)), __popc(b));
#else
return wp::vec_t<2, wp::int32>(0, keep != 0 ? 1 : 0);
#endif
""")
def lane_prefix(keep: int) -> wp.vec2i:
    """(kept lanes below this one, kept lanes in the warp)."""
    ...


@wp.func_native(snippet="""
#if defined(__CUDA_ARCH__)
return __shfl_sync(0xffffffffu, v, src);
#else
return v;
#endif
""")
def lane_bcast(v: float, src: int) -> float:
    """Lane ``src``'s v in every lane."""
    ...


@wp.func
def refresh_trees(w: int, lane: int, nb: int, ntree: int, tr: nk.Trees, ms: nk.Mass, sc: nk.Scratch, out: nk.Dofs):
    """Factor the trees the slots touch (and restart their qacc if ``sc.restart``; the rest keep the main solver's
    qacc): one lane per tree up to ``MAXT8`` dofs (registers), then the whole warp on each larger tree (T09's cable)."""
    for t in range(lane, ntree, nb):
        if sc.need[w, t] != 0 and tr.dofnum[t] <= nk.MAXT8:
            nk.refresh_tree8(tr, t, w, ms, sc, out)
    for t in range(ntree):
        if sc.need[w, t] != 0 and tr.dofnum[t] > nk.MAXT8:
            warp_factor(w, lane, nb, t, tr, ms, sc)
            if sc.restart != 0:
                warp_restart(w, lane, nb, t, tr, ms, sc, out)


@wp.func
def warp_factor(w: int, lane: int, nb: int, t: int, tr: nk.Trees, ms: nk.Mass, sc: nk.Scratch):
    """Dense lower Cholesky factor of tree t's block of M, left-looking, lanes split each column's rows (one lane:
    the serial order)."""
    adr = tr.dofadr[t]
    n = tr.dofnum[t]
    for i in range(lane, n, nb):
        for j in range(i + 1):
            sc.L[w, nk.lidx(sc, t, i, j)] = 0.0
        for e in range(ms.rownnz[adr + i]):
            sp = ms.rowadr[adr + i] + e
            sc.L[w, nk.lidx(sc, t, i, ms.colind[sp] - adr)] = ms.M[w, sp]
    block_sync()
    for j in range(n):
        part = float(0.0)
        for k in range(lane, j, nb):
            part += sc.L[w, nk.lidx(sc, t, j, k)] * sc.L[w, nk.lidx(sc, t, j, k)]
        sq = warp_sum(part)
        d = wp.sqrt(wp.max(sc.L[w, nk.lidx(sc, t, j, j)] - sq, nk.MINVAL))
        for i in range(j + 1 + lane, n, nb):
            v = sc.L[w, nk.lidx(sc, t, i, j)]
            for k in range(j):
                v -= sc.L[w, nk.lidx(sc, t, i, k)] * sc.L[w, nk.lidx(sc, t, j, k)]
            sc.L[w, nk.lidx(sc, t, i, j)] = v / d
        block_sync()  # every lane has read L[j, j]
        if lane == 0:
            sc.L[w, nk.lidx(sc, t, j, j)] = d
        block_sync()


@wp.func
def warp_restart(w: int, lane: int, nb: int, t: int, tr: nk.Trees, ms: nk.Mass, sc: nk.Scratch, out: nk.Dofs):
    """qacc = qacc_smooth + M^-1 qfrc_constraint and Ma = qfrc_smooth + qfrc_constraint on tree t (warp_factor's
    factor): forward and back substitution, each row's dot product split over the lanes."""
    adr = tr.dofadr[t]
    n = tr.dofnum[t]
    base = t * sc.maxt
    for i in range(lane, n, nb):
        sc.V[w, base + i] = out.qfrc_constraint[w, adr + i]
    block_sync()
    for i in range(n):
        part = float(0.0)
        for k in range(lane, i, nb):
            part += sc.L[w, nk.lidx(sc, t, i, k)] * sc.V[w, base + k]
        x = (sc.V[w, base + i] - warp_sum(part)) / sc.L[w, nk.lidx(sc, t, i, i)]
        block_sync()
        if lane == 0:
            sc.V[w, base + i] = x
        block_sync()
    for ii in range(n):
        i = n - 1 - ii
        part = float(0.0)
        for k in range(i + 1 + lane, n, nb):
            part += sc.L[w, nk.lidx(sc, t, k, i)] * sc.V[w, base + k]
        x = (sc.V[w, base + i] - warp_sum(part)) / sc.L[w, nk.lidx(sc, t, i, i)]
        block_sync()
        if lane == 0:
            sc.V[w, base + i] = x
        block_sync()
    for i in range(lane, n, nb):
        out.qacc[w, adr + i] = ms.qacc_smooth[w, adr + i] + sc.V[w, base + i]
        out.Ma[w, adr + i] = ms.qfrc_smooth[w, adr + i] + out.qfrc_constraint[w, adr + i]


__all__ = ["LANES", "block_sync", "lane_bcast", "lane_prefix", "refresh_trees", "warp_sum", "warp_sum5"]
