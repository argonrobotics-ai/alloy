# SPDX-License-Identifier: Apache-2.0
"""J storage moves of the constraint-row sort (``mjw_rows.sort_constraints``), split out of mjw_rows.

- Sparse J, the sort without the route: the live rows' nnz are scanned per world and the entries relaid in the new row
  order (``rebuild_sparse``; its second J is allocated on first use). With the route the storage stays in place and
  only ``J_rowadr`` moves, with the NJMAX_NNZ check from the live total (``_nnz_overflow``).
- Dense J: rows are permuted in place along the permutation's cycles (``_cycle_leaders``, ``_dense_permute``), without
  a second J (coordinator ruling 2026-11-04 05:40Z).
"""

from __future__ import annotations

from typing import Any

import warp as wp

@wp.kernel
def _live_nnz(nefc: wp.array[int], njmax: int, rownnz: wp.array2d[int], flat: wp.array[int]):
    w, r = wp.tid()
    flat[w * njmax + r] = wp.where(r < wp.min(nefc[w], njmax), rownnz[w, r], 0)


@wp.kernel
def _sparse_rowadr(nefc: wp.array[int], njmax: int, scan: wp.array[int], rowadr: wp.array2d[int]):
    w, r = wp.tid()  # rows are contiguous from 0 in the new layout: the exclusive scan within the world
    if r < wp.min(nefc[w], njmax):
        rowadr[w, r] = scan[w * njmax + r] - scan[w * njmax]


@wp.kernel
def _nnz_overflow(nj: int, flat: wp.array[int], scan: wp.array[int], njmax_nnz: int, overflow: wp.array[int]):
    w = wp.tid()  # route only: J storage stays in place, so flag NJMAX_NNZ from the live total (MuJoCo Warp reads
    last = w * nj + nj - 1  # the last row's rowadr + rownnz, which assumes rows are stored in row order)
    if scan[last] + flat[last] - scan[w * nj] > njmax_nnz:
        overflow[w] = overflow[w] | 2  # OverflowType.NJMAX_NNZ


@wp.kernel
def _sparse_move(nefc: wp.array[int], njmax: int, rownnz: wp.array2d[int], old_adr: wp.array2d[int],
                 new_adr: wp.array2d[int], src_j: wp.array3d[float], src_c: wp.array3d[int],
                 dst_j: wp.array3d[float], dst_c: wp.array3d[int]):
    w, r = wp.tid()
    if r >= wp.min(nefc[w], njmax):
        return
    a, b = old_adr[w, r], new_adr[w, r]
    for k in range(rownnz[w, r]):
        dst_j[w, 0, b + k] = src_j[w, 0, a + k]
        dst_c[w, 0, b + k] = src_c[w, 0, a + k]


@wp.kernel
def _cycle_leaders(nefc: wp.array[int], njmax: int, perm: wp.array2d[int], lead: wp.array2d[wp.int8]):
    """One thread per world: mark each cycle of the row permutation once (1 = its first row, 2 = the others)."""
    w = wp.tid()
    n = wp.min(nefc[w], njmax)
    for r in range(n):
        lead[w, r] = wp.int8(0)
    for r in range(n):
        if lead[w, r] == wp.int8(0):
            lead[w, r] = wp.int8(1)
            c = perm[w, r]
            while c != r:
                lead[w, c] = wp.int8(2)
                c = perm[w, c]


@wp.kernel
def _dense_permute(nefc: wp.array[int], njmax: int, perm: wp.array2d[int], lead: wp.array2d[wp.int8],
                   J: wp.array3d[float]):
    """Dense J rows into the canonical order in place (row r <- old row perm[r]), one thread per (world, column):
    each column follows the permutation's cycles with one held value, so no second J is needed (coordinator ruling
    2026-11-04 05:40Z). A pure move, bitwise equal to the gather it replaces."""
    w, q = wp.tid()
    n = wp.min(nefc[w], njmax)
    for r in range(n):
        if lead[w, r] == wp.int8(1):
            held = J[w, r, q]
            cur = r
            nxt = perm[w, r]
            while nxt != r:
                J[w, cur, q] = J[w, nxt, q]
                cur = nxt
                nxt = perm[w, cur]
            J[w, cur, q] = held


def rebuild_sparse(d: Any, s: Any, nj: int) -> None:
    e, nw = d.efc, int(d.nworld)
    if s.j_copy is None:  # only this path (the sort without the route) needs a second sparse J
        s.j_copy, s.c_copy = wp.empty_like(e.J), wp.empty_like(e.J_colind)
    wp.launch(_live_nnz, dim=(nw, nj), inputs=[d.nefc, nj, e.J_rownnz], outputs=[s.nnz_flat])
    wp.utils.array_scan(s.nnz_flat, s.nnz_scan, inclusive=False)  # integer: exact in any order
    wp.launch(_sparse_rowadr, dim=(nw, nj), inputs=[d.nefc, nj, s.nnz_scan], outputs=[e.J_rowadr])
    old_adr = s.copies["J_rowadr"]  # each new row's old storage address
    wp.launch(_sparse_move, dim=(nw, nj), inputs=[d.nefc, nj, e.J_rownnz, old_adr, e.J_rowadr, e.J, e.J_colind],
              outputs=[s.j_copy, s.c_copy])
    # back into place: rows are contiguous in the new layout, so a move with identity addresses copies them back
    wp.launch(_sparse_move, dim=(nw, nj), inputs=[d.nefc, nj, e.J_rownnz, e.J_rowadr, e.J_rowadr, s.j_copy, s.c_copy],
              outputs=[e.J, e.J_colind])
