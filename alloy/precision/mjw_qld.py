# SPDX-License-Identifier: Apache-2.0
"""Ordered sparse L'DL factorization of the mass matrix for Alloy's SolverMuJoCo route (SCR-DC5-4 (ii)).

MuJoCo Warp factors the sparse part of M by tree level (``smooth._factor_i_sparse``). ``m.qLD_updates`` groups the
updates (i, k, Madr_ki) by the depth of the target row i, deepest level first. One ``_qLD_acc`` launch per level runs
one thread per update:
- it subtracts M(k, :) * M(k, i) / M(k, k) from row i with float ``atomic_sub``;
- it then writes the scaled M(k, i).

Every descendant k of i lands in the same level, so on a branching tree or a long chain (a cable) several threads
subtract into the same entries of row i in a racing order.

The replacement runs one thread per target row and applies that row's updates in a fixed order: k descending, which
is MuJoCo C's ``mj_factorI`` order. Rows k are final before their level runs, and no other thread of a level reads
row k, so the result is the sequential algorithm's, in float32, every run. One launch per level, as before.

The solve with that factor (``smooth._solve_LD_sparse_fused``: x = inv(L'DL) y, one block per world, used for
qacc_smooth and by CG for M^-1 grad every iteration) races the same way in its forward pass: x[i] -= L(k, i) x[k]
from every descendant k at once. Its replacement gathers per target i in k-descending order (``mj_solveLD``'s
order); the diagonal and backward passes write each entry once per level and stay as MuJoCo Warp has them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy._vendor.mujoco_warp._src import smooth

from alloy.core.errors import PrecisionError
from alloy.precision.mjw_launch import Launch
from alloy.precision.mjw_rows import block_barrier
from alloy.precision.owned_cache import remember

_GROUPS: dict[int, tuple[wp.array, wp.array, wp.array]] = {}  # level updates ptr -> (rows, start, updates)
_SOLVE: dict[int, tuple[wp.array, wp.array, wp.array, wp.array]] = {}  # all_updates ptr -> (level groups, rows, start, upd)


def _group(updates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(target rows, CSR start offsets, updates sorted by row then k descending) of one level's (i, k, adr) list."""
    order = np.lexsort((-updates[:, 1], updates[:, 0]))
    upd = updates[order]
    rows, first = np.unique(upd[:, 0], return_index=True)
    start = np.append(first, len(upd)).astype(np.int32)
    return rows.astype(np.int32), start, upd.astype(np.int32)


def _check_level(updates: np.ndarray, level: int) -> None:
    """The layout the ordering argument needs (``io.put_model`` groups by the depth of i at the pin): within a level
    each k appears once, and no row a level writes (a target i) is read there as a k."""
    ks, rows = updates[:, 1], updates[:, 0]
    if len(np.unique(ks)) != len(ks) or np.intersect1d(ks, rows).size:
        raise PrecisionError(f"ordered qLD: level {level} of m.qLD_updates is not grouped by target depth")


def prepare_qld(m: Any) -> None:
    """Group every level's updates once per model (eager; before any capture). Raises ``PrecisionError`` (the route
    stays off) when a level does not have the layout the ordering needs."""
    for n, level in enumerate(getattr(m, "qLD_updates", ())):
        if level.ptr in _GROUPS or level.size == 0:
            continue
        updates = level.numpy().reshape(-1, 3)
        _check_level(updates, n)
        rows, start, upd = _group(updates)
        dev = level.device
        remember(_GROUPS, level.ptr, level, (wp.array(rows, dtype=int, device=dev),
                                             wp.array(start, dtype=int, device=dev),
                                             wp.array(upd, dtype=wp.vec3i, device=dev)))
    _prepare_solve(m)


def _prepare_solve(m: Any) -> None:
    """Every level's groups concatenated in level order (the order of ``m.qLD_all_updates``), for the fused solve."""
    all_updates = getattr(m, "qLD_all_updates", None)
    levels = getattr(m, "qLD_updates", ())
    if all_updates is None or not levels or all_updates.ptr in _SOLVE:
        return
    bounds, rows, starts, upds, base = [0], [], [], [], 0
    for level in levels:
        r, st, u = _group(level.numpy().reshape(-1, 3))
        rows.append(r)
        starts.append(st[:-1] + base)
        upds.append(u)
        base += len(u)
        bounds.append(bounds[-1] + len(r))
    start = np.append(np.concatenate(starts), base).astype(np.int32)
    dev = all_updates.device
    remember(_SOLVE, all_updates.ptr, all_updates, (wp.array(np.array(bounds, np.int32), dtype=int, device=dev),
                                                    wp.array(np.concatenate(rows), dtype=int, device=dev),
                                                    wp.array(start, dtype=int, device=dev),
                                                    wp.array(np.concatenate(upds), dtype=wp.vec3i, device=dev)))


@wp.kernel
def _qld_rows(M_rownnz: wp.array[int], M_rowadr: wp.array[int], rows: wp.array[int], start: wp.array[int],
              updates: wp.array[wp.vec3i], L: wp.array2d[float]):
    """One target row per thread: its updates in k-descending order, then the scaled M(k, i)."""
    worldid, g = wp.tid()
    i = rows[g]
    adr_i = M_rowadr[i]
    n = M_rownnz[i]
    for u in range(start[g], start[g + 1]):
        up = updates[u]
        k = up[1]
        adr_ki = up[2]
        adr_k = M_rowadr[k]
        tmp = L[worldid, adr_ki] / L[worldid, adr_k + M_rownnz[k] - 1]
        for j in range(n):
            L[worldid, adr_i + j] = L[worldid, adr_i + j] - L[worldid, adr_k + j] * tmp
        L[worldid, adr_ki] = tmp


def qld_handler(args: tuple, launch: Launch) -> None:
    """Replaces one level's ``_qLD_acc`` launch (route bits leave ATOMICS off, so L_in is L, as on the stock path)."""
    M_rownnz, M_rowadr, updates, L_in = launch.inputs
    L = launch.outputs[0]
    if L_in.ptr != L.ptr:
        raise PrecisionError("ordered qLD: L_in is not L (ATOMICS set on a routed model?)")
    groups = _GROUPS.get(updates.ptr)
    if groups is None:
        raise PrecisionError("ordered qLD: not prepared for this model (mjw_route.enable_route)")
    rows, start, upd = groups
    from alloy.precision.mjw_launch import original_launch

    original_launch(_qld_rows, dim=(int(L.shape[0]), int(rows.shape[0])),
                    inputs=[M_rownnz, M_rowadr, rows, start, upd], outputs=[L], device=L.device)


@wp.kernel
def _solve_ordered(qLD_block_adr: wp.array[int], L: wp.array2d[float], D: wp.array2d[float],
                   all_updates: wp.array[wp.vec3i], level_offsets: wp.array[int], level_groups: wp.array[int],
                   rows: wp.array[int], start: wp.array[int], updates: wp.array[wp.vec3i], y: wp.array2d[float],
                   x: wp.array2d[float]):
    """One block per world: MuJoCo Warp's fused sparse solve with an ordered forward pass."""
    worldid, lane = wp.tid()
    lanes = wp.block_dim()
    nv = qLD_block_adr.shape[0]
    nlevels = level_groups.shape[0] - 1
    for dofid in range(lane, nv, lanes):
        smooth._solve_LD_sparse_copy_fn(qLD_block_adr, y, worldid, dofid, x)
    block_barrier()
    for level in range(nlevels):  # forward, deepest target rows first: x[i] -= L(k, i) x[k], k descending
        n = nlevels - 1 - level
        for g in range(level_groups[n] + lane, level_groups[n + 1], lanes):
            i = rows[g]
            acc = x[worldid, i]
            for u in range(start[g], start[g + 1]):
                up = updates[u]
                acc = acc - L[worldid, up[2]] * x[worldid, up[1]]
            x[worldid, i] = acc
        block_barrier()
    for dofid in range(lane, nv, lanes):
        smooth._solve_LD_sparse_diag_fn(qLD_block_adr, D, x, worldid, dofid, x)
    block_barrier()
    for level in range(nlevels):  # backward: each k once per level (its one ancestor at that depth), no race
        for u in range(level_offsets[level] + lane, level_offsets[level + 1], lanes):
            up = all_updates[u]
            x[worldid, up[1]] = x[worldid, up[1]] - L[worldid, up[2]] * x[worldid, up[0]]
        block_barrier()


def solve_handler(args: tuple, launch: Launch) -> None:
    """Replaces the fused sparse solve launch; a model with no sparse levels launches MuJoCo Warp's kernel as is."""
    from alloy.precision.mjw_launch import launch_as_is

    qLD_block_adr, L, D, all_updates, level_offsets, y = launch.inputs
    groups = _SOLVE.get(all_updates.ptr)
    if groups is None:
        if int(args[1]) == 0:  # nlevels: no sparse-block dofs, nothing races
            launch_as_is(launch)
            return
        raise PrecisionError("ordered qLD solve: not prepared for this model (mjw_route.enable_route)")
    x = launch.outputs[0]
    level_groups, rows, start, upd = groups
    wp.launch_tiled(_solve_ordered, dim=int(x.shape[0]),
                    inputs=[qLD_block_adr, L, D, all_updates, level_offsets, level_groups, rows, start, upd, y],
                    outputs=[x], block_dim=int(launch.block_dim or 64), device=x.device)


def release() -> None:
    _GROUPS.clear()
    _SOLVE.clear()
