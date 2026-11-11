# SPDX-License-Identifier: Apache-2.0
# Portions copied from MuJoCo Warp (Apache-2.0, Copyright The MuJoCo Authors); see NOTICE.
"""Fixed64 sparse Newton Hessian for MuJoCo Warp (P3-1 round 5): an order-free ``solver._JTDACJ_sparse``.

MuJoCo Warp adds each constraint block's J^T D J (or elliptic cone curvature) entry into ``h`` with a float
``atomic_add``, so ``h`` rounds differently from run to run. ``jtdaj_handler`` replaces that launch (through
``mjw_launch``) with two:

1. the factory's entries (body copied from mujoco_warp@86e2fae0 solver.py:3092-3330, ``compact=False``), each added
   as ``int64(h * scale)`` into a packed upper-triangle accumulator. The scale and its bound are ``mjw_fixed``'s
   (computed once per solve). Integer atomics are associative. An entry above its bound bumps an overrun counter.
2. ``_flush``: ``h[w, i, j] += acc / scale`` for i <= j, four packed entries per thread, and ``acc`` back to 0.
"""

from typing import Any

import warp as wp

from alloy._vendor.mujoco_warp._src import math
from alloy._vendor.mujoco_warp._src import solver
from alloy._vendor.mujoco_warp._src import types
from alloy._vendor.mujoco_warp._src.solver import _elliptic_hessian_entry_from_projections

from alloy.core.errors import PrecisionError
from alloy.precision.mjw_fixed import FIX_BITS, scale_of, solve_bounds
from alloy.precision.mjw_launch import Launch, original_launch

_CONE = types.ConstraintState.CONE.value
_QUADRATIC = types.ConstraintState.QUADRATIC.value
_KERNELS: dict[tuple, Any] = {}


FLUSH_RUN = 4  # packed entries per flush thread


@wp.kernel
def _flush(
    npack: int, ctx_done_in: wp.array[bool], hbound_in: wp.array[float], efc_jtdaj_nblock_in: wp.array[int],
    acc_in: wp.array2d[wp.int64], h_out: wp.array3d[float], acc_out: wp.array2d[wp.int64],
):
    worldid, run = wp.tid()
    if ctx_done_in[worldid]:
        return
    scale = float(0.0)
    for q in range(FLUSH_RUN):
        k = run * FLUSH_RUN + q
        if k < npack:
            a = acc_in[worldid, k]
            if a != wp.int64(0):
                if scale == 0.0:
                    scale = scale_of(hbound_in[worldid], efc_jtdaj_nblock_in[worldid])
                j = int((wp.sqrt(float(8 * k + 1)) - 1.0) * 0.5)
                i = k - j * (j + 1) // 2
                h_out[worldid, i, j] = h_out[worldid, i, j] + wp.float32(a) / scale  # one owner per entry
                acc_out[worldid, k] = wp.int64(0)


def _make_terms(condim: int):
    @wp.func
    def func(
        opt_impratio_invsqrt: wp.array[float], efc_D_in: wp.array2d[float], fri: types.vec5,
        ctx_Jaref_in: wp.array2d[float], worldid: int, efcid0: int, block_rows: int,
    ) -> types.vec16:
        mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
        mu2 = mu * mu
        dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))
        if dm == 0.0:
            return types.vec16()
        n = ctx_Jaref_in[worldid, efcid0] * mu
        terms = types.vec16()
        terms[6] = mu
        tt = float(0.0)
        for dim in range(1, wp.static(condim)):
            if dim < block_rows:
                scale = fri[dim - 1]
                u = ctx_Jaref_in[worldid, efcid0 + dim] * scale
                terms[dim] = u
                terms[6 + dim] = scale
                tt += u * u
        t = wp.max(wp.sqrt(tt), types.MJ_MINVAL)
        inv_t = math.safe_div(1.0, t)
        for dim in range(1, wp.static(condim)):
            terms[dim] = terms[dim] * inv_t
        mu_n_over_t = mu * n * inv_t
        terms[12] = dm
        terms[13] = mu
        terms[14] = mu_n_over_t
        terms[15] = mu2 - mu_n_over_t
        return terms

    return func


def _make_entry(condim: int):
    @wp.func
    def func(efc_J_in: wp.array3d[float], terms: Any, rowadr: types.vec6i, worldid: int, pos1: int, pos2: int) -> float:
        z01 = terms[6] * efc_J_in[worldid, 0, rowadr[0] + pos1]
        z02 = terms[6] * efc_J_in[worldid, 0, rowadr[0] + pos2]
        projection1 = float(0.0)
        projection2 = float(0.0)
        tangent_dot = float(0.0)
        for dim in range(1, wp.static(condim)):
            z1 = terms[6 + dim] * efc_J_in[worldid, 0, rowadr[dim] + pos1]
            z2 = terms[6 + dim] * efc_J_in[worldid, 0, rowadr[dim] + pos2]
            projection1 += terms[dim] * z1
            projection2 += terms[dim] * z2
            tangent_dot += z1 * z2
        return _elliptic_hessian_entry_from_projections(
            terms[12], terms[13], terms[14], terms[15], z01, z02, projection1, projection2, tangent_dot
        )

    return func


def _jtdaj_fixed64(elliptic: bool, max_condim: int, block_dim: int) -> Any:
    key = (elliptic, max_condim, block_dim)
    if key not in _KERNELS:
        _KERNELS[key] = _build_jtdaj(elliptic, max_condim, block_dim)
    return _KERNELS[key]


def _build_jtdaj(elliptic: bool, max_condim: int, block_dim: int) -> Any:
    ELLIPTIC = elliptic
    MAX_CONDIM = max_condim
    terms_fn = _make_terms(max_condim)
    entry_fn = _make_entry(max_condim)

    @wp.kernel(module="unique", module_options={"enable_backward": False, "block_dim": block_dim}, grid_stride=True)
    def kernel(
        opt_impratio_invsqrt: wp.array[float], contact_friction_in: wp.array[types.vec5], contact_dim_in: wp.array[int],
        efc_id_in: wp.array2d[int], efc_jtdaj_adr_in: wp.array2d[int], efc_jtdaj_nrow_in: wp.array2d[int],
        efc_jtdaj_nblock_in: wp.array[int], efc_J_rownnz_in: wp.array2d[int], efc_J_rowadr_in: wp.array2d[int],
        efc_J_colind_in: wp.array3d[int], efc_J_in: wp.array3d[float], efc_D_in: wp.array2d[float],
        efc_state_in: wp.array2d[int], dof_cdof_in: wp.array2d[int], ctx_Jaref_in: wp.array2d[float],
        ctx_done_in: wp.array[bool], groups_per_world: int, hbound_in: wp.array[float],
        acc_out: wp.array2d[wp.int64], over_out: wp.array[int],
    ):
        worldid, slot, lane = wp.tid()
        if wp.static(ELLIPTIC):
            lanes = wp.block_dim()
        else:
            lanes = wp.static(solver._JTDAJ_THREADS_PER_GROUP)
        if ctx_done_in[worldid]:
            return
        count = efc_jtdaj_nblock_in[worldid]
        scale = scale_of(hbound_in[worldid], count)
        limit = 2.0 * wp.pow(2.0, FIX_BITS) / float(wp.max(count, 1))
        for groupid in range(slot, count, groups_per_world):
            head_row = efc_jtdaj_adr_in[worldid, groupid]
            block_rows = efc_jtdaj_nrow_in[worldid, groupid]
            head_adr = efc_J_rowadr_in[worldid, head_row]
            support = efc_J_rownnz_in[worldid, head_row]
            n_entries = support * (support + 1) // 2
            is_cone = False
            if wp.static(ELLIPTIC):
                is_cone = efc_state_in[worldid, head_row] == _CONE
                cone_rowadr = types.vec6i(head_adr, head_adr, head_adr, head_adr, head_adr, head_adr)
                if is_cone:
                    conid = efc_id_in[worldid, head_row]
                    for dim in range(1, wp.static(MAX_CONDIM)):
                        if dim < block_rows:
                            cone_rowadr[dim] = head_adr + dim * support
                    local_terms = types.vec16()
                    if lane == 0:
                        local_terms = terms_fn(
                            opt_impratio_invsqrt, efc_D_in, contact_friction_in[conid], ctx_Jaref_in, worldid,
                            head_row, block_rows
                        )
                    cone_terms = wp.tile_zeros(shape=(16,), dtype=float, storage="shared")
                    for term_id in range(1, 16):
                        wp.tile_scatter_masked(cone_terms, term_id, local_terms[term_id], lane == 0)
            for entry in range(lane, n_entries, lanes):
                block_col = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
                block_row = entry - block_col * (block_col + 1) // 2
                dof_row = efc_J_colind_in[worldid, 0, head_adr + block_row]
                dof_col = efc_J_colind_in[worldid, 0, head_adr + block_col]
                hval = float(0.0)
                if wp.static(ELLIPTIC):
                    if is_cone:
                        hval = entry_fn(efc_J_in, cone_terms, cone_rowadr, worldid, block_row, block_col)
                if not is_cone:
                    for member in range(block_rows):
                        member_row = head_row + member
                        if efc_state_in[worldid, member_row] == _QUADRATIC:
                            member_adr = efc_J_rowadr_in[worldid, member_row]
                            hval += (efc_J_in[worldid, 0, member_adr + block_row] * efc_D_in[worldid, member_row]
                                              * efc_J_in[worldid, 0, member_adr + block_col])
                if hval != 0.0:
                    v = hval * scale
                    if wp.abs(v) > limit:
                        wp.atomic_add(over_out, worldid, 1)
                    lo, hi = wp.min(dof_row, dof_col), wp.max(dof_row, dof_col)
                    wp.atomic_add(acc_out, worldid, hi * (hi + 1) // 2 + lo, wp.int64(v))

    return kernel


def jtdaj_handler(args: tuple, launch: Launch) -> None:
    """Replacement for one launch of a ``_JTDACJ_sparse(compact, cone_type, max_condim, block_dim, ...)`` kernel."""
    compact, cone_type, max_condim = args[0], args[1], args[2]
    if compact:
        raise PrecisionError("Fixed64 Hessian: the compact (sleep) path is not covered")
    h = launch.outputs[0]
    nblock, J, done = launch.inputs[6], launch.inputs[10], launch.inputs[15]
    s = solve_bounds(J)
    kernel = _jtdaj_fixed64(cone_type == types.ConeType.ELLIPTIC, int(max_condim), launch.block_dim)
    original_launch(kernel, dim=launch.dim, inputs=launch.inputs + [s["hbound"]], outputs=[s["hacc"], s["over"]],
                    block_dim=launch.block_dim)
    nw, npack = s["hacc"].shape
    original_launch(_flush, dim=(nw, (npack + FLUSH_RUN - 1) // FLUSH_RUN),
                    inputs=[npack, done, s["hbound"], nblock, s["hacc"]], outputs=[h, s["hacc"]])
