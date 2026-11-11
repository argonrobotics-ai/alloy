# SPDX-License-Identifier: Apache-2.0
"""Shared Fixed64 plumbing for the ordered SolverMuJoCo route (P3-1 round 5): bounds, scale rule and scratch.

A Fixed64 sum adds every term as ``int64(term * scale)``. Integer atomics are associative, so the sum does not
depend on the order the terms arrive in. The scale is a power of two per world, 2^(60 - ceil(log2(bound * n))),
where ``bound`` caps one term and ``n`` the number of terms, so no sum can pass 2^60 (``scale_of``).

The bounds depend only on what is fixed during one solve (D, J, contact friction), so they are computed once per
``solver.solve`` (``begin_solve`` marks them stale; the first Fixed64 launch of the solve runs ``_solve_bound``):
- ``hbound``: max over JTDAJ blocks of rows * max |D| * max |J|^2, or F * max |D| * max |J|^2 for an elliptic
  contact block, F = 1 + 2 S + 2 S^2 / mu^2 (S^2: sum of squared tangent scales). In the cone zone mu * n / t lies
  in (-1, mu^2), which bounds the cone curvature entry by F * D * |J|^2; ``max(rows, F)`` covers either state.
- ``rowmax``: max |J| per row, so the per-call qfrc_constraint bound is max |force| * rowmax (``mjw_sums``).
Everything is allocated by ``register`` (eager, before any capture). The registry holds weak references to the
Model and Data, and an entry goes when its Data is freed (``owned_cache.remember``), so a dropped simulation is not
kept alive (WP8 2026-11-05: a long-lived scorer process grew its memory with every build).
"""

import weakref
from typing import Any

import warp as wp

from alloy._vendor.mujoco_warp._src import math
from alloy._vendor.mujoco_warp._src import types

from alloy.core.errors import PrecisionError
from alloy.precision.owned_cache import remember

FIX_BITS = 60.0
_ELLIPTIC = types.ConstraintType.CONTACT_ELLIPTIC.value
_REG: dict[int, dict[str, Any]] = {}  # d.efc.J.ptr -> {"m", "d" (weak refs), scratch arrays, "fresh"}


@wp.func
def scale_of(bound: float, count: int) -> float:
    """2^(60 - ceil(log2(bound * count))) (1 when nothing is bounded)."""
    total = bound * float(wp.max(count, 1))
    if total <= 0.0:
        return 1.0
    return wp.pow(2.0, wp.min(FIX_BITS - wp.ceil(wp.log2(total)), 120.0))


BOUND_LANES = 128  # lanes per world for the per-world bound reductions


@wp.func
def _block_bound(opt_impratio_invsqrt: wp.array[float], contact_friction_in: wp.array[types.vec5],
                 contact_dim_in: wp.array[int], efc_type_in: wp.array2d[int], efc_id_in: wp.array2d[int],
                 efc_jtdaj_adr_in: wp.array2d[int], efc_jtdaj_nrow_in: wp.array2d[int],
                 efc_J_rownnz_in: wp.array2d[int], efc_J_rowadr_in: wp.array2d[int], efc_J_in: wp.array3d[float],
                 efc_D_in: wp.array2d[float], worldid: int, idx: int) -> float:
    """max(rows, F) * max |D| * max |J|^2 of JTDAJ block ``idx``."""
    head = efc_jtdaj_adr_in[worldid, idx]
    rows = efc_jtdaj_nrow_in[worldid, idx]
    support = efc_J_rownnz_in[worldid, head]
    dmax = float(0.0)
    jmax = float(0.0)
    for member in range(rows):
        row = head + member
        dmax = wp.max(dmax, wp.abs(efc_D_in[worldid, row]))
        adr = efc_J_rowadr_in[worldid, row]
        for k in range(support):
            jmax = wp.max(jmax, wp.abs(efc_J_in[worldid, 0, adr + k]))
    factor = float(rows)
    if efc_type_in[worldid, head] == _ELLIPTIC:
        conid = efc_id_in[worldid, head]
        fri = contact_friction_in[conid]
        mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
        s2 = float(0.0)
        for k in range(contact_dim_in[conid] - 1):
            s2 += fri[k] * fri[k]
        factor = wp.max(factor, 1.0 + 2.0 * wp.sqrt(s2) + 2.0 * math.safe_div(s2, mu * mu))
    return factor * dmax * jmax * jmax


@wp.kernel
def _solve_bound(
    opt_impratio_invsqrt: wp.array[float], contact_friction_in: wp.array[types.vec5], contact_dim_in: wp.array[int],
    efc_type_in: wp.array2d[int], efc_id_in: wp.array2d[int], efc_jtdaj_adr_in: wp.array2d[int],
    efc_jtdaj_nrow_in: wp.array2d[int], efc_jtdaj_nblock_in: wp.array[int], nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int], efc_J_rowadr_in: wp.array2d[int], efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float], hbound_out: wp.array[float], rowmax_out: wp.array2d[float],
):
    """One block per world: rowmax per row, and hbound as a tile max over the blocks (no zeroing launch)."""
    worldid, lane = wp.tid()
    lanes = wp.block_dim()
    for idx in range(lane, wp.min(nefc_in[worldid], rowmax_out.shape[1]), lanes):
        adr = efc_J_rowadr_in[worldid, idx]
        jrow = float(0.0)
        for k in range(efc_J_rownnz_in[worldid, idx]):
            jrow = wp.max(jrow, wp.abs(efc_J_in[worldid, 0, adr + k]))
        rowmax_out[worldid, idx] = jrow
    local = float(0.0)
    for idx in range(lane, wp.min(efc_jtdaj_nblock_in[worldid], efc_jtdaj_adr_in.shape[1]), lanes):
        local = wp.max(local, _block_bound(opt_impratio_invsqrt, contact_friction_in, contact_dim_in, efc_type_in,
                                           efc_id_in, efc_jtdaj_adr_in, efc_jtdaj_nrow_in, efc_J_rownnz_in,
                                           efc_J_rowadr_in, efc_J_in, efc_D_in, worldid, idx))
    best = wp.tile_reduce(wp.max, wp.tile(local, preserve_type=True))
    if lane == 0:
        hbound_out[worldid] = best[0]


def register(m: Any, d: Any) -> None:
    """Allocate a Data's Fixed64 scratch (eager; call before any graph capture)."""
    key = d.efc.J.ptr
    if key in _REG:
        return
    nw, nj, nvp, dev = int(d.nworld), int(d.njmax), int(m.nv_pad), d.qpos.device
    s = {"m": weakref.ref(m), "d": weakref.ref(d), "fresh": False,
         "hacc": wp.zeros((nw, nvp * (nvp + 1) // 2), dtype=wp.int64, device=dev),
         "hbound": wp.zeros(nw, dtype=float, device=dev),
         "rowmax": wp.zeros((nw, max(nj, 1)), dtype=float, device=dev),
         "over": wp.zeros(nw, dtype=int, device=dev),
         "qacc": wp.zeros(tuple(d.qfrc_constraint.shape), dtype=wp.int64, device=dev),
         "qbound": wp.zeros(nw, dtype=float, device=dev)}
    remember(_REG, key, d, s)  # dropped when d is freed, before its J's address can be reused


def entry(J: wp.array) -> dict[str, Any]:
    """The registered scratch of the Data whose ``efc.J`` is ``J``."""
    s = _REG.get(J.ptr)
    if s is None:
        raise PrecisionError("Fixed64 scratch not registered for this Data (mjw_route.enable_route prepares it)")
    return s


def begin_solve(m: Any, d: Any, *args: Any, **kwargs: Any) -> None:
    """``solver.solve`` entry hook: this solve's bounds are not computed yet."""
    s = _REG.get(d.efc.J.ptr)
    if s is not None:
        s["fresh"] = False


def solve_bounds(J: wp.array) -> dict[str, Any]:
    """Scratch with this solve's ``hbound`` and ``rowmax``, computed on first use in the solve."""
    s = entry(J)
    if not s["fresh"]:
        m, d = s["m"](), s["d"]()
        if m is None or d is None:
            raise PrecisionError("Fixed64 scratch outlived its Model or Data (a dropped simulation)")
        e = d.efc
        wp.launch_tiled(_solve_bound, dim=int(d.nworld),
                        inputs=[m.opt.impratio_invsqrt, d.contact.friction, d.contact.dim, e.type, e.id,
                                e.jtdaj_adr, e.jtdaj_nrow, e.jtdaj_nblock, d.nefc, e.J_rownnz, e.J_rowadr, e.J, e.D],
                        outputs=[s["hbound"], s["rowmax"]], block_dim=BOUND_LANES)
        s["fresh"] = True
    return s


def overruns() -> int:
    """Hessian entries above their Fixed64 bound since registration (0 expected; a host read)."""
    return sum(int(s["over"].numpy().sum()) for s in _REG.values())


def release() -> None:
    _REG.clear()
