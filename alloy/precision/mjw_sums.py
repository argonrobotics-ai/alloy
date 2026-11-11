# SPDX-License-Identifier: Apache-2.0
"""Order-free replacements for MuJoCo Warp's other float sums on the covered path (P3-1 round 5).

Each handler replaces one launch through ``mjw_launch`` (same inputs, same outputs, a fixed summation order):

- ``grad_handler`` (``solver._update_gradient_grad``): grad_dot is a float ``atomic_add`` over dofs. MuJoCo Warp's
  tiled grad kernel (fixed-order lane sums and tile reduction, used by its CG path) runs instead.
- ``qfrc_handler`` (``solver._update_constraint_init_qfrc_constraint_sparse``): qfrc_constraint = J^T force,
  scattered over dofs with float atomics. Now Fixed64 in one block-per-world launch: the bound (tile max of
  |force| * max |J| of the row, the row max from ``mjw_fixed.solve_bounds``), ``mjw_fixed.scale_of``, int64 atomics,
  and after a block barrier the flush of acc / scale per dof.
- ``jqvel_handler`` (``constraint._efc_contact_jac_sparse``): contact rows' Jqvel is the sum of 8 lanes' partial
  sums, added with float atomics. After the kernel, one thread per contact row rewrites it in row-entry order.
- ``tree_handler`` (``smooth._subtree_com_acc``, ``_crb_accumulate``, ``_cfrc_backward``,
  ``set_const._accumulate_subtreemass`` at mujoco_warp@86e2fae0): each tree level adds every body into its parent with
  a float atomic, one launch per level. Bodies are numbered depth-first, so a subtree is the range [b, b + size); the
  deepest level's launch now runs every range sum (body order) in one launch and the other levels launch nothing.
  Upstream did the same in #1872, after our pin.

The range sums round differently from MuJoCo Warp's level sums (same terms, another order).
"""

from typing import Any

import numpy as np
import warp as wp

from alloy.core.errors import PrecisionError
from alloy.precision.mjw_fixed import BOUND_LANES, scale_of, solve_bounds
from alloy.precision.mjw_launch import Launch, launch_as_is, original_launch
from alloy.precision.mjw_rows import block_barrier
from alloy.precision.owned_cache import remember

_SUBTREE: dict[int, wp.array] = {}  # body_parentid.ptr -> body_subtreenum
_DEEPEST: dict[int, int] = {}  # body_parentid.ptr -> ptr of the deepest body_tree level (the loops start there)
_SNAP: dict[int, wp.array] = {}  # tree array ptr -> scratch for its range sums
_TREE_KERNELS: dict[tuple, Any] = {}
TREE_LANES = 64


GRAD_LANES = 32


def grad_handler(args: tuple, launch: Launch) -> None:
    """Replacement for ``_update_gradient_grad(stable_fast, ...)``: MuJoCo Warp's own tiled grad kernel (lanes sum
    strided dofs in order, then a fixed-tree ``tile_reduce``) writes grad and grad_dot."""
    from alloy._vendor.mujoco_warp._src.solver import _update_gradient_grad_tiled

    if args[0]:
        raise PrecisionError("ordered grad_dot: the stable fast path (incremental H) is not covered")
    nworld, nv = int(launch.dim[0]), int(launch.dim[1])
    smooth, constraint, ma, _changed, done = launch.inputs
    wp.launch_tiled(_update_gradient_grad_tiled, dim=nworld, inputs=[nv, smooth, constraint, ma, done],
                    outputs=launch.outputs, block_dim=GRAD_LANES)




@wp.kernel
def _qfrc_tile(
    nv: int, nefc_in: wp.array[int], rowmax_in: wp.array2d[float], efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int], efc_J_colind_in: wp.array3d[int], efc_J_in: wp.array3d[float],
    efc_force_in: wp.array2d[float], state_changed_count_in: wp.array[int], ctx_done_in: wp.array[bool],
    acc: wp.array2d[wp.int64], qfrc_constraint_out: wp.array2d[float],
):
    """One block per world, one launch: bound (tile max of |force| * max |J|), int64 atomics at the scale, then
    the flush after the block barrier that ``tile_reduce`` makes."""
    worldid, lane = wp.tid()
    if ctx_done_in[worldid] or state_changed_count_in[worldid] == 0:
        return
    n = nefc_in[worldid]
    lanes = wp.block_dim()
    local = float(0.0)
    for efcid in range(lane, n, lanes):
        local = wp.max(local, wp.abs(efc_force_in[worldid, efcid]) * rowmax_in[worldid, efcid])
    best = wp.tile_reduce(wp.max, wp.tile(local, preserve_type=True))
    scale = scale_of(best[0], n)
    for efcid in range(lane, n, lanes):
        force = efc_force_in[worldid, efcid]
        if force != 0.0:
            rowadr = efc_J_rowadr_in[worldid, efcid]
            for i in range(efc_J_rownnz_in[worldid, efcid]):
                v = efc_J_in[worldid, 0, rowadr + i] * force * scale
                wp.atomic_add(acc, worldid, efc_J_colind_in[worldid, 0, rowadr + i], wp.int64(v))
    done = wp.tile_reduce(wp.add, wp.tile(int(0), preserve_type=True))  # block barrier: every add has landed
    if done[0] == 0:
        for dofid in range(lane, nv, lanes):
            a = acc[worldid, dofid]
            if a != wp.int64(0):
                qfrc_constraint_out[worldid, dofid] = qfrc_constraint_out[worldid, dofid] + wp.float32(a) / scale
                acc[worldid, dofid] = wp.int64(0)


def qfrc_handler(args: tuple, launch: Launch) -> None:
    """Replacement for ``_update_constraint_init_qfrc_constraint_sparse(compact, nv, ...)``."""
    if args[0]:
        raise PrecisionError("Fixed64 qfrc_constraint: the compact (sleep) path is not covered")
    nefc, rownnz, rowadr, colind, J, force, _cdof, changed, done = launch.inputs
    qfrc = launch.outputs[0]
    s = solve_bounds(J)
    wp.launch_tiled(_qfrc_tile, dim=int(launch.dim[0]),
                    inputs=[int(qfrc.shape[1]), nefc, s["rowmax"], rownnz, rowadr, colind, J, force, changed, done],
                    outputs=[s["qacc"], qfrc], block_dim=BOUND_LANES)


@wp.kernel
def _contact_jqvel(
    nacon_in: wp.array[int], contact_worldid_in: wp.array[int], contact_efc_address_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int], efc_J_rowadr_in: wp.array2d[int], efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float], qvel_in: wp.array2d[float], efc_Jqvel_out: wp.array2d[float],
):
    conid, dimid = wp.tid()
    if conid >= nacon_in[0]:
        return
    efcid = contact_efc_address_in[conid, dimid]
    if efcid < 0:
        return
    worldid = contact_worldid_in[conid]
    adr = efc_J_rowadr_in[worldid, efcid]
    jqvel = float(0.0)
    for k in range(efc_J_rownnz_in[worldid, efcid]):
        jqvel += efc_J_in[worldid, 0, adr + k] * qvel_in[worldid, efc_J_colind_in[worldid, 0, adr + k]]
    efc_Jqvel_out[worldid, efcid] = jqvel


def jqvel_handler(args: tuple, launch: Launch) -> None:
    """``constraint._efc_contact_jac_sparse(cone, nmaxdim, ...)``: its 8 lanes add partial J * qvel sums into each
    contact row with float atomics. The kernel runs as is (J and colind are written per entry), then one thread per
    contact rewrites Jqvel for its rows, summing the row's entries in order."""
    launch_as_is(launch)
    nacon = launch.arg("nacon_in")
    jqvel = launch.arg("efc_Jqvel_out")
    original_launch(_contact_jqvel, dim=(int(launch.arg("naconmax_in")), int(args[1])),
                    inputs=[nacon, launch.arg("worldid_in"), launch.arg("contact_efc_address_in"),
                            launch.arg("efc_J_rownnz_in"), launch.arg("efc_J_rowadr_in"),
                            launch.arg("efc_J_colind_out"), launch.arg("efc_J_out"), launch.arg("qvel_in")],
                    outputs=[jqvel])


def _tree_kernel(dtype: Any, skip_world_parent: bool) -> Any:
    """Range sums: body b's value becomes the sum of the own values in [b, b + body_subtreenum[b]), in body order
    (the world body's range is every body). ``skip_world_parent``: roots do not add into the world (crb)."""
    key = (dtype, skip_world_parent)
    if key in _TREE_KERNELS:
        return _TREE_KERNELS[key]
    SKIP = skip_world_parent

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(body_subtreenum: wp.array(dtype=int), value_in: wp.array2d(dtype=dtype),
               snap: wp.array2d(dtype=dtype), value_out: wp.array2d(dtype=dtype)):
        worldid, lane = wp.tid()  # one block per world: sums into the scratch, barrier, then back (in == out)
        first = int(0)
        if wp.static(SKIP):
            first = 1
        nbody, lanes = value_in.shape[1], wp.block_dim()
        for bodyid in range(first + lane, nbody, lanes):
            acc = value_in[worldid, bodyid]
            for k in range(bodyid + 1, bodyid + body_subtreenum[bodyid]):
                acc += value_in[worldid, k]
            snap[worldid, bodyid] = acc
        block_barrier()
        for bodyid in range(first + lane, nbody, lanes):
            value_out[worldid, bodyid] = snap[worldid, bodyid]

    _TREE_KERNELS[key] = kernel
    return kernel


def subtree_sizes(parent: np.ndarray) -> np.ndarray:
    """Bodies in each body's subtree; raises unless every subtree is the contiguous range [b, b + size)."""
    n = len(parent)
    size = np.ones(n, dtype=np.int32)
    for b in range(n - 1, 0, -1):
        if not 0 <= parent[b] < b:
            raise PrecisionError(f"body {b} has parent {parent[b]}: bodies are not numbered depth-first")
        size[parent[b]] += size[b]
    for b in range(1, n):
        p = parent[b]
        if not p < b < p + size[p]:
            raise PrecisionError(f"body {b} lies outside its parent's subtree range: not depth-first")
    return size


def prepare_tree(m: Any, values: tuple) -> None:
    """Upload ``body_subtreenum``, note the deepest level and allocate a range-sum scratch per tree array (host
    read; call before any graph capture). ``values``: the arrays the level loops accumulate (subtree_com, crb, cfrc_int,
    body_subtreemass)."""
    parentid = m.body_parentid
    if parentid.ptr not in _SUBTREE:
        size = subtree_sizes(parentid.numpy())
        remember(_SUBTREE, parentid.ptr, parentid, wp.array(size, dtype=int, device=parentid.device))
    if len(m.body_tree):
        remember(_DEEPEST, parentid.ptr, parentid, m.body_tree[len(m.body_tree) - 1].ptr)
    for v in values:
        if v.ptr not in _SNAP:
            remember(_SNAP, v.ptr, v, wp.empty_like(v))


def tree_handler(dtype: Any, skip_world_parent: bool):
    """Handler for the level launches of one tree accumulation over ``dtype`` values: the first (deepest) level
    launch runs every range sum in one block-per-world launch; the other levels launch nothing.
    One launch per stage instead of one per tree level (MuJoCo Warp #1872 does the same upstream)."""
    kernel = _tree_kernel(dtype, skip_world_parent)

    def handle(args: tuple, launch: Launch) -> None:
        parentid, value_in, body_tree = launch.inputs
        value = launch.outputs[0]
        subtree, snap = _SUBTREE.get(parentid.ptr), _SNAP.get(value.ptr)
        if subtree is None or snap is None:
            raise PrecisionError("ordered tree sums: not prepared for this model (mjw_route.enable_route)")
        if body_tree.ptr != _DEEPEST.get(parentid.ptr):
            return  # this level is already in the range sums
        wp.launch_tiled(kernel, dim=int(value.shape[0]), inputs=[subtree, value_in, snap], outputs=[value],
                        block_dim=TREE_LANES)

    return handle


def release() -> None:
    _SUBTREE.clear()
    _DEEPEST.clear()
    _SNAP.clear()
