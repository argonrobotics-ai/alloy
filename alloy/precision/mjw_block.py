# SPDX-License-Identifier: Apache-2.0
"""Block-sparse Newton factorization for MuJoCo Warp (SCR-WP13-SPARSE-1; coordinator ruling 2026-11-08 15:39Z).

MuJoCo Warp factors the Newton Hessian densely, one thread block per world: nv³ per Newton iteration, 69 % of a
settled 50-object pile's GPU time and 92 % at 100 objects (SPEED-1). For models whose kinematic trees have at most 6
dofs each, this factors the 6x6-block sparse Hessian instead (``mjw_block_kernels``): 36-43x faster per factor +
solve at 100 objects x 64 worlds on an L4, the same float32 accuracy (step-1 microbench, scr/WP13.md).

How it runs (no vendored byte changes):
- ``install`` wraps ``mujoco_warp.put_model`` (an eligible model gets ``m.alloy_block_newton``, a ``BlockConfig``)
  and ``put_data`` (an eligible Data gets its buffers then, never inside a graph capture; they are keyed on its
  ``solver_niter`` array: the DISCRETE integrator rebuilds ``d`` with ``dataclasses.replace``, its arrays stay).
- ``alloy.precision.mjw_chol._guarded`` (MuJoCo Warp's ``_cholesky_factorize_solve`` replacement) calls
  ``factor_solve`` when the model carries the config and neither skip flag is set.
- Per call: ``symbolic`` (only where ``solver_niter == 0``: the first factorizations of each solve; the pattern comes
  from every efc row, so it holds while row states flip; it orders only where the tree graph differs from the one
  last ordered, ``graph_changed``: a resting pile keeps its lists, bit for bit; ``ALLOY_MJW_BLOCK_REUSE=0`` orders
  every solve); the float32 factor + solve; the float64 redo of any world float32 flagged (a pivot under mjMINVAL,
  or non-finite sums), which adds 1 to mjw_chol's per-world clamp counter (FALLBACK_USED, as the dense guard
  counts); MuJoCo Warp's dense path for worlds that overflowed a cap.
- Every launch honours the ``ctx.done`` it is handed (WP7's lazy factor passes done | skip).
- The block path never writes the dense ``ctx.hfactor``. Only the incremental Newton path reads it back (the skip
  flags; MuJoCo Warp takes it for every cone but the elliptic one), so a pyramidal-cone model is not eligible.

Eligible: Newton, elliptic cone, sparse constraint Jacobian (MuJoCo Warp's ``is_sparse``), every tree at most 6 dofs,
SLEEP off, nv >= ``NV_MIN``. Off: ``ALLOY_MJW_BLOCK=0``. Other models never see this module and stay bitwise
unchanged.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from typing import Any, Callable

import numpy as np
import warp as wp

from alloy.precision import mjw_block_kernels as K
from alloy.precision.owned_cache import remember

log = logging.getLogger("alloy.precision")

NV_MIN = 96                       # below this the dense path stays (step 3 measures the crossover)
BLOCK_DIM = 64                    # threads per world in the factor kernel (step 1: 32-128 equal at 50-100 trees)
SYM_BLOCK_DIM = 1                 # threads per block, symbolic (L4: 1 is 1.6-2.1x faster than 32 or 256)
GRAPH_LANES = 256                 # threads per world building the graph (a launch where nothing runs stays cheap)
UNSEEN = 2                        # ``seen``'s first value: no graph matches it (graph_changed)
_BUFFERS: dict[int, "Buffers"] = {}
_STATE: dict[str, Any] = {}
_KERNELS: dict[Any, Any] = {}


@dataclasses.dataclass(frozen=True)
class BlockConfig:
    ntree: int
    off_cap: int
    pair_cap: int


def eligible(mjm: Any) -> str | None:
    """None when MuJoCo Warp's Newton Hessian of ``mjm`` (a mujoco.MjModel) can take the block path, else why not."""
    import mujoco  # noqa: PLC0415

    if os.environ.get("ALLOY_MJW_BLOCK", "1") == "0":
        return "ALLOY_MJW_BLOCK=0"
    if mjm.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
        return "not the Newton solver"
    if mjm.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC:
        return "pyramidal cone (the incremental Newton path reuses the dense factor)"
    if mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
        return "sleeping is on"
    if mjm.nv < NV_MIN:
        return f"nv {mjm.nv} < {NV_MIN}"
    if mjm.ntree == 0 or int(np.max(mjm.tree_dofnum[:mjm.ntree])) > K.B:
        return f"a kinematic tree has more than {K.B} dofs"
    from alloy._vendor.mujoco_warp._src import io  # noqa: PLC0415

    return None if io.is_sparse(mjm) else "dense constraint Jacobian"


def config_for(ntree: int) -> BlockConfig:
    off_cap = max(min(ntree * (ntree - 1) // 2, 16 * ntree), 1)
    return BlockConfig(ntree, off_cap, max(64 * ntree, 256))


@dataclasses.dataclass
class Buffers:
    """One Data's symbolic lists and block storage (allocated eagerly, never inside a graph capture)."""

    cfg: BlockConfig
    arrays: dict[str, wp.array]

    @classmethod
    def allocate(cls, cfg: BlockConfig, nworld: int, device: Any) -> "Buffers":
        W, T, O, P = nworld, cfg.ntree, cfg.off_cap, cfg.pair_cap
        i32 = functools.partial(wp.zeros, dtype=wp.int32, device=device)
        i8 = functools.partial(wp.zeros, dtype=wp.int8, device=device)
        a = {"adj": i32((W, T, T)), "edge": i8((W, T, T)), "changed": i32(W),
             "seen": wp.full((W, T, T), UNSEEN, dtype=wp.int8, device=device), "deg": i32((W, T)),
             "left": i32((W, T)), "order": i32((W, T)), "pos": i32((W, T)), "nbuf": i32((W, T)),
             "col_ptr": i32((W, T + 1)), "col_row": i32((W, O)), "row_ptr": i32((W, T + 1)), "row_blk": i32((W, O)),
             "off_row": i32((W, O)), "off_col": i32((W, O)), "level": i32((W, T)), "nlev": i32(W),
             "lvl_node": i32((W, T + 1)), "node_item": i32((W, T)), "lvl_off": i32((W, T + 1)),
             "off_item": i32((W, O)), "opair_ptr": i32((W, O + 1)), "opair_a": i32((W, P)), "opair_b": i32((W, P)),
             "overflow": i32(W), "bad": i32(W),
             "mask": wp.zeros(W, dtype=bool, device=device)}
        for tag, dtype in (("32", wp.float32), ("64", wp.float64)):
            _, M, V = kernels(dtype)
            a.update({f"a_diag{tag}": wp.zeros((W, T), dtype=M, device=device),
                      f"lo{tag}": wp.zeros((W, O), dtype=M, device=device),
                      f"dinv{tag}": wp.zeros((W, T), dtype=M, device=device),
                      f"g{tag}": wp.zeros((W, T), dtype=V, device=device),
                      f"y{tag}": wp.zeros((W, T), dtype=V, device=device),
                      f"x{tag}": wp.zeros((W, T), dtype=V, device=device)})
        return cls(cfg, a)


def kernels(dtype: Any) -> tuple[Any, Any, Any]:
    if dtype not in _KERNELS:
        _KERNELS[dtype] = K.factor_kernel(dtype)
    return _KERNELS[dtype]


def buffers_for(m: Any, d: Any) -> Buffers | None:
    b = _BUFFERS.get(id(d.solver_niter))
    if b is not None or wp.get_device(d.qpos.device).is_capturing:
        return b
    cfg = getattr(m, "alloy_block_newton", None)
    if cfg is None:
        return None
    b = Buffers.allocate(cfg, int(d.nworld), d.qpos.device)
    return remember(_BUFFERS, id(d.solver_niter), d.solver_niter, b)


@wp.kernel
def _dense_mask(done: wp.array[bool], overflow: wp.array[wp.int32], mask: wp.array[bool]):
    w = wp.tid()
    mask[w] = done[w] or overflow[w] == 0             # the dense path runs only where the block path overflowed


# --- the tree graph, and when the symbolic step can be reused (WP7's S1) ---------------------------------
# Where solver_niter == 0 these build the world's tree graph from every efc row into ``edge`` (0/1) and compare it
# with ``seen``, the graph its last ordering used. The same graph: ``changed`` stays 0 and ``K.symbolic`` keeps
# every list it wrote (a function of the graph and the fixed caps alone, so the factor reads what a fresh
# ordering writes, bit for bit). A new graph (or the first: ``seen`` starts at UNSEEN): it goes into ``seen`` and
# ``adj`` and the ordering runs. Decided on the device, so a captured CUDA graph replays it.


@wp.kernel
def graph_clear(ntree: int, lanes: int, solver_niter: wp.array[int], edge: wp.array3d[wp.int8],
                changed: wp.array[wp.int32]):
    """No edges yet, nothing changed yet. A launch where nothing runs costs a few threads."""
    w, lane = wp.tid()
    if solver_niter[w] != 0:
        return
    if lane == 0:
        changed[w] = 0
    for k in range(lane, ntree * ntree, lanes):
        edge[w, k // ntree, k % ntree] = wp.int8(0)


@wp.func
def _row_edges(w: int, r: int, rownnz: wp.array2d[int], rowadr: wp.array2d[int], colind: wp.array3d[int],
               dof_treeid: wp.array[int], edge: wp.array3d[wp.int8]):
    """An edge for every pair of distinct trees row ``r`` touches. A row's dofs come grouped by tree, so runs of one
    tree are skipped (any order stays correct: an edge is only written more than once)."""
    s0 = rowadr[w, r]
    n = rownnz[w, r]
    prev = int(-1)
    for p in range(n):
        ta = dof_treeid[colind[w, 0, s0 + p]]
        if ta != prev:
            last = ta
            for q in range(p + 1, n):
                tb = dof_treeid[colind[w, 0, s0 + q]]
                if tb != last and tb != ta:
                    edge[w, ta, tb] = wp.int8(1)
                    edge[w, tb, ta] = wp.int8(1)
                last = tb
        prev = ta


@wp.kernel
def graph_rows(lanes: int, solver_niter: wp.array[int], nefc: wp.array[int], rownnz: wp.array2d[int],
               rowadr: wp.array2d[int], colind: wp.array3d[int], dof_treeid: wp.array[int],
               edge: wp.array3d[wp.int8]):
    """``lanes`` threads per world stride over its efc rows (every row, whatever its state, so the pattern holds
    while states flip); threads that write the same edge write the same 1."""
    w, lane = wp.tid()
    if solver_niter[w] != 0:
        return
    for r in range(lane, nefc[w], lanes):
        _row_edges(w, r, rownnz, rowadr, colind, dof_treeid, edge)


@wp.kernel
def graph_changed(ntree: int, lanes: int, force: int, solver_niter: wp.array[int], edge: wp.array3d[wp.int8],
                  seen: wp.array3d[wp.int8], changed: wp.array[wp.int32]):
    """``changed`` = 1 where any entry differs from ``seen`` (or ``force``); threads only ever write that same 1."""
    w, lane = wp.tid()
    if solver_niter[w] != 0:
        return
    if force != 0 and lane == 0:
        changed[w] = 1
    for k in range(lane, ntree * ntree, lanes):
        if edge[w, k // ntree, k % ntree] != seen[w, k // ntree, k % ntree]:
            changed[w] = 1


@wp.kernel
def graph_take(ntree: int, lanes: int, solver_niter: wp.array[int], changed: wp.array[wp.int32],
               edge: wp.array3d[wp.int8], seen: wp.array3d[wp.int8], adj: wp.array3d[wp.int32]):
    """Where the graph changed: it becomes ``seen`` and the symbolic step's ``adj``."""
    w, lane = wp.tid()
    if solver_niter[w] != 0 or changed[w] == 0:
        return
    for k in range(lane, ntree * ntree, lanes):
        a = k // ntree
        b = k % ntree
        e = edge[w, a, b]
        seen[w, a, b] = e
        adj[w, a, b] = wp.int32(e)


def symbolic(m: Any, d: Any, b: Buffers) -> None:
    """The symbolic step's launches (each a no-op where ``solver_niter != 0``): the tree graph from every efc row in
    parallel, compared with the one last ordered (the graph kernels above), then one thread per world whose graph
    changed orders and lists it (one thread per block: the worlds spread over the multiprocessors, each world's lists
    in its own cache)."""
    a, c, dev, W = b.arrays, b.cfg, d.qpos.device, int(d.nworld)
    lanes = (W, GRAPH_LANES)
    force = int(os.environ.get("ALLOY_MJW_BLOCK_REUSE", "1") == "0")
    wp.launch(graph_clear, dim=lanes, inputs=[c.ntree, GRAPH_LANES, d.solver_niter],
              outputs=[a["edge"], a["changed"]], device=dev)
    wp.launch(graph_rows, dim=lanes, device=dev,
              inputs=[GRAPH_LANES, d.solver_niter, d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind,
                      m.dof_treeid],
              outputs=[a["edge"]])
    wp.launch(graph_changed, dim=lanes, inputs=[c.ntree, GRAPH_LANES, force, d.solver_niter, a["edge"], a["seen"]],
              outputs=[a["changed"]], device=dev)
    wp.launch(graph_take, dim=lanes, inputs=[c.ntree, GRAPH_LANES, d.solver_niter, a["changed"], a["edge"]],
              outputs=[a["seen"], a["adj"]], device=dev)
    wp.launch(K.symbolic, dim=W, device=dev, block_dim=SYM_BLOCK_DIM,
              inputs=[c.ntree, c.off_cap, c.pair_cap, d.solver_niter, a["changed"]],
              outputs=[a[k] for k in ("adj", "deg", "left", "order", "pos", "nbuf", "col_ptr", "col_row", "row_ptr",
                                      "row_blk", "off_row", "off_col", "level", "nlev", "lvl_node", "node_item",
                                      "lvl_off", "off_item", "opair_ptr", "opair_a", "opair_b", "overflow")])


def launch_factor(dtype: Any, m: Any, d: Any, ctx: Any, b: Buffers, clamps: wp.array) -> None:
    """One factor + solve launch: float32 (every world not done, not overflowed) or float64 (the worlds float32
    flagged in ``bad``)."""
    a, c, tag = b.arrays, b.cfg, "64" if dtype == wp.float64 else "32"
    wp.launch_tiled(kernels(dtype)[0], dim=d.nworld, block_dim=BLOCK_DIM, device=d.qpos.device,
                    inputs=[c.ntree, ctx.done, a["overflow"], a["bad"], clamps, ctx.h, ctx.grad, a["order"],
                            m.tree_dofadr, m.tree_dofnum, a["nlev"], a["lvl_node"], a["node_item"], a["lvl_off"],
                            a["off_item"], a["col_ptr"], a["off_row"], a["off_col"], a["row_ptr"], a["row_blk"],
                            a["opair_ptr"], a["opair_a"], a["opair_b"], a[f"a_diag{tag}"], a[f"lo{tag}"],
                            a[f"dinv{tag}"], a[f"g{tag}"], a[f"y{tag}"], a[f"x{tag}"]],
                    outputs=[ctx.search, ctx.search_dot, ctx.newton_decrement])


def factor_solve(m: Any, d: Any, ctx: Any, clamps: wp.array, dense: Callable[[Any], None]) -> bool:
    """The block path for one ``_cholesky_factorize_solve`` call. False (nothing launched) when this Data has no
    buffers (first seen inside a capture), or when the model's options moved onto MuJoCo Warp's incremental path
    after ``put_model`` (a cone switched to pyramidal: that path reads the dense factor back): the caller runs the
    dense path then."""
    from alloy._vendor.mujoco_warp._src import solver as mjw_solver  # noqa: PLC0415

    b = buffers_for(m, d)
    if b is None or mjw_solver._use_incremental(m):
        return False
    symbolic(m, d, b)
    a, dev = b.arrays, d.qpos.device
    for dtype in (wp.float32, wp.float64):
        launch_factor(dtype, m, d, ctx, b, clamps)
    wp.launch(_dense_mask, dim=d.nworld, inputs=[ctx.done, a["overflow"]], outputs=[a["mask"]], device=dev)
    dense(dataclasses.replace(ctx, done=a["mask"]))
    return True


def _put_model(put_model: Any) -> Any:
    @functools.wraps(put_model)
    def wrapped(mjm: Any, *args: Any, **kwargs: Any) -> Any:
        m = put_model(mjm, *args, **kwargs)
        why = eligible(mjm)
        if why is None:
            object.__setattr__(m, "alloy_block_newton", config_for(int(mjm.ntree)))
            log.info("mjw_block: block-sparse Newton factorization on (nv %d, %d trees)", mjm.nv, mjm.ntree)
        else:
            log.debug("mjw_block: dense Newton factorization (%s)", why)
        return m

    wrapped.alloy_block = True
    return wrapped


def _put_data(put_data: Any) -> Any:
    @functools.wraps(put_data)
    def wrapped(mjm: Any, *args: Any, **kwargs: Any) -> Any:
        d = put_data(mjm, *args, **kwargs)
        if eligible(mjm) is None and id(d.solver_niter) not in _BUFFERS:
            b = Buffers.allocate(config_for(int(mjm.ntree)), int(d.nworld), d.qpos.device)
            remember(_BUFFERS, id(d.solver_niter), d.solver_niter, b)
        return d

    wrapped.alloy_block = True
    return wrapped


_WRAPS = (("put_model", _put_model), ("put_data", _put_data))


def install() -> bool:
    """Wrap ``mujoco_warp.put_model`` and ``put_data`` (idempotent). False when the MuJoCo Warp sources this relies
    on moved."""
    from alloy._vendor import mujoco_warp  # noqa: PLC0415
    from alloy.precision.mjw_pin import BLOCK, changed  # noqa: PLC0415

    moved = changed(BLOCK)
    if moved:
        log.warning("mjw_block: mujoco_warp differs from the pin in %s; block-sparse Newton not installed",
                    ", ".join(moved))
        return False
    for name, wrap in _WRAPS:
        current = getattr(mujoco_warp, name)
        if not getattr(current, "alloy_block", False):
            _STATE[name] = current
            setattr(mujoco_warp, name, wrap(current))
    return True


def uninstall() -> None:
    """Restore what ``install`` wrapped (call before any wrapper installed under it is restored)."""
    from alloy._vendor import mujoco_warp  # noqa: PLC0415

    for name, _ in _WRAPS:
        if _STATE.get(name) is not None:
            setattr(mujoco_warp, name, _STATE.pop(name))


__all__ = ["BlockConfig", "NV_MIN", "eligible", "factor_solve", "install", "uninstall"]
