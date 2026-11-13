# SPDX-License-Identifier: Apache-2.0
"""MuJoCo C's pivot clamp for MuJoCo Warp's Newton Cholesky (BUL-0016; coordinator ruling 2026-11-04 09:32Z).

MuJoCo C factors the Newton Hessian with ``mju_cholFactor(H, nv, mjMINVAL)``: a pivot below ``mjMINVAL`` is raised to
``mjMINVAL`` (and the rank count drops). MuJoCo Warp's ``solver._cholesky_factorize_solve`` has no such clamp. Its
float32 tile Cholesky (cuSolverDx on a MathDx build, Warp's scalar column loop otherwise) turns a finite, positive
definite but ill-conditioned Hessian into NaN when rounding makes a pivot negative (E9 LEAP: pivot 15 = -4.8e-7,
condition 2.8e7). The NaN reaches ``qacc``.

The guard replaces ``solver._cholesky_factorize_solve`` module-wide (every SolverMuJoCo Newton path, route on or off;
CG never calls it). MuJoCo Warp's own launch runs first, unchanged; then one thread per world checks it:
- nv <= 32 (one tile; the factor lives in shared memory only): the outputs (search, search_dot, newton_decrement) must
  be finite. A negative or zero float32 pivot makes them NaN or inf.
- Larger nv (blocked; factor kept in ``ctx.hfactor``): the outputs must be finite and every factor diagonal
  >= sqrt(mjMINVAL).
- A world that fails is refactored on that thread with MuJoCo C's ``mju_cholFactor`` in float64, MuJoCo C's precision
  (``mjw_chol_f64``; a float32 clamp overflowed E9 100072:1890), and solved again. The blocked path stores the factor
  back into ``ctx.hfactor`` (padding identity) so the skip-unchanged path can reuse it.
- Every world that passes keeps MuJoCo Warp's result bit for bit. (A copy of the one-tile kernel with an in-kernel
  diagonal check was not bitwise at nv 32 on the L4, so the guard copies no MuJoCo Warp kernel.)
- Each refactored factorization adds 1 to a per-world int32 counter ("clamps"). HealthMonitor reads the counter as a
  FALLBACK_USED (DEGRADED) source, so a world that ran a clamped factorization reads DEGRADED from then on.
- A model marked with ``set_pivot_floor`` (SCR-WP3-CHAINGRIP-GPU: a CHAIN the chain-grip rule covers) floors the redo's
  pivots at H's float32 rounding, c * nv * eps32 * max diag H, instead of mjMINVAL (``mjw_chol_f64.factor64_floor``):
  its float32 H can be indefinite by rounding alone, and mjMINVAL then makes the step inf. Unmarked models run
  ``_check``, unchanged.

Scratch (the counter and the float64 factor, [W, nv + 1, nv]) is allocated per Data when ``mujoco_warp.put_data``
creates it (SolverMuJoCo's constructor, never inside a graph capture; ``install`` wraps it), or by ``prepare``. It is
keyed on the Data's ``solver_niter`` array, which every ``dataclasses.replace`` copy shares: MuJoCo Warp's DISCRETE
solve and its compact solve (sleeping, ``nvmax``: a copy at ``nvmax_pad`` every step) run on such copies (SCR-WP5-18;
WP11 found CPL-8's sleeping worlds NaN on rc-1109: keyed on id(Data), each captured compact step missed its scratch and
ran unguarded, and each eager one allocated a new one). A Data made some other way is prepared on its first call,
unless a CUDA graph is being captured; then it runs the stock path, unguarded, with one warning.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import warp as wp

from alloy.precision.mjw_chol_f64 import SQRT_MINVAL, factor64, factor64_floor, solve64, store_factor

from alloy.precision.owned_cache import remember

log = logging.getLogger("alloy.precision")

# id(_owner(Data)): its solver_niter (shared by its copies), else the Data -> (counter [W] int32, f64 [W, nv + 1, nv])
_SCRATCH: dict[int, tuple[wp.array, wp.array]] = {}
_STATE: dict[str, Any] = {"original": None, "warned": False}
_FLOORS: dict[int, float] = {}   # id(solver_niter) -> pivot-floor factor (set_pivot_floor)
FLOOR_ATTR = "alloy_pivot_floor"   # SCR-WP3-CHAINGRIP-GPU: a model (or its opt) carrying c > 0 takes _check_floor


@wp.func
def _outputs_finite(w: int, nv: int, search: wp.array2d[float], search_dot: wp.array[float],
                    newton_decrement: wp.array[float]) -> bool:
    if not (wp.isfinite(search_dot[w]) and wp.isfinite(newton_decrement[w])):
        return False
    for i in range(nv):
        if not wp.isfinite(search[w, i]):
            return False
    return True


@wp.func
def _factor_ok(w: int, hfactor: wp.array3d[float]) -> bool:
    for j in range(hfactor.shape[1]):
        if not (hfactor[w, j, j] >= SQRT_MINVAL):  # a pivot below mjMINVAL, or NaN
            return False
    return True


@wp.kernel
def _check(nv: int, done: wp.array[bool], h: wp.array3d[float], hfactor: wp.array3d[float], grad: wp.array2d[float],
           clamps: wp.array[int], f64: wp.array3d[wp.float64], search: wp.array2d[float], search_dot: wp.array[float],
           newton_decrement: wp.array[float]):
    """After MuJoCo Warp's own factorization and solve: a world whose outputs are non-finite, or (blocked path) whose
    factor has a pivot below mjMINVAL, is refactored in float64 with MuJoCo C's clamp and solved again (blocked: the
    factor goes back into ``hfactor`` for the skip-unchanged path). Every other world keeps MuJoCo Warp's result."""
    w = wp.tid()
    if done[w]:
        return
    blocked = hfactor.shape[1] > 0
    if _outputs_finite(w, nv, search, search_dot, newton_decrement):
        if not blocked or _factor_ok(w, hfactor):
            return
    clamps[w] = clamps[w] + 1
    factor64(h, w, nv, f64)
    solve64(w, nv, f64, grad, search, search_dot, newton_decrement)
    if blocked:
        store_factor(w, nv, f64, hfactor)


@wp.kernel
def _check_floor(nv: int, done: wp.array[bool], h: wp.array3d[float], hfactor: wp.array3d[float],
                 grad: wp.array2d[float], clamps: wp.array[int], f64: wp.array3d[wp.float64], c: wp.float64,
                 search: wp.array2d[float], search_dot: wp.array[float], newton_decrement: wp.array[float]):
    """``_check`` whose float64 redo floors a pivot at H's float32 rounding (``factor64_floor``), for a model marked
    with ``set_pivot_floor``."""
    w = wp.tid()
    if done[w]:
        return
    blocked = hfactor.shape[1] > 0
    if _outputs_finite(w, nv, search, search_dot, newton_decrement):
        if not blocked or _factor_ok(w, hfactor):
            return
    clamps[w] = clamps[w] + 1
    factor64_floor(h, w, nv, f64, c)
    solve64(w, nv, f64, grad, search, search_dot, newton_decrement)
    if blocked:
        store_factor(w, nv, f64, hfactor)


def set_pivot_floor(m: Any, c: float, d: Any = None) -> None:
    """Mark a mujoco_warp Model: its float64 redo floors pivots at c * nv * eps32 * max diag H instead of mjMINVAL
    (``mjw_chol_f64.factor64_floor``). The mark sits on the model, its ``opt`` and, given its Data ``d``, on the Data's
    ``solver_niter`` array: the compact solve replaces both the model and its ``opt``, but its Data copy shares that
    array. Unmarked models keep MuJoCo C's clamp, bit for bit."""
    setattr(m, FLOOR_ATTR, float(c))
    setattr(m.opt, FLOOR_ATTR, float(c))
    niter = getattr(d, "solver_niter", None)
    if niter is not None:
        remember(_FLOORS, id(niter), niter, float(c))


def pivot_floor(m: Any, d: Any = None) -> float:
    """The pivot-floor factor of a call on (m, d) (0: MuJoCo C's mjMINVAL clamp)."""
    for c in (getattr(m, FLOOR_ATTR, None), getattr(getattr(m, "opt", None), FLOOR_ATTR, None)):
        if c is not None:
            return float(c)
    niter = getattr(d, "solver_niter", None)
    return 0.0 if niter is None else _FLOORS.get(id(niter), 0.0)


def prepare(m: Any, d: Any) -> tuple[wp.array, wp.array]:
    """Allocate the guard's scratch for one Data (eager; before any capture). Returns (fallback counter, float64
    factor scratch)."""
    return _allocate(int(m.nv), d)


def _owner(d: Any) -> Any:
    """What a Data's scratch lives with: its ``solver_niter`` array, which every ``dataclasses.replace`` copy of it
    shares (the DISCRETE and compact solves run on a new copy every step), else the Data itself."""
    niter = getattr(d, "solver_niter", None)
    return d if niter is None else niter


def _allocate(nv: int, d: Any) -> tuple[wp.array, wp.array]:
    owner = _owner(d)
    s = _SCRATCH.get(id(owner))
    if s is not None and s[1].shape[2] >= nv:
        return s
    nw, dev = int(d.nworld), d.qpos.device
    cm = getattr(d, "cM", None)
    if cm is not None and cm.size > 0:                    # compaction on: its copy runs at nvmax_pad (WP9's review)
        nv = max(nv, int(getattr(d, "nvmax_pad", 0) or 0))
    counter = s[0] if s is not None else wp.zeros(nw, dtype=wp.int32, device=dev)   # a wider slab keeps the count
    return remember(_SCRATCH, id(owner), owner, (counter, wp.zeros((nw, nv + 1, nv), dtype=wp.float64, device=dev)))


def _prepared_put_data(put_data: Any) -> Any:
    """``mujoco_warp.put_data`` that also allocates the guard's scratch for the new Data (nv = qvel's width)."""
    @functools.wraps(put_data)
    def put_data_prepared(*args: Any, **kwargs: Any) -> Any:
        d = put_data(*args, **kwargs)
        try:
            _allocate(int(d.qvel.shape[1]), d)
        except (AttributeError, RuntimeError) as exc:  # stays lazy: prepared on the first uncaptured call
            log.warning("mjw_chol: could not prepare a new mujoco_warp Data (%s)", exc)
        return d

    put_data_prepared.alloy_chol = True
    return put_data_prepared


def clamp_counts(d: Any) -> wp.array | None:
    """The per-world count of the guard's refactored factorizations of one Data and its copies (None if the guard
    never saw it)."""
    s = _SCRATCH.get(id(_owner(d)))
    return None if s is None else s[0]


def _scratch_for(m: Any, d: Any) -> tuple[wp.array, wp.array] | None:
    s = _SCRATCH.get(id(_owner(d)))
    if s is not None and s[1].shape[2] >= int(m.nv):         # never a slab narrower than this call's nv
        return s
    if wp.get_device(d.qpos.device).is_capturing:
        if not _STATE["warned"]:
            log.warning("mjw_chol: a Data first seen inside a graph capture runs the unguarded Newton Cholesky "
                        "(prepare it before capturing)")
            _STATE["warned"] = True
        return None
    return prepare(m, d)


def _guarded(m: Any, d: Any, ctx: Any, skip_unchanged: bool = False, skip_noflip: bool = False) -> None:
    """``solver._cholesky_factorize_solve`` with MuJoCo C's pivot clamp. A model that ``alloy.precision.mjw_block``
    marked (SCR-WP13-SPARSE-1: elliptic cone; trees over 6 dofs split into runs since 0.1.2) takes the block-sparse
    factorization instead, with the same clamp counter; a call with ``skip_unchanged`` / ``skip_noflip`` (the
    incremental path, which reuses the dense ``ctx.hfactor``) stays dense. A model marked with ``set_pivot_floor``
    takes the block path too, its float64 redo floored the same way (it stayed dense before)."""
    original = _STATE["original"]
    s = _scratch_for(m, d)
    if s is None:
        return original(m, d, ctx, skip_unchanged, skip_noflip)
    clamps, f64 = s

    floor = pivot_floor(m, d)

    def dense(c: Any) -> None:
        original(m, d, c, skip_unchanged, skip_noflip)
        if floor > 0.0:                     # SCR-WP3-CHAINGRIP-GPU: a marked model (set_pivot_floor)
            wp.launch(_check_floor, dim=d.nworld, inputs=[int(m.nv), c.done, c.h, c.hfactor, c.grad, clamps, f64,
                                                          wp.float64(floor)],
                      outputs=[c.search, c.search_dot, c.newton_decrement], device=d.qpos.device)
            return
        wp.launch(_check, dim=d.nworld, inputs=[int(m.nv), c.done, c.h, c.hfactor, c.grad, clamps, f64],
                  outputs=[c.search, c.search_dot, c.newton_decrement], device=d.qpos.device)

    block = getattr(m, "alloy_block_newton", None) is not None
    if block and not (skip_unchanged or skip_noflip):
        from alloy.precision import mjw_block  # noqa: PLC0415

        if mjw_block.factor_solve(m, d, ctx, clamps, dense, floor):
            return
    dense(ctx)


def install() -> bool:
    """Replace ``mujoco_warp._src.solver._cholesky_factorize_solve`` (idempotent). False (stock path kept, one
    warning) when mujoco_warp's Cholesky sources differ from the pin the guard copies (``mjw_pin.GUARD``)."""
    from alloy._vendor.mujoco_warp._src import solver

    from alloy.precision.mjw_pin import GUARD, PIN, changed

    if "pin_ok" not in _STATE:  # checked once, against mujoco_warp's own functions (before the first patch)
        moved = changed(GUARD)
        _STATE["pin_ok"] = not moved
        if moved:
            log.warning("mjw_chol: mujoco_warp differs from %s in %s; Newton Cholesky pivot clamp not installed",
                        PIN[:8], ", ".join(moved))
    if not _STATE["pin_ok"]:
        return False
    if _STATE["original"] is None:
        _STATE["original"] = solver._cholesky_factorize_solve
        _guarded.__wrapped__ = _STATE["original"]  # inspect.unwrap (mjw_pin) hashes mujoco_warp's source
    if solver._cholesky_factorize_solve is not _guarded:
        solver._cholesky_factorize_solve = _guarded
        log.info("mjw_chol: MuJoCo C's pivot clamp guards MuJoCo Warp's Newton Cholesky")
    from alloy._vendor import mujoco_warp  # SolverMuJoCo calls mujoco_warp.put_data(...) through the module

    if not getattr(mujoco_warp.put_data, "alloy_chol", False):
        _STATE["put_data"] = mujoco_warp.put_data
        mujoco_warp.put_data = _prepared_put_data(mujoco_warp.put_data)
    from alloy.precision import mjw_block  # noqa: PLC0415  (SCR-WP13-SPARSE-1: marks eligible models at put_model)

    mjw_block.install()
    return True


def uninstall() -> None:
    from alloy._vendor import mujoco_warp
    from alloy._vendor.mujoco_warp._src import solver

    from alloy.precision import mjw_block  # noqa: PLC0415

    mjw_block.uninstall()                             # first: its put_data wraps this module's
    if _STATE["original"] is not None:
        solver._cholesky_factorize_solve = _STATE["original"]
    if _STATE.get("put_data") is not None:
        mujoco_warp.put_data = _STATE["put_data"]
