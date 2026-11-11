# SPDX-License-Identifier: Apache-2.0
"""Skip the last Newton iteration's unused Hessian factorization (SCR-WP0-RIGIDDT-2, lever L1).

MuJoCo Warp's Newton solver runs ``_update_gradient`` in every iteration: the gradient, then H = M + JᵀDJ (with the
elliptic cone's term), its Cholesky and Mgrad = H⁻¹ grad. Only then does ``_solve_done`` decide whether the world is
done. done is any of:
- alpha == 0;
- improvement < tol;
- gradient < tol;
- model improvement (½ grad·Mgrad) < tol.

Only the last criterion needs the factor. When one of the first three already holds, that iteration's H, factor and
Mgrad are never read: the world is done, and the next step's ``init_context`` rebuilds them. On XB4-9's resting stacks
(elliptic, niter 1) that is the second of the step's two factorizations: 27 % of an L4 step at 3,072 worlds.

**The patch.** It covers Newton iterations on the non-incremental path (elliptic cones); pyramidal's incremental H is
left alone.
1. Run the gradient first (MuJoCo Warp's own two kernels).
2. Mark the worlds a cheap criterion ends, with ``_solve_done``'s own float32 arithmetic: since MuJoCo Warp 3.17 its
   stop tests scale by the per-world ``ctx.inertia_scale`` (1 / trace M), read here from the same array.
3. Call MuJoCo Warp's ``_update_gradient`` with done | skip as its done mask. Its H, factor and Mgrad kernels, and
   mjw_chol's check, then pass those worlds by.

done, solver_niter, qacc and the state stay bit for bit the stock solver's. For a skipped world ``_solve_done`` reads
the same grad_dot and the same cheap criterion. Its newton_decrement is 0 there and does not change the decision.

**Scratch.** The skip mask and a per-world count of skipped factorizations are allocated per Data when
``mujoco_warp.put_data`` makes it (``install`` wraps it), or on the first uncaptured call. A Data first seen inside a
graph capture runs the stock path, with one warning. They are keyed on the Data's ``solver_niter`` array, not the Data
object: the discrete integrator's ``_solve`` hands the iterations a ``dataclasses.replace`` copy of the Data each step,
which shares that array (WP0's review).

**Pin.** The functions it reads or wraps (``mjw_pin.LAZY``) are hashed at mujoco_warp@f34c6dd5 (v3.17.0; refit from
86e2fae0's ``_rescale(nv, meaninertia)`` for the Newton 1.8 bump) in ``mjw_pin.SOURCES``. A
change disables the patch with one warning, as mjw_chol does. Kernel factories (``_update_gradient_grad``) are looked
up on the module at each call, so the ordered route's substitutions apply here as in MuJoCo Warp's own code.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from typing import Any

import warp as wp

from alloy._vendor.mujoco_warp._src import solver as S
from alloy._vendor.mujoco_warp._src import types as T
from alloy.precision.owned_cache import remember

log = logging.getLogger("alloy.precision")

_SCRATCH: dict[int, tuple[wp.array, wp.array]] = {}   # id(solver_niter) -> (skip [W] bool, skipped [W] int32)
_STATE: dict[str, Any] = {"iteration": None, "update_gradient": None, "put_data": None, "in_iteration": False,
                          "warned": False}


@wp.kernel(enable_backward=False)
def _cheap_done(opt_tolerance: wp.array(dtype=float), inertia_scale: wp.array(dtype=float),
                grad_dot: wp.array(dtype=float), improvement: wp.array(dtype=float), alpha: wp.array(dtype=float),
                done: wp.array(dtype=bool), skip: wp.array(dtype=bool), skipped: wp.array(dtype=wp.int32)):
    """skip = done, or a criterion of ``_solve_done`` that needs no factor (the same arithmetic) already holds."""
    w = wp.tid()
    skip[w] = done[w]
    if done[w]:
        return
    tol = opt_tolerance[w % opt_tolerance.shape[0]]
    scale = inertia_scale[w]                            # _solve_done's per-world 1 / trace(M) (MuJoCo Warp 3.17)
    impr = scale * improvement[w]
    grad = scale * wp.sqrt(grad_dot[w])
    if (alpha[w] == 0.0) or (impr > 0.0 and impr < tol) or (grad < tol):
        skip[w] = True
        skipped[w] = skipped[w] + 1


def _applies(m: Any) -> bool:
    return m.opt.solver == T.SolverType.NEWTON and not S._use_incremental(m)


def _owner(d: Any) -> wp.array:
    """What the scratch is keyed on: an array every ``dataclasses.replace`` copy of the Data shares."""
    return d.solver_niter


def _allocate(d: Any) -> tuple[wp.array, wp.array]:
    s = _SCRATCH.get(id(_owner(d)))
    if s is not None:
        return s
    nw, dev = int(d.nworld), d.qpos.device
    s = (wp.zeros(nw, dtype=bool, device=dev), wp.zeros(nw, dtype=wp.int32, device=dev))
    return remember(_SCRATCH, id(_owner(d)), _owner(d), s)


def _scratch_for(d: Any) -> tuple[wp.array, wp.array] | None:
    s = _SCRATCH.get(id(_owner(d)))
    if s is not None:
        return s
    if wp.get_device(d.qpos.device).is_capturing:
        if not _STATE["warned"]:
            log.warning("mjw_lazy_factor: a Data first seen inside a graph capture runs every factorization")
            _STATE["warned"] = True
        return None
    return _allocate(d)


def skipped_counts(d: Any) -> wp.array | None:
    """Per world, the factorizations skipped so far for one Data (None if the patch never saw it)."""
    s = _SCRATCH.get(id(_owner(d)))
    return None if s is None else s[1]


def _iteration(m: Any, d: Any, ctx: Any, nsolving: wp.array, compact: bool = False) -> None:
    _STATE["in_iteration"] = True
    try:
        return _STATE["iteration"](m, d, ctx, nsolving, compact=compact)
    finally:
        _STATE["in_iteration"] = False


def _update_gradient(m: Any, d: Any, ctx: Any, compact: bool = False) -> None:
    original = _STATE["update_gradient"]
    s = _scratch_for(d) if _STATE["in_iteration"] and _applies(m) else None
    if s is None:
        return original(m, d, ctx, compact=compact)
    skip, skipped = s
    det = bool(m.opt.deterministic & T.DeterminismType.ATOMICS)
    wp.launch(S._update_gradient_zero_grad_dot(False), dim=d.nworld, inputs=[d.nefc, ctx.alpha, ctx.done],
              outputs=[ctx.grad_dot, ctx.newton_decrement, ctx.grad_scale, ctx.search_unchanged])
    wp.launch(S._update_gradient_grad(False, det), dim=(d.nworld, m.nv),
              inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, d.nefc, ctx.done], outputs=[ctx.grad, ctx.grad_dot])
    wp.launch(_cheap_done, dim=d.nworld, inputs=[m.opt.tolerance, ctx.inertia_scale, ctx.grad_dot, ctx.improvement,
                                                 ctx.alpha, ctx.done, skip, skipped])
    return original(m, d, dataclasses.replace(ctx, done=skip), compact=compact)


def _prepared_put_data(put_data: Any) -> Any:
    @functools.wraps(put_data)
    def put_data_prepared(*args: Any, **kwargs: Any) -> Any:
        d = put_data(*args, **kwargs)
        try:
            _allocate(d)
        except (AttributeError, RuntimeError) as exc:      # stays lazy: prepared on the first uncaptured call
            log.warning("mjw_lazy_factor: could not prepare a new mujoco_warp Data (%s)", exc)
        return d

    put_data_prepared.alloy_lazy_factor = True
    return put_data_prepared


def install() -> bool:
    """Wrap ``solver._solver_iteration`` and ``solver._update_gradient`` (idempotent). False when mujoco_warp's
    sources differ from LAZY_SOURCES (stock path kept, one warning)."""
    from alloy.precision.mjw_pin import LAZY, PIN, changed

    if "pin_ok" not in _STATE:              # checked once, against mujoco_warp's own functions (before the first patch)
        moved = changed(LAZY)
        _STATE["pin_ok"] = not moved
        if moved:
            log.warning("mjw_lazy_factor: mujoco_warp differs from %s in %s; every factorization runs", PIN[:8],
                        ", ".join(moved))
    if not _STATE["pin_ok"]:
        return False
    if _STATE["iteration"] is None:
        _STATE["iteration"], _STATE["update_gradient"] = S._solver_iteration, S._update_gradient
        _iteration.__wrapped__ = _STATE["iteration"]            # inspect.unwrap (mjw_pin) hashes mujoco_warp's source
        _update_gradient.__wrapped__ = _STATE["update_gradient"]
    if S._solver_iteration is _iteration:
        return True
    S._solver_iteration, S._update_gradient = _iteration, _update_gradient
    from alloy._vendor import mujoco_warp

    if not getattr(mujoco_warp.put_data, "alloy_lazy_factor", False):
        _STATE["put_data"] = mujoco_warp.put_data
        mujoco_warp.put_data = _prepared_put_data(mujoco_warp.put_data)
    log.info("mjw_lazy_factor: the last Newton iteration's factorization is skipped where a cheap criterion ends it")
    return True


def uninstall() -> None:
    if _STATE["iteration"] is not None:
        S._solver_iteration, S._update_gradient = _STATE["iteration"], _STATE["update_gradient"]
    if _STATE["put_data"] is not None:
        from alloy._vendor import mujoco_warp

        mujoco_warp.put_data = _STATE["put_data"]
        _STATE["put_data"] = None


__all__ = ["install", "skipped_counts", "uninstall"]
