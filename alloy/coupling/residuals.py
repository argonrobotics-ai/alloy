# SPDX-License-Identifier: Apache-2.0
"""Per-endpoint interface residuals, per-world convergence and feedback freezing (SPEC §3.6, coupling-time §4.4.2).

Every reduction over endpoints into a per-world value is an ``atomic_max`` on float32 or an integer ``atomic_add``,
both order-independent, so a world's residuals do not depend on the batch it runs in (D9, CPL-9). The opt-in L2 force
norm (``InterfaceTolerance.force_norm``) sums in float64 with one thread per world over a world-sorted CSR instead,
which is order-independent too.

World convergence codes (``world_converged``): 0 = active, 1 = converged in this pass (transient), 2 = frozen.
A frozen world keeps the feedback its source used in the converging pass, so re-stepping it reproduces that pass
bitwise; the blended value it would have carried is held in ``frozen_out`` and restored when the exchange ends.
"""

from __future__ import annotations

from typing import Any

import warp as wp

ACTIVE = wp.constant(0)
JUST_CONVERGED = wp.constant(1)
FROZEN = wp.constant(2)

# InterfaceTolerance.force_norm: the per-endpoint max (default), or one aggregate per world over all endpoints ("l2")
# or over the free ones (meff > 0, "l2_free"): ||F_raw - F_used||_2 / max(||F_raw||_2, force_floor·sqrt(n)), the
# per-endpoint vec3 residuals flattened into one vector (WP11, SPH-VBD P2 L7)
FORCE_MAX = wp.constant(0)
FORCE_L2 = wp.constant(1)
FORCE_L2_FREE = wp.constant(2)
FORCE_NORMS = {"max": 0, "l2": 1, "l2_free": 2}


@wp.kernel(enable_backward=False)
def gather_feedback_kernel(
    proxy_global: wp.array[int],
    coupling_forces: wp.array[Any],
    out: wp.array[Any],
):
    """out[i] = coupling_forces[proxy_global[i]] (the feedback a source is about to use)."""
    i = wp.tid()
    out[i] = coupling_forces[proxy_global[i]]


@wp.kernel(enable_backward=False)
def scatter_frozen_kernel(
    proxy_global: wp.array[int],
    endpoint_world: wp.array[int],
    world_converged: wp.array[wp.int32],
    values: wp.array[Any],
    coupling_forces: wp.array[Any],
):
    """coupling_forces[proxy_global[i]] = values[i] for endpoints of frozen worlds."""
    i = wp.tid()
    if world_converged[endpoint_world[i]] == FROZEN:
        coupling_forces[proxy_global[i]] = values[i]


@wp.kernel(enable_backward=False)
def freeze_new_worlds_kernel(
    proxy_global: wp.array[int],
    endpoint_world: wp.array[int],
    world_converged: wp.array[wp.int32],
    f_used: wp.array[Any],
    coupling_forces: wp.array[Any],
    frozen_out: wp.array[Any],
):
    """For worlds that converged in this pass: keep their blended value aside and pin the source feedback."""
    i = wp.tid()
    if world_converged[endpoint_world[i]] == JUST_CONVERGED:
        g = proxy_global[i]
        frozen_out[i] = coupling_forces[g]
        coupling_forces[g] = f_used[i]


for _dtype in (wp.vec3, wp.spatial_vector):
    wp.overload(gather_feedback_kernel, {"coupling_forces": wp.array[_dtype], "out": wp.array[_dtype]})
    wp.overload(scatter_frozen_kernel, {"values": wp.array[_dtype], "coupling_forces": wp.array[_dtype]})
    wp.overload(
        freeze_new_worlds_kernel,
        {"f_used": wp.array[_dtype], "coupling_forces": wp.array[_dtype], "frozen_out": wp.array[_dtype]},
    )


@wp.func
def _force_residual(raw: wp.vec3, used: wp.vec3, meff: float, gravity: float, floor: float) -> float:
    scale = wp.max(wp.max(wp.length(raw), meff * gravity), floor)
    return wp.length(raw - used) / scale


@wp.struct
class ResidualInputs:
    """Per-pass constants of the residual kernels."""

    gravity: float
    force_floor: float
    staggered: int          # 1: kinematic residual = change of the source end state since the previous pass
    # contact gate (lever #2, contact_gate.py): 1 when this set's proxies are gated; a released (unarmed) endpoint then
    # adds no residual. The gate released it because it cannot reach its destination within the period (the safety
    # net re-runs the period if it touched), so it carries no force and its lag behind its source changes nothing;
    # a lagged pair's velocity residual is that lag (a dt_x, the same in every pass) and kept worlds iterating to k_max
    gated: int
    armed: wp.array[wp.int32]           # (gate entries,) 1 = armed (ContactGate.armed)
    gate_index: wp.array[wp.int32]      # (P,) this set's endpoint -> gate entry


@wp.func
def _released(inp: ResidualInputs, i: int) -> bool:
    if inp.gated == 0:
        return False
    return inp.armed[inp.gate_index[i]] == 0


@wp.func
def _kinematic(staggered: int, first_pass: int, x_src: wp.vec3, v_src: wp.vec3, x_dst: wp.vec3, v_dst: wp.vec3,
               x_prev: wp.vec3, v_prev: wp.vec3) -> wp.vec2:
    """(|Δv|, |Δx|): lagged compares source and proxy end states; staggered compares the source end state with the
    previous pass (the proxy starts from the source end state there, so source-vs-proxy is dt·v by construction)."""
    if staggered == 0:
        return wp.vec2(wp.length(v_src - v_dst), wp.length(x_src - x_dst))
    if first_pass != 0:
        return wp.vec2(0.0, 0.0)
    return wp.vec2(wp.length(v_src - v_prev), wp.length(x_src - x_prev))


@wp.kernel(enable_backward=False)
def body_residual_kernel(
    inp: ResidualInputs,
    iteration: wp.array[wp.int32],
    src_local: wp.array[int],
    proxy_local: wp.array[int],
    proxy_global: wp.array[int],
    endpoint_world: wp.array[int],
    endpoint_meff: wp.array[float],
    world_converged: wp.array[wp.int32],
    src_q: wp.array[wp.transform],
    src_qd: wp.array[wp.spatial_vector],
    dst_q: wp.array[wp.transform],
    dst_qd: wp.array[wp.spatial_vector],
    coupling_forces: wp.array[wp.spatial_vector],
    f_used: wp.array[wp.spatial_vector],
    f_raw: wp.array[wp.spatial_vector],
    x_prev: wp.array[wp.vec3],
    v_prev: wp.array[wp.vec3],
    res_vel: wp.array[float],
    res_pos: wp.array[float],
    res_force: wp.array[float],
):
    """Record the raw harvest and max-reduce the kinematic and relative force residuals per world."""
    i = wp.tid()
    w = endpoint_world[i]
    if world_converged[w] != ACTIVE:
        return
    raw = coupling_forces[proxy_global[i]]
    f_raw[i] = raw
    s = src_local[i]
    p = proxy_local[i]
    x_src = wp.transform_get_translation(src_q[s])
    v_src = wp.spatial_top(src_qd[s])
    kin = _kinematic(inp.staggered, int(iteration[0] == 0), x_src, v_src, wp.transform_get_translation(dst_q[p]),
                     wp.spatial_top(dst_qd[p]), x_prev[i], v_prev[i])
    x_prev[i] = x_src
    v_prev[i] = v_src
    if _released(inp, i):
        return
    wp.atomic_max(res_vel, w, kin[0])
    wp.atomic_max(res_pos, w, kin[1])
    if endpoint_meff[i] > 0.0:              # a pinned source (meff 0): its feedback moves nothing, so no force gate
        rf = _force_residual(wp.spatial_top(raw), wp.spatial_top(f_used[i]), endpoint_meff[i], inp.gravity,
                             inp.force_floor)
        wp.atomic_max(res_force, w, rf)


@wp.kernel(enable_backward=False)
def particle_residual_kernel(
    inp: ResidualInputs,
    iteration: wp.array[wp.int32],
    src_local: wp.array[int],
    proxy_local: wp.array[int],
    proxy_global: wp.array[int],
    endpoint_world: wp.array[int],
    endpoint_meff: wp.array[float],
    world_converged: wp.array[wp.int32],
    src_q: wp.array[wp.vec3],
    src_qd: wp.array[wp.vec3],
    dst_q: wp.array[wp.vec3],
    dst_qd: wp.array[wp.vec3],
    coupling_forces: wp.array[wp.vec3],
    f_used: wp.array[wp.vec3],
    f_raw: wp.array[wp.vec3],
    x_prev: wp.array[wp.vec3],
    v_prev: wp.array[wp.vec3],
    res_vel: wp.array[float],
    res_pos: wp.array[float],
    res_force: wp.array[float],
):
    """Particle-proxy version of ``body_residual_kernel``."""
    i = wp.tid()
    w = endpoint_world[i]
    if world_converged[w] != ACTIVE:
        return
    raw = coupling_forces[proxy_global[i]]
    f_raw[i] = raw
    s = src_local[i]
    p = proxy_local[i]
    kin = _kinematic(inp.staggered, int(iteration[0] == 0), src_q[s], src_qd[s], dst_q[p], dst_qd[p], x_prev[i],
                     v_prev[i])
    x_prev[i] = src_q[s]
    v_prev[i] = src_qd[s]
    if _released(inp, i):
        return
    wp.atomic_max(res_vel, w, kin[0])
    wp.atomic_max(res_pos, w, kin[1])
    if endpoint_meff[i] > 0.0:              # pinned proxy vertices (meff 0) hold the max but cannot move (WP11, P2 L7)
        rf = _force_residual(raw, f_used[i], endpoint_meff[i], inp.gravity, inp.force_floor)
        wp.atomic_max(res_force, w, rf)


@wp.func
def _force_part(f: wp.vec3) -> wp.vec3:
    return f


@wp.func
def _force_part(f: wp.spatial_vector) -> wp.vec3:
    return wp.spatial_top(f)


@wp.kernel(enable_backward=False)
def force_l2_sums_kernel(
    free_only: int,
    csr_start: wp.array[wp.int32],
    csr_index: wp.array[wp.int32],
    endpoint_meff: wp.array[float],
    world_converged: wp.array[wp.int32],
    f_raw: wp.array[Any],
    f_used: wp.array[Any],
    l2_sums: wp.array2d[wp.float64],
):
    """One thread per world, ordered float64 sums over its endpoints: (|F_raw - F_used|², |F_raw|², n)."""
    w = wp.tid()
    if world_converged[w] != ACTIVE:
        return
    for j in range(csr_start[w], csr_start[w + 1]):
        i = csr_index[j]
        if free_only == 0 or endpoint_meff[i] > 0.0:
            raw = _force_part(f_raw[i])
            d = raw - _force_part(f_used[i])
            l2_sums[w, 0] = l2_sums[w, 0] + wp.float64(wp.dot(d, d))
            l2_sums[w, 1] = l2_sums[w, 1] + wp.float64(wp.dot(raw, raw))
            l2_sums[w, 2] = l2_sums[w, 2] + wp.float64(1.0)


for _dtype in (wp.vec3, wp.spatial_vector):
    wp.overload(force_l2_sums_kernel, {"f_raw": wp.array[_dtype], "f_used": wp.array[_dtype]})


@wp.func
def _l2_force_residual(num: wp.float64, den: wp.float64, n: wp.float64, floor: float) -> float:
    if n <= wp.float64(0.0):
        return 0.0
    scale = wp.max(wp.sqrt(den), wp.float64(floor) * wp.sqrt(n))
    return float(wp.sqrt(num) / scale)


@wp.kernel(enable_backward=False)
def reset_active_residuals_kernel(
    world_converged: wp.array[wp.int32],
    res_vel: wp.array[float],
    res_pos: wp.array[float],
    res_force: wp.array[float],
):
    w = wp.tid()
    if world_converged[w] == ACTIVE:
        res_vel[w] = 0.0
        res_pos[w] = 0.0
        res_force[w] = 0.0


@wp.kernel(enable_backward=False)
def reset_active_l2_sums_kernel(world_converged: wp.array[wp.int32], l2_sums: wp.array2d[wp.float64]):
    w = wp.tid()
    if world_converged[w] == ACTIVE:
        for k in range(3):
            l2_sums[w, k] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def begin_exchange_kernel(
    world_converged: wp.array[wp.int32],
    iters_used: wp.array[wp.int32],
    start_frozen: wp.array[wp.int32],
):
    """Start an exchange: every world active except those whose interface is asleep (start frozen)."""
    w = wp.tid()
    iters_used[w] = 0
    if start_frozen[w] != 0:
        world_converged[w] = FROZEN
    else:
        world_converged[w] = ACTIVE


@wp.struct
class ConvergenceParams:
    k_min: int
    k_max: int
    tol_vel: float
    tol_pos: float
    tol_force: float
    check_force: int
    force_norm: int               # FORCE_MAX / FORCE_L2 / FORCE_L2_FREE
    force_floor: float


@wp.kernel(enable_backward=False)
def convergence_kernel(
    params: ConvergenceParams,
    iteration: wp.array[wp.int32],
    res_vel: wp.array[float],
    res_pos: wp.array[float],
    res_force: wp.array[float],
    l2_sums: wp.array2d[wp.float64],
    world_converged: wp.array[wp.int32],
    iters_used: wp.array[wp.int32],
    n_unconverged: wp.array[wp.int32],
):
    """After pass ``iteration[0]`` (0-based): count the pass, mark newly converged worlds, count the rest. Under an
    L2 force norm, ``res_force`` becomes the world's aggregate first."""
    w = wp.tid()
    if world_converged[w] != ACTIVE:
        return
    if params.force_norm != FORCE_MAX:
        res_force[w] = _l2_force_residual(l2_sums[w, 0], l2_sums[w, 1], l2_sums[w, 2], params.force_floor)
    iters_used[w] = iters_used[w] + 1
    k = iteration[0] + 1
    ok = k >= params.k_min and res_vel[w] <= params.tol_vel and res_pos[w] <= params.tol_pos
    if params.check_force != 0:
        ok = ok and res_force[w] <= params.tol_force
    if ok:
        world_converged[w] = JUST_CONVERGED
    else:
        wp.atomic_add(n_unconverged, 0, 1)


@wp.kernel(enable_backward=False)
def settle_new_worlds_kernel(world_converged: wp.array[wp.int32]):
    w = wp.tid()
    if world_converged[w] == JUST_CONVERGED:
        world_converged[w] = FROZEN


@wp.kernel(enable_backward=False)
def advance_loop_kernel(
    k_max: int,
    n_unconverged: wp.array[wp.int32],
    iteration: wp.array[wp.int32],
    condition: wp.array[wp.int32],
):
    """iteration += 1; condition = any world unconverged and iteration < k_max (the capture_while test)."""
    k = iteration[0] + 1
    iteration[0] = k
    if n_unconverged[0] > 0 and k < k_max:
        condition[0] = 1
    else:
        condition[0] = 0
