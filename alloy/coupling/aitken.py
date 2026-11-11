# SPDX-License-Identifier: Apache-2.0
"""Per-world Aitken relaxation for staggered proxy coupling (SPEC §3.6, coupling-time §4.4.4).

Newton keeps one Aitken scalar per proxy mapping, summed with float ``atomic_add`` over every world
(newton@ac7f7830:newton/_src/solvers/coupled/solver_coupled_proxy.py:167-182), so one diverging world changes the
relaxation of all others. Here every world owns its relaxation ``omega[w]`` and its statistics ``stats[w] =
(r_prev·Δr, |Δr|²)`` in float64, summed by one thread per world over that world's endpoints in a fixed order (a
world-sorted CSR), so the result is bitwise independent of batch size and launch order (D9, CPL-9).

Update (Irons-Tuck vector Δ²): ω ← clamp(-ω·(r_prev·Δr)/|Δr|², ω_min, ω_max) when |Δr|² > 1e-20.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.coupling.residuals import ACTIVE, FROZEN

AITKEN_DENOM_MIN = wp.constant(1.0e-20)


@dataclass
class WorldCSR:
    """Endpoints grouped by world: endpoints of world w are ``index[start[w]:start[w+1]]`` in ascending order."""

    start: wp.array  # (W+1,) int32
    index: wp.array  # (P,) int32


def build_world_csr(endpoint_world: np.ndarray, world_count: int, device: Any) -> WorldCSR:
    """Stable sort of endpoints by world (global endpoints with world < 0 are dropped from the CSR)."""
    worlds = np.asarray(endpoint_world, dtype=np.int64)
    order = np.argsort(worlds, kind="stable")
    order = order[worlds[order] >= 0]
    counts = np.bincount(worlds[order], minlength=world_count)[:world_count]
    start = np.zeros(world_count + 1, dtype=np.int32)
    np.cumsum(counts, out=start[1:])
    return WorldCSR(
        start=wp.array(start, dtype=wp.int32, device=device),
        index=wp.array(order.astype(np.int32), dtype=wp.int32, device=device),
    )


@wp.func
def _dot64(a: wp.spatial_vector, b: wp.spatial_vector) -> wp.float64:
    return wp.float64(wp.dot(a, b))


@wp.func
def _dot64(a: wp.vec3, b: wp.vec3) -> wp.float64:
    return wp.float64(wp.dot(a, b))


@wp.kernel(enable_backward=False)
def aitken_begin_kernel(
    carry: int,
    omega0: float,
    carry_min: float,
    omega_max: float,
    omega: wp.array[float],
    has_prev: wp.array[wp.int32],
):
    """Start of an exchange: ω = clamp(ω_last, carry_min, ω_max) when carrying, else ω0; no previous residual."""
    w = wp.tid()
    if carry != 0:
        omega[w] = wp.clamp(omega[w], carry_min, omega_max)
    else:
        omega[w] = omega0
    has_prev[w] = 0


@wp.kernel(enable_backward=False)
def aitken_update_kernel(
    csr_start: wp.array[wp.int32],
    csr_index: wp.array[wp.int32],
    world_converged: wp.array[wp.int32],
    f_raw: wp.array[Any],
    f_used: wp.array[Any],
    r_prev: wp.array[Any],
    omega_min: float,
    omega_max: float,
    stats: wp.array2d[wp.float64],
    omega: wp.array[float],
    has_prev: wp.array[wp.int32],
):
    """One thread per world: ordered float64 sums of r_prev·Δr and |Δr|², then the ω update."""
    w = wp.tid()
    if world_converged[w] != ACTIVE:
        return
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    if has_prev[w] != 0:
        for j in range(csr_start[w], csr_start[w + 1]):
            i = csr_index[j]
            r = f_raw[i] - f_used[i]
            d = r - r_prev[i]
            s0 = s0 + _dot64(r_prev[i], d)
            s1 = s1 + _dot64(d, d)
        if s1 > wp.float64(AITKEN_DENOM_MIN):
            candidate = -wp.float64(omega[w]) * s0 / s1
            omega[w] = wp.clamp(float(candidate), omega_min, omega_max)
    stats[w, 0] = s0
    stats[w, 1] = s1
    has_prev[w] = 1


@wp.kernel(enable_backward=False)
def aitken_blend_kernel(
    proxy_global: wp.array[int],
    endpoint_world: wp.array[int],
    world_converged: wp.array[wp.int32],
    f_raw: wp.array[Any],
    f_used: wp.array[Any],
    omega: wp.array[float],
    r_prev: wp.array[Any],
    coupling_forces: wp.array[Any],
):
    """F_k = F_used + ω[w]·(R_k − F_used) for active worlds; frozen worlds keep F_used."""
    i = wp.tid()
    w = endpoint_world[i]
    state = world_converged[w]
    if state == FROZEN:
        coupling_forces[proxy_global[i]] = f_used[i]
        return
    if state != ACTIVE:
        return
    r = f_raw[i] - f_used[i]
    coupling_forces[proxy_global[i]] = f_used[i] + omega[w] * r
    r_prev[i] = r


for _dtype in (wp.vec3, wp.spatial_vector):
    wp.overload(
        aitken_update_kernel,
        {"f_raw": wp.array[_dtype], "f_used": wp.array[_dtype], "r_prev": wp.array[_dtype]},
    )
    wp.overload(
        aitken_blend_kernel,
        {
            "f_raw": wp.array[_dtype],
            "f_used": wp.array[_dtype],
            "r_prev": wp.array[_dtype],
            "coupling_forces": wp.array[_dtype],
        },
    )
