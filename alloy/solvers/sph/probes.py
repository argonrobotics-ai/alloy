# SPDX-License-Identifier: Apache-2.0
"""Device-side measurements of a SolverDFSPH run for law twins and bully targets (no host sync per step).

Every per-world reduction is deterministic: maxima via ``atomic_max``, masses via int64 fixed-point atomics
(2^40 kg units, as the mass ledger), volumes via the solver's fixed-order float64 WorldReducer.
"""

from __future__ import annotations

import warp as wp

from .pressure import Particles, is_fluid

MASS_SCALE = wp.constant(1099511627776.0)  # 2^40 per kg
MASS_SCALE_HOST = float(2**40)


@wp.func
def _fx(x: float, scale: float) -> wp.int64:
    return wp.int64(wp.round(wp.float64(x) * wp.float64(scale)))


@wp.func
def _unfx(q: wp.int64, scale: float) -> float:
    return float(wp.float64(q) / wp.float64(scale))


@wp.kernel(enable_backward=False)
def front_kernel(p: Particles, z_max: float, out: wp.array[wp.float32]):
    """out[w] = max x over active liquid with z < z_max (the leading edge on the floor); ``out`` pre-filled."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    x = p.q[i]
    if x[2] < z_max:
        wp.atomic_max(out, p.world[i], x[0])


@wp.kernel(enable_backward=False)
def volume_kernel(p: Particles, ratio: wp.array[wp.float32], out: wp.array[wp.float64]):
    """LIQ-2 SPH volume per particle: m / (rho0 max(ratio, 1)) (deficient surface particles count at rho0)."""
    i = wp.tid()
    v = wp.float64(0.0)
    if is_fluid(p, i):
        v = wp.float64(p.mass[i]) / (wp.float64(p.rho0[i]) * wp.float64(wp.max(ratio[i], 1.0)))
    out[i] = v


@wp.func
def _inside(x: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> bool:
    return x[0] >= lo[0] and x[0] <= hi[0] and x[1] >= lo[1] and x[1] <= hi[1] and x[2] >= lo[2] and x[2] <= hi[2]


@wp.kernel(enable_backward=False)
def region_mass_kernel(p: Particles, box_lo: wp.vec3, box_hi: wp.vec3, body_q: wp.array[wp.transform],
                       body_of_world: wp.array[wp.int32], local_lo: wp.vec3, local_hi: wp.vec3,
                       out: wp.array2d[wp.int64]):
    """Per world, fixed point: [0] liquid mass inside the world-frame box, [1] inside the local box of the world's
    body (``body_of_world[w] < 0``: none), [2] all active liquid. ``out`` zeroed before the launch."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    w = p.world[i]
    x = p.q[i]
    m = _fx(float(p.mass[i]), MASS_SCALE)
    wp.atomic_add(out, w, 2, m)
    if _inside(x, box_lo, box_hi):
        wp.atomic_add(out, w, 0, m)
    b = body_of_world[w]
    if b >= 0:
        if _inside(wp.transform_point(wp.transform_inverse(body_q[b]), x), local_lo, local_hi):
            wp.atomic_add(out, w, 1, m)


@wp.kernel(enable_backward=False)
def store_column_kernel(values: wp.array[wp.float64], col: int, out: wp.array2d[wp.float32]):
    w = wp.tid()
    out[w, col] = wp.float32(values[w])


@wp.kernel(enable_backward=False)
def store_max_kernel(values: wp.array[wp.float32], col: int, offset: float, out: wp.array2d[wp.float32]):
    w = wp.tid()
    out[w, col] = values[w] + offset


@wp.kernel(enable_backward=False)
def store_fixed_kernel(values: wp.array2d[wp.int64], slot: int, scale: float, col: int,
                       out: wp.array2d[wp.float32]):
    w = wp.tid()
    out[w, col] = wp.float32(_unfx(values[w, slot], scale))


@wp.kernel(enable_backward=False)
def store_meter_kernel(meter: wp.array2d[wp.int64], shape_of_world: wp.array[wp.int32], comp: int, scale: float,
                       col: int, out: wp.array2d[wp.float32]):
    """Cumulative fluid impulse component ``comp`` on each world's metered shape [N·s]."""
    w = wp.tid()
    out[w, col] = wp.float32(_unfx(meter[shape_of_world[w], comp], scale))


def liquid_volume(solver, out: wp.array) -> None:
    """Per-world LIQ-2 SPH volume [m^3] into ``out`` (float64 [W]) from the density of the last step."""
    p = solver._particles_model()
    if p is None:
        out.zero_()
        return
    wp.launch(volume_kernel, dim=solver.layout.n, inputs=[p, solver._w.ratio, solver._tmp_e], device=solver.device)
    solver._reducer.sum(solver._tmp_e, out)

@wp.kernel(enable_backward=False)
def rest_terms_kernel(p: Particles, spacing: float, nx: int, ny: int, out: wp.array2d[wp.float64],
                      top: wp.array2d[wp.float32]):
    """Per particle, for the resting-jitter law: out[0] |v|^2 and out[1] 1 over active liquid (0 elsewhere, summed per
    world in fixed order); top[w, ix ny + iy] = max z over the liquid of lattice column (ix, iy) of the glass
    (atomic max, order-independent; ``top`` pre-filled with -1e9)."""
    i = wp.tid()
    out[0, i] = wp.float64(0.0)
    out[1, i] = wp.float64(0.0)
    if not is_fluid(p, i):
        return
    v = p.qd[i]
    out[0, i] = wp.float64(wp.dot(v, v))
    out[1, i] = wp.float64(1.0)
    x = p.q[i]
    ix = wp.clamp(int(wp.floor(x[0] / spacing)), 0, nx - 1)
    iy = wp.clamp(int(wp.floor(x[1] / spacing)), 0, ny - 1)
    wp.atomic_max(top, p.world[i], ix * ny + iy, x[2])


@wp.kernel(enable_backward=False)
def rest_finish_kernel(sums: wp.array2d[wp.float64], top: wp.array2d[wp.float32], k: int,
                       it_den: wp.array[wp.int32], it_div: wp.array[wp.int32], err_den: wp.array[wp.float32],
                       err_div: wp.array[wp.float32], rms: wp.array[wp.float64], zstd: wp.array[wp.float64],
                       solve: wp.array2d[wp.float64]):
    """Per world: velocity RMS over the liquid, std of the free surface (top particle per column) over the columns,
    and the last step's solves: solve[0..3, w] = density / divergence iterations, density / divergence residual."""
    w = wp.tid()
    rms[w] = wp.sqrt(sums[0, w] / wp.max(sums[1, w], wp.float64(1.0)))
    n = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    for c in range(k):
        z = top[w, c]
        if z > -1.0e8:
            n += wp.float64(1.0)
            s1 += wp.float64(z)
            s2 += wp.float64(z) * wp.float64(z)
    n = wp.max(n, wp.float64(1.0))
    zstd[w] = wp.sqrt(wp.max(s2 / n - (s1 / n) * (s1 / n), wp.float64(0.0)))
    solve[0, w] = wp.float64(it_den[w])
    solve[1, w] = wp.float64(it_div[w])
    solve[2, w] = wp.float64(err_den[w])
    solve[3, w] = wp.float64(err_div[w])


def rest_stats(solver, state, nx: int, ny: int, terms: wp.array, sums: wp.array, top: wp.array, rms: wp.array,
               zstd: wp.array, solve: wp.array) -> None:
    """Resting-jitter probes of ``state`` per world (deterministic): ``terms`` float64 [2, n], ``sums`` float64
    [2, W], ``top`` float32 [W, nx ny] (columns of the glass on the 2r lattice from x = y = 0); outputs float64 [W]
    and ``solve`` float64 [4, W] (rest_finish_kernel)."""
    if solver.layout.n == 0:
        return
    p, st = solver._particles(state), solver.stats
    top.fill_(-1.0e9)
    wp.launch(rest_terms_kernel, dim=solver.layout.n, inputs=[p, solver.config.spacing, nx, ny, terms, top],
              device=solver.device)
    for k in range(2):
        solver._reducer.sum(terms[k], sums[k])
    wp.launch(rest_finish_kernel, dim=solver.layout.world_count,
              inputs=[sums, top, nx * ny, st.iters_den, st.iters_div, st.den_err, st.div_err, rms, zstd, solve],
              device=solver.device)


__all__ = ["MASS_SCALE_HOST", "front_kernel", "region_mass_kernel", "store_column_kernel", "store_fixed_kernel",
           "store_max_kernel", "store_meter_kernel", "volume_kernel", "liquid_volume", "rest_stats"]
