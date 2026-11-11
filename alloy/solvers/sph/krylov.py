# SPDX-License-Identifier: Apache-2.0
"""Projected preconditioned conjugate gradient for the DFSPH pressure solves, one CG per world.

With equal particle masses and rest densities the DFSPH operator A k = -dt^n Ar(k) equals dt^n J J^T k (J = dC/dx,
the density-constraint Jacobian including the variational wall term): symmetric positive semidefinite. Each solve is
the bound-constrained problem
    w = A k - b,   k >= k_lo,   w >= 0 where k = k_lo,   w = 0 where k > k_lo     (b: predicted compression)
whose residual is the same complementarity measure the Jacobi path uses. Relaxed Jacobi needs O(N^2) iterations for
N layers of liquid; CG propagates pressure through the column in O(N).

Active set F = {k > k_lo} U {w < 0}. Each iteration takes the CG step on F and projects onto the bound; the clamped
part c = k_new - (k + a p) (nonzero only where a variable hit its bound) gets its own product A c so that w stays
exact (that second product runs only in worlds where something was clamped). A clamped variable leaves F, an
off-F variable with w < 0 enters it, and either event restarts the world's direction (beta = 0). Preconditioner
M = diag(A) = dt^n / alpha. Dot products are per-world float64 sums in fixed order (reduce.WorldReducer); worlds
iterate in lockstep under wp.capture_while and freeze when converged (NATIVE, D9).
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .pressure import MODE_DIVERGENCE, Particles, Slots, SolveArrays, accel_divergence, is_fluid, pressure_accel
from .reduce import WorldReducer


@dataclass(frozen=True)
class SolveControl:
    """Per-world control arrays of a pressure solve (owned by SolverDFSPH, shared with the Jacobi path)."""

    converged: wp.array  # int32 [W]
    iters: wp.array  # int32 [W]
    err_mean: wp.array  # float32 [W]
    err_max: wp.array  # float32 [W]
    n_unconv: wp.array  # int32 [1]
    n_active: wp.array  # float64 [W]


@dataclass(frozen=True)
class SolveSpec:
    mode: int
    dt: float
    tol: float
    tol_max: float
    min_iterations: int
    max_iterations: int
    min_neighbors: float
    tol_scale: wp.array  # float32 [W]: per-world factor of tol and tol_max (resting-jitter rule, solve_driver.py)


@wp.func
def _op_scale(mode: int, dt: float) -> float:
    """A = -scale * Ar."""
    if mode == MODE_DIVERGENCE:
        return dt
    return dt * dt


@wp.func
def _bias(mode: int, src: float) -> float:
    """b: divergence Dr/Dt (src), density r* - 1."""
    if mode == MODE_DIVERGENCE:
        return src
    return src - 1.0


@wp.func
def _skip(p: Particles, i: int, converged: wp.array[wp.int32], restart: wp.array[wp.int32], gated: int) -> bool:
    wi = p.world[i]
    if converged[wi] != 0:
        return True
    return gated != 0 and restart[wi] == 0


@wp.kernel(enable_backward=False)
def cg_accel_kernel(p: Particles, sl: Slots, h: float, kind: int, vec: wp.array[wp.float32],
                    converged: wp.array[wp.int32], restart: wp.array[wp.int32], gated: int, acc: wp.array[wp.vec3]):
    """acc = pressure acceleration of the field ``vec`` (only in clamped worlds when ``gated``)."""
    i = wp.tid()
    if not is_fluid(p, i):
        acc[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    if _skip(p, i, converged, restart, gated):
        return
    acc[i] = pressure_accel(p, sl, vec, i, h, kind)


@wp.kernel(enable_backward=False)
def cg_apply_kernel(p: Particles, sl: Slots, h: float, kind: int, mode: int, dt: float, acc: wp.array[wp.vec3],
                    converged: wp.array[wp.int32], restart: wp.array[wp.int32], gated: int, bias: int,
                    accumulate: int, vec: wp.array[wp.float32], src: wp.array[wp.float32],
                    out: wp.array[wp.float32], pair: wp.array[wp.vec2]):
    """out (+)= A vec (- b when ``bias``); pair.x = vec_i (A vec)_i for the p.Ap reduction."""
    i = wp.tid()
    if not is_fluid(p, i):
        out[i] = 0.0
        pair[i] = wp.vec2(0.0, 0.0)
        return
    if _skip(p, i, converged, restart, gated):
        return
    v = -_op_scale(mode, dt) * (accel_divergence(p, sl, acc, i, h, kind) - sl.reg[i] * vec[i])  # + compliance
    pair[i] = wp.vec2(vec[i] * v, 0.0)
    if bias != 0:
        v -= _bias(mode, src[i])
    if accumulate != 0:
        v += out[i]
    out[i] = v


@wp.kernel(enable_backward=False)
def cg_init_kernel(p: Particles, mode: int, dt: float, kappa: wp.array[wp.float32], lower: wp.array[wp.float32],
                   alpha: wp.array[wp.float32], w: wp.array[wp.float32], act: wp.array[wp.int32],
                   z: wp.array[wp.float32], pd: wp.array[wp.float32], pair: wp.array[wp.vec2]):
    """Active set, z = M^-1 r on F (r = -w) and the first direction p = z; pair.x = r.z."""
    i = wp.tid()
    if not is_fluid(p, i):
        act[i] = 0
        z[i] = 0.0
        pd[i] = 0.0
        pair[i] = wp.vec2(0.0, 0.0)
        return
    a = int(0)
    zi = float(0.0)
    if kappa[i] > lower[i] or w[i] < 0.0:
        a = 1
        zi = -w[i] * alpha[i] / _op_scale(mode, dt)
    act[i] = a
    z[i] = zi
    pd[i] = zi
    pair[i] = wp.vec2(-w[i] * zi, 0.0)


@wp.kernel(enable_backward=False)
def cg_first_rz_kernel(sums: wp.array[wp.vec2d], rz: wp.array[wp.float64], restart: wp.array[wp.int32]):
    wi = wp.tid()
    rz[wi] = sums[wi][0]
    restart[wi] = 0


@wp.func
def _cg_alpha(rz: wp.array[wp.float64], sums: wp.array[wp.vec2d], wi: int) -> float:
    """alpha = r.z / p.Ap of world wi (0 when p.Ap <= 1e-30), evaluated per particle in ``cg_step_kernel``
    (SCR-WP8-7 L3: one launch fewer per iteration; the same float64 division as the per-world kernel it replaced)."""
    pap = sums[wi][0]
    a = wp.float64(0.0)
    if pap > wp.float64(1.0e-30):
        a = rz[wi] / pap
    return float(a)


@wp.kernel(enable_backward=False)
def cg_step_kernel(p: Particles, rz: wp.array[wp.float64], sums: wp.array[wp.vec2d], converged: wp.array[wp.int32],
                   pd: wp.array[wp.float32], ap: wp.array[wp.float32], lower: wp.array[wp.float32],
                   act: wp.array[wp.int32], kappa: wp.array[wp.float32], w: wp.array[wp.float32],
                   corr: wp.array[wp.float32], restart: wp.array[wp.int32]):
    """k += a p on F, projected onto k >= k_lo (the clamped part goes to ``corr``); w += a A p."""
    i = wp.tid()
    if not is_fluid(p, i):
        corr[i] = 0.0
        return
    wi = p.world[i]
    if converged[wi] != 0:
        return
    a = _cg_alpha(rz, sums, wi)
    c = float(0.0)
    if act[i] != 0:
        k = kappa[i] + a * pd[i]
        if k < lower[i]:
            c = lower[i] - k
            k = lower[i]
            act[i] = 0
            wp.atomic_add(restart, wi, 1)
        kappa[i] = k
    corr[i] = c
    w[i] = w[i] + a * ap[i]


@wp.kernel(enable_backward=False)
def cg_finish_kernel(p: Particles, mode: int, dt: float, min_neighbors: float, neff: wp.array[wp.float32],
                     alpha: wp.array[wp.float32], lower: wp.array[wp.float32], converged: wp.array[wp.int32],
                     kappa: wp.array[wp.float32], w: wp.array[wp.float32], act: wp.array[wp.int32],
                     z: wp.array[wp.float32], restart: wp.array[wp.int32], pair: wp.array[wp.vec2],
                     err: wp.array[wp.float32]):
    """Activation of compressed bound variables, z = M^-1 r on F, and per-particle (r.z, error); the world's error
    max comes from the reduction of ``pair`` (SCR-WP8-7 L2)."""
    i = wp.tid()
    if not is_fluid(p, i):
        pair[i] = wp.vec2(0.0, 0.0)
        err[i] = 0.0
        return
    wi = p.world[i]
    if converged[wi] != 0:
        return
    wv = w[i]
    if act[i] == 0 and wv < 0.0:
        act[i] = 1
        wp.atomic_add(restart, wi, 1)
    zi = float(0.0)
    if act[i] != 0:
        zi = -wv * alpha[i] / _op_scale(mode, dt)
    z[i] = zi
    e = wp.max(-wv, 0.0)
    if kappa[i] > lower[i]:
        e = wp.abs(wv)
    if mode == MODE_DIVERGENCE:
        if neff[i] < min_neighbors:
            e = 0.0
        e *= dt
    err[i] = e
    pair[i] = wp.vec2(-wv * zi, e)


@wp.kernel(enable_backward=False)
def cg_beta_kernel(sums: wp.array[wp.vec2d], n_active: wp.array[wp.float64], tol: float, tol_max: float,
                   tol_scale: wp.array[wp.float32], min_it: int, max_it: int, err_max: wp.array[wp.float32],
                   rz: wp.array[wp.float64], restart: wp.array[wp.int32], step: wp.array[wp.float64],
                   converged: wp.array[wp.int32], iters: wp.array[wp.int32], err_mean: wp.array[wp.float32],
                   n_unconverged: wp.array[wp.int32]):
    """Per world: stop test (as check_kernel), then beta = r.z_new / r.z (0 after a set change) into ``step``."""
    wi = wp.tid()
    if converged[wi] != 0:
        return
    it = iters[wi] + 1
    iters[wi] = it
    rz_new = sums[wi][0]
    mean = float(sums[wi][1] / wp.max(n_active[wi], wp.float64(1.0)))
    err_mean[wi] = mean
    beta = wp.float64(0.0)
    if restart[wi] == 0 and rz[wi] > wp.float64(1.0e-30):
        beta = rz_new / rz[wi]
    restart[wi] = 0
    rz[wi] = rz_new
    step[wi] = beta
    ts = tol_scale[wi]
    if (it >= min_it and mean <= tol * ts and err_max[wi] <= tol_max * ts) or it >= max_it:
        converged[wi] = 1
    else:
        wp.atomic_add(n_unconverged, 0, 1)


@wp.kernel(enable_backward=False)
def cg_direction_kernel(p: Particles, step: wp.array[wp.float64], converged: wp.array[wp.int32],
                        act: wp.array[wp.int32], z: wp.array[wp.float32], pd: wp.array[wp.float32]):
    i = wp.tid()
    if not is_fluid(p, i):
        return
    wi = p.world[i]
    if converged[wi] != 0:
        return
    if act[i] != 0:
        pd[i] = z[i] + float(step[wi]) * pd[i]
    else:
        pd[i] = 0.0


class ProjectedCG:
    """Fixed-capacity projected PCG; ``solve`` is capture-safe (no allocation, no host sync)."""

    def __init__(self, n: int, world_count: int, device: wp.context.Device, reducer: WorldReducer,
                 work: SolveArrays, lower: wp.array, ctl: SolveControl):
        n, W = max(n, 1), max(world_count, 1)
        f32 = lambda: wp.zeros(n, dtype=wp.float32, device=device)  # noqa: E731
        self.n, self.world_count, self.device = n, world_count, device
        self.reducer, self.work, self.lower, self.ctl = reducer, work, lower, ctl
        self.pd, self.z, self.w, self.ap, self.corr = f32(), f32(), f32(), f32(), f32()
        self.act = wp.zeros(n, dtype=wp.int32, device=device)
        self.pair = wp.zeros(n, dtype=wp.vec2, device=device)
        self.rz = wp.zeros(W, dtype=wp.float64, device=device)
        self.step = wp.zeros(W, dtype=wp.float64, device=device)
        self.sums = wp.zeros(W, dtype=wp.vec2d, device=device)
        self.restart = wp.zeros(W, dtype=wp.int32, device=device)
        self.wall_inertia = None  # soft_inertia.SoftInertia: the shell's interlinked term (SPH-VBD P2)

    def _launch(self, kernel, inputs: list, dim: int | None = None) -> None:
        wp.launch(kernel, dim=self.n if dim is None else dim, inputs=inputs, device=self.device)

    def _product(self, p: Particles, sl: Slots, h: float, kind: int, spec: SolveSpec, vec: wp.array,
                 out: wp.array, gated: int, bias: int, accumulate: int) -> None:
        """out (+)= A vec (- b): pressure acceleration of ``vec``, then its divergence."""
        wk, c = self.work, self.ctl
        self._launch(cg_accel_kernel, [p, sl, h, kind, vec, c.converged, self.restart, gated, wk.acc])
        self._launch(cg_apply_kernel, [p, sl, h, kind, spec.mode, spec.dt, wk.acc, c.converged, self.restart, gated,
                                       bias, accumulate, vec, wk.src, out, self.pair])
        if self.wall_inertia is not None:
            self.wall_inertia.product(p, spec.mode, spec.dt, vec, out, self.pair, c.converged, self.restart, gated)

    def _init(self, p: Particles, sl: Slots, h: float, kind: int, spec: SolveSpec) -> None:
        wk = self.work
        self.restart.zero_()
        self._product(p, sl, h, kind, spec, wk.kappa, self.w, gated=0, bias=1, accumulate=0)
        self._launch(cg_init_kernel, [p, spec.mode, spec.dt, wk.kappa, self.lower, wk.alpha, self.w, self.act, self.z,
                                      self.pd, self.pair])
        self.reducer.sum2(self.pair, self.sums)
        self._launch(cg_first_rz_kernel, [self.sums, self.rz, self.restart], dim=self.world_count)

    def _iteration(self, p: Particles, sl: Slots, h: float, kind: int, spec: SolveSpec) -> None:
        wk, c, W = self.work, self.ctl, self.world_count
        self._product(p, sl, h, kind, spec, self.pd, self.ap, gated=0, bias=0, accumulate=0)
        self.reducer.sum2(self.pair, self.sums)
        self._launch(cg_step_kernel, [p, self.rz, self.sums, c.converged, self.pd, self.ap, self.lower, self.act,
                                      wk.kappa, self.w, self.corr, self.restart])
        self._product(p, sl, h, kind, spec, self.corr, self.w, gated=1, bias=0, accumulate=1)
        self._launch(cg_finish_kernel, [p, spec.mode, spec.dt, spec.min_neighbors, wk.neff, wk.alpha, self.lower,
                                        c.converged, wk.kappa, self.w, self.act, self.z, self.restart, self.pair,
                                        wk.err])
        self.reducer.sum2(self.pair, self.sums, c.err_max)  # sums and the error max (SCR-WP8-7 L2)
        c.n_unconv.zero_()
        self._launch(cg_beta_kernel, [self.sums, c.n_active, spec.tol, spec.tol_max, spec.tol_scale,
                                      spec.min_iterations, spec.max_iterations, c.err_max, self.rz, self.restart,
                                      self.step, c.converged, c.iters, c.err_mean, c.n_unconv], dim=W)
        self._launch(cg_direction_kernel, [p, self.step, c.converged, self.act, self.z, self.pd])

    def solve(self, p: Particles, sl: Slots, h: float, kind: int, spec: SolveSpec) -> None:
        """Iterate from the warm start in ``work.kappa`` until every world converges (begin_solve_kernel ran)."""
        self._init(p, sl, h, kind, spec)
        wp.capture_while(self.ctl.n_unconv, lambda: self._iteration(p, sl, h, kind, spec))
