# SPDX-License-Identifier: Apache-2.0
"""Implicit viscosity, Weiler et al. 2018 ("A physically consistent implicit viscosity solver for SPH fluids"),
following SPlisHSPlasH@dba6b26:SPlisHSPlasH/Viscosity/Viscosity_Weiler2018.cpp (matrix-free CG, 3x3 block-Jacobi
preconditioner).

Solve A v = v* with  (A v)_i = v_i - dt / rho_i * sum_j d mu_ij (m_j / rho_j) / (|x_ij|^2 + 0.01 h^2) * gradW_ij
(x_ij . (v_i - v_j)),  d = 2 (dim + 2) = 10, mu_ij = (mu_i + mu_j) / 2. With equal masses A is symmetric positive
definite; rigid motion (translation, rotation) is in its null space, so viscosity conserves linear and angular
momentum. The wall term sees the relative velocity after the step's pressure guess (``wall_shift_kernel``,
SCR-WP9-11): v* holds the step's gravity, which the pressure solve cancels in a liquid at rest, so on v* alone the
wall term read g dt as shear and carried 5-8 % of a sealed column at rest. The guess (the density solve's warm start,
plus the air push's residual offset in sealed liquid), averaged over the liquid around each wall particle, shifts
the wall velocity by -dt a_p; at rest it cancels g dt, along a draining film it is ~0, and the liquid-liquid term
(Galilean invariant) is untouched, so scenes without wall viscosity are bitwise. (Adding the guess to v* for the
whole solve instead also held the walls at 0, but viscosity then damped the pressure's restoring flow: a free viscous
drop's l = 2 frequency fell 11 %.) Walls carry no viscous term unless ``Config.boundary_viscosity`` is set
(SCR-SC1-1, wall_viscosity.py: the SPlisHSPlasH volume-map wall term; with None the kernels take the has_wall = 0
path and every scene is bitwise unchanged).
Per-world CG with fixed-order float64 dot products (reduce.WorldReducer), stop at |r| <= tol |b| per world, worlds
frozen when converged (NATIVE, capture-safe). The coefficient d is divided by the operator's rest-lattice factor
(kernels.laplacian_lattice_factor: 0.70 for the cubic kernel at h = 2 spacings), so the card's viscosity is the
effective one; measured on a shear mode it is otherwise damped at ~60 % of the rate nu k^2.
"""

from __future__ import annotations

import warp as wp

from .kernels import kernel_grad, kernel_w
from .pressure import Particles, Slots, is_fluid, nbr_base, nbr_j, pressure_accel
from .reduce import WorldReducer
from .wall_viscosity import WallSlots, wall_reaction_kernel, wall_terms_kernel

LAPLACE_D = 10.0  # 2 (dim + 2)


@wp.func
def _visc_coeff(p: Particles, mu: wp.array[wp.float32], ratio: wp.array[wp.float32], i: int, j: int,
                xij: wp.vec3, h: float, d: float) -> float:
    rho_j = ratio[j] * p.rho0[j]
    if rho_j <= 0.0:
        return 0.0
    mu_ij = 0.5 * (mu[i] + mu[j])
    return d * mu_ij * (p.mass[j] / rho_j) / (wp.dot(xij, xij) + 0.01 * h * h)


@wp.kernel(enable_backward=False)
def visc_apply_kernel(p: Particles, mu: wp.array[wp.float32], ratio: wp.array[wp.float32], h: float, kind: int,
                      d: float, dt: float, converged: wp.array[wp.int32], vec: wp.array[wp.vec3], out: wp.array[wp.vec3],
                      pair: wp.array[wp.vec2], wall_m: wp.array[wp.mat33], has_wall: int):
    """out = A vec; pair.x = vec_i . out_i. Walls (has_wall): -dt/rho_i M_i vec_i (wall_viscosity.py)."""
    i = wp.tid()
    if not is_fluid(p, i):
        out[i] = wp.vec3(0.0, 0.0, 0.0)
        pair[i] = wp.vec2(0.0, 0.0)
        return
    if converged[p.world[i]] != 0:
        return
    xi = p.q[i]
    vi = vec[i]
    a = wp.vec3(0.0, 0.0, 0.0)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        xij = xi - p.q[j]
        a += _visc_coeff(p, mu, ratio, i, j, xij, h, d) * wp.dot(xij, vi - vec[j]) * kernel_grad(xij, h, kind)
    rho_i = wp.max(ratio[i] * p.rho0[i], 1.0e-6)
    r = vi - (dt / rho_i) * a
    if has_wall != 0:
        r = r - (dt / rho_i) * (wall_m[i] * vi)
    out[i] = r
    pair[i] = wp.vec2(wp.dot(vi, r), 0.0)


@wp.kernel(enable_backward=False)
def visc_init_kernel(p: Particles, mu: wp.array[wp.float32], ratio: wp.array[wp.float32], h: float, kind: int,
                     d: float, dt: float, b: wp.array[wp.vec3], ax: wp.array[wp.vec3], minv: wp.array[wp.mat33],
                     r: wp.array[wp.vec3], z: wp.array[wp.vec3], pd: wp.array[wp.vec3], pair: wp.array[wp.vec2],
                     wall_m: wp.array[wp.mat33], wall_w: wp.array[wp.vec3], has_wall: int):
    """Block-diagonal preconditioner, r = b - A x, z = M^-1 r, p = z; pair = (r.z, b.b). Walls (has_wall): the
    block gets -dt/rho_i M_i and b_i -= dt/rho_i w_i (moving walls drag the liquid)."""
    i = wp.tid()
    if not is_fluid(p, i):
        r[i] = wp.vec3(0.0, 0.0, 0.0)
        z[i] = wp.vec3(0.0, 0.0, 0.0)
        pd[i] = wp.vec3(0.0, 0.0, 0.0)
        pair[i] = wp.vec2(0.0, 0.0)
        return
    xi = p.q[i]
    rho_i = wp.max(ratio[i] * p.rho0[i], 1.0e-6)
    blk = wp.identity(n=3, dtype=float)
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        xij = xi - p.q[j]
        blk -= (dt / rho_i) * _visc_coeff(p, mu, ratio, i, j, xij, h, d) * wp.outer(kernel_grad(xij, h, kind), xij)
    if has_wall != 0:
        blk -= (dt / rho_i) * wall_m[i]
        b[i] = b[i] - (dt / rho_i) * wall_w[i]
    mi = wp.inverse(blk)
    minv[i] = mi
    ri = b[i] - ax[i]
    zi = mi * ri
    r[i] = ri
    z[i] = zi
    pd[i] = zi
    pair[i] = wp.vec2(wp.dot(ri, zi), wp.dot(b[i], b[i]))


@wp.kernel(enable_backward=False)
def visc_begin_kernel(sums: wp.array[wp.vec2d], n_active: wp.array[wp.float64], rz: wp.array[wp.float64],
                      bb: wp.array[wp.float64], converged: wp.array[wp.int32], iters: wp.array[wp.int32],
                      n_unconverged: wp.array[wp.int32]):
    w = wp.tid()
    rz[w] = sums[w][0]
    bb[w] = sums[w][1]
    iters[w] = 0
    if n_active[w] > wp.float64(0.0) and sums[w][0] > wp.float64(0.0):
        converged[w] = 0
        wp.atomic_add(n_unconverged, 0, 1)
    else:
        converged[w] = 1


@wp.kernel(enable_backward=False)
def visc_alpha_kernel(rz: wp.array[wp.float64], sums: wp.array[wp.vec2d], converged: wp.array[wp.int32],
                      step: wp.array[wp.float64]):
    w = wp.tid()
    if converged[w] != 0:
        return
    pap = sums[w][0]
    a = wp.float64(0.0)
    if pap > wp.float64(1.0e-30):
        a = rz[w] / pap
    step[w] = a


@wp.kernel(enable_backward=False)
def visc_update_kernel(p: Particles, step: wp.array[wp.float64], converged: wp.array[wp.int32],
                       minv: wp.array[wp.mat33], pd: wp.array[wp.vec3], ap: wp.array[wp.vec3], x: wp.array[wp.vec3],
                       r: wp.array[wp.vec3], z: wp.array[wp.vec3], pair: wp.array[wp.vec2]):
    i = wp.tid()
    if not is_fluid(p, i):
        pair[i] = wp.vec2(0.0, 0.0)
        return
    w = p.world[i]
    if converged[w] != 0:
        return
    a = float(step[w])
    x[i] = x[i] + a * pd[i]
    ri = r[i] - a * ap[i]
    zi = minv[i] * ri
    r[i] = ri
    z[i] = zi
    pair[i] = wp.vec2(wp.dot(ri, zi), wp.dot(ri, ri))


@wp.kernel(enable_backward=False)
def visc_check_kernel(sums: wp.array[wp.vec2d], bb: wp.array[wp.float64], tol: float, max_it: int,
                      rz: wp.array[wp.float64], step: wp.array[wp.float64], converged: wp.array[wp.int32],
                      iters: wp.array[wp.int32], n_unconverged: wp.array[wp.int32]):
    """Stop at |r|^2 <= tol^2 |b|^2; beta = r.z_new / r.z into ``step``."""
    w = wp.tid()
    if converged[w] != 0:
        return
    it = iters[w] + 1
    iters[w] = it
    rz_new = sums[w][0]
    beta = wp.float64(0.0)
    if rz[w] > wp.float64(1.0e-30):
        beta = rz_new / rz[w]
    rz[w] = rz_new
    step[w] = beta
    if sums[w][1] <= wp.float64(tol * tol) * bb[w] or it >= max_it:
        converged[w] = 1
    else:
        wp.atomic_add(n_unconverged, 0, 1)


@wp.kernel(enable_backward=False)
def visc_direction_kernel(p: Particles, step: wp.array[wp.float64], converged: wp.array[wp.int32],
                          z: wp.array[wp.vec3], pd: wp.array[wp.vec3]):
    i = wp.tid()
    if not is_fluid(p, i):
        return
    w = p.world[i]
    if converged[w] != 0:
        return
    pd[i] = z[i] + float(step[w]) * pd[i]


@wp.kernel(enable_backward=False)
def visc_store_kernel(p: Particles, x: wp.array[wp.vec3]):
    i = wp.tid()
    if is_fluid(p, i):
        p.qd[i] = x[i]


class ImplicitViscosity:
    """Fixed-capacity Weiler 2018 solver; ``solve`` overwrites the fluid velocities in ``p.qd`` (capture-safe)."""

    def __init__(self, n: int, world_count: int, device, reducer: WorldReducer, mu: wp.array, n_active: wp.array,
                 tol: float = 1.0e-3, max_iterations: int = 100, laplace_d: float = LAPLACE_D,
                 wall_coeff: float = 0.0):
        n, W = max(n, 1), max(world_count, 1)
        v3 = lambda: wp.zeros(n, dtype=wp.vec3, device=device)  # noqa: E731
        f64 = lambda: wp.zeros(W, dtype=wp.float64, device=device)  # noqa: E731
        self.n, self.world_count, self.device = n, world_count, device
        self.reducer, self.mu, self.n_active = reducer, mu, n_active
        self.tol, self.max_iterations, self.d = float(tol), int(max_iterations), float(laplace_d)
        self.b, self.x, self.r, self.z, self.pd, self.ap = v3(), v3(), v3(), v3(), v3(), v3()
        self.minv = wp.zeros(n, dtype=wp.mat33, device=device)
        self.pair = wp.zeros(n, dtype=wp.vec2, device=device)
        self.sums = wp.zeros(W, dtype=wp.vec2d, device=device)
        self.rz, self.bb, self.step = f64(), f64(), f64()
        self.converged = wp.zeros(W, dtype=wp.int32, device=device)
        self.iters = wp.zeros(W, dtype=wp.int32, device=device)
        self.n_unconv = wp.zeros(1, dtype=wp.int32, device=device)
        self.wall_coeff = float(wall_coeff)  # mu_b x wall_coefficient_scale; 0 = free-slip walls
        nw = n if self.wall_coeff > 0.0 else 1
        self.wall_m = wp.zeros(nw, dtype=wp.mat33, device=device)
        self.wall_w = wp.zeros(nw, dtype=wp.vec3, device=device)

    @property
    def has_wall(self) -> int:
        return int(self.wall_coeff > 0.0)

    def _launch(self, kernel, inputs: list, dim: int | None = None) -> None:
        wp.launch(kernel, dim=self.n if dim is None else dim, inputs=inputs, device=self.device)

    def solve(self, p: Particles, ratio: wp.array, h: float, kind: int, dt: float, walls: WallSlots | None = None,
              radius: float = 0.0, slot_impulse: wp.array | None = None) -> None:
        """Solve for the new velocities (in p.qd). With wall viscosity on, ``walls`` and ``radius`` give the wall
        term and ``slot_impulse`` receives the impulses the walls took."""
        W = self.world_count
        hw = self.has_wall if walls is not None else 0
        wall = (self.wall_m, hw)
        wp.copy(self.b, p.qd, count=self.n)
        wp.copy(self.x, p.qd, count=self.n)
        self.converged.zero_()
        if hw:
            self._launch(wall_terms_kernel, [p, walls, radius, h, kind, self.wall_coeff, self.wall_m, self.wall_w])
        self._launch(visc_apply_kernel, [p, self.mu, ratio, h, kind, self.d, dt, self.converged, self.x, self.ap,
                                         self.pair, *wall])
        self._launch(visc_init_kernel, [p, self.mu, ratio, h, kind, self.d, dt, self.b, self.ap, self.minv, self.r, self.z,
                                        self.pd, self.pair, self.wall_m, self.wall_w, hw])
        self.reducer.sum2(self.pair, self.sums)
        self.n_unconv.zero_()
        self._launch(visc_begin_kernel, [self.sums, self.n_active, self.rz, self.bb, self.converged, self.iters,
                                         self.n_unconv], dim=W)

        def iteration() -> None:
            self._launch(visc_apply_kernel, [p, self.mu, ratio, h, kind, self.d, dt, self.converged, self.pd,
                                             self.ap, self.pair, *wall])
            self.reducer.sum2(self.pair, self.sums)
            self._launch(visc_alpha_kernel, [self.rz, self.sums, self.converged, self.step], dim=W)
            self._launch(visc_update_kernel, [p, self.step, self.converged, self.minv, self.pd, self.ap, self.x,
                                              self.r, self.z, self.pair])
            self.reducer.sum2(self.pair, self.sums)
            self.n_unconv.zero_()
            self._launch(visc_check_kernel, [self.sums, self.bb, self.tol, self.max_iterations, self.rz, self.step,
                                             self.converged, self.iters, self.n_unconv], dim=W)
            self._launch(visc_direction_kernel, [p, self.step, self.converged, self.z, self.pd])

        wp.capture_while(self.n_unconv, iteration)
        self._launch(visc_store_kernel, [p, self.x])
        if hw and slot_impulse is not None:
            self._launch(wall_reaction_kernel, [p, walls, ratio, radius, h, kind, self.wall_coeff, dt, slot_impulse])


@wp.kernel(enable_backward=False)
def pressure_guess_kernel(p: Particles, kd: wp.array[wp.float32], dt: float, warm: float, kappa: wp.array[wp.float32]):
    """kappa = the density solve's warm start (pressure.solve_init_kernel, density branch): the step's pressure guess."""
    i = wp.tid()
    kappa[i] = 0.0
    if is_fluid(p, i):
        kappa[i] = warm * wp.min(kd[i], 2.5e-4) / (dt * dt)


@wp.kernel(enable_backward=False)
def guess_shift_kernel(p: Particles, sl: Slots, h: float, kind: int, kappa: wp.array[wp.float32], dt: float,
                       resid: wp.array[wp.vec3], has_resid: int, raw: wp.array[wp.vec3]):
    """raw = -dt a, a the pressure guess's acceleration (kappa; + the air push's residual offset of sealed liquid,
    ambient.py)."""
    i = wp.tid()
    raw[i] = wp.vec3(0.0, 0.0, 0.0)
    if not is_fluid(p, i):
        return
    a = pressure_accel(p, sl, kappa, i, h, kind)
    if has_resid != 0:
        a = a + resid[i]
    raw[i] = -dt * a


@wp.kernel(enable_backward=False)
def wall_shift_kernel(p: Particles, sl: Slots, h: float, kind: int, raw: wp.array[wp.vec3],
                      shift: wp.array[wp.vec3]):
    """shift = the Shepard average of ``raw`` over the liquid around a particle beside a wall (the wall term sees
    v_i - (v_b + shift) = v_i + dt a - v_b); the average keeps the per-particle pressure noise out of the walls."""
    i = wp.tid()
    shift[i] = wp.vec3(0.0, 0.0, 0.0)
    if not is_fluid(p, i):
        return
    walls = int(0)
    for k in range(4):
        if sl.shape[i, k] >= 0:
            walls += 1
    if walls == 0:
        return
    xi = p.q[i]
    tot = p.mass[i] / p.rho0[i] * kernel_w(0.0, h, kind)
    acc = tot * raw[i]
    b_ = nbr_base(p, i)
    for k in range(p.count[i]):
        j = nbr_j(p, b_, i, k)
        if is_fluid(p, j):
            wj = p.mass[j] / p.rho0[j] * kernel_w(wp.length(xi - p.q[j]), h, kind)
            acc += wj * raw[j]
            tot += wj
    shift[i] = acc / tot


__all__ = ["ImplicitViscosity", "LAPLACE_D", "guess_shift_kernel", "pressure_guess_kernel", "wall_shift_kernel"]
