# SPDX-License-Identifier: Apache-2.0
"""Pressure-solve driver of SolverDFSPH (split from solver.py): one divergence or density solve per call with per-world
convergence under ``wp.capture_while``, and the per-world density tolerance of the resting-jitter rule.

Resting-jitter rule (SCR-SC-C2, ``DFSPHConfig.rest_jitter_speed``): the density solve stops once a world's mean
compression is at most its tolerance, and the next step pushes that compression back out, so a settled column of depth
H jitters at about tol·H/dt (measured RMS 0.5-1.3 of it at dt 1.25 ms in wp9-results.md, far below it at smaller
dt). Per world and step the tolerance is min(max_density_error, v·dt/H), floored at min_density_error, with
v = max(rest_jitter_speed, rest_jitter_flow_ratio·U), U = |mean velocity| of the liquid in the state the step starts
from: bulk flow (a dam break at its CFL dt, a pour, a carried glass) keeps about the base tolerance's cost, a resting
liquid gets rest_jitter_speed. The mean, not an RMS speed: particle jitter averages out of it, so the allowance cannot
sustain its own jitter (with U an RMS speed and ratio 0.25 a 120 mm column stalled at 20 mm/s on CPU, wp9-results.md).
H is the liquid's hydrostatic depth
from the density warm start k = kd_store / dt^2 = p/rho0 [m^2/s^2] (the pressure the solve starts from: the previous
step's, or ``prime``'s): H = 1.5 sum k^2 / (|g| sum k) over the world's liquid, exact for a resting column (k linear
in depth) and insensitive to single pressure spikes (a max over particles read 1.7x the depth in a 100 mm glass, GPU)
and to unpressurised particles (a falling stream); zero pressure or no gravity: no tightening. The head is read from
the existing warm-start store, so the rule adds no state (snapshots, resets and coupling restarts are unchanged).
The factor of the configured tolerance is ``tol_scale`` [W], device data computed inside the step (graph capture
keeps it live); the divergence solve keeps scale 1.
"""

from __future__ import annotations

import math

import warp as wp

from . import pressure as pr
from .kernels import lattice_diagonal
from .krylov import SolveSpec


@wp.kernel(enable_backward=False)
def head_terms_kernel(p: pr.Particles, kd_store: wp.array[wp.float32], use_abs: int, terms: wp.array[wp.vec2]):
    """terms_i = (kd, kd^2) of the liquid's positive warm-start pressure (kd_store = k dt^2 [m^2]), else 0; with
    ``use_abs`` (ambient pressure: kd_store holds the gauge part, ambient.py) of |kd|, the gauge head of hanging or
    resting liquid."""
    i = wp.tid()
    kd = float(0.0)
    if pr.is_fluid(p, i):
        if use_abs != 0:
            kd = wp.abs(kd_store[i])
        else:
            kd = wp.max(kd_store[i], 0.0)
    terms[i] = wp.vec2(kd, kd * kd)


@wp.kernel(enable_backward=False)
def flow_terms_kernel(p: pr.Particles, ta: wp.array[wp.vec2], tb: wp.array[wp.vec2]):
    """ta_i = (v_z, 1), tb_i = (v_x, v_y) over the liquid, else 0: per-world sums give its mean velocity."""
    i = wp.tid()
    a = wp.vec2(0.0, 0.0)
    b = wp.vec2(0.0, 0.0)
    if pr.is_fluid(p, i):
        v = p.qd[i]
        a = wp.vec2(v[2], 1.0)
        b = wp.vec2(v[0], v[1])
    ta[i] = a
    tb[i] = b


@wp.kernel(enable_backward=False)
def tol_scale_kernel(head: wp.array[wp.vec2d], flow_z: wp.array[wp.vec2d], flow_xy: wp.array[wp.vec2d],
                     gravity: wp.array[wp.vec3], dt: float, speed: float, ratio: float, tol: float, tol_min: float,
                     scale: wp.array[wp.float32]):
    """scale[w] in [tol_min / tol, 1]: tol scale[w] = min(tol, v dt / H), H = 1.5 S2 / (S1 dt^2 |g|) [m],
    v = max(speed, ratio U), U = |mean velocity| from the flow sums (flow_terms_kernel; unused at ratio 0)."""
    w = wp.tid()
    s = float(1.0)
    g = wp.length(gravity[w])
    s1 = float(head[w][0])
    s2 = float(head[w][1])
    v = speed
    n = flow_z[w][1]
    if ratio > 0.0 and n > wp.float64(0.0):
        u = wp.vec3(float(flow_xy[w][0] / n), float(flow_xy[w][1] / n), float(flow_z[w][0] / n))
        v = wp.max(v, ratio * wp.length(u))
    if g > 0.0 and s1 > 0.0 and s2 > 0.0:
        depth = 1.5 * s2 / (s1 * dt * dt * g)
        s = wp.clamp(v * dt / (depth * tol), tol_min / tol, 1.0)
    scale[w] = s


class PressureSolves:
    """Mixin of SolverDFSPH: ``_solve`` / ``_jacobi`` and the resting-jitter tolerance (module doc)."""

    def _alloc_tolerance(self) -> None:
        W, d = max(self.layout.world_count, 1), self.device
        cfg = self.config
        self._lattice_grad = math.sqrt(lattice_diagonal(cfg.spacing, cfg.h, cfg.kernel))  # sqrt(D) [1/m]
        self._tol_scale = wp.ones(W, dtype=wp.float32, device=d)
        self._tol_one = wp.ones(W, dtype=wp.float32, device=d)
        self._head_terms = wp.zeros(max(self.layout.n, 1), dtype=wp.vec2, device=d)  # scratch (kd, kd^2)
        self._head = wp.zeros(W, dtype=wp.vec2d, device=d)  # scratch: per-world sums of the terms
        self._flow_terms = wp.zeros(max(self.layout.n, 1), dtype=wp.vec2, device=d)  # scratch (v_x, v_y)
        self._flow_z = wp.zeros(W, dtype=wp.vec2d, device=d)  # per world (sum v_z, count) at the start of the step
        self._flow_xy = wp.zeros(W, dtype=wp.vec2d, device=d)  # per world (sum v_x, sum v_y)

    def _flow_speed(self, p: pr.Particles) -> None:
        """Mean-velocity sums of the liquid in the step's start state (before the solves and this step's g dt)."""
        cfg = self.config
        if cfg.rest_jitter_speed and cfg.rest_jitter_flow_ratio > 0.0:
            self._launch(flow_terms_kernel, [p, self._head_terms, self._flow_terms])
            self._reducer.sum2(self._head_terms, self._flow_z)
            self._reducer.sum2(self._flow_terms, self._flow_xy)

    def _density_tolerance(self, p: pr.Particles, dt: float) -> None:
        """Per-world factor of max_density_error for this step's density solve (module doc)."""
        cfg = self.config
        if not cfg.rest_jitter_speed:
            return
        self._launch(head_terms_kernel, [p, self._kd, int(self._ambient is not None), self._head_terms])
        self._reducer.sum2(self._head_terms, self._head)
        self._launch(tol_scale_kernel, [self._head, self._flow_z, self._flow_xy, self.model.gravity, dt,
                                        float(cfg.rest_jitter_speed),
                                        float(cfg.rest_jitter_flow_ratio), cfg.max_density_error,
                                        min(cfg.min_density_error, cfg.max_density_error), self._tol_scale],
                     dim=self.layout.world_count)

    def _solve(self, p: pr.Particles, sl: pr.Slots, mode: int, dt: float) -> None:
        """One pressure solve with per-world convergence under wp.capture_while."""
        cfg, w, W = self.config, self._w, self.layout.world_count
        div = mode == pr.MODE_DIVERGENCE
        store = self._kv if div else self._kd
        tol = cfg.max_divergence_error if div else cfg.max_density_error
        if not div:
            self._density_tolerance(p, dt)
            if self._ambient is not None:  # the air's push on the free surface (ambient.py)
                self._ambient.push(p, sl, self._h, self._kind, dt, self._kd, self._vtmp, self.layout.n)
        scale = self._tol_one if div else self._tol_scale
        mn = float(cfg.min_neighbors_divergence)
        h, kind = self._h, self._kind
        warm = cfg.warm_start_divergence if div else cfg.warm_start_density
        self._launch(pr.solve_init_kernel, [p, sl, h, kind, mode, dt, mn, warm, cfg.warm_start_density,
                                            int(cfg.divergence_two_sided), w.ratio,
                                            w.neff, store, self._kd, w.src, w.kappa, self._lower,
                                            self._relief(dt, div)])
        if not div and self._ambient is not None:  # open walls keep P' >= p_a (ambient.py)
            self._ambient.bound(p, sl, self._lower, w.kappa, self.layout.n)
        self._n_unconv.zero_()
        self._launch(pr.begin_solve_kernel, [self._n_active, self._converged, self._iters, self._err_mean,
                                             self._n_unconv], dim=W)

        spec = SolveSpec(mode, dt, tol, tol * cfg.max_error_factor, cfg.min_iterations, cfg.max_iterations, mn,
                         scale)
        if self._cg is not None:
            self._cg.solve(p, sl, h, kind, spec)
        else:
            self._jacobi(p, sl, spec)
        self._launch(pr.contact_limit_kernel, [p, sl, dt, cfg.contact_separation_speed, w.kappa])
        self._launch(pr.apply_pressure_kernel, [p, sl, h, kind, mode, dt, w.kappa, store, self._slot_impulse])
        if cfg.wall_friction:  # lone particles: Coulomb friction on the wall's push (pressure.lone_friction_kernel)
            sa = self.shapes.arrays
            self._launch(pr.lone_friction_kernel, [p, sl, dt, w.kappa, sa.mu, sa.soft.base, self._slot_impulse])
        if not div and self._ambient is not None:  # walls get the gauge pressure; cavitation count (ambient.py)
            self._ambient.gauge(p, sl, dt, self._slot_impulse, self._kd, self.layout.n)
            self._ambient.count_cavitating(p, self._lower, w.kappa, self.stats.cavitating, self.layout.n)
        self.stats.record_solve(div, self._iters, self._err_mean)
        if not div:
            self.stats.record_max_density_error(p, w.err)
            self.stats.record_density_tolerance(self._tol_scale, tol)

    def _relief(self, dt: float, div: bool) -> float:
        """Cap of the existing compression in the density source: v dt sqrt(D) (pressure.py, compression relief)."""
        v = self.config.compression_relief_speed
        if div or not v:
            return 1.0e30
        return float(v) * dt * self._lattice_grad

    def _jacobi(self, p: pr.Particles, sl: pr.Slots, spec: SolveSpec) -> None:
        """Relaxed Jacobi (SPlisHSPlasH) iterations with per-world convergence under wp.capture_while."""
        w, c, W, h, kind = self._w, self._ctl, self.layout.world_count, self._h, self._kind

        def iteration() -> None:
            self._launch(pr.accel_kernel, [p, sl, h, kind, w.kappa, c.converged, w.acc])
            self._launch(pr.jacobi_kernel, [p, sl, h, kind, spec.mode, spec.dt, spec.min_neighbors, w.alpha, w.neff,
                                            w.src, w.acc, c.converged, self.config.jacobi_omega, self._lower,
                                            w.kappa, w.err])
            self._reducer.sum(w.err, self._err_sum, c.err_max)  # sum and max (SCR-WP8-7 L2)
            c.n_unconv.zero_()
            self._launch(pr.check_kernel, [self._err_sum, c.n_active, c.err_max, spec.tol, spec.tol_max,
                                           spec.tol_scale, spec.min_iterations, spec.max_iterations, c.converged,
                                           c.iters, c.err_mean, c.n_unconv], dim=W)

        wp.capture_while(c.n_unconv, iteration)
