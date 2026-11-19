# SPDX-License-Identifier: Apache-2.0
"""Hydrostatic priming of SolverDFSPH (``SolverDFSPH.prime``): a fluid at rest starts with its hydrostatic pressure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags

from .pressure import MODE_DIVERGENCE, scale_kernel

if TYPE_CHECKING:
    from .solver import SolverDFSPH

_ACTIVE = int(ParticleFlags.ACTIVE)


def hydrostatic_guess(solver: SolverDFSPH, state) -> np.ndarray:
    """Host: k_i = |g| (z_top - z_i + r) per (world, horizontal 2r bin) column of active liquid, measured along
    gravity; the initial guess that ``SolverDFSPH.prime`` refines (k = p / rho0 [m^2/s^2])."""
    n = solver.layout.n
    out = np.zeros(max(n, 1), dtype=np.float32)
    if n == 0:
        return out
    m = solver.model
    q = state.particle_q.numpy().astype(np.float64)
    act = solver.layout.phase.numpy() >= 1
    act &= (m.particle_flags.numpy() & _ACTIVE) != 0
    world = m.particle_world.numpy()
    grav = m.gravity.numpy().astype(np.float64)
    sp = solver.config.spacing
    for w in np.unique(world[act]):
        g = grav[w]
        gn = float(np.linalg.norm(g))
        if gn == 0.0:
            continue
        up = -g / gn
        sel = np.nonzero(act & (world == w))[0]
        z = q[sel] @ up
        e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        key = np.stack([np.floor(q[sel] @ e1 / sp), np.floor(q[sel] @ e2 / sp)], 1).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        top = np.full(inv.max() + 1, -np.inf)
        np.maximum.at(top, inv, z)
        out[sel] = (gn * np.maximum(top[inv] - z + 0.5 * sp, 0.0)).astype(np.float32)
    return out


def prime_hydrostatic(solver: SolverDFSPH, state, dt: float, tol: float = 1.0e-6,
                  max_iterations: int = 2000) -> int:
    """Host helper: give a fluid at rest its hydrostatic pressure before the first step.

    Solves the divergence constraint (no compression rate anywhere) for the velocity one gravity step would add,
    which stops the whole column (a one-step density solve would only stop the layers near the floor), and stores
    that pressure as the density warm start; velocities are left unchanged. With an ambient pressure and seal walls
    (ambient.py) the solve is two-sided on sealed liquid off its free surface, down to -p_a, so a column hanging
    under its piston starts at its hydrostatic tension. Returns the iterations used."""
    if solver.layout.n == 0:
        return 0
    cfg = solver.config
    saved = (cfg.max_divergence_error, cfg.max_iterations, cfg.warm_start_divergence, cfg.min_neighbors_divergence,
             cfg.max_error_factor, cfg.divergence_two_sided, cfg.warm_start_density)
    qd = wp.clone(state.particle_qd)
    p = solver._particles(state)
    solver._prepare_density(state, p)
    solver._kv.assign(hydrostatic_guess(solver, state) * float(dt))
    amb = solver._ambient
    try:
        cfg.max_divergence_error, cfg.max_iterations = tol, max_iterations
        cfg.warm_start_divergence, cfg.min_neighbors_divergence, cfg.max_error_factor = 1.0, 0.0, 1.0e9
        if amb is not None and amb.any_seal:  # sealed liquid may start in tension (ambient.py)
            amb.prime_bounds(p, solver._slot_struct(), solver._h, solver._kind, dt, solver._kd, solver.layout.n)
            cfg.divergence_two_sided, cfg.warm_start_density = True, 1.0
        solver._nonpressure(p, state, dt, *solver._body_state(state), predict=False)  # no pressure guess yet
        solver._solve(p, solver._slot_struct(), MODE_DIVERGENCE, dt)
    finally:
        (cfg.max_divergence_error, cfg.max_iterations, cfg.warm_start_divergence, cfg.min_neighbors_divergence,
         cfg.max_error_factor, cfg.divergence_two_sided, cfg.warm_start_density) = saved
        wp.copy(state.particle_qd, qd)
        solver._slot_impulse.zero_()
    wp.launch(scale_kernel, dim=solver.layout.n, inputs=[solver._kv, float(dt), solver._kd], device=solver.device)
    solver._kv.zero_()
    return int(solver.stats.iters_div.numpy()[: solver.layout.world_count].max())
