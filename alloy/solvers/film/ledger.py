# SPDX-License-Identifier: Apache-2.0
"""Per-world film energy ledger (SPEC D20 conventions: float64, one thread per world looping its film particles in
index order, so the sums are bitwise batch-invariant).

- ``air_work[w]``: Σ f_air·(x_out − x_in), the air force's work over each step by the displacement rule the D20
  external-work kernel uses (alloy/coupling/energy.py). Booked as dissipation; it is ≤ 0 unless the wind blows.
- ``projection_dE[w]``: change of KE + gravity PE made by the strain-limit projection, summed over steps
  (Δx from the projection, v_after = v_before + Δx/dt).
- ``projection_gain_max[w]``: the largest single-step positive projection ΔE (J).
- ``air_since[w]``: air work since the last ``SolverFilm.export_air_work(health)``, which adds it to the D20
  monitor's W_ext (``HealthMonitor.add_external_work``, WP5 ruling 2026-11-03) and zeroes it: device only.
- ``step_gain_max[w]``: the largest single-step gain of film KE + gravity PE net of the air work over the whole
  film step (VBD + projection + hinge), the D20 per-step check restricted to film particles (J). In scenes where
  the film touches other bodies, contact work is not in it (the coupler's ledger covers that).
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.func
def world_g(gravity: wp.array[wp.vec3], w: int) -> wp.vec3:
    if gravity.shape[0] > 1:
        return gravity[w]
    return gravity[0]


@wp.kernel(enable_backward=False)
def film_energy_kernel(
    world_start: wp.array[wp.int32],
    particles: wp.array[wp.int32],
    particle_mass: wp.array[float],
    gravity: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    out: wp.array[wp.float64],
):
    """KE + gravity PE (datum: the origin) of each world's film particles."""
    w = wp.tid()
    g = world_g(gravity, w)
    acc = wp.float64(0.0)
    for n in range(world_start[w], world_start[w + 1]):
        p = particles[n]
        m = wp.float64(particle_mass[p])
        v = wp.vec3d(qd[p])
        acc = acc + wp.float64(0.5) * m * wp.dot(v, v) - m * wp.float64(wp.dot(g, q[p]))
    out[w] = acc


@wp.kernel(enable_backward=False)
def air_work_kernel(
    world_start: wp.array[wp.int32],
    particles: wp.array[wp.int32],
    f_air: wp.array[wp.vec3],
    x_in: wp.array[wp.vec3],
    x_out: wp.array[wp.vec3],
    air_work: wp.array[wp.float64],
    air_step: wp.array[wp.float64],
    air_since: wp.array[wp.float64],
):
    w = wp.tid()
    acc = wp.float64(0.0)
    for n in range(world_start[w], world_start[w + 1]):
        p = particles[n]
        acc = acc + wp.float64(wp.dot(f_air[p], x_out[p] - x_in[p]))
    air_work[w] = air_work[w] + acc
    air_step[w] = acc
    air_since[w] = air_since[w] + acc


@wp.kernel(enable_backward=False)
def step_gain_kernel(e_pre: wp.array[wp.float64], e_post: wp.array[wp.float64], air_step: wp.array[wp.float64],
                     gain_max: wp.array[wp.float64]):
    w = wp.tid()
    gain = e_post[w] - e_pre[w] - air_step[w]
    if gain > gain_max[w]:
        gain_max[w] = gain


@wp.kernel(enable_backward=False)
def projection_energy_kernel(
    world_start: wp.array[wp.int32],
    particles: wp.array[wp.int32],
    particle_mass: wp.array[float],
    gravity: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
    qd_after: wp.array[wp.vec3],
    inv_dt: float,
    projection_dE: wp.array[wp.float64],
    gain_max: wp.array[wp.float64],
):
    w = wp.tid()
    g = world_g(gravity, w)
    acc = wp.float64(0.0)
    for n in range(world_start[w], world_start[w + 1]):
        p = particles[n]
        m = wp.float64(particle_mass[p])
        va = wp.vec3d(qd_after[p])
        vb = wp.vec3d(qd_after[p] - dx[p] * inv_dt)
        dke = wp.float64(0.5) * m * (wp.dot(va, va) - wp.dot(vb, vb))
        dpe = -m * wp.float64(wp.dot(g, dx[p]))
        acc = acc + dke + dpe
    projection_dE[w] = projection_dE[w] + acc
    if acc > gain_max[w]:
        gain_max[w] = acc


class FilmLedger:
    """[W] float64 accumulators (module doc) and per-step scratch; ``numbers()`` is a host read for tests and
    reports."""

    def __init__(self, world_count: int, device) -> None:
        z = lambda: wp.zeros(world_count, dtype=wp.float64, device=device)  # noqa: E731
        self.air_work = z()
        self.projection_dE = z()
        self.projection_gain_max = z()
        self.step_gain_max = z()
        self.e_pre, self.e_post, self.air_step = z(), z(), z()
        self.air_since = z()     # air work since the last export to the D20 monitor (solver.export_air_work)

    def arrays(self) -> dict[str, wp.array]:
        return {"film_air_work": self.air_work, "film_projection_dE": self.projection_dE,
                "film_projection_gain_max": self.projection_gain_max, "film_step_gain_max": self.step_gain_max}

    def zero(self) -> None:
        for a in (*self.arrays().values(), self.e_pre, self.e_post, self.air_step, self.air_since):
            a.zero_()

    def numbers(self) -> dict[str, np.ndarray]:
        return {k: v.numpy().copy() for k, v in self.arrays().items()}
