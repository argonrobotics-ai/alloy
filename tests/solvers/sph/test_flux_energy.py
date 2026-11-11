# SPDX-License-Identifier: Apache-2.0
"""Emitted and retired liquid as external work for the health monitor (SolverDFSPH.export_flux_energy, OSS-3's pour):
the energy check (E - W_ext, D20) must not read an emission or a retirement as an energy jump."""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.solvers.sph.build import add_fluid_points, add_spares
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH

R = 2.0e-3


class _Health:
    """The two HealthMonitor calls the export uses, for one world at a given z_ref."""

    def __init__(self, z_ref: float):
        self.z = wp.array([z_ref], dtype=wp.float64, device="cpu")
        self.w = 0.0

    def state_arrays(self):
        return {"health.energy_z_ref": self.z}

    def add_external_work(self, w_ext):
        self.w += float(w_ext.numpy()[0])


def _energy(sol, st, z_ref: float) -> float:
    out = wp.zeros(1, dtype=wp.float64, device="cpu")
    sol.alloy_energy(st, wp.array([z_ref], dtype=wp.float64, device="cpu"), out)
    return float(out.numpy()[0])


def test_emitted_energy_is_external_work():
    """Zero g, one layer emitted at 0.5 m/s in the first step (no pressure acts on a single layer): the exported W_ext
    equals the liquid's energy gain."""
    cfg = DFSPHConfig(particle_radius=R, world_bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    add_spares(b, cfg, 64, park=(0.0, 0.0, -0.5))
    b.end_world()
    m = b.finalize(device="cpu")
    sol = SolverDFSPH(m, cfg)
    sol.add_emitter(0, wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()), wp.vec3(0.006, 0.006, 0.002), 1.0,
                    wp.vec3(0.0, 0.0, -0.5))
    a, bb = m.state(), m.state()
    h = _Health(0.0)
    e0 = _energy(sol, a, 0.0)
    sol.step(a, bb, None, None, 1.0e-3)
    sol.export_flux_energy(h)
    e1 = _energy(sol, bb, 0.0)
    n = int(sol.report().mass_in[0] / float(m.particle_mass.numpy()[0]) + 0.5)
    assert n > 0
    assert e0 == 0.0 and np.isclose(h.w, e1, rtol=1e-6)
    assert np.isclose(h.w, n * float(m.particle_mass.numpy()[0]) * 0.5 * 0.25, rtol=1e-6)
    sol.export_flux_energy(h)  # the ledger was zeroed
    assert np.isclose(h.w, e1, rtol=1e-6)


def test_retired_energy_leaves_at_the_monitors_z_ref():
    """Gravity on, z_ref 0.05 m: a particle outside the escape box retires at the step start, and W_ext is minus its
    energy about z_ref, so E - W_ext does not jump."""
    cfg = DFSPHConfig(particle_radius=R, world_bounds=((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    add_fluid_points(b, cfg, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.15]]), velocity=(0.3, 0.0, 1.2))
    b.end_world()
    m = b.finalize(device="cpu")
    sol = SolverDFSPH(m, cfg)
    a, bb = m.state(), m.state()
    h = _Health(0.05)
    mass = float(m.particle_mass.numpy()[1])
    e_out = mass * (0.5 * (0.3**2 + 1.2**2) + 9.81 * (0.15 - 0.05))
    e0 = _energy(sol, a, 0.05)
    sol.step(a, bb, None, None, 1.0e-3)
    sol.export_flux_energy(h)
    assert sol.report().escaped.tolist() == [1]
    assert np.isclose(h.w, -e_out, rtol=1e-6)
    e1 = _energy(sol, bb, 0.05)
    assert abs((e1 - h.w) - e0) < 1e-3 * e_out  # the kept particle's one free-fall step is all that changes
