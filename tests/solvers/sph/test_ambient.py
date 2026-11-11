# SPDX-License-Identifier: Apache-2.0
"""Ambient pressure for sealed liquid (SCR-SC10-1, alloy/solvers/sph/ambient.py), CPU checks.

The laws (A13.sealed_column, A13.piston_displacement: seal_twins.py) run at C18's particle size; these tests are the
small, fast versions: liquid that is not sealed behaves as without an ambient pressure (the offset's force is
cancelled, walls see gauge), a sealed column hangs from its piston while an unsealed piston is never pulled, the floor
cavitates when the ambient is below the column's need, the residual offset carries no net load, the prime starts a
sealed column in tension, the inputs are validated and a state of another model is refused.
"""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.seal_twins import add_sealed_bore
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.sph import default_config, water_column

R_BORE = 0.075e-3  # C18's particle radius
DT_BORE = 0.02 / 512


def _resting(ambient: float | None, steps: int = 30):
    cfg = default_config(2.0e-3, ambient_pressure=ambient)
    sc = water_column(6, 6, 10, cfg=cfg, device="cpu")
    s = sc.make_solver()
    a, c = sc.model.state(), sc.model.state()
    dt = 1.25e-3
    s.prime(a, dt)
    s.reset_shape_meters()
    for _ in range(steps):
        s.step(a, c, None, None, dt)
        a, c = c, a
    floor = -s.shape_impulse()[sc.tank[0]["floor"], 2] / (steps * dt)
    weight = float(sc.model.particle_mass.numpy().sum()) * 9.81
    return a.particle_q.numpy(), floor, weight, s


def test_unsealed_liquid_behaves_as_without_ambient():
    """No seal shape: every particle's offset force is cancelled and its gauge kept >= 0, so a resting column is the
    solver without an ambient pressure shifted by p_a: same positions to the solve tolerance, the floor carries the
    weight in gauge, no particle counted at the cavitation floor."""
    q_off, f_off, weight, _ = _resting(None)
    q_on, f_on, _, s = _resting(500.0)
    assert np.abs(q_on - q_off).max() < 0.02 * 4.0e-3, np.abs(q_on - q_off).max()  # measured 0.005 spacing
    assert abs(f_on - weight) < 0.02 * weight and abs(f_on - f_off) < 0.01 * weight, (f_on, f_off, weight)
    assert int(s.stats.diagnostics()["cavitating"][0]) == 0


def _bore(ambient: float | None, seal: bool, steps: int, nz: int = 16):
    cfg = DFSPHConfig(particle_radius=R_BORE, surface_tension="csf", viscosity="xsph", xsph_c=0.3,
                      ambient_pressure=ambient, integrate_bodies="none", max_density_error=1.0e-4,
                      world_bounds=((-0.01, -0.01, -0.02), (0.01, 0.01, 0.02)))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    ids = add_sealed_bore(b, cfg, 6, nz, seal and ambient is not None)
    b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()
    s.prime(a, DT_BORE)
    z0 = a.particle_q.numpy()[:, 2].copy()
    for k in range(steps):
        if k == steps // 2:
            s.reset_shape_meters()  # the piston's force over the second half
        s.step(a, c, None, None, DT_BORE)
        a, c = c, a
    cav = int(s.stats.diagnostics()["cavitating"][0])  # the last step's
    gauge = s.shape_impulse()[ids["shape"], 2] / ((steps - steps // 2) * DT_BORE) / ids["A"]
    drop = float(z0.mean() - a.particle_q.numpy()[:, 2].mean()) / cfg.spacing
    return gauge, drop, cav, 1000.0 * 9.81 * ids["H"], s


def test_sealed_column_hangs_and_open_piston_never_pulls():
    """A 2.4 mm column in a 0.9 mm bore at r = 0.075 mm (csf, 90 deg walls): sealed under its piston (p_a 500 Pa) it
    hangs and the piston is pulled at about -rho g H over the second half of 10 ms (measured -0.960 rho g H, drop
    0.002 spacing; the rest is the water meniscus, SCR-WP9-7). With the seal off the piston carries no tension (gauge >= 0: the column is free to leave it;
    csf at this R/h 1.5 is outside its law range, so the open column's motion itself is not checked here: it matches
    the solver without an ambient to the solve tolerance until csf's own instability takes over)."""
    steps = 256  # 10 ms
    gauge, drop, cav, rgh, _s = _bore(500.0, True, steps)
    assert -1.3 * rgh < gauge < -0.7 * rgh, (gauge, rgh)
    assert abs(drop) < 0.3, drop
    assert cav == 0
    gauge_open, _d, _c, _r, _s = _bore(500.0, False, steps)
    assert gauge_open > -0.1 * rgh, gauge_open


def test_residual_offset_carries_no_net_load():
    """SCR-WP9-12: the offset's force the push leaves in sealed liquid below 0 gauge, -2 k_a (1 - w) g, sums to zero
    over the world's sealed liquid off its free surface (before, it held 3-11 % of C18's column with no reaction on
    any shape), while it still acts on single particles (the local repulsion that keeps tension from pairing)."""
    _g, _d, _c, rgh, s = _bore(500.0, True, 64)
    p = s._particles_model()
    m = s.model.particle_mass.numpy()
    surf = s._ambient.surf.numpy()
    fluid = (s.model.particle_flags.numpy() & 1) != 0
    sealed = fluid & (surf == 0)
    r = s._ambient.resid.numpy()
    assert p is not None and sealed.sum() > 100, sealed.sum()
    net = (m[sealed, None] * r[sealed]).sum(axis=0)
    local = (m[sealed] * np.linalg.norm(r[sealed], axis=1)).sum()
    assert local > 1.0e-3 * m[sealed].sum() * 9.81, local  # the residual is there, particle by particle
    assert np.abs(net).max() < 1.0e-4 * m[sealed].sum() * 9.81, (net, local)


def test_prime_starts_a_sealed_column_in_tension():
    """prime_hydrostatic is two-sided on sealed liquid: the warm start (gauge part, ambient.py) of the top layers of a
    hanging column is near -rho g H right away, and the free-surface layer at the mouth stays at >= 0 gauge."""
    cfg = DFSPHConfig(particle_radius=R_BORE, surface_tension="csf", viscosity="xsph", xsph_c=0.3,
                      ambient_pressure=500.0, integrate_bodies="none", max_density_error=1.0e-4,
                      world_bounds=((-0.01, -0.01, -0.02), (0.01, 0.01, 0.02)))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    ids = add_sealed_bore(b, cfg, 6, 16, True)
    b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    a = m.state()
    s.prime(a, DT_BORE)
    z = a.particle_q.numpy()[:, 2]
    gauge = 1000.0 * s._kd.numpy()[: m.particle_count] / DT_BORE**2
    rgh = 1000.0 * 9.81 * ids["H"]
    top = gauge[z > ids["H"] - cfg.spacing].mean()
    assert -1.3 * rgh < top < -0.6 * rgh, (top, rgh)
    assert gauge[z < cfg.spacing].min() >= -1.0e-3 * rgh


def test_prime_refuses_a_state_of_another_model():
    """A state whose bodies are not the solver's (a coupled parent's, SolverCoupledProxy) is refused: its body poses
    would place the walls wrong (SC10: prime(sim.state) put the capillary at the robot base)."""
    cfg = DFSPHConfig(particle_radius=R_BORE, ambient_pressure=500.0, integrate_bodies="none")
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    add_sealed_bore(b, cfg, 4, 4, True, contact_angle=None, sigma=0.0)
    b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    a = m.state()
    a.body_q = wp.zeros(m.body_count + 2, dtype=wp.transform, device="cpu")
    with pytest.raises(ValueError, match="entry's state"):
        s.prime(a, DT_BORE)


def test_cavitation_floor_is_counted():
    """An ambient below the column's need (5 Pa < rho g H = 24 Pa): the liquid under the piston reaches the floor,
    gauge -p_a, and diagnostics() counts it."""
    _g, drop, cav, rgh, _s = _bore(5.0, True, 128)
    assert rgh > 5.0 and cav > 0, (cav, rgh)


def test_ambient_pressure_must_be_positive_or_none():
    DFSPHConfig(ambient_pressure=None).validate()
    DFSPHConfig(ambient_pressure=500.0).validate()
    with pytest.raises(ValueError, match="ambient_pressure"):
        DFSPHConfig(ambient_pressure=0.0).validate()


def test_seal_flags_need_an_ambient_pressure():
    """sph:shape_seal is read only with ambient_pressure set; a flagged shape without it is refused."""
    cfg = DFSPHConfig(particle_radius=R_BORE)
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    b.begin_world()
    add_sealed_bore(b, cfg, 4, 4, True, contact_angle=None, sigma=0.0)
    b.end_world()
    m = b.finalize(device="cpu")
    with pytest.raises(ValueError, match="shape_seal"):
        SolverDFSPH(m, cfg)
