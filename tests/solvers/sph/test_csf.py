# SPDX-License-Identifier: Apache-2.0
"""Continuum surface force with the Breinlinger contact angle (SCR-WP9-7, csf.py): constants, inputs, the capillary
step report, and the laws (Laplace, Rayleigh l = 2, returns to round, sessile contact angle; slow / gpu)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.solvers.sph import csf
from alloy.solvers.sph.build import add_fluid_points, wall_box
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.drops import VISCOUS, WATER, drop_points, laplace_drop, rayleigh_drop, sessile_drop
from alloy.solvers.sph.solver import SolverDFSPH

R = 1.0e-4


def _drop(mode: str = "csf", angle: float | None = None, adhesion: float = 0.0, radius_cells: float = 4.0,
          sigma: float = 0.072):
    """A small free drop over a plate with sph:shape_contact_angle ``angle`` [rad] (None: free)."""
    cfg = DFSPHConfig(particle_radius=R, surface_tension=mode)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    wall_box(b, (0.0, 0.0, -cfg.h), (0.004, 0.004, cfg.h), contact_angle=angle)
    pts = drop_points(radius_cells, cfg.spacing) + np.array([0.0, 0.0, (radius_cells + 1) * cfg.spacing])
    add_fluid_points(b, cfg, pts, 1000.0, 1.0e-3, surface_tension=sigma, adhesion=adhesion)
    b.end_world()
    m = b.finalize(device="cpu")
    return m, SolverDFSPH(m, cfg)


def test_flat_lattice_constants():
    I, c_s = csf.flat_lattice_constants(2.0 * R, 4.0 * R)
    assert 0.3 < I < 1.0 and 0.4 < c_s < 0.8, (I, c_s)
    I2, c_s2 = csf.flat_lattice_constants(2.0e-3, 4.0e-3)
    assert np.isclose(I, I2, rtol=1e-6) and np.isclose(c_s, c_s2, rtol=1e-6)  # scale-free


def test_contact_angle_inputs():
    m, s = _drop(angle=math.radians(60.0))
    cos, sin = (a.numpy() for a in csf.shape_angles(m, "cpu"))
    assert np.isclose(cos[0], 0.5, atol=1e-6) and np.isclose(sin[0], math.sqrt(3) / 2, atol=1e-6)
    m, _ = _drop()
    assert csf.shape_angles(m, "cpu")[1].numpy()[0] == -1.0  # default -1: a free contact line
    a = m.sph.shape_contact_angle.numpy()
    a[0] = 4.0
    m.sph.shape_contact_angle.assign(a)
    with pytest.raises(ValueError, match="shape_contact_angle"):
        csf.shape_angles(m, "cpu")


def test_inputs_the_model_does_not_read_are_refused():
    with pytest.raises(ValueError, match="shape_contact_angle"):
        _drop(mode="none", angle=1.0, sigma=0.0)  # a contact angle without csf would be dropped
    with pytest.raises(ValueError, match="shape_contact_angle"):
        _drop(mode="akinci2013", angle=1.0)
    with pytest.raises(ValueError, match="adhesion"):
        _drop(mode="csf", adhesion=0.5)  # sph:adhesion is Akinci's


def test_report_and_capillary_step():
    m, s = _drop()
    assert s._xsph_c == csf.XSPH_MIN and isinstance(s._tension, csf.CSFTension)
    a, c = m.state(), m.state()
    s.step(a, c, None, None, s.config.dt_max)
    rep = s.report()
    assert rep.tension == "csf" and not rep.tension_qualitative and rep.capillary_dt_violations == 0
    assert rep.tension_rh_range == csf.RH_RANGE_XSPH
    s.step(c, a, None, None, 2.0 * s.config.dt_max)  # above the limit: counted and warned, never silent
    assert s.report().capillary_dt_violations == 1
    _, sa = _drop(mode="akinci2013")
    assert sa.report().tension_qualitative and sa._xsph_c == DFSPHConfig().xsph_c


def test_low_viscosity_drops_warn(caplog):
    with caplog.at_level("WARNING", logger="alloy.sph"):
        _drop()
    assert any("l = 2 oscillation grows" in r.message for r in caplog.records)


def test_interior_particles_get_no_force():
    m, s = _drop(radius_cells=6.0)
    a, c = m.state(), m.state()
    s.step(a, c, None, None, s.config.dt_max)
    q = c.particle_q.numpy()
    rr = np.linalg.norm(q - q.mean(axis=0), axis=1)
    surf = s._tension.surf.numpy()[: m.particle_count]
    assert surf.any() and not surf[rr < 6.0 * 2.0 * R - 2.0 * s.config.h].any()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("radius_cells", [7.0, 10.0])
def test_laplace_pressure(radius_cells):
    """Interior pressure of a resting water drop within 10 % of 2 sigma / R at R/h 3.5 and 5 (no fitted constant)."""
    o = laplace_drop(radius_cells, WATER)
    assert abs(o["ratio"] - 1.0) <= 0.10, o


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("radius_cells", [7.0, 10.0])
def test_rayleigh_l2_frequency(radius_cells):
    """Undamped l = 2 frequency of a water drop within 10 % of Rayleigh's at R/h 3.5 and 5."""
    o = rayleigh_drop(radius_cells, WATER, settle=600)
    assert abs(o["ratio_0"] - 1.0) <= 0.10, o


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_oscillating_drop_returns_to_round():
    """No permanent set: a viscous drop's l = 2 amplitude decays to the noise floor around round (offset ~ 0)."""
    o = rayleigh_drop(7.0, VISCOUS, settle=600, periods=8.0)
    assert o["amp"][-1] < 0.05 * o["amp"][0] and abs(o["offset"]) < 0.05 * o["amp"][0], o
    assert abs(o["ratio_0"] - 1.0) <= 0.10, o


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("angle", [60.0, 120.0])
def test_sessile_contact_angle(angle):
    """Static contact angle within 5 deg of sph:shape_contact_angle at R/h 5 (both measures, sessile_drop)."""
    o = sessile_drop(10.0, WATER, contact_angle=angle, settle=2000)
    assert o["lost"] == 0 and abs(o["theta"] - angle) <= 5.0 and abs(o["theta_fit"] - angle) <= 5.0, o


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("angle", [60.0, 90.0, 120.0])
def test_sessile_contact_angle_finer(angle):
    """The same at R/h 7 (5,730 particles; CUDA)."""
    o = sessile_drop(14.0, WATER, contact_angle=angle, start_angle=120.0 if angle == 90.0 else 90.0, device="cuda:0")
    assert o["lost"] == 0 and abs(o["theta"] - angle) <= 5.0 and abs(o["theta_fit"] - angle) <= 5.0, o


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("radius_cells", [14.0])
def test_laws_finer(radius_cells):
    """Laplace and Rayleigh within 10 % at R/h 7 (water, CUDA)."""
    lap = laplace_drop(radius_cells, WATER, device="cuda:0")
    ray = rayleigh_drop(radius_cells, WATER, settle=600, device="cuda:0")
    assert abs(lap["ratio"] - 1.0) <= 0.10 and abs(ray["ratio_0"] - 1.0) <= 0.10, (lap, ray)

