# SPDX-License-Identifier: Apache-2.0
"""Pendant and sessile drops on C18's materials (SCR-SC10-2; alloy/solvers/sph/drops.py), csf with no added force.

Laws: (5) the largest pendant drop a round polypropylene tip (100 deg) holds matches Tate with the Harkins-Brown
correction within 15 %: 0.85 x is held and 1.15 x falls at tip radii 0.75 and 1.25 mm (r = 0.075 mm; rows by
viscosity: 0.05 Pa s and C18's 0.06 Pa s glycerol-water, open and sealed tip; CUDA; water fails, SCR-WP9-7).
Young-Laplace's largest static drop is 0.94 x Tate-HB on both tips (25.0 and 37.6 uL of water), inside that band.
(6) the sessile contact angle is within 5 deg of the plate's (polypropylene 100 deg, polystyrene 88 deg), with the
plate open or sealed (ambient.py). (7) csf is unchanged: nothing in csf.py changed for SCR-SC10-2 (wall adhesion
was built, measured and dropped: wp9-results.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.solvers.sph.drops import C18_GLYCEROL, C18_VISCOUS, WATER, pendant_drop, sessile_drop, tate_volume

PP, PS = 100.0, 88.0  # water on polypropylene / untreated polystyrene [deg] (SC10's C18 tip and well)


def test_tate_volume_reference():
    """Tate with Harkins-Brown (Lando-Oakley fit): F(x) x V = r sigma / (rho g) at the fixed point; 26.5 / 40.3 uL."""
    for r, ul in ((0.75e-3, 26.5), (1.25e-3, 40.3)):
        v = tate_volume(r)
        x = r / v ** (1.0 / 3.0)
        assert np.isclose(v * (0.14782 + 0.27896 * x - 0.166 * x * x), r * 0.072 / (1000.0 * 9.81), rtol=1e-9)
        assert abs(v * 1.0e9 - ul) < 0.05


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("angle", [PP, PS])
@pytest.mark.parametrize("ambient", [None, 500.0])
def test_sessile_angle_c18_materials(angle, ambient):
    """Law (6): the static contact angle within 5 deg of the plate's at R/h 5, plate open or sealed (both measures)."""
    o = sessile_drop(10.0, WATER, contact_angle=angle, settle=2000, ambient_pressure=ambient)
    assert o["lost"] == 0 and abs(o["theta"] - angle) <= 5.0 and abs(o["theta_fit"] - angle) <= 5.0, o


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("liquid", "ambient"), [(C18_VISCOUS, None), (C18_GLYCEROL, None), (C18_GLYCEROL, 500.0)],
                         ids=["mu0.05", "c18_glycerol_mu0.06", "c18_glycerol_mu0.06_sealed"])
@pytest.mark.parametrize("tip_radius", [0.75e-3, 1.25e-3])
def test_pendant_drop_tate(tip_radius, liquid, ambient):
    """Law (5), by viscosity: a polypropylene round tip holds 0.85 x Tate-Harkins-Brown's volume (of the liquid) for
    0.5 s and loses 1.15 x of it; 0.05 Pa s with water's rho and sigma, and C18's 80 % glycerol-water (0.06 Pa s),
    open and sealed as C18's capillary (p_a 500 Pa). Water fails (SCR-WP9-7, wp9-results.md): not tested here."""
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    v = tate_volume(tip_radius, liquid.sigma, liquid.rho0)
    held = pendant_drop(tip_radius, 0.85 * v, liquid, contact_angle=PP, ambient_pressure=ambient, device="cuda:0")
    lost = pendant_drop(tip_radius, 1.15 * v, liquid, contact_angle=PP, ambient_pressure=ambient, device="cuda:0")
    assert held["held"] and not lost["held"], (held, lost)
