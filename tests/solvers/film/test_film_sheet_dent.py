# SPDX-License-Identifier: Apache-2.0
"""Law S2 of J2 metal sheets (SCR-WP7-15 P3, sheet_dent.py): a steel strip on two supports dented by a rigid cylinder
and released. Hinge by hinge at the end of the hold, the plane-strain section's moment at each hinge's curvature is the
statics' R a, the kept curvature is an elastic unloading from there, and the strip settles to the lattice sum of its
kept rest angles. The set against the continuous beam integral at the same peak is a measured ratio (the module doc
says why it is not a check: 3-point bending is ill-conditioned, and the hinge network is more compliant than the
plate). CPU, mass-scaled quasi-static."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.solvers.film import sheet_dent as S
from alloy.solvers.film.sheet_laws import metal_spec, section

pytestmark = pytest.mark.compile_heavy


def _check(r: S.DentRow, width: float, statics: float, springback: float, settle: float) -> None:
    D = section(metal_spec("steel_carbon", 1.0e-3))[3]
    for a, k, m, kept in zip(r.lever, r.k_hold, r.m_hold, r.k_kept):
        assert m / (r.reaction * a / width) == pytest.approx(1.0, abs=statics), (a, r)
        assert abs(kept - (k - m / D)) <= springback * k, (a, kept, k - m / D)
    assert r.set == pytest.approx(r.set_rest, rel=settle), r
    assert r.max_strain > 0.0 and r.k_kept[0] > 0.5 * r.k_hold[0], r          # the mid hinge yielded and kept it


@pytest.mark.law
def test_s2_a_short_strip_follows_the_statics_and_settles_to_its_rest_angles():
    """60 mm span, 1 mm steel, 2.5 mm of cylinder travel: the mid hinge yields to 6 x the first-yield curvature."""
    r = S.s2_strip_dent(span=0.06, width=0.005, depth=2.5e-3, speed=5.0e-3, hold=0.2, settle=0.6)
    _check(r, 0.005, statics=0.04, springback=0.04, settle=0.03)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1200)
def test_s2_steel_strip_dent_row():
    """The S2 row: 100 mm span, 10 mm wide, 6 mm of travel (peak 5.9 mm, mid hinge at 7 x first yield). Measured
    2026-11-08: statics within 2.1 %, springback within 1.6 % of κ, set 2.080 mm vs the lattice sum 2.074 mm, and
    set / beam integral at the same peak 0.77 (20 and 60 iterations alike)."""
    r = S.s2_strip_dent()
    _check(r, 0.01, statics=0.03, springback=0.02, settle=0.01)
    assert 0.70 < r.set / r.set_beam < 0.85, r


def test_the_beam_lattice_converges_to_the_beam_integral():
    """The hinge-lattice sums (each hinge bent by its own lever's κ(M) over l_h = dx) tend to the continuous
    integral as dx shrinks, and the elastic profile is M / D."""
    sets = [S.beam_dent("steel_carbon", 1.0e-3, 0.1, 0.01, 5.5e-3, dx)[1] for dx in (5e-3, 2.5e-3, 1.25e-3)]
    exact = S.beam_dent("steel_carbon", 1.0e-3, 0.1, 0.01, 5.5e-3)[1]
    errors = np.abs(np.array(sets) - exact)
    assert errors[0] < 0.03 * exact and np.all(np.diff(errors) < 0.0), (sets, exact)
    D = section(metal_spec("steel_carbon", 1.0e-3))[3]
    k, kr = S.beam_profile("steel_carbon", 1.0e-3, 0.01, 1.0, np.array([0.01, 0.02]))     # 1 N: far below yield
    assert np.allclose(k, np.array([0.01, 0.02]) / 0.01 / D, rtol=1e-9) and np.allclose(kr, 0.0, atol=1e-12)
