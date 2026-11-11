# SPDX-License-Identifier: Apache-2.0
"""cloth_accuracy on the VBD entry (alloy.scene.cloth_accuracy, SCR-WP2-13): a tagged entry runs SolverVBDAlloy's
adaptive start + Chebyshev at the setting's iterations ("super" wins over "default", "default" over "fast"); "fast"
runs VBD's own solve; an untagged entry (no cloth) is unchanged; a higher iteration count (the "accurate" cloth
profile's) is kept."""

from __future__ import annotations

import pytest

from alloy.scene import cloth_accuracy as CA


def test_untagged_entry_kwargs_are_unchanged():
    kw = {"iterations": 10, "particle_enable_self_contact": True}
    assert CA.entry_kwargs(kw, set()) == kw


def test_super_runs_the_adaptive_start_at_more_iterations():
    out = CA.entry_kwargs({"iterations": 10}, {CA.SUPER_EFFECT})
    assert out["alloy_initial_guess"] == "adaptive" and out["alloy_chebyshev_rho"] == 0.9
    assert out["iterations"] == CA.SUPER_ITERATIONS == 60


def test_default_runs_the_adaptive_start_at_its_iterations():
    out = CA.entry_kwargs({"iterations": 10}, {CA.DEFAULT_EFFECT})
    assert out["alloy_initial_guess"] == "adaptive" and out["alloy_chebyshev_rho"] == 0.9
    assert out["iterations"] == CA.DEFAULT_ITERATIONS == 40
    assert CA.entry_kwargs({"iterations": 10}, {CA.DEFAULT_EFFECT, CA.SUPER_EFFECT})["iterations"] == 60


def test_fast_runs_vbds_own_solve():
    """cloth_accuracy="fast" (ruling 2026-11-08 15:58Z): VBD's inertial start, no Chebyshev, its own 10 iterations."""
    out = CA.entry_kwargs({"iterations": 10}, {CA.FAST_EFFECT})
    assert out["alloy_initial_guess"] == "inertial" and out["alloy_chebyshev_rho"] is None
    assert out["iterations"] == CA.FAST_ITERATIONS == 10
    assert CA.entry_kwargs({"iterations": 10}, {CA.FAST_EFFECT, CA.DEFAULT_EFFECT})["iterations"] == 40
    assert CA.tagged({CA.FAST_EFFECT}) == "fast" and CA.COST["fast"] < 1.0 and "12 %" in CA.ACCURACY["fast"]


def test_every_setting_keeps_a_higher_iteration_count():
    for effect in CA.EFFECT.values():
        assert CA.entry_kwargs({"iterations": 6000}, {effect})["iterations"] == 6000


def test_checked_names_the_choices():
    assert CA.checked("super") == "super" and CA.checked("fast") == "fast"
    with pytest.raises(ValueError, match="cloth_accuracy must be one of"):
        CA.checked("high")
