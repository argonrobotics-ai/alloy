# SPDX-License-Identifier: Apache-2.0
"""Laws of the film's stretch limit as a force carrier, on ``FilmConfig.limiter="stress"`` (alloy/solvers/film/
limiter_laws.py states each bound; the default projection fails every one of them but L4's force: a strip under
2-500 g stood 11.5-34x past its cap, the pins read 0.64-1.01 of the weight, the load moved at 10-83 mm/s)."""

from __future__ import annotations

import pytest

from alloy.solvers.film import limiter_laws as L
from alloy.solvers.film.config import FilmConfig

pytestmark = pytest.mark.compile_heavy
LIMITER = "stress"


@pytest.mark.parametrize("load_kg", L.LOADS)
def test_hang_holds_the_load_at_the_films_elastic_stretch(load_kg: float) -> None:
    r = L.measure_hang(LIMITER, load_kg)
    assert r.stretch_ratio <= L.stretch_bound(load_kg), r
    assert r.force_low <= r.force_ratio <= 1.0 + L.FORCE_OVER, r
    assert r.speed <= L.SPEED_MAX and r.particle_speed <= L.PARTICLE_SPEED_MAX, r


def test_light_hang_far_from_the_origin_reads_the_same_force() -> None:
    """float32 positions 1 m out: a particle's force is quantised above a pin's share; the pins' sum still holds."""
    r, r0 = L.measure_hang(LIMITER, 0.002, x0=1.0), L.measure_hang(LIMITER, 0.002)
    assert r.ok, r
    assert abs(r.force_ratio - r0.force_ratio) <= 0.01, (r, r0)


def test_off_centre_load_moment_is_carried_by_the_pins() -> None:
    r = L.measure_moment(LIMITER)
    assert r.residual <= L.MOMENT_TOL, r


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_pendulum_keeps_period_and_energy() -> None:
    r = L.measure_pendulum(LIMITER)
    assert abs(r.period / r.period_ref - 1.0) <= L.PERIOD_TOL, r
    assert r.energy_gain <= L.GAIN_TOL, r
    assert r.loss_per_period <= L.LOSS_MAX, r


def test_released_load_is_let_go_at_the_films_elastic_recoil() -> None:
    r = L.measure_release(LIMITER)
    assert r.force_error <= L.RELEASE_FORCE_TOL, r
    assert r.speed <= r.speed_bound, r


def test_film_config_checks_the_limiter() -> None:
    assert FilmConfig().limiter == "projection"                        # the stress limiter is opt-in
    assert FilmConfig(limiter=LIMITER).stress_entry == 0.05
    with pytest.raises(ValueError, match="limiter"):
        FilmConfig(limiter="xpbd")
    for bad in (1.0, -0.1):
        with pytest.raises(ValueError, match="stress_entry"):
            FilmConfig(limiter=LIMITER, stress_entry=bad)


def test_the_stress_limiter_is_named_and_pulls_on_the_pins() -> None:
    s = L.make_strip(LIMITER, 0.050)
    assert "elastic membrane" in s.rig.solver.alloy_route_notes()["film_limiter"]
    s.rig.run(0.3)
    assert s.rig.solver.limiter_force.numpy()[s.top][:, 2].sum() < 0.0   # the pins are pulled down

