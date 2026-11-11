# SPDX-License-Identifier: Apache-2.0
"""Film laws FILM-1..5 (design note §9). The full-size runs are ``gpu`` + ``slow`` (SC8 VM:
``python -m pytest tests/solvers/film -m gpu``); FILM-4a (the air model against the measured coffee filter) and the
heavy-elastica target are CPU-cheap and run in the gate.

Measured numbers per law and the commit they were taken at: docs/notes/sc8-film-laws.md.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.film import aero, laws
from alloy.solvers.film.config import FilmConfig
from alloy.solvers.film.materials import HDPE_15UM, LLDPE_30UM, PAPER_100UM
from alloy.solvers.film.rigs import frustum_mesh

FILTER = {"mass": 0.86e-3, "r_bottom": 0.0435, "r_top": 0.0755, "height": 0.047}   # arXiv 2206.14902
FILTER_VT = (1.06, 1.12)                                                         # m/s, same source


def test_heavy_elastica_is_d1388s_41_5_degrees() -> None:
    assert abs(laws.elastica_chord_angle(2.0) - 41.5) < 0.05


def _filter_air_force(speed: float) -> float:
    v, t = frustum_mesh(FILTER["r_bottom"], FILTER["r_top"], FILTER["height"])
    p = aero.AirParams()
    c = FilmConfig()
    p.rho, p.cd, p.ct, p.wind, p.dt, p.cap = c.air_density, c.drag_normal, c.drag_tangent, wp.vec3(0.0, 0.0, 0.0), 1e-3, 0
    qd = np.zeros_like(v, dtype=np.float32)
    qd[:, 2] = -speed
    f = wp.zeros(len(t), dtype=wp.vec3, device="cpu")
    wp.launch(aero.tri_air_force, dim=len(t), inputs=[p, wp.array(v.astype(np.float32), dtype=wp.vec3, device="cpu"),
                                                      wp.array(qd, dtype=wp.vec3, device="cpu"),
                                                      wp.array(t, dtype=wp.int32, device="cpu"),
                                                      wp.array(np.ones(len(t), np.float32), dtype=float, device="cpu"),
                                                      f], device="cpu")
    return float(f.numpy()[:, 2].sum())


@pytest.mark.law
def test_film4a_air_model_terminal_speed_of_the_coffee_filter() -> None:
    """The per-triangle air force on the filter's shape balances its weight within 10 % of the measured speed."""
    weight = FILTER["mass"] * 9.81
    lo, hi = 0.0, 3.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _filter_air_force(mid) < weight else (lo, mid)
    v_model = 0.5 * (lo + hi)
    assert 0.9 * FILTER_VT[0] <= v_model <= 1.1 * FILTER_VT[1], v_model


@pytest.mark.parametrize("spec", [HDPE_15UM], ids=["hdpe"])
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_film1_cantilever_bending_length(spec) -> None:
    r = laws.cantilever(spec, device="cuda:0", seconds=2.5)
    assert abs(r["angle_deg"] - r["target_deg"]) <= 5.0, r
    assert r["step_gain_max_rel"] < 1e-2


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_film2_hanging_sheet_and_loaded_handle_stay_within_the_cap() -> None:
    r = laws.hang(HDPE_15UM, device="cuda:0", seconds=1.5)
    assert r["strain_p99_over_cap"] <= 1.1 and r["step_gain_max_rel"] < 1e-2, r
    load = 0.5 * HDPE_15UM.yield_stress * 0.10 * HDPE_15UM.thickness / 9.81           # σ = σ_y/2 on the strip
    r = laws.hang(HDPE_15UM, device="cuda:0", seconds=1.5, load_kg=load, film=FilmConfig(mass_ratio_cap=10.0))
    assert r["strain_p99_over_cap"] <= 1.35 and r["step_gain_max_rel"] < 1e-2, r


@pytest.mark.parametrize("spec", [HDPE_15UM, LLDPE_30UM], ids=["hdpe", "lldpe"])
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_film3_fold_release_return_angle(spec) -> None:
    r = laws.fold_release(spec, device="cuda:0")
    assert abs(r["return_angle_deg"] - r["target_return_deg"]) <= 15.0, r


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_film4b_falling_paper_filter() -> None:
    """Free film components take A17's per-component translation step (SolverVBDMomentum base, item 8 G2): stock
    VBD let the 0.1 mm filter fall at ~0.65 m/s (docs/notes/sc8-vbd-momentum.md); with the step 1.05 m/s on CPU."""
    r = laws.falling_cone(PAPER_100UM, device="cuda:0", **FILTER)
    assert 0.9 * FILTER_VT[0] <= r["terminal_mps"] <= 1.1 * FILTER_VT[1], r


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(900)
def test_film4b_falling_paper_filter_on_cpu() -> None:
    """FILM-4b on CPU, 10 iterations (~70 s): 1.048 m/s with the momentum step (target 1.06-1.12 m/s)."""
    r = laws.falling_cone(PAPER_100UM, device="cpu", iterations=10, **FILTER)
    assert 0.9 * FILTER_VT[0] <= r["terminal_mps"] <= 1.1 * FILTER_VT[1], r
    assert r["step_gain_max_rel"] < 1e-2, r
