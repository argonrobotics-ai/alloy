# SPDX-License-Identifier: Apache-2.0
"""CARD-8: a sheet routed CLOTH_VBD sags under self-weight like a Navier simply supported plate (≤ 15 %).

Square plate a = 0.2 m, h = 2 mm, E = 3 GPa, ν = 0.3, ρ = 1200 kg/m³ (polycarbonate-like): the router sends it to
CLOTH_VBD (δ_b at 2 N = 0.18 mm > τ), the emitter maps E, ν, h to the 2D Lame pair (derive.cloth_stiffness) and
the plate hinge map edge_ke = c·D·L/(A1+A2) (VBD's hinge energy is ½·edge_ke·L·θ²), and Alloy's VBD starts the
pinned plate's iterations from x + h·v (particle_pinned_piece_start; the inertial start settled it 30x past Navier
at 10 iterations; see docs/notes/wp11-results.md). Boundary vertices are pinned (positions fixed, rotation free:
boundary edges carry no bending element = simply supported).
Load q = ρ·h·g; w_max/h ≈ 0.035, inside linear plate theory. Navier series (Timoshenko & Woinowsky-Krieger, Theory
of Plates and Shells, 1959, §30): w(a/2, a/2) = 16q/(π⁶D) Σ_{m,n odd} (−1)^((m+n)/2−1) / (m·n·(m²/a² + n²/a²)²)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.cards.types import Budget
from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.testing import route as T
from alloy.testing import scene as S

A, H, E, NU, RHO, G = 0.2, 2.0e-3, 3.0e9, 0.3, 1200.0, 9.81
TOL = 0.15


def navier_centre(q: float, a: float, D: float, terms: int = 99) -> float:
    s = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            s += (-1) ** ((m + n) // 2 - 1) / (m * n * (m * m / a**2 + n * n / a**2) ** 2)
    return 16.0 * q / (math.pi**6 * D) * s


def test_navier_series_matches_table():
    # TWK table 8: α = 0.00406 for a simply supported square plate, w = α·q·a⁴/D
    D = E * H**3 / (12 * (1 - NU**2))
    assert navier_centre(1.0, 1.0, 1.0) == pytest.approx(0.00406, rel=2e-3)
    assert navier_centre(RHO * H * G, A, D) / H == pytest.approx(0.035, rel=0.05)


#: CARD-8 also under SolverVBDAlloy's accelerators (alloy.scene.cloth_accuracy). Chebyshev mixed positions until
#: 2026-11-08 and froze this plate at w = 0 (a 0.007 um iteration step is below float32's ulp at z 0.5 m; WP11).
ACCEL = {"stock": {}, "chebyshev": {"alloy_chebyshev_rho": 0.9},
         "adaptive_chebyshev_20": {"alloy_initial_guess": "adaptive", "alloy_chebyshev_rho": 0.9, "iterations": 20}}


@pytest.mark.compile_heavy
@pytest.mark.timeout(180)
@pytest.mark.parametrize("accel", list(ACCEL))
def test_card8_plate_deflection(accel):
    from alloy.integration.throughput import with_entry_kwargs

    pc = T.material("pc", density=RHO, E=E, nu=NU, mu=0.4, e=0.2, beta=2e-3)
    plate = T.box("plate", pc, 0.5 * A, 0.5 * A, 0.5 * H, budget=Budget(particles=17 * 17))
    card, lib = S.scene([plate], [S.place("plate", z=0.5)], worlds=1)
    with isolated_registry(empty=True):
        built = SceneAssembler(lib, Router()).assemble(card, "cpu")
        assert built.handles["plate"].decision.route == Route.CLOTH_VBD
        if ACCEL[accel]:
            built = with_entry_kwargs(built, ACCEL[accel], "vbd")
        m = built.model
        q0 = m.particle_q.numpy()
        edge = (np.abs(np.abs(q0[:, 0]) - 0.5 * A) < 1e-6) | (np.abs(np.abs(q0[:, 1]) - 0.5 * A) < 1e-6)
        mass, inv = m.particle_mass.numpy(), m.particle_inv_mass.numpy()
        mass[edge], inv[edge] = 0.0, 0.0
        m.particle_mass.assign(mass)
        m.particle_inv_mass.assign(inv)
        sim = Simulation(built)
        for _ in range(40):                                      # 0.8 s with β damping
            sim.step_period()
        q = sim.state.particle_q.numpy()
    centre = int(np.argmin(np.hypot(q0[:, 0], q0[:, 1])))
    w = float(q0[centre, 2] - q[centre, 2])
    D = E * H**3 / (12 * (1 - NU**2))
    ref = navier_centre(RHO * H * G, A, D)
    err = w / ref - 1.0
    S.record("CARD-8.navier" + ("" if accel == "stock" else f".{accel}"), err, "pass" if abs(err) <= TOL else "fail",
             w_m=w, navier_m=ref, grid=int(round(math.sqrt(len(q0)))), device="cpu")
    assert abs(err) <= TOL, (w, ref)
