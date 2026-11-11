# SPDX-License-Identifier: Apache-2.0
"""k_real_mujoco vs a MuJoCo drop at 3 dts (WP2 acceptance, D22): a free 1 kg sphere settles on a table through
the pair-table per-contact stiffness (Newton-contacts mode, elliptic cone); measured static stiffness m·g/δ vs the
formula, within 10 %. dts span unclamped (τ_req > 2dt), borderline and clamped (τ = 2dt) rows.

A free body is used so MuJoCo's body_invweight0 (its contact-mass estimate, mujoco/kernels.py:663-669) is 1/m;
for constrained bodies m_eff must be MuJoCo's 1/invweight0, not the physical mass (noted in the results)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.cards.types import Primitive
from alloy.core.registry import isolated_registry
from alloy.precision.types import PrecisionConfig
from alloy.route.router import Router
from alloy.scene import derive
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.testing import route as T
from alloy.testing import scene as S

R, TOP, MASS, K_MAT, G = 0.05, 0.02, 1.0, 2.0e6, 9.81
RESULTS: dict[float, tuple[float, float]] = {}


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def settle(dt: float, t_end: float = 0.4) -> tuple[float, float, float]:
    rho = MASS / (4.0 / 3.0 * math.pi * R**3)
    ball = T.obj("ball", Primitive("sphere", (R,)), T.material("ball", density=rho, E=1e9, mu=0.5, e=0.0, k=K_MAT))
    tab = T.box("table", T.material("tab", density=600.0, E=1e10, mu=0.5, e=0.0, k=K_MAT), 0.2, 0.2, 0.02,
                motion="static")
    card, lib = S.scene([tab, ball], [S.place("table"), S.place("ball", z=TOP + R - MASS * G / (0.5 * K_MAT))],
                        worlds=1)
    built = SceneAssembler(lib, Router()).assemble(card, "cpu")
    sim = Simulation(built, precision=PrecisionConfig(native_contacts="newton"))
    for _ in range(int(round(t_end / dt))):
        sim._exchange(dt)
    z = float(sim.state.body_q.numpy()[0, 2])
    imp = float(built.model.mujoco.geom_solimp.numpy().reshape(-1, 5)[0, 1])
    k = float(built.pair_table.stiffness.numpy()[0, 1])
    return TOP + R - z, imp, k


@pytest.mark.parametrize("dt", [1.25e-3, 2.5e-3, 5.0e-3])
def test_k_real_matches_drop(dt):
    delta, imp, k = settle(dt)
    kd = derive.contact_damping(0.0, k, MASS, imp)
    tau, zeta = derive.mujoco_solref_from_force(k, kd, MASS, imp)
    k_formula = derive.k_real_mujoco(MASS, imp, dt, tau, zeta)
    k_meas = MASS * G / delta
    RESULTS[dt] = (k_meas, k_formula)
    err = k_meas / k_formula - 1.0
    S.record("k_real_vs_drop", err, "pass" if abs(err) <= 0.10 else "fail", dt=dt, k_meas=k_meas,
             k_formula=k_formula, k_declared=k, tau_req=tau, clamped=tau < 2 * dt)
    assert delta > 0
    assert k_meas == pytest.approx(k_formula, rel=0.10), (dt, k_meas, k_formula, tau, 2 * dt)


def test_rows_cover_both_regimes():
    """The three dts include an unclamped (K_real = K) and a clamped (K_real < K) row."""
    k = 0.5 * K_MAT
    imp = 0.95
    kd = derive.contact_damping(0.0, k, MASS, imp)
    tau, _ = derive.mujoco_solref_from_force(k, kd, MASS, imp)
    assert tau > 2 * 1.25e-3 and tau < 2 * 5.0e-3
