# SPDX-License-Identifier: Apache-2.0
"""alloy.scene.derive against hand-computed values (formula in each comment)."""

from __future__ import annotations

import math

import pytest

from alloy.scene import derive as d


def test_lame_steel():
    # E = 200 GPa, ν = 0.3: μ = E/(2·1.3) = 76.923 GPa, λ = E·0.3/(1.3·0.4) = 115.385 GPa
    mu, lam = d.lame(200e9, 0.3)
    assert mu == pytest.approx(200e9 / 2.6, rel=1e-12)
    assert lam == pytest.approx(200e9 * 0.3 / (1.3 * 0.4), rel=1e-12)


def test_lame_rejects_bad_poisson():
    with pytest.raises(ValueError):
        d.lame(1e6, 0.5)


def test_cloth_stiffness_is_the_plane_stress_lame_pair():
    """cotton: E = 1 MPa, nu = 0.3, h = 0.5 mm. Newton's VBD membrane takes tri_ke = mu and tri_ka = lambda (2D Lame),
    so mu = E h / (2(1+nu)) = 192.31 N/m and lambda = E h nu / (1-nu^2) = 164.84 N/m; its 2D modulus
    4 mu (mu + lambda) / (2 mu + lambda) and Poisson lambda / (2 mu + lambda) give back E h = 500 N/m and nu = 0.3.
    (Before 2026-11-08: Newton's USD importer map E h / (1-nu^2), 0, a 2.2x stiffer membrane with Poisson 0.)
    edge_ke = D = E h^3 / (12(1-nu^2)) = 1.1447e-5 N m."""
    mu, lam, edge_ke = d.cloth_stiffness(1e6, 0.3, 5e-4)
    assert mu == pytest.approx(500.0 / 2.6, rel=1e-12)
    assert lam == pytest.approx(500.0 * 0.3 / 0.91, rel=1e-12)
    assert 4 * mu * (mu + lam) / (2 * mu + lam) == pytest.approx(500.0, rel=1e-12)
    assert lam / (2 * mu + lam) == pytest.approx(0.3, rel=1e-12)
    assert edge_ke == pytest.approx(1e6 * 1.25e-10 / (12 * 0.91), rel=1e-12)


def test_rod_section_circle():
    # r = 1 mm, l = 1 cm, E = 1 GPa, G = 0.4 GPa: EA/l = 1e9·π·1e-6/1e-2; EI/l = 1e9·π·1e-12/4/1e-2; GJ = G·π·r⁴/2
    ea, ei, gj = d.rod_section(1e9, 0.4e9, 1e-3, 1e-2)
    assert ea == pytest.approx(1e9 * math.pi * 1e-6 / 1e-2)
    assert ei == pytest.approx(1e9 * math.pi * 1e-12 / 4 / 1e-2)
    assert gj == pytest.approx(0.4e9 * math.pi * 1e-12 / 2 / 1e-2)


def test_torsional_lever_hertz():
    # μ 0.5, F 20 N, R 10 mm, E* 1 GPa: a = (3·20·0.01/(4e9))^(1/3) = 5.313e-4 m; lever = 3π/16·0.5·a
    a = (3 * 20 * 0.01 / 4e9) ** (1 / 3)
    assert d.torsional_lever(0.5, 20.0, 0.01, 1e9) == pytest.approx(3 * math.pi / 16 * 0.5 * a, rel=1e-12)
    assert d.torsional_lever(0.5, 0.0, 0.01, 1e9) == 0.0


def test_pair_stiffness_series():
    assert d.pair_stiffness(2e6, 2e6) == pytest.approx(1e6)
    assert d.pair_stiffness(1e6, math.inf) == 1e6
    assert d.pair_stiffness(0.0, 1e6) == 0.0


def test_harmonic_mu():
    assert d.harmonic_mu(0.4, 0.8) == pytest.approx(2 * 0.32 / 1.2)
    assert d.harmonic_mu(0.0, 0.0) == 0.0


def test_mu_mix_rules():
    assert d.mu_mix("max", 0.3, 0.7) == 0.7
    assert d.mu_mix("mean", 0.3, 0.7) == pytest.approx(0.5)
    assert d.mu_mix("geometric", 0.25, 1.0) == pytest.approx(0.5)
    assert d.mu_mix("harmonic", 0.4, 0.8) == pytest.approx(d.harmonic_mu(0.4, 0.8))
    assert d.mu_mix("none", 0.4, 0.8) == 0.0


def test_friction_scale_never_zero():
    assert d.friction_scale(0.35, 0.7) == pytest.approx(0.5)
    assert d.friction_scale(0.0, 0.7) == d.TINY_SCALE   # 0 would mean "unset" in Newton
    assert d.friction_scale(0.3, 0.0) == 0.0


@pytest.mark.parametrize("e", [0.1, 0.3, 0.5, 0.8, 0.95])
def test_zeta_from_restitution_roundtrip(e):
    # underdamped impact: e = exp(−ζπ/√(1−ζ²))
    z = d.zeta_from_restitution(e)
    assert math.exp(-z * math.pi / math.sqrt(1 - z * z)) == pytest.approx(e, rel=1e-12)


def test_zeta_limits():
    assert d.zeta_from_restitution(0.0) == 1.0
    assert d.zeta_from_restitution(1.0) == 0.0
    with pytest.raises(ValueError):
        d.zeta_from_restitution(1.5)


def test_contact_damping_mujoco_correction():
    # kd = 2ζ√(K m); MuJoCo divides by √(1−imp)
    K, m, e = 1e5, 0.2, 0.5
    plain = d.contact_damping(e, K, m)
    assert plain == pytest.approx(2 * d.zeta_from_restitution(e) * math.sqrt(K * m))
    assert d.contact_damping(e, K, m, imp=0.95) == pytest.approx(plain / math.sqrt(0.05))
    assert d.contact_damping(1.0, K, m) == d.TINY_DAMPING   # e = 1 must not write 0 (= critical in Newton)


def test_k_real_unclamped_equals_declared():
    # 35 g peg, K = 2e6, imp 0.99, dt 0.3125 ms: τ_req = √(m/(K(1−imp))) = 1.32 ms > 2dt → K_real = K
    m, K, imp, dt = 0.035, 2e6, 0.99, 3.125e-4
    tau, zeta = d.mujoco_solref_from_force(K, 0.0, m, imp)
    assert tau == pytest.approx(math.sqrt(m / (K * (1 - imp))))
    assert d.k_real_mujoco(m, imp, dt, tau, zeta) == pytest.approx(K, rel=1e-12)


def test_k_real_clamped_spec_example():
    # D22: 35 g peg at imp 0.95, dt 0.3125 ms: τ clamped to 2dt → K_real = m/((1−imp)·4dt²) ≈ 1.79e6 N/m
    m, imp, dt = 0.035, 0.95, 3.125e-4
    tau, zeta = d.mujoco_solref_from_force(1e7, 0.0, m, imp)
    assert d.k_real_mujoco(m, imp, dt, tau, zeta) == pytest.approx(m / (0.05 * 4 * dt * dt), rel=1e-12)
    assert d.k_real_mujoco(m, imp, dt, tau, zeta) == pytest.approx(1.792e6, rel=1e-3)


def test_k_real_discrete_cap():
    # discrete: k = 1/(imp²τ²ζ²) capped at 1/(dt²·imp) → K = imp·m/((1−imp)·dt²)
    m, imp, dt = 0.035, 0.95, 3.125e-4
    tau, zeta = d.mujoco_solref_from_force(1e9, 0.0, m, imp)
    assert d.k_real_mujoco(m, imp, dt, tau, zeta, integrator="discrete") == pytest.approx(
        imp * m / ((1 - imp) * dt * dt), rel=1e-12)
    tau2, zeta2 = d.mujoco_solref_from_force(1e5, 0.0, m, imp)   # soft: no cap → K
    assert d.k_real_mujoco(m, imp, dt, tau2, zeta2, integrator="discrete") == pytest.approx(1e5, rel=1e-12)


def test_solref_with_damping_keeps_static_k():
    # τ²ζ² = 1/ke' whatever kd is, so the unclamped static stiffness stays K
    m, K, imp = 0.5, 1e5, 0.9
    kd = d.contact_damping(0.5, K, m, imp)
    tau, zeta = d.mujoco_solref_from_force(K, kd, m, imp)
    assert d.k_real_mujoco(m, imp, 1e-5, tau, zeta) == pytest.approx(K, rel=1e-12)
