# SPDX-License-Identifier: Apache-2.0
"""alloy.route.checks and alloy.route.rodref against hand-computed values and the SPEC §3.3 sanity cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.route import checks as c
from alloy.route import rodref as r


def test_morphology():
    assert c.morphology((0.2, 0.01, 0.01), False, False)[0] == "rod"
    assert c.morphology((0.2, 0.2, 0.002), False, False)[0] == "sheet"
    assert c.morphology((0.1, 0.08, 0.05), False, False)[0] == "volume"
    assert c.morphology((0.1, 0.08, 0.05), True, False)[0] == "rod"
    assert c.morphology((0.1, 0.08, 0.05), False, True)[0] == "sheet"


def test_local_squeeze_flat_punch():
    # δ = F(1−ν²)/(2aE), a = min(l3/2, 5 mm): 20 N on rubber E = 2 MPa, ν 0.45, l3 = 4 mm → a = 2 mm
    assert c.local_squeeze(20.0, 2e6, 0.45, 4e-3) == pytest.approx(20 * (1 - 0.2025) / (2 * 2e-3 * 2e6))
    # a capped at 5 mm
    assert c.local_squeeze(20.0, 2e6, 0.45, 0.1) == pytest.approx(20 * (1 - 0.2025) / (2 * 5e-3 * 2e6))


def test_spec_sanity_steel_rod_rigid():
    # SPEC §3.3: steel rod 10 × 200 mm at 2 N: δ_b = 2·0.2³/(3·200e9·π·0.005⁴/4) = 54 µm ≤ τ = 100 µm
    sec = c.rod_section("cylinder", (0.005, 0.1), (0.2, 0.01, 0.01))
    assert c.rod_bend(2.0, sec.length, 200e9 * sec.inertia) == pytest.approx(54.3e-6, rel=2e-3)


def test_spec_sanity_tray_rigid():
    # 2 mm aluminium tray, 200 mm: R = 0.1, D = 70e9·8e-9/(12(1−0.33²)), δ = 2·R²/(16πD) = 7.6 µm
    D = c.plate_rigidity(70e9, 0.33, 2e-3)
    assert c.sheet_bend(2.0, 0.1, D) == pytest.approx(7.6e-6, rel=1e-2)


def test_spec_sanity_screwdriver_and_pencil():
    shaft = c.rod_section("cylinder", (0.0025, 0.05), (0.1, 0.005, 0.005))
    assert c.rod_bend(2.0, shaft.length, 200e9 * shaft.inertia) == pytest.approx(0.109e-3, rel=1e-2)
    pencil = c.rod_section("cylinder", (0.0035, 0.085), (0.17, 0.007, 0.007))
    assert c.rod_bend(2.0, pencil.length, 10e9 * pencil.inertia) == pytest.approx(2.8e-3, rel=2e-2)


def test_rod_section_box_weak_axis():
    sec = c.rod_section("box", (0.1, 0.01, 0.002), (0.2, 0.02, 0.004))
    assert sec.inertia == pytest.approx(0.02 * 0.004**3 / 12)
    assert sec.area == pytest.approx(0.02 * 0.004)


def test_gamma_and_bending_length():
    # γ = ρAgL³/EI; c = (EI/(ρAg))^(1/3) → γ = (L/c)³
    rho, area, L, EI = 1200.0, math.pi * 9e-6, 0.5, 0.08
    g = c.rod_gamma(rho, area, L, EI)
    assert g == pytest.approx((L / c.rod_bending_length(rho, area, EI)) ** 3, rel=1e-12)


def test_chain_links():
    assert c.chain_links(1.0, 0.17, 0.002) == (7, 7)                # stiff cable: the sag rule (SCR-DC3-7, no floor)
    assert c.chain_links(1.0, 0.17, 0.002, override=12) == (12, 7)  # card features.chain_links
    links, req = c.chain_links(3.0, 0.08, 0.002)                    # limp long cable: required 39
    assert req == 39 and links == 39
    assert c.chain_links(2.0, 0.02, 0.002)[1] == 101               # → ROD_VBD branch (> cap)
    assert c.chain_links(0.05, 0.01, 0.004) == (6, 6)
    assert c.chain_links(0.05, 0.01, 0.004, override=30)[0] == 7    # never shorter than one diameter
    assert c.chain_links(0.3, 1.0, 0.002)[0] == c.CHAIN_LINKS_MIN   # a stiff short cord: at least 3


def test_threshold_margin():
    t = c.Threshold("x", 0.109e-3, 1e-4)
    assert not t.passed
    assert t.margin == pytest.approx(math.log10(1.09))
    assert c.min_margin([t, c.Threshold("y", 1.0, 100.0)]) == pytest.approx(math.log10(1.09))


def test_sph_radius_and_counts():
    assert c.sph_radius(None) == 2e-3
    assert c.sph_radius(0.016) == pytest.approx(1e-3)       # passage/16
    assert c.sph_particle_count(1e-3, 1e-3) == 125_000     # 1 L at 2 mm spacing
    assert c.grain_count(1e-6, 1e-3) == math.ceil(0.6 * 1e-6 / (math.pi * 1e-9 / 6))
    assert c.mpm_voxel(0.04, 5e-3) == 5e-3 and c.mpm_voxel(0.01, 5e-3) == 2.5e-3


# --- rod references ------------------------------------------------------------------------------------------

L, R, RHO = 0.5, 0.003, 1200.0
M_LIN = RHO * math.pi * R**2


def test_elastica_small_deflection_limit():
    EI = M_LIN * 9.81 * L**3 / 0.01        # γ = 0.01: linear theory wL⁴/8EI within 1e-5 relative
    assert r.elastica_tip_drop(L, M_LIN, EI) == pytest.approx(r.linear_tip_drop(L, M_LIN, EI), rel=1e-4)


def test_elastica_large_deflection_known_value():
    # γ = wL³/EI = 5: tip drop / L = 0.4959 (regression value of the N = 4001 chain, not an independent table)
    EI = M_LIN * 9.81 * L**3 / 5.0
    assert r.elastica_tip_drop(L, M_LIN, EI) / L == pytest.approx(0.4959, abs=5e-4)


def _moment_residual(lay, th, g=9.81):
    """Hinge moment balance M_j = Σ_{i≥j} w_i·(x_ci − x_j) vs k_j·(θ_j − θ_{j−1}), relative to the root moment."""
    lens, w = lay.lengths, lay.masses * g
    c = np.cos(th)
    x_node = lay.root + np.concatenate([[0.0], np.cumsum(lens * c)])[:-1]
    x_c = x_node + 0.5 * lens * c
    moment = np.cumsum((w * x_c)[::-1])[::-1] - np.cumsum(w[::-1])[::-1] * x_node
    return float(np.max(np.abs(lay.k * np.diff(np.concatenate([[0.0], th])) - moment)) / np.max(np.abs(moment)))


@pytest.mark.parametrize("gamma", [0.5, 5.0, 50.0, 1944.0])
def test_static_reference_balances_moments_at_any_gamma(gamma):
    # v1's fixed point diverged above γ ≈ 10 (the 1 m USB cable is γ = 1944); Newton must satisfy the hinge balance
    EI = M_LIN * 9.81 * L**3 / gamma
    lay = r.chain_layout(30, L, M_LIN, EI)
    th = r.static_angles(lay)
    assert _moment_residual(lay, th) < 1e-9
    assert 0.0 < r.static_tip_drop(lay) < L


def test_static_reference_approaches_hanging_limit():
    drops = [r.elastica_tip_drop(L, M_LIN, M_LIN * 9.81 * L**3 / g) / L for g in (10.0, 100.0, 1e3, 1e4, 1e5)]
    assert all(b > a for a, b in zip(drops, drops[1:], strict=False)) and 0.99 < drops[-1] < 1.0


@pytest.mark.parametrize("n", [20, 30, 40, 80])
def test_midpoint_chain_converges_second_order(n):
    EI = M_LIN * 9.81 * L**3 / 0.5
    err = r.static_tip_drop(r.chain_layout(n, L, M_LIN, EI)) / r.elastica_tip_drop(L, M_LIN, EI) - 1.0
    assert abs(err) < 0.6 / (n - 1) ** 2      # midpoint rule: |err| ≈ 0.5/(N−1)² at γ = 0.5


def test_release_reference_static_and_period():
    EI = M_LIN * 9.81 * L**3 / 0.5
    lay = r.chain_layout(30, L, M_LIN, EI, R)
    t = np.linspace(0.0, 1.0, 2001)
    ref = r.release_reference(lay, L, M_LIN, EI, t)
    assert ref.drop[0] == pytest.approx(0.0, abs=1e-12)
    assert ref.static_drop == pytest.approx(r.static_tip_drop(lay), rel=5e-3)   # linear vs exact at γ 0.5
    assert ref.period == pytest.approx(ref.period_continuum, rel=2e-3)
    assert 1.8 * ref.static_drop < ref.peak_drop < 2.1 * ref.static_drop       # undamped release overshoot


def test_hanging_chain_and_catenary():
    assert r.hanging_chain_w1(1.0, 9.81) == pytest.approx(1.2024 * math.sqrt(9.81), rel=1e-4)
    sag = r.catenary_sag(0.8, 1.0)
    # independent check: recover the catenary parameter a from the sag, then its length 2a·sinh(span/2a) = 1 m
    lo, hi = 1e-3, 1e3
    for _ in range(200):
        a = math.sqrt(lo * hi)
        if a * (math.cosh(0.4 / a) - 1) > sag:
            lo = a
        else:
            hi = a
    assert 2 * a * math.sinh(0.4 / a) == pytest.approx(1.0, rel=1e-9)


def test_euler_buckling():
    assert r.euler_buckling_load(1.0, 1.0) == pytest.approx(math.pi**2)
    assert r.euler_buckling_load(1.0, 1.0, 2.0) == pytest.approx(math.pi**2 / 4)
