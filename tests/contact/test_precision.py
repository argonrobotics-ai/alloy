# SPDX-License-Identifier: Apache-2.0
"""alloy.contact.precision: D22 stiffness, K_real, gap-ladder rungs, L2 sizing (hand-computed values)."""

from __future__ import annotations

import math

import pytest

from alloy.contact import precision as pr
from alloy.contact.types import PrecisionPairSpec
from alloy.core.errors import PrecisionSpecError

PEG = PrecisionPairSpec(0, 1, mode="sdf", clearance=2e-4)   # 12.0 in 12.2 mm: max_pen = min(5e-5, 0.5*1e-4) = 5e-5


def test_pair_stiffness_default_and_override():
    # K = 2 F / max_pen = 2*50/5e-5 = 2e6 N/m (D22)
    assert pr.pair_stiffness(PEG, 50.0) == pytest.approx(2.0e6)
    assert pr.pair_stiffness(PrecisionPairSpec(0, 1, pair_stiffness=3e6), 50.0) == 3e6
    with pytest.raises(PrecisionSpecError):
        pr.pair_stiffness(PEG, 0.0)


def test_k_real_matches_formula_and_clamp():
    # m/((1-imp) tau^2 zeta^2): 0.035/(0.05 * (2*3.125e-4)^2) = 1.792e6 (D22: "35 g peg ... K_real ~ 1.8e6 at imp 0.95")
    k = pr.k_real_mujoco(0.035, 0.95, 3.125e-4, tau_req=1e-5)
    assert k == pytest.approx(0.035 / (0.05 * (6.25e-4) ** 2))
    assert k == pytest.approx(1.792e6, rel=1e-3)
    # tau_req above the clamp is used as is
    assert pr.k_real_mujoco(0.035, 0.95, 1e-5, tau_req=1e-3) == pytest.approx(0.035 / (0.05 * 1e-6))
    with pytest.raises(ValueError):
        pr.k_real_mujoco(0.035, 1.0, 1e-4, 1e-3)


def test_tau_for_stiffness_round_trip():
    tau = pr.tau_for_stiffness(2e6, 0.026, 0.99)
    assert tau == pytest.approx(math.sqrt(0.026 / (0.01 * 2e6)))
    assert pr.k_real_mujoco(0.026, 0.99, 1e-6, tau) == pytest.approx(2e6)


def test_check_k_real_raises_when_dt_too_coarse():
    # tau for 2e6 at 26 g, imp 0.99 = 1.14e-3 s; at dt 1.25e-3 the clamp tau=2.5e-3 gives K_real = 4.16e5 < 1.6e6
    with pytest.raises(PrecisionSpecError, match="K_real"):
        pr.check_k_real(2e6, 0.026, 0.99, 1.25e-3, label="peg/hole")
    assert pr.check_k_real(2e6, 0.026, 0.99, 3.125e-4) == pytest.approx(2e6)


def test_ladder_rungs_per_integrator():
    assert pr.ladder_rungs(PEG, "euler") == ("L0", "L1", "L2", "L3", "L4")
    assert pr.ladder_rungs(PrecisionPairSpec(0, 1), "euler") == ("L1", "L2", "L3", "L4")      # hydroelastic: no L0
    assert pr.ladder_rungs(PEG, "implicitfast", native="newton") == ("L0", "L1", "L2", "L3", "L4")
    assert pr.ladder_rungs(PEG, "discrete") == ("L0", "L2", "L3", "L4")
    assert pr.ladder_rungs(PEG, "euler", append_l1=False) == ("L0", "L2", "L3", "L4")
    # no L0 without ccd: the pipeline expands no gap by velocity
    assert pr.ladder_rungs(PrecisionPairSpec(0, 1, mode="sdf", ccd="off"), "euler") == ("L2", "L3", "L4")
    for bad in ("rk4", "implicit"):
        with pytest.raises(NotImplementedError):
            pr.ladder_rungs(PEG, bad)


def test_l2_impact_dt_matches_spec_example():
    # SPEC §3.5 L2: 50 um at 0.5 m/s -> e*5e-5/(2*0.5) = 1.36e-4 s
    assert pr.l2_impact_dt(5e-5, 0.5) == pytest.approx(math.e * 5e-5 / 1.0)
    assert pr.l2_impact_dt(5e-5, 0.5) == pytest.approx(1.359e-4, rel=1e-3)
    assert pr.l2_explicit_dt(1e6, 0.01) == pytest.approx(0.5 / 1e4)


def test_size_l2_snaps_to_grid():
    # impact 1.36e-4 -> grid 0.02/2^8 = 7.8125e-5 s (SPEC: "1.36e-4 s -> 7.8e-5 s on the grid")
    s = pr.size_l2(PEG, m_eff=0.026, design_force=50.0, contact_speed=0.5, control_dt=0.02)
    assert s.dt == pytest.approx(0.02 / 2**8)
    assert s.k_declared == pytest.approx(2e6)
    assert s.k_real == pytest.approx(2e6)
    assert s.static_ok
    # slow approach (CT-3, 0.05 m/s): impact bound 1.36e-3; stiffness bound 0.5*tau(1.6e6) = 6.37e-4 -> 3.125e-4 grid
    slow = pr.size_l2(PEG, m_eff=0.026, design_force=50.0, contact_speed=0.05, control_dt=0.02)
    assert slow.dt == pytest.approx(0.02 / 2**6)


def test_size_l2_raises_below_base_tick():
    tiny = PrecisionPairSpec(0, 1, max_penetration=1e-7)
    with pytest.raises(PrecisionSpecError, match="base tick"):
        pr.size_l2(tiny, m_eff=0.026, design_force=50.0, contact_speed=0.5, control_dt=0.02)


def test_supported_natives_route_away_measured_failures():
    """SB-3: hydroelastic precision pairs only on native='mujoco'; sdf pairs on both paths."""
    assert pr.supported_natives(PrecisionPairSpec(0, 1, mode="hydroelastic")) == ("mujoco",)
    assert pr.supported_natives(PrecisionPairSpec(0, 1, mode="sdf")) == ("mujoco", "newton")
    assert "CT-1" in pr.UNSUPPORTED_NATIVE[("hydroelastic", "newton")]
