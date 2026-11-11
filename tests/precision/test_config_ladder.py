# SPDX-License-Identifier: Apache-2.0
"""Presets (SPEC §3.7 table), GPU tiers and the §4.3 severity ladder as code."""

from __future__ import annotations

import pytest
import warp as wp

from alloy.contact.types import PrecisionPairSpec
from alloy.precision.config import PRESETS, far_is_invalid, gpu_tier, preset
from alloy.precision.ladder import DEGRADED_MASK, INVALID_MASK, LADDER, pair_limits, row, severity_of, thresholds, NOTICE_MASK
from alloy.precision.types import (
    CheckPolicy, Determinism, FramePolicy, GpuTier, HealthFlags, PrecisionConfig, Severity,
)


def test_cpu_tier() -> None:
    assert gpu_tier("cpu") == GpuTier.CPU


@pytest.mark.parametrize("name", PRESETS)
def test_presets_follow_the_table(name: str) -> None:
    for tier in GpuTier:
        cfg = PrecisionConfig.preset(name, tier)  # WP0 delegates to alloy.precision.config.preset
        assert cfg == preset(name, tier)
        assert cfg.determinism == (Determinism.NONE if name == "fast" else Determinism.RUN_TO_RUN)
        assert cfg.accum.scatter == ("atomic" if name == "fast" else "fixed64")
        assert cfg.accum.positions == "f32"  # D14: compensated/f64 positions deferred
        assert cfg.frames.worlds_at_origin


def test_preset_checks() -> None:
    fast, std, prec, ref = (preset(n, GpuTier.B) for n in PRESETS)
    assert (fast.checks.level, fast.checks.every, fast.checks.full_every) == ("full", 10, 10)   # joints too (ERR-13)
    assert (std.checks.level, std.checks.every, std.checks.full_every) == ("full", 1, 10)
    assert (prec.checks.level, prec.checks.full_every, prec.checks.on_invalid) == ("full", 1, "raise")
    assert ref.checks.level == "full" and ref.checks.full_every == 1
    assert prec.accum.hessian == "f32_refine" and preset("precision", GpuTier.A).accum.hessian == "f64"
    assert ref.accum.hessian == "f64" and std.accum.hessian == "auto" and fast.accum.hessian == "f32"


def test_unknown_preset() -> None:
    with pytest.raises(ValueError):
        preset("turbo", GpuTier.B)


def test_digest_distinguishes_presets() -> None:
    digests = {preset(n, GpuTier.B).digest() for n in PRESETS}
    assert len(digests) == len(PRESETS)


def test_every_flag_has_one_row() -> None:
    flagged = [r.flag for r in LADDER if r.flag is not None]
    assert sorted(int(f) for f in flagged) == sorted(int(f) for f in HealthFlags)
    assert INVALID_MASK & DEGRADED_MASK == 0
    assert INVALID_MASK | DEGRADED_MASK | NOTICE_MASK == sum(int(f) for f in HealthFlags)    # notices: severity OK


def test_severity_of() -> None:
    assert severity_of(0) == Severity.OK
    assert severity_of(int(HealthFlags.PEN_WARN | HealthFlags.UNCONVERGED)) == Severity.DEGRADED
    assert severity_of(int(HealthFlags.PEN_FAIL | HealthFlags.PEN_WARN)) == Severity.INVALID
    assert severity_of(int(HealthFlags.ENERGY_JUMP)) == Severity.DEGRADED
    assert severity_of(int(HealthFlags.ENERGY_JUMP), energy_s1=True) == Severity.INVALID
    assert severity_of(int(HealthFlags.FAR_FROM_ORIGIN), far_fail=True) == Severity.INVALID
    assert row("bully:teleport").bully == "S1" and row("JOINT_FAIL").severity == Severity.DEGRADED


def test_pair_limits_d2_d3() -> None:
    # 12.0 mm peg in 12.2 mm hole: radial 0.1 mm -> max pen 0.05 mm, S2 0.1 mm, S1 0.1 mm
    assert pair_limits(PrecisionPairSpec(0, 1, clearance=2e-4)) == pytest.approx((1e-4, 1e-4))
    # M10 6H/6g: 32 um diametral -> radial 16 um: max pen 8 um, S2 16 um, S1 16 um
    assert pair_limits(PrecisionPairSpec(0, 1, clearance=3.2e-5)) == pytest.approx((1.6e-5, 1.6e-5))
    assert pair_limits(PrecisionPairSpec(0, 1)) == pytest.approx((1e-4, 5e-4))


def test_thresholds_follow_policies() -> None:
    t = thresholds(CheckPolicy(), FramePolicy(), [PrecisionPairSpec(0, 1, clearance=2e-4),
                                                  PrecisionPairSpec(2, 3, clearance=3.2e-5)])
    assert (t.pen_warn_m, t.pen_fail_m, t.speed_fail_mps) == (1e-3, 1e-2, 20.0)
    assert t.pair_warn_m == pytest.approx(1e-4) and t.pair_fail_m == pytest.approx(1e-4)  # loosest pair
    assert (t.r_warn_m, t.r_fail_m) == (4.0, 16.0)
    assert t.far_invalid == far_is_invalid(FramePolicy())
    empty = thresholds(CheckPolicy(), FramePolicy())
    assert empty.pair_warn_m == float("inf")
