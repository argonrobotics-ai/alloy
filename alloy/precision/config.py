# SPDX-License-Identifier: Apache-2.0
"""Precision presets and GPU tiers (SPEC §3.7 preset table, D14; precision research §4.1, §4.7).

v1 presets keep positions float32 everywhere (D14 defers compensated/f64 positions inside Newton solvers); the
precision preset gets its accuracy from frames (origin + task-centred origin), pair tolerances, Fixed64 scatter,
full health checks every step and determinism. ``AccumPolicy.hessian`` applies to Alloy-owned solves only.
"""

from __future__ import annotations

import logging
from typing import Literal

import warp as wp

from alloy.precision.types import (
    AccumPolicy,
    CheckPolicy,
    Determinism,
    FramePolicy,
    GpuTier,
    PrecisionConfig,
)

log = logging.getLogger("alloy.precision")

PresetName = Literal["fast", "standard", "precision", "reference"]
PRESETS: tuple[str, ...] = ("fast", "standard", "precision", "reference")
TIER_A_ARCHS = frozenset({80, 90})  # A100, H100: FP64 at 1:2 (SPEC §3.7 GpuTier)


def gpu_tier(device: wp.context.Devicelike = None) -> GpuTier:
    """CPU for the Warp CPU device; A for sm_80/sm_90; B for every other CUDA device (L4 = sm_89)."""
    dev = wp.get_device(device)
    if not dev.is_cuda:
        return GpuTier.CPU
    return GpuTier.A if int(dev.arch) in TIER_A_ARCHS else GpuTier.B


def _hessian(name: str, tier: GpuTier) -> str:
    if name == "fast":
        return "f32"
    if name == "standard":
        return "auto"
    if name == "reference":
        return "f64"
    return "f32_refine" if tier == GpuTier.B else "f64"


def _checks(name: str) -> CheckPolicy:
    if name == "fast":
        # joints and welds too (audit ERR-13): "cheap" never measured them, so JOINT_FAIL could not fire under fast
        return CheckPolicy(level="full", every=10, full_every=10)
    if name == "standard":
        return CheckPolicy(level="full", every=1, full_every=10)
    if name == "precision":
        return CheckPolicy(level="full", every=1, full_every=1, on_invalid="raise", start_pen_invalid=True)
    return CheckPolicy(level="full", every=1, full_every=1)


def preset(name: str, tier: GpuTier = GpuTier.B) -> PrecisionConfig:
    """The preset table of SPEC §3.7 (``PrecisionConfig.preset`` delegates here)."""
    if name not in PRESETS:
        raise ValueError(f"unknown precision preset {name!r}; expected one of {PRESETS}")
    precise = name in ("precision", "reference")
    det = Determinism.NONE if name == "fast" else Determinism.RUN_TO_RUN
    accum = AccumPolicy(hessian=_hessian(name, tier), positions="f32",
                        scatter="atomic" if name == "fast" else "fixed64")
    frames = _frame_policy(precise)
    cfg = PrecisionConfig(determinism=det, frames=frames, accum=accum, checks=_checks(name), fuse_fp=True)
    log.debug("precision preset %s tier %s -> %s", name, tier.name, cfg)
    return cfg


def _frame_policy(precise: bool) -> FramePolicy:
    """Origin everywhere; beyond r_fail is INVALID only under the precision/reference presets (§4.3 row)."""
    return FramePolicy(far_invalid=precise)


def far_is_invalid(frames: FramePolicy) -> bool:
    """True when |p| > r_fail must make a world INVALID (precision/reference presets)."""
    return bool(frames.far_invalid)
