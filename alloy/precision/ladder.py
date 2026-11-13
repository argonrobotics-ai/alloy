# SPDX-License-Identifier: Apache-2.0
"""The severity ladder of SPEC §4.3 (D3) as code: one row per ``HealthFlags`` bit and per bully monitor.

``CheckPolicy`` and ``BullyLimits`` carry the same numbers; WP7's agreement test reads ``LADDER`` and
``thresholds``. Two flags have two levels on one bit, so their INVALID level is decided by a value, not the bit:
``ENERGY_JUMP`` (per-step gain = S2/DEGRADED, cumulative gain > ``energy_s1_rel`` = S1/INVALID) and
``FAR_FROM_ORIGIN`` (> r_warn = a notice, > r_fail under the precision presets = S1/INVALID).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from alloy.contact.types import PrecisionPairSpec
from alloy.precision.types import CheckPolicy, FramePolicy, HealthFlags, Severity

F = HealthFlags


@dataclass(frozen=True)
class LadderRow:
    key: str                      # HealthFlags name, or "bully:<monitor>"
    flag: HealthFlags | None      # None for bully-only monitors
    condition: str
    bully: str                    # "S1", "S2", "S3", "S3/S1" or "-"
    severity: Severity            # StepHealth severity of the bit (the lower level for two-level rows)
    upper: Severity | None = None  # the second level, decided by value (ENERGY_JUMP, FAR_FROM_ORIGIN)


LADDER: tuple[LadderRow, ...] = (
    LadderRow("NONFINITE_Q", F.NONFINITE_Q, "any non-finite body/joint state", "S1", Severity.INVALID),
    LadderRow("NONFINITE_QD", F.NONFINITE_QD, "any non-finite body/joint velocity", "S1", Severity.INVALID),
    LadderRow("NONFINITE_PARTICLE", F.NONFINITE_PARTICLE, "any non-finite active particle", "S1", Severity.INVALID),
    LadderRow("CONTACT_OVERFLOW", F.CONTACT_OVERFLOW, "dropped contacts (capacity)", "S1", Severity.INVALID),
    LadderRow("ROW_OVERFLOW", F.ROW_OVERFLOW, "dropped constraint rows (capacity)", "S1", Severity.INVALID),
    LadderRow("bully:teleport", None, "reset or jump > 10·v_max·dt", "S1", Severity.INVALID),
    LadderRow("PEN_FAIL", F.PEN_FAIL, "penetration > pen_s1_m (1e-2 m) or > half the thinner part; a soft body's "
              "particle inside another soft body (SCR-SOFT-3-1)", "S1",
              Severity.INVALID),
    LadderRow("PAIR_PEN_FAIL", F.PAIR_PEN_FAIL, "precision pair > min(5e-4 m, radial clearance)", "S1",
              Severity.INVALID),
    LadderRow("SPEED_FAIL", F.SPEED_FAIL, "any body or active particle > speed_fail_mps (20 m/s)", "S1",
              Severity.INVALID),
    LadderRow("bully:impossible_motion", None, "free body > 10 m/s without robot contact", "S1", Severity.INVALID),
    LadderRow("ENERGY_JUMP", F.ENERGY_JUMP, "per-step gain net of W_ext > energy_jump_rel·E_scale (S2); cumulative "
              "unexplained gain > energy_s1_rel·E_scale (S1)", "S2/S1", Severity.DEGRADED, Severity.INVALID),
    LadderRow("PEN_WARN", F.PEN_WARN, "penetration > pen_limit_m (1e-3 m)", "S2", Severity.DEGRADED),
    LadderRow("PAIR_PEN_WARN", F.PAIR_PEN_WARN, "precision pair > 2·max_penetration (the max gate)", "S2",
              Severity.DEGRADED),
    LadderRow("JOINT_FAIL", F.JOINT_FAIL, "joint/weld violation > max(1e-3 m, 3x solref steady state at design load)",
              "S2", Severity.DEGRADED),
    LadderRow("bully:non_target_displacement", None, "non-target object moved > 0.05 mm", "S2", Severity.DEGRADED),
    LadderRow("UNCONVERGED", F.UNCONVERGED, "solver iteration limit or residual > tolerance", "-", Severity.DEGRADED),
    # Notices (coordinator ruling 2026-11-10 02:23Z): listed in the flags, severity OK. A world whose only flags are
    # notices is healthy; every GPU rigid world carries DET_DOWNGRADED (docs/known_gaps/determinism-gpu-rigid.md).
    # FAR_FROM_ORIGIN joined them (TRAIN-FIX 4c, physics 2026-11-11): a walking robot passes 4 m in seconds, so as
    # DEGRADED it marked every locomotion world (TA-3 F4); past r_fail under the precision preset it is still INVALID.
    LadderRow("FAR_FROM_ORIGIN", F.FAR_FROM_ORIGIN, "|p| > r_warn (4 m): a notice; > r_fail (16 m) under the precision "
              "preset: INVALID", "-/S1", Severity.OK, Severity.INVALID),
    LadderRow("FALLBACK_USED", F.FALLBACK_USED, "a fallback path ran (notice)", "-", Severity.OK),
    LadderRow("DET_DOWNGRADED", F.DET_DOWNGRADED, "runs on this device do not repeat bit for bit: effective "
              "determinism below the requested mode, or below RUN_TO_RUN on a GPU (notice)", "-", Severity.OK),
    LadderRow("RUNG_PINNED", F.RUNG_PINNED, "a pinned entry's stability estimate asks a smaller dt than its one step "
              "per exchange (precision rung pinned; SCR-JOB2-1)", "-", Severity.DEGRADED),
    LadderRow("STRAIN_RANGE", F.STRAIN_RANGE, "a soft tet's principal strain > soft_strain_range (60 %): outside the "
              "soft FEM model's range (ruling 2026-11-06 14:17Z; known gap: large compression under a flat tool)", "S2",
              Severity.DEGRADED),
    LadderRow("START_PEN", F.START_PEN, "bodies overlap > pen_limit_m at the start (sticky; past pen_fail: check() not "
              "ok, INVALID under the precision preset; SCR-WP5-14)", "S2", Severity.DEGRADED),
    LadderRow("SPIN", F.SPIN, "a body turns > spin_rad_per_step (0.5 rad) in one step: contacts linearised over that "
              "rotation can inject energy (SCR-WP5-14 addendum; WP3's marker report)", "S2", Severity.DEGRADED),
    LadderRow("CLOTH_STRETCH", F.CLOTH_STRETCH, "a cloth triangle's largest principal stretch > cloth_stretch_warn "
              "(50 %); past cloth_stretch_fail (500 %) the world is INVALID: a blown sheet (SCR-WP5-17)", "S2",
              Severity.DEGRADED),
    LadderRow("SELF_CONTACT_OVERFLOW", F.SELF_CONTACT_OVERFLOW, "a VBD vertex or edge found more self-contact pairs "
              "(within one object or between two in the entry) than its row buffer holds (grown to the scene, up to "
              "256): the rest are dropped, so those surfaces can pass into each other (sticky until a reset; "
              "gap #13); INVALID once they cross, under CheckPolicy.self_contact_cross='invalid'", "S2",
              Severity.DEGRADED),
    LadderRow("FAST_CONTACT", F.FAST_CONTACT, "a body closed on a static box, sphere, capsule or cylinder by more than "
              "5 mm in one rigid step, the shape is thinner along the approach than twice that travel, and the body ended "
              "within 5 mm of it: it can pass through (held 64 rigid steps, then clears; GAP-E2)",
              "S2", Severity.DEGRADED),
    LadderRow("TUNNEL", F.TUNNEL, "a point inside a body passed through a static shape (in and out of the opposite "
              "side, out of another side from 2 mm deep, across it in one step, or the body ended wholly inside it): "
              "INVALID, frozen (sticky; GAP-E2)", "S1",
              Severity.INVALID),
    LadderRow("QUASI_STATIC", F.QUASI_STATIC, "a j2 solid under quasi-static mass scaling moves as a whole faster "
              "than half its declared tool speed, or a body touching it faster than 12x that speed; DEGRADED while "
              "it lasts (GAP #6 slice 2b, health_plugin)", "S2",
              Severity.DEGRADED),
    LadderRow("LATCHED", F.LATCHED, "world frozen after INVALID (sticky)", "-", Severity.INVALID),
)


def _mask(sev: Severity) -> int:
    return int(sum(int(r.flag) for r in LADDER if r.flag is not None and r.severity == sev))


INVALID_MASK: int = _mask(Severity.INVALID)
DEGRADED_MASK: int = _mask(Severity.DEGRADED)
NOTICE_MASK: int = _mask(Severity.OK)        # FALLBACK_USED | DET_DOWNGRADED | FAR_FROM_ORIGIN: they leave a world OK


def row(key: str) -> LadderRow:
    """The ladder row for a flag name or ``bully:<monitor>`` key."""
    for r in LADDER:
        if r.key == key:
            return r
    raise KeyError(key)


def severity_of(flags: int, *, energy_s1: bool = False, far_fail: bool = False) -> Severity:
    """Severity of a flag word; the two-level rows reach INVALID only through their value conditions."""
    if flags & INVALID_MASK or energy_s1 or far_fail:
        return Severity.INVALID
    if flags & DEGRADED_MASK:
        return Severity.DEGRADED
    return Severity.OK


@dataclass(frozen=True)
class Thresholds:
    """Every numeric limit the HealthMonitor uses, from one CheckPolicy + FramePolicy + the precision pairs."""

    pen_warn_m: float
    pen_fail_m: float
    pair_warn_m: float            # loosest pair S2 = max over pairs of 2·max_penetration (inf without pairs)
    pair_fail_m: float            # loosest pair S1 = max over pairs of min(5e-4, radial clearance)
    joint_m: float | None         # None -> per constraint (max(1e-3, 3x solref steady state))
    joint_rad: float | None
    speed_fail_mps: float
    energy_jump_rel: float
    energy_s1_rel: float
    r_warn_m: float
    r_fail_m: float
    far_invalid: bool


def pair_limits(pair: PrecisionPairSpec) -> tuple[float, float]:
    """(S2, S1) of one precision pair (D2, D3), from the pair's own helpers."""
    return pair.s2_limit(), pair.s1_limit()


def thresholds(policy: CheckPolicy, frames: FramePolicy, pairs: Sequence[PrecisionPairSpec] = ()) -> Thresholds:
    limits = [pair_limits(p) for p in pairs]
    return Thresholds(
        pen_warn_m=policy.pen_limit_m, pen_fail_m=policy.pen_s1_m,
        # the per-world max over pairs only bounds the loosest pair; the GapMonitor classifies each pair exactly
        pair_warn_m=max((s2 for s2, _ in limits), default=float("inf")),
        pair_fail_m=max((s1 for _, s1 in limits), default=float("inf")),
        joint_m=policy.joint_limit_m, joint_rad=policy.joint_limit_rad,
        speed_fail_mps=policy.speed_fail_mps, energy_jump_rel=policy.energy_jump_rel,
        energy_s1_rel=policy.energy_s1_rel, r_warn_m=frames.r_warn_m, r_fail_m=frames.r_fail_m,
        far_invalid=bool(frames.far_invalid),
    )
