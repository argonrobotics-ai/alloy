# SPDX-License-Identifier: Apache-2.0
"""SPEC §4.3 severity ladder as data, the bully classifier and the B1 badness objective.

Two row sets cover the table: ``FLAG_ROWS`` maps every ``HealthFlags`` bit (set by WP5's ``HealthMonitor`` from the
same numbers, ``CheckPolicy``) and ``MONITOR_ROWS`` maps every bench monitor (``BenchMonitors.report`` keys) to a
threshold field of ``BullyLimits`` and a severity. ``classify`` takes the worst (lowest non-zero) severity per world
and fails closed: a world whose ``HealthReport.severity`` is INVALID is at least S1 even if no S1 row fired
(D3: INVALID <=> an S1 condition; this covers the S1 levels of ENERGY_JUMP and FAR_FROM_ORIGIN, which share one bit
with ENERGY_JUMP's S2 level and FAR_FROM_ORIGIN's notice).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from alloy.bully.types import BullyLimits, BullySeverity
from alloy.core.errors import BullyError
from alloy.precision.types import HealthFlags, HealthReport, Severity

S1, S2, S3 = BullySeverity.S1, BullySeverity.S2, BullySeverity.S3
NONE = 0  # no finding in this world
NONFINITE_BADNESS = 1.0e3  # a non-finite world outranks every finite monitor ratio in B1

# Rows for bits another package is adding to HealthFlags; each applies once the bit exists, so either merge order
# keeps main green. RUNG_PINNED (SCR-JOB2-1, WP5 for WP12): a pinned entry's stability estimate asks a smaller dt than
# its one step per exchange; DEGRADED, flag only. STRAIN_RANGE (coordinator ruling 2026-11-06T14:17Z, WP5 with WP10's
# alloy.topology.soft_strain): a soft tet's principal strain is past the model range (60 % to start); DEGRADED, a
# model-limit report, so S2: S1 is not redefined, a blow-up still trips the INVALID rules. START_PEN (SCR-WP5-14): the
# start state overlaps; a placement error, not an engine finding, so flag only (a deeper overlap still trips PEN_FAIL
# S1 at the check step). SPIN (WP5, WP3 reviewed): a body turns > 0.5 rad per physics step; a precursor, flag only
# (the blow-up it warns of trips the S1 / INVALID rows). Both DEGRADED. CLOTH_STRETCH (SCR-WP5-17): a cloth triangle's
# largest principal stretch above 50 %, DEGRADED and sticky; a model-limit report like STRAIN_RANGE, so S2 (past
# 500 % the world is INVALID through classify, which arrives as S1 by the fail-closed rule). SELF_CONTACT_OVERFLOW
# (gap #13, WP1): VBD self-contact pairs dropped over a row buffer, DEGRADED and sticky. S2: a local loss of VBD
# self-contact pairs (the world's other contacts are kept), unlike CONTACT_OVERFLOW's S1 (WP7 review). TUNNEL
# (GAP-E2): a body passed through a static shape, INVALID and frozen: S1 like PEN_FAIL. FAST_CONTACT (GAP-E2): a body
# closed on a static shape by more than the contact gap in one step, DEGRADED: S2 like PEN_WARN.
_ADDED_FLAG_ROWS: Mapping[str, BullySeverity | None] = {"RUNG_PINNED": None, "STRAIN_RANGE": S2, "START_PEN": None,
                                                        "SPIN": None, "CLOTH_STRETCH": S2,
                                                        "SELF_CONTACT_OVERFLOW": S2, "FAST_CONTACT": S2,
                                                        "TUNNEL": S1, "QUASI_STATIC": S2}

# HealthFlags bit -> bully severity (None: flag-only, no bully finding) — §4.3, one row per flag
FLAG_ROWS: Mapping[HealthFlags, BullySeverity | None] = {
    HealthFlags.NONFINITE_Q: S1,
    HealthFlags.NONFINITE_QD: S1,
    HealthFlags.NONFINITE_PARTICLE: S1,
    HealthFlags.CONTACT_OVERFLOW: S1,
    HealthFlags.ROW_OVERFLOW: S1,
    HealthFlags.PEN_FAIL: S1,
    HealthFlags.PAIR_PEN_FAIL: S1,
    HealthFlags.SPEED_FAIL: S1,
    HealthFlags.ENERGY_JUMP: S2,  # S1 level (cumulative > 0.5 E_scale) arrives as INVALID, see module doc
    HealthFlags.PEN_WARN: S2,
    HealthFlags.PAIR_PEN_WARN: S2,
    HealthFlags.JOINT_FAIL: S2,
    HealthFlags.FAR_FROM_ORIGIN: None,  # a notice (TRAIN-FIX 4c); S1 under the precision preset arrives as INVALID
    HealthFlags.UNCONVERGED: None,
    HealthFlags.FALLBACK_USED: None,
    HealthFlags.DET_DOWNGRADED: None,
    HealthFlags.LATCHED: None,
    **{HealthFlags[n]: lvl for n, lvl in _ADDED_FLAG_ROWS.items() if n in HealthFlags.__members__},
}


@dataclass(frozen=True)
class MonitorRow:
    """One bench monitor: value > threshold(limits) trips ``severity``; ``None`` severity = diagnostic only."""

    name: str
    threshold: Callable[[BullyLimits], float]
    severity: BullySeverity | None
    condition: str


MONITOR_ROWS: tuple[MonitorRow, ...] = (
    MonitorRow("nonfinite", lambda lim: 0.5, S1, "any non-finite body or active particle state (value 1)"),
    MonitorRow("capacity_overflow", lambda lim: 0.5, S1, "solver contact or row capacity overflowed (value 1)"),
    MonitorRow("teleport", lambda lim: 1.0, S1, "|dx| > 10 v_max dt (value = |dx| / (10 v_max dt))"),
    MonitorRow("max_speed", lambda lim: lim.speed_fail, S1, "any body or active particle > 20 m/s (SPEED_FAIL)"),
    MonitorRow("impossible_speed", lambda lim: lim.v_max, S1,
               "free body speed not explained by robot contact or free fall (SCR-WP7-7): in contact or one substep "
               "after, its speed > v_max; in flight, |v| > max(v_max, |v_exit| + g t + 0.5 m/s)"),
    MonitorRow("pen_s1_ratio", lambda lim: 1.0, S1, "penetration > 1e-2 m or > half the thinner part (PEN_FAIL)"),
    MonitorRow("e_gain_cum", lambda lim: lim.e_gain_s1_rel, S1, "cumulative unexplained gain > 0.5 E_scale"),
    # Target rows (replay.target_monitors: the target's sim reports [W] counts over the episode), agreed with the
    # class owners for SB-7 (coordinator 2026-11-05): cutting with WP10, cloth with WP11
    MonitorRow("cut_pool_overflow", lambda lim: 0.5, S1, "cutting: the topology controller's spare pool overflowed"),
    MonitorRow("inverted_tets", lambda lim: 0.5, S1, "cutting / soft bodies: a tet with det F <= 0"),
    MonitorRow("cloth_pass_through", lambda lim: 0.5, S1, "cloth: a body passed through the cloth"),
    MonitorRow("cloth_crossings", lambda lim: 0.5, S1, "cloth: edge-triangle self-crossings at a control period"),
    MonitorRow("cut_face_deviation", lambda lim: np.inf, None,
               "cutting: episode max distance (m) of a newly opened twin face from the blade's swept "
               "surface; diagnostic while seams are pre-split (WP10, engine item 9)"),
    MonitorRow("max_pen", lambda lim: lim.pen_limit, S2, "penetration > 1e-3 m (PEN_WARN)"),
    MonitorRow("e_gain", lambda lim: lim.e_gain_s2_rel, S2, "per-step unexplained gain > 1e-2 E_scale"),
    MonitorRow("joint_sep", lambda lim: lim.joint_sep_s2, S2, "joint / weld anchor separation > 1e-3 m"),
    MonitorRow("nontarget_disp", lambda lim: lim.rest_disp_max, S2, "non-target displacement > 0.05 mm"),
    MonitorRow("contact_free_accel", lambda lim: np.inf, None, "|a - g| of a free body in no contact (diagnostic)"),
    MonitorRow("impossible_speed_d12", lambda lim: np.inf, None,
               "D12 before SCR-WP7-7: free body > v_max without robot contact, free fall included (diagnostic)"),
)
MONITORS: Mapping[str, MonitorRow] = {row.name: row for row in MONITOR_ROWS}


def _worst(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise worst severity: the lowest non-zero value (0 = none)."""
    a0 = np.where(a == NONE, 99, a)
    b0 = np.where(b == NONE, 99, b)
    out = np.minimum(a0, b0)
    return np.where(out == 99, NONE, out).astype(np.int32)


def classify_flags(health: HealthReport) -> np.ndarray:
    """[W] severity from the HealthFlags rows plus the INVALID fail-closed rule."""
    flags = np.asarray(health.flags).astype(np.int64)
    sev = np.zeros(flags.shape[0], dtype=np.int32)
    for flag, level in FLAG_ROWS.items():
        if level is not None:
            sev = _worst(sev, np.where(flags & int(flag), int(level), NONE))
    invalid = np.asarray(health.severity) == int(Severity.INVALID)
    return _worst(sev, np.where(invalid, int(S1), NONE))


def classify_monitors(extra: Mapping[str, np.ndarray], limits: BullyLimits) -> np.ndarray:
    """[W] severity from bench monitor values; unknown keys raise, absent rows are skipped; NaN counts as tripped."""
    unknown = set(extra) - set(MONITORS)
    if unknown:
        raise BullyError(f"no §4.3 row for bench monitor(s) {sorted(unknown)}")
    sev: np.ndarray | None = None
    for name, values in extra.items():
        row = MONITORS[name]
        v = np.asarray(values, dtype=np.float64)
        sev = np.zeros(v.shape[0], dtype=np.int32) if sev is None else sev
        if row.severity is not None:
            tripped = ~(v <= row.threshold(limits))  # NaN trips
            sev = _worst(sev, np.where(tripped, int(row.severity), NONE))
    return np.zeros(0, dtype=np.int32) if sev is None else sev


def classify(health: HealthReport, extra: Mapping[str, np.ndarray], limits: BullyLimits) -> np.ndarray:
    """[W] severity per world: 0 none, else the worst ``BullySeverity`` over health flags and bench monitors."""
    sev = classify_flags(health)
    if extra:
        mon = classify_monitors(extra, limits)
        if mon.shape != sev.shape:
            raise BullyError(f"health has {sev.shape[0]} worlds, monitors {mon.shape[0]}")
        sev = _worst(sev, mon)
    return sev


def monitor_ratios(extra: Mapping[str, np.ndarray], limits: BullyLimits) -> dict[str, np.ndarray]:
    """value / threshold per monitor with a severity row, capped at ``NONFINITE_BADNESS``; non-finite values and a
    tripped ``nonfinite`` take the cap, so every blown-up world ranks at the top of B1."""
    out: dict[str, np.ndarray] = {}
    for name, values in extra.items():
        row = MONITORS[name]
        if row.severity is None:
            continue
        v = np.asarray(values, dtype=np.float64)
        r = v / row.threshold(limits)
        if name in ("nonfinite", "capacity_overflow"):
            r = np.where(v > 0.5, NONFINITE_BADNESS, 0.0)
        out[name] = np.clip(np.where(np.isfinite(r), r, NONFINITE_BADNESS), 0.0, NONFINITE_BADNESS)
    return out


S1_BADNESS_OFFSET = 10.0  # SCR-WP7-6: a tripped S1 row scores 10 + r
S2_BADNESS_CAP = 10.0  # SCR-WP7-6: an S2 row scores min(r, 10)


def _weighted(severity: BullySeverity | None, r: np.ndarray) -> np.ndarray:
    if severity == S1:
        return np.where(r > 1.0, S1_BADNESS_OFFSET + r, r)
    return np.minimum(r, S2_BADNESS_CAP)


def badness(extra: Mapping[str, np.ndarray], limits: BullyLimits) -> np.ndarray:
    """B1 objective per world (§3.11 as amended by SCR-WP7-6): b = max_m w_m·r_m + 0.1 · (sum of the others), with
    w·r = 10 + r for a tripped S1 row and min(r, 10) for an S2 row, so S1 worlds rank above S2 worlds."""
    ratios = monitor_ratios(extra, limits)
    if not ratios:
        raise BullyError("badness needs at least one bench monitor with a severity row")
    stack = np.stack([_weighted(MONITORS[name].severity, r) for name, r in ratios.items()], axis=0)
    top = stack.max(axis=0)
    return top + 0.1 * (stack.sum(axis=0) - top)


INVALID_CAUSE = "INVALID"  # a world INVALID by health with no S1 row set (fail-closed rule)


def world_cause(health: HealthReport, extra: Mapping[str, np.ndarray], limits: BullyLimits, world: int
                ) -> tuple[str, float, float, int]:
    """(monitor, value, threshold, severity) of the row that sets ``world``'s severity: the most severe tripped row,
    largest value/threshold among equals. Health flags appear by name (value 1, threshold 0.5); a fail-closed
    INVALID world without an S1 row is ``INVALID``. Raises BullyError if nothing tripped."""
    causes: list[tuple[int, float, str, float, float]] = []
    for name, values in extra.items():
        row = MONITORS[name]
        v, thr = float(np.asarray(values)[world]), float(row.threshold(limits))
        if row.severity is not None and not v <= thr:
            ratio = np.inf if not np.isfinite(v) else v / thr
            causes.append((int(row.severity), ratio, name, v, thr))
    bits = int(np.asarray(health.flags)[world])
    causes += [(int(lv), 1.0 / 0.5, f.name, 1.0, 0.5) for f, lv in FLAG_ROWS.items() if lv is not None and bits & int(f)]
    if int(np.asarray(health.severity)[world]) == int(Severity.INVALID) and not any(c[0] == int(S1) for c in causes):
        causes.append((int(S1), 1.0 / 0.5, INVALID_CAUSE, 1.0, 0.5))
    if not causes:
        raise BullyError(f"world {world} tripped nothing")
    sev, _, name, value, thr = min(causes, key=lambda c: (c[0], -c[1]))
    return name, value, thr, sev
