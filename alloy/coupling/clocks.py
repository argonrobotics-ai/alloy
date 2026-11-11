# SPDX-License-Identifier: Apache-2.0
"""Exchange clock and per-entry substep rungs on the D4 grid (SPEC §3.6, D4, D21; coupling-time §4.4.3).

Pure host logic, float64. Every exchange interval is ``control_dt / 2**j`` with ``0 <= j <= j_max``; every entry
substep count is a rung of ``ClockPolicy.substep_ladder``.

- interface bound: ``dt_bound = safety · omega_dt_max · sqrt(m_eff / k_c)`` (explicit-force hold limit ω·dt; the
  proxy blew up near ω·dt ≈ 2.1-2.3 on box-on-cloth, coupling-time §3.2), and for contact damping c
  ``safety · omega_dt_max · m_eff / c`` when ``ClockPolicy.damping_bound`` (opt-in; the held damping force c·v over
  dt: rate c/m_eff; a 1.2 g piston on a VBD body at Newton's default kd gained energy on every contact until
  dt_x ≤ m/c, WP4 P3 law, 2026-11-06);
- ``dt_x`` = largest grid value ≤ min(dt_bound, dt_x_max), never coarser than an escalation floor;
- ``n_e`` = smallest rung with ``dt_x / n_e ≤ safety · stable_dt_e`` (lockstep: ``stable_dt_e`` is the min over
  worlds);
- raise (finer) immediately; lower (coarser) one grid step or one rung after ``downshift_quiet_periods``
  consecutive periods in which the coarser value would still satisfy the bound. Escalations are sticky.
- graph-budget caps (opt-in ``ClockPolicy.graph_budget_rungs``): an entry capped at rung ``c`` (``cap_rung``, after an
  exchange's graph went over the node budget) also bounds ``dt_x <= c · safety · stable_dt_e``, so its rung never
  passes ``c``; a new cap refines dt_x at once, the rungs following (the gate-jump rule), with each entry's own step
  unchanged. Caps are sticky (the budget is).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from alloy.core.timebase import grid_dt, grid_index, snap_to_grid

if TYPE_CHECKING:
    from alloy.coupling.types import ClockPolicy

log = logging.getLogger("alloy.coupling")


def interface_bound(k_c: float, m_eff: float, safety: float, omega_dt_max: float) -> float:
    """Largest exchange interval [s] that keeps ω·dt ≤ safety·omega_dt_max, ω = sqrt(k_c/m_eff); inf if unknown."""
    if not (k_c > 0.0 and m_eff > 0.0) or not math.isfinite(k_c) or not math.isfinite(omega_dt_max):
        return math.inf
    return safety * omega_dt_max * math.sqrt(m_eff / k_c)


def damping_bound(c_d: float, m_eff: float, safety: float, omega_dt_max: float) -> float:
    """Largest exchange interval [s] that keeps the held contact damping's rate c_d/m_eff times dt ≤
    safety·omega_dt_max; inf if unknown or undamped."""
    if not (c_d > 0.0 and m_eff > 0.0) or not math.isfinite(c_d) or not math.isfinite(omega_dt_max):
        return math.inf
    return safety * omega_dt_max * m_eff / c_d


def grid_level(cap: float, control_dt: float, j_max: int) -> tuple[int, bool]:
    """Grid index of the largest ``control_dt/2**j <= cap`` (j <= j_max) and whether ``cap`` was finer than the
    finest allowed value (then j = j_max)."""
    finest = grid_dt(control_dt, j_max)
    if not cap >= finest:
        return j_max, True
    snapped = snap_to_grid(min(cap, control_dt), control_dt)
    return grid_index(snapped, control_dt), False


def entry_safety(estimator: object, clock: "ClockPolicy", partners: Collection[str | None] = ()) -> float:
    """The safety ``pick_rung`` applies to an entry's estimate: ``clock.safety``, or with ``clock.own_cfl`` the
    estimator's own ``alloy_clock_safety`` when it declares one (an estimate that already carries its margin, such
    as SPH's CFL step), unless the entry is paired with a capability in the estimator's
    ``alloy_clock_safety_off_beside`` (``partners``: the capabilities of the entries it is paired with)."""
    own = getattr(estimator, "alloy_clock_safety", None) if clock.own_cfl else None
    if own is not None and set(partners) & set(getattr(estimator, "alloy_clock_safety_off_beside", ())):
        own = None
    return float(clock.safety) if own is None else float(own)


def entry_safeties(estimators: Mapping[str, object], entries: Mapping[str, Any], pairs: Sequence[Any],
                   clock: "ClockPolicy") -> dict[str, float]:
    """``entry_safety`` of every entry with an estimator, given the capabilities of the entries it is paired with
    (``entries``: name -> AlloyEntry; ``pairs``: PairPolicy). Example: ``entry_safeties(est, entries, pairs, pol)``."""
    def partners(n: str) -> set[str | None]:
        return {entries[p.destination if p.source == n else p.source].capability for p in pairs
                if n in (p.source, p.destination) and {p.source, p.destination} <= set(entries)}

    return {n: entry_safety(e, clock, partners(n)) for n, e in estimators.items() if e is not None}


def pick_rung(dt_x: float, stable_dt: float, safety: float, ladder: Sequence[int]) -> int:
    """Smallest ladder rung n with dt_x/n <= safety·stable_dt; the largest rung if none qualifies."""
    if math.isnan(stable_dt) or stable_dt == math.inf:
        return ladder[0]
    if stable_dt <= 0.0:
        return ladder[-1]
    limit = safety * stable_dt
    for n in ladder:
        if dt_x / n <= limit:
            return n
    return ladder[-1]


@dataclass
class _Hysteresis:
    """One integer level with immediate raises and delayed one-step lowering."""

    value: int
    quiet: int = 0

    def update(self, desired: int, lower_to: int, quiet_needed: int) -> bool:
        """Return True if the level changed. ``lower_to`` is the next level down (used only when lowering)."""
        if desired > self.value:
            self.value, self.quiet = desired, 0
            return True
        if desired < self.value:
            self.quiet += 1
            if self.quiet >= quiet_needed:
                self.value, self.quiet = lower_to, 0
                return True
            return False
        self.quiet = 0
        return False

    def jump(self, desired: int) -> bool:
        """Go straight to ``desired`` (no delay either way); True if the level changed."""
        changed, self.value, self.quiet = desired != self.value, desired, 0
        return changed


@dataclass(frozen=True)
class ClockDecision:
    """Result of one period's clock choice."""

    dt_x: float
    j: int
    substeps: dict[str, int]
    rung_changes: int
    bound_unmet: bool = False       # the interface bound is finer than control_dt / 2**j_max
    escalation_exhausted: bool = False


@dataclass
class ClockController:
    """Holds the exchange level j and per-entry rungs across periods (host state, one instance per Scheduler)."""

    policy: "ClockPolicy"
    initial_rungs: Mapping[str, int]
    adaptive: Mapping[str, bool]
    j: int | None = None
    j_floor: int = 0                                  # sticky escalation floor (finer than this never relaxes)
    safety_for: Mapping[str, float] = field(default_factory=dict)   # per-entry safety (entry_safety); else policy's
    rung_changes_total: int = 0
    rung_cap: dict[str, int] = field(default_factory=dict)          # graph-budget caps (cap_rung), for the run
    _rungs: dict[str, _Hysteresis] = field(default_factory=dict)
    _level: _Hysteresis | None = None
    _cap_jump: bool = False                                          # a new cap: the next decide refines at once
    _stable_last: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ladder = tuple(self.policy.substep_ladder)
        if not ladder or list(ladder) != sorted(set(ladder)) or ladder[0] < 1:
            raise ValueError(f"substep_ladder must be strictly increasing positive ints, got {ladder}")
        for name, n in self.initial_rungs.items():
            if n not in ladder:
                raise ValueError(f"initial rung {n} of entry {name!r} is not on the ladder {ladder}")
            self._rungs[name] = _Hysteresis(n)

    @property
    def rungs(self) -> dict[str, int]:
        return {name: h.value for name, h in self._rungs.items()}

    def force_rung(self, name: str, n: int) -> None:
        """Set an entry's rung directly (manual override; hysteresis restarts)."""
        if n not in self.policy.substep_ladder:
            raise ValueError(f"rung {n} for entry {name!r} is not on the ladder {self.policy.substep_ladder}")
        self._rungs[name] = _Hysteresis(n)

    def escalate(self) -> bool:
        """Halve dt_x for every later period (sticky). Returns False if already at j_max."""
        current = self.j if self.j is not None else 0
        if current >= self.policy.j_max:
            return False
        self.j_floor = current + 1
        log.info("coupling clock escalation: dt_x level j %d -> %d", current, self.j_floor)
        return True

    def cap_rung(self, substeps: Mapping[str, int]) -> str | None:
        """An exchange at ``substeps`` was over the graph node budget: cap the adaptive entry with the most substeps
        (and a stable-dt estimate) one ladder rung lower for the rest of the run, so later periods take a finer dt_x
        with fewer substeps (the same entry step) instead of an eager exchange. Returns that entry; None when no entry
        can go lower."""
        ladder = tuple(self.policy.substep_ladder)
        known = {n: int(r) for n, r in substeps.items() if n in self._rungs and self.adaptive.get(n, True)
                 and math.isfinite(self._stable_last.get(n, math.nan)) and int(r) in ladder[1:]}
        if not known:
            return None
        name = max(known, key=lambda n: (known[n], n))
        cap = ladder[ladder.index(known[name]) - 1]
        self.rung_cap[name] = min(cap, self.rung_cap.get(name, cap))
        self._cap_jump = True
        log.info("coupling clock: an exchange at substeps %s is over the graph node budget; entry %s capped at %d "
                 "substeps (finer dt_x instead)", dict(substeps), name, self.rung_cap[name])
        return name

    def _clamp_caps(self, dt_x: float, stable_dt: Mapping[str, float]) -> int:
        """A capped rung held above its cap by the rung hysteresis (no jump: dt_x already met the cap bound) drops to the
        cap at once; safe, since there the estimator's own rung is within the cap (WP4's review). Returns changes."""
        changes = 0
        for name, cap in self.rung_cap.items():
            h, est = self._rungs.get(name), stable_dt.get(name, math.nan)
            if h is None or h.value <= cap or not (math.isfinite(est) and est > 0.0):
                continue
            if dt_x <= cap * self.safety_for.get(name, self.policy.safety) * est:
                changes += h.jump(cap)
        return changes

    def _cap_bound(self, stable_dt: Mapping[str, float]) -> float:
        """Largest dt_x [s] at which each capped entry's rung stays within its cap: cap x safety x stable dt."""
        out = math.inf
        for name, cap in self.rung_cap.items():
            est = stable_dt.get(name, math.nan)
            if math.isfinite(est) and est > 0.0:
                out = min(out, cap * self.safety_for.get(name, self.policy.safety) * est)
        return out

    def decide(self, control_dt: float, bound_dt: float, stable_dt: Mapping[str, float], *,
               jump: bool = False) -> ClockDecision:
        """Choose dt_x and rungs for the next period from the interface bound and per-entry stable dt [s].
        ``jump``: lever #2's gate changed which pairs are armed (its hold periods are the hysteresis), so dt_x goes
        straight to what the new bound needs instead of one grid step per quiet stretch, and when that makes dt_x finer
        the rungs follow at once (``_update_rungs``). Without it a re-armed pair ran 1/16 of the released dt_x with the
        released rungs (VBD 8 substeps where it needs 1): every exchange in contact cost several times today's, and
        its graph keys were new (eager). With ``ClockPolicy.rungs_follow_dt_x`` the rungs follow any dt_x that a bound
        (interface, contact band, graph-budget cap) makes finer; in a decide where the sticky escalation floor rose,
        alone or with a bound, the rungs hold as today, so a watchdog trip never gets less of its remedy (WP4)."""
        pol = self.policy
        j_need, bound_unmet = grid_level(min(bound_dt, pol.dt_x_max), control_dt, pol.j_max)
        self._stable_last = dict(stable_dt)
        cap_dt = self._cap_bound(stable_dt)
        j_cap = grid_level(min(cap_dt, pol.dt_x_max), control_dt, pol.j_max)[0] if cap_dt < math.inf else 0
        desired = max(j_need, j_cap, self.j_floor)
        jump = jump or (self._cap_jump and self._level is not None and desired > self._level.value)
        self._cap_jump = False
        changes = 0
        level_before = self._level.value if self._level is not None else None
        before = grid_dt(control_dt, level_before) if level_before is not None else None
        if self._level is None:
            self._level = _Hysteresis(desired)
        elif jump:
            changes += self._level.jump(desired)
        elif self._level.update(desired, max(self._level.value - 1, self.j_floor), pol.downshift_quiet_periods):
            changes += 1
        self.j = self._level.value
        dt_x = grid_dt(control_dt, self.j)
        # a bound made dt_x finer, not the sticky floor rising in this decide (a watchdog trip keeps all its remedy)
        follow = pol.rungs_follow_dt_x and level_before is not None and self.j_floor <= level_before
        refined = (jump or follow) and before is not None and dt_x < before
        changes += self._update_rungs(dt_x, stable_dt, before if refined else None)
        changes += self._clamp_caps(dt_x, stable_dt)
        self.rung_changes_total += changes
        return ClockDecision(dt_x=dt_x, j=self.j, substeps=self.rungs, rung_changes=changes,
                             bound_unmet=bound_unmet, escalation_exhausted=self.j_floor >= pol.j_max)

    def _update_rungs(self, dt_x: float, stable_dt: Mapping[str, float], refined_from: float | None = None) -> int:
        """Each adaptive rung from its entry's stable-dt estimate: raises at once, lowers one ladder step per quiet
        stretch. ``refined_from``: the dt_x before a gate jump made it finer. Then each rung goes straight to the larger
        of what its estimator wants and the rung that keeps its own step (dt_x / rung) no coarser than before, so the
        released rungs do not carry into contact, and no entry's own step coarsens through a gate change (an entry
        whose rung rose on its own estimate keeps its step: WP4's review)."""
        pol = self.policy
        ladder = tuple(pol.substep_ladder)
        changes = 0
        for name, h in self._rungs.items():
            estimate = stable_dt.get(name, math.nan)
            if not self.adaptive.get(name, True) or math.isnan(estimate):
                continue                      # fixed entries and entries without an estimator hold their rung
            desired = pick_rung(dt_x, estimate, self.safety_for.get(name, pol.safety), ladder)
            idx = ladder.index(h.value)
            lower = ladder[max(idx - 1, 0)]
            if refined_from is not None:
                own = refined_from / h.value                      # the entry's own step before the jump
                keep = next((n for n in ladder if dt_x / n <= own * (1.0 + 1.0e-9)), ladder[-1])
                changed = h.jump(max(desired, keep))
            else:
                changed = h.update(desired, lower, pol.downshift_quiet_periods)
            if changed:
                log.debug("entry %s substeps -> %d (dt_x %.6g s)", name, h.value, dt_x)
                changes += 1
        return changes
