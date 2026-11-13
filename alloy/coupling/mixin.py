# SPDX-License-Identifier: Apache-2.0
"""SchedulerMixin: clock, watchdog, estimators, report and snapshots shared by Scheduler and SchedulerADMM.

Placed before the Newton coupler in the MRO (``class Scheduler(SchedulerMixin, SolverCoupledProxy)``), it wraps
``step`` (one exchange, SPEC D18: no period loop here) and provides ``choose_clock``, ``report``, ``state_arrays``,
``on_rollback`` and ``frozen_contacts``. The coupler-specific iteration lives in ``proxy_loop`` / ``admm_loop``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from alloy.core.errors import CouplingError
from alloy.core.registry import estimator_for, state_arrays_for, state_world_maps_for
from alloy.core.timebase import grid_dt
from alloy.coupling import capture_passes as cpm
from alloy.coupling.capture_passes import short_worlds
from alloy.coupling.clocks import ClockController, damping_bound, entry_safeties, interface_bound
from alloy.coupling.contact_band import BandRerun, ContactBandBound, read_band
from alloy.coupling.entry_history import EntryHistory
from alloy.coupling.ladder import PeriodLadder, escalate
from alloy.coupling.onset import OnsetRefiner, escalated_beyond_onset
from alloy.coupling.parked import stable_host
from alloy.coupling.static_walls import narrow_static_walls, static_wall_shapes
from alloy.coupling.watchdog import EnergyWatchdog, WatchConfig

if TYPE_CHECKING:
    from alloy.coupling.types import AlloyEntry, CouplingReport, PairPolicy, SchedulerPolicy

log = logging.getLogger("alloy.coupling")

NON_PRECISION_TOL_SCALE = 10.0   # InterfaceTolerance holds the precision values; x10 elsewhere (SPEC §4.2)
RES_ESCALATE_FACTOR = 10.0       # escalate when the final-pass source-proxy gap exceeds 10x the tolerance


def _host(arr: wp.array) -> np.ndarray:
    """Owned host copy (on CPU ``numpy()`` is a live view of device memory)."""
    return np.array(arr.numpy(), copy=True)


__all__ = ["SchedulerMixin", "escalated_beyond_onset"]


class SchedulerMixin:
    """Overrides _step_coupled (residual-gated, per-world freeze, capture_while on CUDA, host loop on CPU) and
    _blend_proxy_feedback (lagged omega=1; staggered per-world Aitken from (W,2) float64 stats via ordered per-world
    sums, never float atomic_add, D9)."""

    # ------------------------------------------------------------------ construction
    def _init_scheduler(self, entries: Sequence["AlloyEntry"], pairs: Sequence["PairPolicy"],
                        policy: "SchedulerPolicy") -> None:
        """Call after the Newton coupler's __init__ (entries, views and feedback buffers exist)."""
        self._alloy_policy = policy
        self._alloy_entries = {ae.entry.name: ae for ae in entries}
        self._alloy_pairs = tuple(pairs)
        self._check_runtime_substeps()
        initial = {name: int(ae.initial_rung) for name, ae in self._alloy_entries.items()}
        adaptive = {name: bool(ae.adaptive) for name, ae in self._alloy_entries.items()}
        self._apply_substeps(initial)
        self._clock = ClockController(policy.clock, initial, adaptive)
        wd = policy.watchdog
        self._watchdog = EnergyWatchdog(self.model, WatchConfig(
            wd.window, wd.warn_gain_per_min, wd.escalate_gain_per_window, wd.abort_speed,
            res_pos_escalate=RES_ESCALATE_FACTOR * self._position_tolerance()))
        self._entry_order = tuple(self._alloy_entries)
        self._estimators = {n: estimator_for(self._entries[n].solver) for n in self._entry_order}
        self._clock.safety_for = entry_safeties(self._estimators, self._alloy_entries, self._alloy_pairs, policy.clock)
        shape = (len(self._entry_order), int(self.model.world_count))
        self._stable_dt = wp.full(shape, math.inf, dtype=float, device=self.model.device)
        self._stable_host: dict[str, float] = {}
        self._dt_x: float | None = None
        self._substeps_now = dict(initial)
        self._rollbacks = self._rollbacks_period = 0
        self._rerun = False
        self._short_rerun = False          # lever #1: this re-run only completes worlds a capped capture left short
        self._gate_rerun = False           # lever #2: this re-run arms proxies that touched unarmed
        self._period_state = self._last_out = None      # lever #2: the state the next period starts from
        self._steps_seen, self._steps_captured, self._steps_expected = 0, False, 0
        self._attempt_limit = math.inf     # the last attempt's force-jump limit (a short re-run replays it)
        self._attempt_watch: dict[str, wp.array] | None = None     # watchdog arrays at the attempt's start
        self._onset = OnsetRefiner(wd)     # SCR-WP11-4: force-jump re-runs at a finer dt_x
        self._ladder = PeriodLadder(wd.per_period_ladder, wd.window)   # SCR-WP3-C3RAND: a first trip, its period only
        self._last_report: "CouplingReport | None" = None
        self._rung_changes = 0
        self._strong: set[tuple[str, str]] = set()
        self._frozen_contacts = None
        self._band = ContactBandBound(self._contact_routes(), self.model.device, shape[1])   # R2-E3: the band's dt_x
        self._band_bound, self._control_dt = math.inf, math.nan
        self._band_rerun = BandRerun(policy.clock.band_rerun)      # R2-E3 follow-up: a period the bound outran
        log.info("scheduler %s: entries %s, pairs %s, initial rungs %s", type(self).__name__,
                 list(self._entry_order), [(p.source, p.destination, p.mode) for p in self._alloy_pairs], initial)

    def _position_tolerance(self) -> float:
        """Interface position tolerance [m]: InterfaceTolerance.pos for precision pairs, x10 elsewhere."""
        precise = any(p.precision for p in self._alloy_pairs)
        return self._alloy_policy.tol.pos * (1.0 if precise else NON_PRECISION_TOL_SCALE)

    def _check_runtime_substeps(self) -> None:
        """D21: an adaptive entry needs the substep scratch state, which Newton allocates only at substeps > 1."""
        ladder = tuple(self._alloy_policy.clock.substep_ladder)
        for name, ae in self._alloy_entries.items():
            entry = self._entries[name]
            if ae.initial_rung not in ladder:
                raise CouplingError(f"entry {name!r}: initial_rung {ae.initial_rung} is not on the ladder {ladder}")
            if ae.adaptive and not entry.in_place and entry.state_tmp is None and max(ladder) > 1:
                raise CouplingError(
                    f"entry {name!r} is adaptive but was built with substeps={ae.entry.substeps}: construct it at "
                    f"the max rung ({max(ladder)}) so Newton allocates the substep scratch state (SPEC D21)")
            if not ae.adaptive and ae.initial_rung > ae.entry.substeps and entry.state_tmp is None:
                raise CouplingError(f"entry {name!r}: initial_rung {ae.initial_rung} needs substep scratch state")

    def _apply_substeps(self, substeps: Mapping[str, int]) -> None:
        """Set the private SolverEntry.substeps from the host mirror (D21, pinned hook ("SolverCoupled", "_entries"))."""
        for name, n in substeps.items():
            self._entries[name].substeps = int(n)
        self._substeps_now = dict(substeps)

    def set_substeps(self, substeps: Mapping[str, int]) -> None:
        """Manual rung override between exchanges (takes effect at the next ``step``; D21)."""
        for name, n in substeps.items():
            if name not in self._alloy_entries:
                raise CouplingError(f"unknown entry {name!r}; entries are {list(self._alloy_entries)}")
            entry = self._entries[name]
            if n > 1 and entry.state_tmp is None and not entry.in_place:
                raise CouplingError(f"entry {name!r} has no substep scratch state; build it at the max rung (D21)")
            self._clock.force_rung(name, int(n))
        self._apply_substeps({**self._substeps_now, **{k: int(v) for k, v in substeps.items()}})

    # ------------------------------------------------------------------ entry views (Newton hooks, SCR-JOB2-5)
    def _entry_visible_shapes(self, cfg: Any, visible_bodies: set[int]) -> set[int]:
        """Newton's visible shapes, plus every static wall (``alloy.coupling.static_walls``) for a particle entry. Runs
        inside the Newton coupler's __init__ (collision mask, compaction, shape metadata). Private Newton hook, listed in
        ``alloy.core.newton_compat.PRIVATE_HOOKS`` (signature pinned by tests/core/test_newton_contract.py)."""
        shapes = super()._entry_visible_shapes(cfg, visible_bodies)
        if not cfg.particles:
            return shapes
        walls = self.__dict__.get("_static_walls")
        if walls is None:
            walls = self.__dict__["_static_walls"] = static_wall_shapes(self.model)
        added = {int(w) for w in walls} - shapes
        if added:
            self.__dict__.setdefault("_static_added", {}).setdefault(cfg.name, set()).update(added)
        return shapes | added

    def _customize_compact_view(self, view: Any) -> None:
        """The static walls a particle entry sees keep only their particle-collision bit (SCR-JOB2-5). Private Newton
        hook (as above); runs after the view's compaction or its full-layout fallback."""
        super()._customize_compact_view(view)
        added = self.__dict__.get("_static_added", {}).get(view.name)
        if added:
            narrow_static_walls(view, sorted(added))
            log.info("entry %r: %d static wall shape(s) visible to its particles (SCR-JOB2-5)", view.name, len(added))

    # ------------------------------------------------------------------ one exchange
    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float, *,
             observables: Any = None) -> None:
        """One exchange of length dt (Newton SolverBase contract)."""
        if not self._watchdog.has_reference:
            self.reset_energy_reference(state_in)
        self._note_states(state_in, state_out)
        self._watchdog.begin_exchange()
        super().step(state_in, state_out, control, contacts, dt, observables=observables)
        self._entry_history().commit(self._entries)       # MPM history state_1 -> state_0 (entry_history)
        self._watchdog.after_exchange(state_in, state_out, control, dt, self._loop_arrays()["res_pos"])
        self._run_estimators(contacts)
        self._band.accumulate(self._entries, state_in, self.model)       # R2-E3: bodies at the exchange start
        for factory in self._contact_factories():
            factory.update_overflow()

    def _entry_history(self) -> EntryHistory:
        """Created on first use: the entries exist only after Newton's constructor."""
        hist = self.__dict__.get("_history")
        if hist is None:
            hist = self.__dict__["_history"] = EntryHistory()
        return hist

    def _contact_factories(self) -> list[Any]:
        """Proxy collision pipelines built by ``ProxyContactPipeline`` (SCR-DC5-19); none for ADMM."""
        return []

    def _contact_routes(self) -> list[tuple[Any, str]]:
        """(proxy pipeline, destination entry) per pair with one (``contact_band``); none for ADMM."""
        return []

    def capacity_flags(self) -> list[Any]:
        """For HealthMonitor(capacity_flags=...): int32[1] device counters of dropped or missed proxy contacts
        (nonzero raises CONTACT_OVERFLOW, INVALID, on every world: the proxy buffer is global, SC0-1), and per
        pipeline the per-world pass-through flags with PAIR_PEN_FAIL (SCR-SC4-4, ``pass_through``)."""
        flags: list[Any] = [f.overflow for f in self._contact_factories() if f.overflow is not None]
        return flags + [f.pass_through.health_source() for f in self._contact_factories()   # SCR-SC4-4: per world
                        if f.pass_through is not None]

    def _run_estimators(self, contacts: Any) -> None:
        for e, name in enumerate(self._entry_order):
            est = self._estimators.get(name)
            if est is not None:
                est.alloy_stable_dt(self._entries[name].state_1, contacts, self._stable_dt[e])

    def reset_energy_reference(self, state: Any) -> None:
        """Re-base the D20 datum, E₀ and E_scale (call at reset, before graph capture)."""
        self._watchdog.reset_reference(state)
        self._period_state, self._last_out = state, None

    def _note_states(self, state_in: Any, state_out: Any) -> None:
        """Python sees an eager exchange, or one exchange of a capture (lever #2's gate reads the period start)."""
        if self._steps_seen == 0:
            self._period_state = state_in
        self._last_out = state_out
        self._steps_seen += 1
        self._steps_captured = self._steps_captured or cpm.capturing(self.model.device)

    def clock_mode(self) -> dict[str, Any] | None:
        """None under today's clock; with lever #2's contact gate on, what it gates (``ContactGate.describe``)."""
        gate = getattr(self, "_gate", None)
        return None if gate is None else gate.describe()

    def _gate_state(self) -> Any:
        """The state this period starts from: the last exchange's output after an eager period (any exchange count),
        else the object the last captured or partial period started from (graph periods keep the State objects)."""
        eager = not self._steps_captured and self._steps_seen >= self._steps_expected > 0
        return self._last_out if eager and self._last_out is not None else self._period_state

    def reset(self, state: Any, world_mask: Any = None, flags: Any = None) -> None:
        super().reset(state, world_mask=world_mask, flags=flags)
        self._watchdog.reset_reference(state)
        if world_mask is None:
            for factory in self._contact_factories():
                factory.reset_overflow()
        self.reset_world_flags(world_mask)

    def reset_world_flags(self, world_mask: Any = None) -> None:
        """Clear the per-world health flags this stepper raises (pass-through, SCR-SC4-4; subclasses add theirs)
        for the masked worlds (all when None), without touching the coupling state: call it when a world is
        reset or restored."""
        for factory in self._contact_factories():
            if factory.pass_through is not None:
                factory.pass_through.reset(world_mask)

    # ------------------------------------------------------------------ clock (host, once per period)
    def choose_clock(self, control_dt: float, last: "CouplingReport | None") -> tuple[float, dict[str, int]]:
        """Pick dt_x and per-entry substeps for the next control period; sets the private substeps (D21)."""
        refine, rerun, short_rerun, gate_rerun = self._onset.pending, self._rerun, self._short_rerun, self._gate_rerun
        self._rerun = self._short_rerun = self._gate_rerun = False
        band_rerun, self._control_dt = self._band_rerun.begin_attempt(), control_dt     # R2-E3 follow-up
        cp, gate = self._captured_passes(), getattr(self, "_gate", None)
        if cp is not None:
            cp[0].begin_attempt(rerun, cp[1])
        if gate is not None:
            gate.begin_attempt()
        if short_rerun and self._dt_x is not None:
            return self._replay_attempt()
        jump = gate_rerun                                  # lever #2: the re-run armed more pairs
        if not rerun:
            self._rollbacks_period = 0
            self._band_rerun.new_period()
            self._ladder.new_period()
            self._onset.new_period()
            self._watchdog.begin_period(control_dt)
            jump = self._gate_evaluate(gate, control_dt)
        escalated = escalated_beyond_onset(last)
        if not refine and not gate_rerun and not band_rerun and escalated.any():     # re-runs: their report decides
            if not self._ladder.within_period(escalated, rerun, control_dt):        # ladder.py: else this period only
                escalate(self, escalated)
            self._watchdog.rebase_window()
        decision = self._clock.decide(control_dt, self._interface_bound(), self._stable_host, jump=jump)
        j_max = self._alloy_policy.clock.j_max
        j = self._ladder.j(decision.j, j_max)
        if decision.bound_unmet:
            log.warning("interface bound %.3g s is finer than control_dt/2^j_max; running at %.3g s",
                        self._interface_bound(), decision.dt_x)
        self._apply_substeps(decision.substeps)
        self._dt_x, self._attempt_limit = self._onset.attempt(control_dt, j, decision.dt_x if j == decision.j
                                                              else grid_dt(control_dt, j), j_max)
        self._watchdog.begin_attempt(self._attempt_limit, self._onset.armed(self._watchdog.world_count))
        if (cp is not None and cp[0].active) or gate is not None or self._band_rerun.on:   # what a re-run starts from
            self._onset_last = self._onset.last.copy()
            self._save_attempt_watch()
        self._rung_changes += decision.rung_changes
        self._steps_seen, self._steps_captured = 0, False
        self._steps_expected = max(int(round(control_dt / self._dt_x)), 1)
        return self._dt_x, dict(decision.substeps)

    def _gate_evaluate(self, gate: Any, control_dt: float) -> bool:
        """Lever #2: arm the pairs near contact at the period start; True when the armed set changed."""
        state = self._gate_state()
        if gate is None or state is None:
            return False
        return gate.evaluate(state, control_dt)

    def _save_attempt_watch(self) -> None:
        """The watchdog's per-world arrays as this attempt starts (not the period start's): a re-run's (lever #1/#2)."""
        arrays = self._watchdog.state_arrays()
        if self._attempt_watch is None:
            self._attempt_watch = {k: wp.empty_like(v) for k, v in arrays.items()}
        for k, v in arrays.items():
            wp.copy(self._attempt_watch[k], v)

    def _restore_attempt(self, sel: np.ndarray) -> None:
        """Short re-run (host): the rolled-back ``sel`` worlds' watchdog arrays and force-jump cooldowns as their
        attempt began (all worlds when Simulation rolled every world back: WP4's review)."""
        before = getattr(self, "_onset_last", None)
        if before is not None and before.shape == sel.shape == self._onset.last.shape:
            self._onset.last[sel] = before[sel]
        for k, v in self._watchdog.state_arrays().items():
            saved = (self._attempt_watch or {}).get(k)
            if saved is None or v.shape[:1] != sel.shape:
                continue
            cur = v.numpy().copy()
            cur[sel] = saved.numpy()[sel]
            v.assign(cur)

    def replay_period(self) -> None:
        """The driver re-runs the period it just ran, unchanged, every world back at its start (WP12's MPM grid growth,
        grid_growth.py): the same dt_x, substeps and force-jump trigger; nothing is counted or escalated."""
        self._restore_attempt(np.ones(self._watchdog.world_count, dtype=bool))
        self._band.vmax.zero_()                               # the discarded attempt's contact speeds
        self._rerun = self._short_rerun = True

    def _replay_attempt(self) -> tuple[float, dict[str, int]]:
        """A short re-run (lever #1) or ``replay_period`` replays the last attempt: same dt_x, substeps and trigger."""
        self._watchdog.begin_attempt(self._attempt_limit, self._onset.armed(self._watchdog.world_count))
        self._steps_seen, self._steps_captured = 0, False
        return float(self._dt_x), dict(self._substeps_now)

    def _escalate_mode(self) -> bool:
        """Coupler-specific second rung (proxy: staggered + Aitken ω0_strong; ADMM: ρ×4). False if exhausted."""
        return False

    def _interface_bound(self) -> float:
        """min over pairs of safety·ω_max·sqrt(m_eff/k_c) and safety·ω_max·m_eff/c_d [s], and the contact-band bound
        from the last period (``contact_band``, R2-E3)."""
        clock = self._alloy_policy.clock
        bound = math.inf
        for k_c, m_eff in self._pair_stiffness_mass():
            bound = min(bound, interface_bound(k_c, m_eff, clock.safety, clock.omega_dt_max))
        for c_d, m_eff in (self._pair_damping_mass() if clock.damping_bound else ()):
            bound = min(bound, damping_bound(c_d, m_eff, clock.safety, clock.omega_dt_max))
        return min(bound, self._band_bound)                  # R2-E3: contacts cross <= band_travel of their band

    def _pair_stiffness_mass(self) -> list[tuple[float, float]]:
        return []

    def _pair_damping_mass(self) -> list[tuple[float, float]]:
        return []

    # ------------------------------------------------------------------ rollback (Simulation owns it, D18)
    def rollback_requested(self, report: "CouplingReport") -> bool:
        """True if a world escalated and this period still has rollbacks left (≤ max_rollbacks_per_period), or every
        escalation was a force jump that a finer re-run can still shrink (SCR-WP11-4, no budget used)."""
        self._last_report = report                            # the driver's view (merged per world) for on_rollback
        if self._band_rerun.wanted:                           # R2-E3 follow-up: the band bound outran this attempt
            return True
        if getattr(self, "_gate", None) is not None and (report.notes or {}).get("gate_rerun"):
            return True                                      # lever #2: a proxy touched unarmed: the whole period
        cp = self._captured_passes()
        if cp is not None and cp[0].wants_rerun(report):     # lever #1: worlds a capped capture left short go first
            return True
        if not escalated_beyond_onset(report).any():         # none, or only force jumps (SCR-WP11-4)
            return self._onset.wants_rerun(report)
        return self._escalation_budget_left()

    def _escalation_budget_left(self) -> bool:
        return self._rollbacks_period < self._alloy_policy.watchdog.max_rollbacks_per_period

    def _captured_passes(self) -> tuple[Any, int, int] | None:
        """(CapturedPasses, k_min, k_max) of this coupler's iteration loop (lever #1); None without one."""
        loop = getattr(self, "iteration_loop", None)
        cp = getattr(loop, "captured_passes", None)
        return None if cp is None else (cp, int(loop.cfg.k_min), int(loop.cfg.k_max))

    def allow_capped_captures(self) -> None:
        """The driver re-runs worlds a capped capture leaves short, and whole periods (``IntegratedSimulation``):
        captures may run only the passes that converge worlds (lever #1; else they unroll k_max), band re-runs too."""
        self._band_rerun.enabled = True                      # R2-E3 follow-up (contact_band.BandRerun)
        cp = self._captured_passes()
        if cp is not None:
            cp[0].driver = True

    def graph_key(self) -> tuple:
        """What a capture of one exchange bakes in beyond (dt_x, substeps): the captured coupling passes, and the sticky
        ladder's step (its staggered / ω0_strong inputs; that switch can come in a period with no rollback, ladder.py)."""
        cp, trips = self._captured_passes(), getattr(self, "_trips", 0)
        return (() if cp is None else cp[0].graph_key(cp[1], cp[2])) + (("ladder", trips) if trips else ())

    def on_rollback(self, world_mask: Any = None) -> None:
        """Called by Simulation after it restored the period snapshot; the next choose_clock re-plans the rerun.
        Worlds a capped capture left short (lever #1) are handled first, alone: Simulation rolls back only them, they
        take back their watchdog arrays and cooldowns as the attempt began, and the re-run replays the attempt with the
        full loop. Their escalations and force jumps in the capped attempt are discarded; the replay's own report
        decides those, as the uncapped run's attempt would. No rollback budget is used."""
        cp, report, gate = self._captured_passes(), self._last_report, getattr(self, "_gate", None)
        if self._band_rerun.wanted and report is not None:    # R2-E3 follow-up: every world, at the measured bound
            self._band_rerun.accept(self._band.last_speed, self._dt_x, self._band_bound)
            self._restore_attempt(np.ones(self._watchdog.world_count, dtype=bool))
            self._rerun = True
            self._gate_rerun = bool(gate is not None and (report.notes or {}).get("gate_rerun") and gate.accept())
            return
        if gate is not None and report is not None and (report.notes or {}).get("gate_rerun") and gate.accept():
            self._restore_attempt(np.ones(self._watchdog.world_count, dtype=bool))   # every world went back
            self._rerun = self._gate_rerun = True
            return
        if cp is not None and report is not None and cp[0].accept(report, cp[1], cp[2]):
            short = short_worlds(report)               # world_mask None: every world went back (no per-world axis)
            self._restore_attempt(np.ones_like(short) if world_mask is None else np.asarray(world_mask).astype(bool))
            self._rerun = self._short_rerun = True
            return
        if self._last_report is not None and self._onset.accept(self._last_report):
            self._rerun = True
            self._watchdog.mark_rolled_back(None if world_mask is None else np.asarray(world_mask))
            return
        self._rollbacks += 1
        self._rollbacks_period += 1
        self._rerun = True
        self._watchdog.mark_rolled_back(None if world_mask is None else np.asarray(world_mask))
        log.info("coupling rollback %d (this period %d)", self._rollbacks, self._rollbacks_period)

    # ------------------------------------------------------------------ outputs
    def _loop_arrays(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def report(self) -> "CouplingReport":
        """Host copy, read once per period."""
        from alloy.coupling.types import CouplingReport

        loop = self._loop_arrays()
        wd = self._watchdog
        stable = self._stable_dt.numpy()
        self._stable_host = stable_host(self, stable)          # min over worlds, without parked ones (parked.py)
        self._band_bound = read_band(self)          # R2-E3: the next dt_x and the re-run verdict, once per attempt
        self._last_report = report = CouplingReport(
            dt_x=float(self._dt_x) if self._dt_x is not None else math.nan,
            substeps=dict(self._substeps_now),
            res_vel=_host(loop["res_vel"]), res_pos=_host(loop["res_pos"]), res_force=_host(loop["res_force"]),
            iters_used=_host(loop["iters_used"]), watch_state=_host(wd.watch_state),
            interface_work=_host(wd.interface_work), energy=_host(wd.energy),
            rollbacks=self._rollbacks, rung_changes=self._rung_changes,
            strong_pairs=tuple(sorted(self._strong)),
            notes={"stable_dt": {n: stable[e].copy() for e, n in enumerate(self._entry_order)   # [W] s, SCR-JOB2-1
                                 if self._estimators.get(n) is not None},
                   **self._onset.notes(wd.onset_only(), _host(wd.exchange_peak)),               # SCR-WP11-4
                   **self._capture_notes(), **self._ladder.notes(),                              # levers, ladder
                   "band": {"v_rel": self._band.last_speed.copy(), "bound_s": self._band_bound,      # R2-E3
                            **self._band_rerun.notes()}, **self._band_rerun.whole()},    # a band re-run: every world
        )
        return report

    def _capture_notes(self) -> dict[str, Any]:
        cp, gate = self._captured_passes(), getattr(self, "_gate", None)
        out = {} if cp is None else cp[0].notes()
        if gate is not None:                             # lever #2: a touched-unarmed attempt re-runs every world
            g = gate.notes()
            out.update(g, rerun_all=g["gate_rerun"])
        return out

    def period_refines(self) -> list[int]:
        """Per world: periods re-run finer, that period only, for its escalation (``alloy.coupling.ladder``)."""
        return self._ladder.counts(self._watchdog.world_count)

    def energy_ledger(self) -> dict[str, np.ndarray]:
        """Host copy of the D20 ledger: E, E_ref, E_scale, z_ref, W_ext, ΔE_int, passive gain G [J or m]."""
        wd = self._watchdog
        return {"energy": _host(wd.energy), "energy_ref": _host(wd.energy_ref), "e_scale": _host(wd.e_scale),
                "z_ref": _host(wd.z_ref), "external_work": _host(wd.external_work),
                "interface_work": _host(wd.interface_work), "gain": _host(wd.gain)}

    def state_arrays(self) -> Mapping[str, wp.array]:
        """Lagged feedback, Aitken ω/stats, energies and watch state (for WorldSnapshot)."""
        out = dict(self._coupler_state_arrays())
        out.update(self._watchdog.state_arrays())
        out.update(self._entry_history().snapshot_arrays(self._entries))     # MPM history in state_0
        for name in self._entry_order:                       # inner solvers are invisible to Simulation
            for key, arr in state_arrays_for(self._entries[name].solver).items():
                out[f"{name}.{key}"] = arr
        return out

    def _coupler_state_arrays(self) -> Mapping[str, wp.array]:
        return {}

    def state_world_maps(self) -> Mapping[str, wp.array]:
        """Item->world maps for the per-item arrays of ``state_arrays`` (SCR-WP4-3); (W,) arrays need none."""
        out = dict(self._coupler_world_maps())
        out.update(self._entry_history().snapshot_world_maps(self._entries))
        for name in self._entry_order:
            for key, arr in state_world_maps_for(self._entries[name].solver).items():
                out[f"{name}.{key}"] = arr
        return out

    def _coupler_world_maps(self) -> Mapping[str, wp.array]:
        return {}

    def frozen_contacts(self) -> Any:
        """Unsolved clone of the last contacts of sleeping islands, for observers (PRF-11); None if sleeping is off."""
        return self._frozen_contacts
