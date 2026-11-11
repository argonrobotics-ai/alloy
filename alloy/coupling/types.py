# SPDX-License-Identifier: Apache-2.0
"""Coupling and clock types (SPEC §3.6; Scheduler, clocks, watchdog and sleeping are WP4)."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from alloy.core.defaults import DT_X_MAX, SPEED_FAIL_MPS
from alloy.core.newton_compat import SolverCoupled
from alloy.core.timebase import TICK_EXP


@dataclass(frozen=True)
class ClockPolicy:
    dt_x_max: float = DT_X_MAX              # s, bound; dt_x = largest grid value <= it (1.25e-3 at control_dt 0.02)
    j_max: int = TICK_EXP                   # smallest dt_x = control_dt / 2**j_max (D4)
    omega_dt_max: float = 1.0
    safety: float = 0.5
    substep_ladder: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    downshift_quiet_periods: int = 10
    # opt-in (WP4 2026-11-06): also bound dt_x by the held contact damping, safety x omega_dt_max x m_eff / c_d. It
    # stops light proxies being thrown off soft bodies (P3 law rig), but the bound takes the lightest proxy whether or
    # not it touches anything: on dc5_like it is 17x finer than today's dt_x. Default-on waits for the reconciliation
    # census ([internal note])
    damping_bound: bool = False
    # WP9 S4, 2026-11-09: an entry whose estimator declares alloy_clock_safety (its estimate already carries its own
    # margin: SPH's CFL step at cfl 0.2) takes its rung at that safety instead of `safety` on top
    # (alloy.coupling.clocks.entry_safety). On by default since the coupled SPH suite passed with it on the CPU and an
    # L4 ([internal note] scr/WP9.md, S4); False gives the earlier clock
    own_cfl: bool = True
    # R2-E3 (WP4 2026-11-08): dt_x <= band_travel x a proxy pipeline's detection band / the fastest contact's speed
    # relative to its body point over the last period (alloy.coupling.contact_band); None turns it off
    band_travel: float | None = 1.0 / 6.0
    # R2-E3 follow-up (WP4 2026-11-08): re-run a period, every world, when its own band bound is under dt_x / band_rerun
    # (the bound reaches the clock one period late; contact_band.BandRerun); None turns it off
    band_rerun: float | None = 3.0
    # opt-in and experimental, not advertised (coordinator 2026-11-08; WP10 lever #2, [internal note]
    # notes/mixed-speed-1107.md): keep a coupled pair's stiffness/damping bound only
    # while one of its movable proxies is in or near its destination (contact_gate.py); then the whole pair is armed,
    # so in contact the clock is today's. No gain on main since wp4's MuJoCo -> VBD -> SPH chain (3dddcdfaa): a Panda
    # reaching a towel beside water (L4) ran its released periods 6x slower than today's clock at 4 worlds (10.5 vs
    # 1.7 s) and 1.3-3.3x slower at 64; before the chain the same scene ran 1.7-2.7x and 4-5x faster. Cost
    # (docs/known_gaps/coupling-contact-gate-settling.md): cloth resting on fixed geometry while the robot is away
    # settles at the coarser VBD step (the towel is 0.6 mm off when the arm arrives). No drift while in contact.
    # Margin: static gap [m] plus the travel the pair can close in a period; an armed pair stays armed
    # gate_hold_periods periods after it was last near
    contact_gate: bool = False
    gate_margin: float = 0.01
    gate_hold_periods: int = 2
    # opt-in, experimental, not advertised (speed-now S2, WP10 2026-11-09): a coupled VBD entry runs fewer iterations
    # at the finer exchange step, n_route sqrt(h / 0.625 ms), never under 10, and Chebyshev rho e/(1+e)
    # (alloy.coupling.vbd_iterations). It FAILS the settled-drape law: a towel resting on a still arm slides off (1 of
    # 8 worlds kept it, against 8 of 8 with the defaults; the rho part alone at 40 iterations: 5 of 8). Keep it off
    # until a law passes
    vbd_step_iterations: bool = False
    # opt-in (speed-now S2, WP10 2026-11-09): when one exchange's graph is over the node budget (it would run eagerly),
    # cap the entry with the most substeps one rung lower, so the clock takes a finer dt_x at the same entry step
    # (alloy.coupling.clocks.ClockController.cap_rung; IntegratedSimulation reports the key). With it on, dt_x depends
    # on graph node counts: CPU, graph=False and GPU runs take different clocks (same device and seed stay bitwise).
    # Settled drape with the contact gate: FAILS the rate rule (2026-11-10; 9 L4 runs of 8 worlds, seeds 100 / 200 /
    # 300, tree main 2c76e7b6e before the Newton 1.8 bump): 2 of 72 towels off the arm against the defaults' 3, but
    # 48 at rest against 64 (in 3 runs every towel was still moving at 2 s). The lever itself was inert in the
    # earlier single runs (alloy.coupling.graph_budget_rungs); rows [internal note] notes/wp10-runs/speed1109/out_rates
    graph_budget_rungs: bool = False
    # opt-in (speed-now S2, WP10 2026-11-09; WP4 reviewed the rule): when a bound (interface, contact band, graph-budget
    # cap) makes dt_x finer, each adaptive rung goes at once to the larger of what its estimate wants and the rung that
    # keeps its own step no coarser than before, as through a contact-gate jump. Today the rung holds for
    # downshift_quiet_periods, so a band trip from 78 to 19.5 us ran VBD at 4 substeps (4.9 us) where its bound asked
    # for 1. In a period where the watchdog's sticky escalation raised the floor the rungs hold as today.
    # DO NOT TURN ON: it failed its drape gate (2026-11-10; settled drape, 9 L4 runs of 8 worlds per arm, seeds
    # 100 / 200 / 300; tree main 2c76e7b6e + the flag, before the Newton 1.8 bump). Towels off the arm / at rest, of
    # 72: defaults 3 / 64, with the flag 11 / 56. Two of its six runs with a band trip ended with a world INVALID at
    # the trip (period 9: dt_x 78 -> 19.5 -> 9.8 us with VBD at 1 substep) and the other seven never at rest; none
    # of the defaults' four did. Run to run, 2 bad of 9 against 1 of 9 is not a difference; by the rate rule it
    # fails. Kept only so the rows can be repeated ([internal note] notes/wp10-runs/speed1109/out_rates)
    rungs_follow_dt_x: bool = False


@dataclass(frozen=True)
class InterfaceTolerance:
    vel: float = 1.0e-3                     # m/s (precision); elsewhere x10 via PairPolicy
    pos: float = 1.0e-4                     # m; a precision pair spanning two entries needs pos <= 0.25·max_penetration
    force_rel: float = 1.0e-2               # -
    force_floor: float = 1.0e-3             # N
    # force gate: "max" per endpoint (default); "l2" / "l2_free" one aggregate per world over all / free (meff > 0)
    # endpoints, ||F_raw - F_used||_2 / max(||F_raw||_2, force_floor sqrt(n)) (opt-in, WP11 for SPH-VBD P2 pairs)
    force_norm: Literal["max", "l2", "l2_free"] = "max"
    k_min: int = 1
    k_max: int = 8


@dataclass(frozen=True)
class AitkenPolicy:
    omega0: float = 0.5
    omega0_strong: float = 0.1
    omega_min: float = 0.05
    omega_max: float = 1.0
    carry_across_steps: bool = True


@dataclass(frozen=True)
class WatchdogPolicy:
    window: float = 1.0                     # s
    warn_gain_per_min: float = 0.01         # fraction of E_scale (D20)
    escalate_gain_per_window: float = 0.05  # fraction of E_scale (D20)
    abort_speed: float = SPEED_FAIL_MPS     # m/s (D12)
    max_rollbacks_per_period: int = 2
    # SCR-WP11-4: one exchange whose power bond injects more than this fraction of E_scale (a force jump, e.g. first
    # contact) re-runs its period at dt_x / 2, / 4, ... up to onset_refine_levels (that period only; inf: off)
    onset_gain_per_exchange: float = 1.0e-4
    onset_refine_levels: int = 3
    onset_cooldown_s: float = 0.5           # a world re-runs for a force jump at most once per this sim time
    # SCR-WP3-C3RAND (WP4 2026-11-08): a world's first ladder escalation re-runs its period finer, that period only;
    # the sticky ladder (clock floor, strong mode) only when a world trips again within ``window`` or a period ends
    # still escalated (alloy.coupling.ladder). False: every escalation takes the sticky ladder at once
    per_period_ladder: bool = True


@dataclass(frozen=True)
class SleepPolicy:
    enabled: bool = True                    # forced False when native_contacts="newton" (D10) and in drift tests
    lin_tol: float = 1.0e-3                 # m/s
    lin_tol_precision: float = 1.0e-4       # m/s, precision scenes
    accel_tol_g: float = 1.0e-3             # Alloy-controlled sleep also needs |unbalanced accel| < 1e-3·g
    quiet_time: float = 0.02
    init_asleep: bool = True
    never: tuple[str, ...] = ("actuated", "precision_pair_in_contact", "precision_pair_within_gap", "applied_wrench")
    freeze_interface: bool = True
    frozen_contact_cache: bool = True
    mujoco_nvmax_full_below: int = 128


@dataclass(frozen=True)
class PairPolicy:
    source: str
    destination: str
    mode: Literal["lagged", "staggered", "admm", "monolithic"] = "lagged"
    proxy: Any | None = None                # SolverCoupledProxy.Proxy for lagged/staggered
    precision: bool = False                 # tighter InterfaceTolerance
    added_mass_ratio_max: float = 0.5
    saturating_actuator: bool = False
    static_hold: bool = False               # must hold static friction below CREEP_GENERAL (§4.2) -> strong
    added_mass: float | None = None         # kg; fluid pairs: C_a·ρ_f·V_sub (D5)
    interface_stiffness: float | None = None  # N/m, k_c of the exchange-clock bound (WP4, D17 additive); None: derived
    one_way: bool = False                   # destination follows the source, never feeds back (SCR-DC5-6, WP4)


@dataclass(frozen=True)
class AlloyEntry:
    entry: SolverCoupled.Entry              # constructed at substeps = max rung so scratch exists (D21)
    initial_rung: int = 1
    adaptive: bool = True                   # False for dt-dependent material scenes (CON-76)
    capability: str | None = None           # solver capability name (make_stepper sets it; SCR-JOB2-3)


@dataclass(frozen=True)
class SchedulerPolicy:
    clock: ClockPolicy = ClockPolicy()
    tol: InterfaceTolerance = InterfaceTolerance()
    aitken: AitkenPolicy = AitkenPolicy()
    watchdog: WatchdogPolicy = WatchdogPolicy()
    sleep: SleepPolicy = SleepPolicy()
    lagged_iterations: int = 2              # exactly 2, omega = 1 (coupling-time §3.2)
    admm_k: tuple[int, int] = (2, 20)
    admm_eps_abs: float = 1e-4
    admm_eps_rel: float = 1e-2
    # SchedulerADMM's penalty as rho * dt_x (set every exchange): Newton's rows apply rho W^2 (u - Jv) as a force with
    # W^2 the pair's reduced mass, so an iteration removes rho dt_x of a contact's approach speed (Newton's rho 1:
    # 0.125 % at 1.25 ms, a 20 g box through a pad particle; notes/wp4-admm-rho). Escalation raises it once, to 1.
    admm_rho_dt: float = 0.25
    proxy_soft_contacts_per_world: int | None = None   # SCR-DC5-19 proxy soft-contact capacity; None: candidates
    proxy_rigid_contacts_per_world: int | None = None  # SCR-DC5-19 proxy rigid-contact capacity; None: Newton's
    # SCR-SC4-4: edge/face soft contacts on the proxy (rigid features passing between cloth vertices); None: on for
    # destinations that consume them (SolverVBD and its subclasses), off otherwise; True elsewhere raises
    proxy_full_surface: bool | None = None
    # SCR-SC4-4d (opt-in, coordinator 2026-11-06; costs in alloy.coupling.stiffness_cap and docs/notes/wp10-sc4-4d.md):
    # a lagged proxy body's touching-record stiffness capped at c m_eff / dt_x^2 on VBD destinations (c 4: the
    # explicit oscillator limit, omega dt_x < 2); None: off
    exchange_stiffness_c: float | None = None
    # ... and its touching-record contact damping at c_d m_eff / dt_x (C7: 0.5); None: off
    exchange_damping_c: float | None = None
    # SCR-SC4-4d: a lagged body proxy's feedback may change by at most c_s m_eff v_s / dt_x per exchange (static forces
    # untouched; alloy.coupling.slew); None: off
    exchange_slew_c: float | None = None


class WatchState(enum.IntEnum):
    """``CouplingReport.watch_state`` codes."""

    OK = 0
    WARN = 1
    ESCALATED = 2
    ROLLED_BACK = 3
    FAILED = 4


@dataclass(frozen=True, eq=False)
class CouplingReport:                       # host copy, read once per period
    dt_x: float
    substeps: Mapping[str, int]
    res_vel: np.ndarray                     # [W]
    res_pos: np.ndarray                     # [W]
    res_force: np.ndarray                   # [W]
    iters_used: np.ndarray                  # [W]
    watch_state: np.ndarray                 # [W] WatchState
    interface_work: np.ndarray              # [W] J
    energy: np.ndarray                      # [W] J
    rollbacks: int
    rung_changes: int
    strong_pairs: tuple[tuple[str, str], ...]
    notes: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def quiet(cls, dt_x: float, substeps: Mapping[str, int], world_count: int) -> CouplingReport:
        """A converged, uneventful report (single-entry steppers and fakes)."""
        z = np.zeros(world_count)
        return cls(dt_x, dict(substeps), z.copy(), z.copy(), z.copy(), np.ones(world_count, np.int32),
                   np.zeros(world_count, np.int32), z.copy(), z.copy(), 0, 0, ())
