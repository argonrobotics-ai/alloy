# SPDX-License-Identifier: Apache-2.0
"""The timed run behind the dynamic checks: build the integrated simulation (health on), step it ``seconds`` with
no control input, and keep what the checks read.

Kept: body and particle poses at four times (start, end of settling, middle, end); the deepest penetration per world
from the health record after every control period, with the object pair behind it (a geometric re-collide whenever
the deepest value grows past the warn limit) and the depth at the last period (what ``sim.health()`` reports after
the same run); the final health report; the bench monitors' energy ledger
(``alloy.bully.monitors.BenchMonitors``: unexplained energy gain over E_scale, non-finite state, teleports).

Example::

    >>> res = run_scene(built, index, RunOptions(seconds=0.5))     # doctest: +SKIP
    >>> res.periods, res.pen_max.shape
    (25, (4,))
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.check.contacts import ContactProbe, PairDepth
from alloy.check.objects import ObjectIndex
from alloy.core.defaults import PEN_LIMIT_M

log = logging.getLogger("alloy.check")

SETTLE_S = 0.25         # the first 0.25 s (at most a quarter of the run) is settling
GROW = 1.25             # re-attribute penetration when the deepest value grows by 25 %
MAX_ATTRIBUTIONS = 8
MIN_PERIODS = 4         # a run stopped by its wall budget still runs this many periods ...
HARD_CAP = 4.0          # ... unless it has used HARD_CAP x the budget: then it stops after any period


@dataclass(frozen=True)
class RunOptions:
    """``seconds`` of simulated time; ``profile`` an integration profile ("standard", "fast", "precision");
    ``budget_s`` the wall time the run may take (None: no limit): the run stops after the period that uses it up (at
    least ``MIN_PERIODS``, unless ``HARD_CAP`` x the budget is gone first), and ``RunResult.truncated`` says so.
    ``contacts``: the contact pipeline as ``alloy.build(contacts=)`` takes it ("auto", "mujoco", "newton"); None
    keeps the profile's own.

    Example::

        >>> RunOptions(seconds=0.5).budget_s
        25.0
    """

    seconds: float = 2.0
    profile: str = "standard"
    seed: int = 0
    budget_s: float | None = 25.0
    contacts: str | None = None

    def overrides(self) -> dict[str, Any]:
        """Profile fields for ``contacts``, as ``alloy.build`` sets them. Example: ``RunOptions(contacts="newton")``."""
        if self.contacts is None:
            return {}
        return {"contacts_auto": True} if self.contacts == "auto" else {"native_contacts": self.contacts}


@dataclass
class Sample:
    """Body poses and velocities and particle positions of every world at time ``t`` [s], as float64 copies.

    Example::

        >>> s = Sample(0.0, np.zeros((2, 7)), np.zeros((2, 6)), np.zeros((0, 3)))
        >>> s.t, s.body_q.shape
        (0.0, (2, 7))
    """
    t: float
    body_q: np.ndarray        # [B, 7] float64
    body_qd: np.ndarray       # [B, 6]
    particle_q: np.ndarray    # [P, 3]


@dataclass
class RunResult:
    """What the dynamic checks read (see the module docstring); ``error`` is set when the run stopped early.

    Example::

        >>> RunResult(periods=40, control_dt=0.0125).seconds_run
        0.5
    """

    samples: list[Sample] = field(default_factory=list)
    pen_max: np.ndarray | None = None          # [W] m, NaN where never measured
    pen_t: np.ndarray | None = None            # [W] s, time of pen_max
    pen_end: np.ndarray | None = None          # [W] m, depth at the last period (sim.health()'s max_penetration_m)
    pair_pen_max: np.ndarray | None = None     # [W] m, precision pairs
    attributions: list[tuple[float, PairDepth]] = field(default_factory=list)
    health: Any = None                          # HealthReport
    monitors: dict[str, np.ndarray] | None = None
    e_scale: np.ndarray | None = None
    control_dt: float = 0.0
    periods: int = 0
    steps: int = 0
    native: str = ""
    error: str | None = None
    error_t: float | None = None
    seconds_asked: float = 0.0
    truncated: bool = False                     # shortened to fit RunOptions.budget_s
    graph_error: str | None = None              # a CUDA graph capture that failed; the run was redone eagerly

    @property
    def seconds_run(self) -> float:
        """Simulated seconds the run covered (periods stepped times the control period).

        Example::

            >>> RunResult(periods=40, control_dt=0.0125).seconds_run
            0.5
        """
        return self.periods * self.control_dt


def run_scene(built: Any, index: ObjectIndex, opts: RunOptions, probe: ContactProbe | None = None) -> RunResult:
    """Step ``built`` for ``opts.seconds`` (or less, to fit ``opts.budget_s``) and collect a ``RunResult``; never
    raises for a failing scene. A run whose CUDA graph capture fails (a part of the scene reads the host inside a step)
    runs again without graphs; ``graph_error`` keeps why.

    Example::

        >>> res = run_scene(built, index, RunOptions(seconds=0.5))  # doctest: +SKIP
        >>> res.error is None
        True
    """
    res = _run(built, index, opts, probe)
    if res.error is None or not res.error.startswith("GraphCaptureError"):
        return res
    log.warning("check run: a part of this scene reads the host inside a CUDA graph capture (traceback above); "
                "running it again without graphs (slower)")
    failed, res = res.error, None
    gc.collect()
    res = _run(built, index, opts, probe, graph="off")
    res.graph_error = failed
    return res


def _run(built: Any, index: ObjectIndex, opts: RunOptions, probe: ContactProbe | None, **overrides: Any) -> RunResult:
    from alloy.integration.wiring import build_simulation  # noqa: PLC0415

    res = RunResult(control_dt=float(built.control_dt), seconds_asked=float(opts.seconds))
    try:
        sim = build_simulation(built, opts.profile, seed=opts.seed, **opts.overrides(), **overrides)
    except Exception as exc:  # noqa: BLE001 - a simulation that cannot be built (out of memory) is a finding
        _stopped(res, exc, 0)
        return res
    res.native = str(getattr(sim, "native", ""))
    mon = _monitors(sim, index)
    n = max(1, math.ceil(opts.seconds / built.control_dt - 1e-9))
    cps, taken = checkpoints(n, built.control_dt), {0: _sample(sim, 0.0)}
    _track_init(res, built.world_count)
    t0 = time.perf_counter()
    for k in range(1, n + 1):
        if not _step(sim, res, k):
            break
        probe = _track(sim, res, index, k * res.control_dt, probe)
        over = k < n and over_budget(time.perf_counter() - t0, k, opts.budget_s)
        if k in cps or over:
            taken[k] = _sample(sim, k * res.control_dt)
        if over:
            res.truncated = True
            log.warning("check run: %.0f s wall budget used after %d of %d periods", opts.budget_s, k, n)
            break
    res.periods, res.steps = int(sim.periods), int(sim.steps)
    res.samples = pick_samples(taken, res.periods, built.control_dt)
    res.health = _poll(sim)
    if mon is not None:
        res.monitors, res.e_scale = mon.report(), np.asarray(mon.e_scale, np.float64)
    log.info("check run: %d periods, %d steps, native %s%s%s", res.periods, res.steps, res.native,
             ", shortened to fit the wall budget" if res.truncated else "",
             f", stopped: {res.error}" if res.error else "")
    return res


def over_budget(spent: float, k: int, budget: float | None) -> bool:
    """True when a run that has stepped ``k`` periods in ``spent`` wall seconds must stop.

    Example::

        >>> over_budget(30.0, 2, 25.0), over_budget(30.0, 4, 25.0), over_budget(120.0, 1, 25.0)
        (False, True, True)
    """
    if budget is None:
        return False
    return (spent > budget and k >= MIN_PERIODS) or spent > HARD_CAP * budget


def sample_periods(n: int, control_dt: float) -> set[int]:
    """Periods after which poses are sampled: end of settling, middle, end (period 0 is sampled before stepping).

    Example::

        >>> sorted(sample_periods(40, 0.0125))
        [10, 20, 40]
    """
    settle = min(max(1, round(SETTLE_S / control_dt)), max(1, n // 4))
    return {settle, max(settle, n // 2), n}


def checkpoints(n: int, control_dt: float) -> set[int]:
    """``sample_periods(n)`` plus periods 1, 2, 3, 4, 6, 8, 12, 16, 24, ... (ratio <= 1.5), so a run stopped early
    still has a sample near its own settle and middle marks.

    Example::

        >>> sorted(checkpoints(40, 0.0125))
        [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40]
    """
    out, k = set(sample_periods(n, control_dt)), 1
    while k <= n:
        out.add(k)
        k = k + 1 if k < 4 else (k * 3) // 2 if k & (k - 1) == 0 else (k * 4) // 3
    return out


def pick_samples(taken: dict[int, Sample], periods: int, control_dt: float) -> list[Sample]:
    """Start, settle, middle and end samples of a run of ``periods`` from the checkpoints taken (the nearest one at
    or before each mark).

    Example::

        >>> [s.t for s in pick_samples(taken, 40, 0.0125)]  # doctest: +SKIP
        [0.0, 0.125, 0.25, 0.5]
    """
    if periods not in taken:                    # a run that stopped on an error: its last checkpoint is the end
        periods = max(taken)
    marks = sorted(sample_periods(max(periods, 1), control_dt) | {periods})
    keys = sorted(taken)
    chosen = [0] + [max(k for k in keys if k <= m) for m in marks]
    return [taken[k] for k in sorted(set(chosen))]


def _step(sim: Any, res: RunResult, k: int) -> bool:
    try:
        sim.step_period()
    except Exception as exc:  # noqa: BLE001 - a crashing scene is a finding, not a crash of the check
        _stopped(res, exc, k)
        return False
    return True


def _stopped(res: RunResult, exc: Exception, k: int) -> None:
    res.error, res.error_t = f"{type(exc).__name__}: {exc}", k * res.control_dt
    log.warning("check run stopped at period %d: %s", k, res.error, exc_info=exc)


def _monitors(sim: Any, index: ObjectIndex) -> Any:
    """BenchMonitors (energy ledger, non-finite, teleport) as a step observer; None if it cannot be built."""
    from alloy.bully.monitors import BenchMonitors, shape_work_sources  # noqa: PLC0415
    from alloy.bully.types import BullyLimits  # noqa: PLC0415
    from alloy.core.registry import estimator_for  # noqa: PLC0415

    solvers = list(getattr(getattr(sim, "parts", None), "solvers", {}).values())
    solver = solvers[0] if len(solvers) == 1 else None
    robots = [i for n in index.names if index.motion.get(n) == "robot"
              for i in sim.built.handles[n].template.body]
    try:
        mon = BenchMonitors(sim.model, BullyLimits(), robot_bodies=robots,
                            energy_source=estimator_for(solver) if solver is not None else None,
                            work_sources=shape_work_sources(solver) if solver is not None else ())
        mon.reset(sim.state, sim.control)
    except Exception as exc:  # noqa: BLE001 - the energy check is skipped, the others still run
        log.warning("check: bench monitors unavailable (%s: %s); energy check skipped", type(exc).__name__, exc)
        return None
    sim.add_step_observer(mon.observer)
    return mon


def _sample(sim: Any, t: float) -> Sample:
    m, s = sim.model, sim.state
    bq = s.body_q.numpy().astype(np.float64) if m.body_count else np.zeros((0, 7))
    bqd = s.body_qd.numpy().astype(np.float64) if m.body_count else np.zeros((0, 6))
    pq = s.particle_q.numpy().astype(np.float64) if m.particle_count else np.zeros((0, 3))
    return Sample(t, bq, bqd, pq)


def _track_init(res: RunResult, worlds: int) -> None:
    res.pen_max = np.full(worlds, np.nan)
    res.pen_t = np.full(worlds, np.nan)
    res.pen_end = np.full(worlds, np.nan)
    res.pair_pen_max = np.zeros(worlds)


def _track(sim: Any, res: RunResult, index: ObjectIndex, t: float, probe: ContactProbe | None) -> ContactProbe | None:
    """Fold this period's health record into the per-world maxima; attribute a new deepest contact (the probe is
    built on first use and returned for the next period)."""
    h = _poll(sim)
    if h is None:
        return probe
    pen = np.where(_measured(h), np.asarray(h.max_pen_m, np.float64), np.nan)
    before = float(np.nanmax(res.pen_max)) if np.isfinite(res.pen_max).any() else 0.0
    grew = np.isfinite(pen) & ~(pen <= res.pen_max)          # NaN-safe: first measurement counts as growth
    res.pen_max = np.where(grew, pen, res.pen_max)
    res.pen_t = np.where(grew, t, res.pen_t)
    res.pen_end = pen
    res.pair_pen_max = np.maximum(res.pair_pen_max, np.nan_to_num(np.asarray(h.max_pair_pen_m, np.float64)))
    now = float(np.nanmax(pen)) if np.isfinite(pen).any() else 0.0
    if now > PEN_LIMIT_M and now > GROW * before and len(res.attributions) < MAX_ATTRIBUTIONS:
        probe = probe if probe is not None else ContactProbe(sim.built)
        rows = probe.depths(sim.state, index)
        if rows:
            res.attributions.append((t, rows[0]))
    return probe


def _measured(h: Any) -> np.ndarray:
    m = getattr(h, "pen_measured", None)
    return np.ones(len(h.severity), bool) if m is None else np.asarray(m, bool)


def _poll(sim: Any) -> Any:
    from alloy.core.errors import PrecisionError  # noqa: PLC0415

    health = getattr(sim, "health", None)
    if health is None:
        return None
    try:
        return health.poll()
    except PrecisionError as exc:              # on_invalid="raise" presets: the report rides on the error
        return getattr(exc, "report", None)


__all__ = ["RunOptions", "RunResult", "Sample", "run_scene", "sample_periods"]
