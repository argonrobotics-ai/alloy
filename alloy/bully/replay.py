# SPDX-License-Identifier: Apache-2.0
"""Bully episodes: run a legal action trace in a batch of worlds and classify every world (SPEC §3.11).

``EpisodeRunner`` builds the target's simulation once and reuses it (reset between episodes), so a B1 search pays
the solver build once. ``run_episode`` is the one-shot form the spec names. ``replay_trace`` repeats one trace
``runs`` times in a one-world simulation and counts how often the finding's monitor trips (10/10 on CPU,
>= 9/10 on the same GPU type, bench §4.7).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.bully.actions import HookedForceActions, HookedPoseActions, bind_actions, legalize
from alloy.bully.monitors import BenchMonitors, shape_work_sources
from alloy.bully.severity import INVALID_CAUSE, MONITORS, badness, classify
from alloy.bully.types import BullyLimits, BullyTarget, EpisodeResult, ReplayResult
from alloy.core.errors import BullyError
from alloy.core.registry import estimator_for
from alloy.precision.types import HealthFlags, HealthReport, Severity

log = logging.getLogger("alloy.bully")

_PERIOD_FIELDS = ("body_q", "body_qd", "particle_q", "particle_qd")


def sim_model(sim: Any) -> newton.Model:
    """The Model behind a simulation (``sim.model``, or ``sim.built.model`` for ``alloy.sim.Simulation``)."""
    model = getattr(sim, "model", None)
    if model is None and getattr(sim, "built", None) is not None:
        model = sim.built.model
    if model is None:
        raise BullyError(f"{type(sim).__name__} exposes no model (SCR-WP7-1: SimulationLike.model)")
    return model


def _health(sim: Any, worlds: int) -> HealthReport:
    rep: Mapping[str, Any] = sim.report() or {}
    health = rep.get("health")
    return health if isinstance(health, HealthReport) else HealthReport.ok(worlds)


def target_monitors(sim: Any) -> dict[str, np.ndarray]:
    """A target's own per-world S1 counters (SB-7 cloth and cutting classes): ``sim.bully_monitors()`` returns
    {row name: [W] values} for rows in ``severity.MONITORS`` (an unknown name fails classification). The sim keeps
    them for the episode and clears them on ``reset``. Absent on every other sim."""
    fn = getattr(sim, "bully_monitors", None)
    return {} if not callable(fn) else {str(k): np.asarray(v, dtype=np.float64) for k, v in fn().items()}


class EpisodeRunner:
    """One simulation of ``worlds`` worlds for one target; ``run`` plays a [W, T, A] trace from the reset state."""

    def __init__(self, target: BullyTarget, worlds: int, device: str, limits: BullyLimits,
                 check_every: int = 1) -> None:
        self.target, self.worlds, self.device, self.limits = target, int(worlds), device, limits
        self.check_every = max(1, int(check_every))
        self.sim = target.make_sim(self.worlds, device)
        self.model = sim_model(self.sim)
        if self.model.world_count != self.worlds:
            raise BullyError(f"target {target.name} built {self.model.world_count} worlds, asked for {self.worlds}")
        solver = getattr(self.sim, "solver", None)
        source = estimator_for(solver) if isinstance(solver, SolverBase) else None
        overflow = getattr(getattr(solver, "mjw_data", None), "overflow", None)  # MJWarp per-step capacity bits
        self.monitors = BenchMonitors(self.model, limits, target.non_target_bodies, robot_bodies=target.robot_bodies,
                                      thin_parts=target.thin_parts, energy_source=source, overflow=overflow,
                                      work_sources=shape_work_sources(solver))
        hook = getattr(self.sim, "add_step_observer", None)
        self.per_step = hook is not None
        if hook is not None:
            hook(self.monitors.observer)
        else:
            log.warning("target %s: no step observer hook; monitors run once per control period", target.name)
            self._prev = self.model.state()
        self.episodes = 0

    def run(self, actions: np.ndarray, *, reset: bool = True) -> EpisodeResult:
        """Legalise ``actions`` ([W, T, A] or [T, A] broadcast), play it, return per-world severity and badness."""
        trace = self._legal(actions)
        if reset and self.episodes:
            self.sim.reset()
        self.monitors.reset(self.sim.state, self.sim.control)
        bound = bind_actions(self.sim, self.model, _kind(self.target), _index(self.target), trace)
        severity = np.zeros(self.worlds, dtype=np.int32)
        first = np.full(self.worlds, -1, dtype=np.int64)
        extra: dict[str, np.ndarray] = {}
        health = HealthReport.ok(self.worlds)
        for tick in range(bound.ticks):
            bound.apply(tick)
            self._step_period()
            if (tick + 1) % self.check_every == 0 or tick + 1 == bound.ticks:
                extra, health = {**self.monitors.report(), **target_monitors(self.sim)}, _health(self.sim, self.worlds)
                severity = classify(health, extra, self.limits)
                first[(first < 0) & (severity > 0)] = tick
        if isinstance(bound, (HookedForceActions, HookedPoseActions)):
            bound.detach()
        self.episodes += 1
        return EpisodeResult(severity, first, extra, health, badness(extra, self.limits))

    def _legal(self, actions: np.ndarray) -> np.ndarray:
        sp = self.target.actions
        arr = np.asarray(actions, dtype=np.float64)
        if arr.ndim == 2:
            arr = np.broadcast_to(arr, (self.worlds, *arr.shape))
        if arr.shape[0] != self.worlds or arr.shape[2] != sp.dim:
            raise BullyError(f"actions {arr.shape} do not fit {self.worlds} worlds x {sp.dim} actuators")
        legal = legalize(arr, sp.low, sp.high, sp.max_delta_per_tick, sp.initial)
        if legal.clips.any():
            log.info("target %s: %d worlds had clipped commands (max %d)", self.target.name,
                     int((legal.clips > 0).sum()), int(legal.clips.max()))
        return legal.trace

    def _step_period(self) -> None:
        if self.per_step:
            self.sim.step_period()
            return
        for name in _PERIOD_FIELDS:
            src = getattr(self.sim.state, name, None)
            if src is not None and src.shape[0]:
                wp.copy(getattr(self._prev, name), src)
        self.sim.step_period()
        dt = float(getattr(getattr(self.sim, "clock", None), "control_dt", getattr(self.sim, "control_dt", 0.02)))
        self.monitors.update(self._prev, self.sim.state, self.sim.control, None, dt, getattr(self.sim, "contacts", None))


def run_episode(target: BullyTarget, actions: np.ndarray, device: str, limits: BullyLimits) -> EpisodeResult:
    """One batched episode, one world per row of ``actions`` ([W, T, A])."""
    worlds = 1 if np.ndim(actions) == 2 else int(np.shape(actions)[0])
    return EpisodeRunner(target, worlds, device, limits).run(actions)


def tripped(result: EpisodeResult, monitor: str, limits: BullyLimits) -> np.ndarray:
    """[W] bool: did ``monitor`` (a bench monitor name or a HealthFlags name) trip in each world?"""
    if monitor in MONITORS:
        values = np.asarray(result.monitors.get(monitor, np.zeros_like(result.severity)), dtype=np.float64)
        return ~(values <= MONITORS[monitor].threshold(limits))
    if monitor == INVALID_CAUSE:
        return np.asarray(result.health.severity) == int(Severity.INVALID)
    flag = HealthFlags[monitor]
    return (np.asarray(result.health.flags).astype(np.int64) & int(flag)) != 0


def replay_trace(target: BullyTarget, trace: np.ndarray, monitor: str, device: str, limits: BullyLimits,
                 runs: int = 10, finding_id: str = "") -> ReplayResult:
    """Replay one trace [T, A] ``runs`` times (one world, reset between runs) and count trips of ``monitor``."""
    runner = EpisodeRunner(target, 1, device, limits)
    trips = 0
    for k in range(runs):
        hit = bool(tripped(runner.run(trace[None]), monitor, limits)[0])
        trips += int(hit)
        log.debug("replay %s run %d/%d: %s", finding_id or target.name, k + 1, runs, "trip" if hit else "clean")
    needed = runs if device == "cpu" else int(np.ceil(0.9 * runs))
    return ReplayResult(finding_id, trips, runs, device, trips >= needed)


def _kind(target: BullyTarget) -> str:
    return target.action_kind


def _index(target: BullyTarget) -> tuple[int, ...]:
    return tuple(target.action_index) or tuple(range(target.actions.dim))
