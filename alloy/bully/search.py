# SPDX-License-Identifier: Apache-2.0
"""B1 batched search (SPEC §3.11, bench §4.7): CMA-ES over legal action knots, one candidate per world.

Each generation is one batched episode: ``worlds`` candidates, ``knots`` Catmull-Rom knots per actuator over
``horizon_s``, normalised to the action box. The mean starts at B0's worst trace (highest badness) and the step
size at ``sigma0`` of the range. The objective is ``-badness``. Every world that trips is a candidate; the worst
candidate per (monitor) is replayed (10 runs; 10/10 on CPU, >= 9/10 on the same GPU type) and becomes a ``Finding``
only if the replay counts. The search stops at ``generations`` or when the wall clock reaches ``budget_gpu_h``.
IPOP restarts (SCR-WP7-6): once the best b has improved by under 1 % in 25 generations, CMA restarts from the best
candidate with twice the population (run as consecutive batches of ``worlds``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alloy.bully.actions import knots_to_trace, legalize, tick_count, trace_to_knots
from alloy.bully.cma import BatchedCMAES
from alloy.bully.replay import EpisodeRunner, replay_trace
from alloy.bully.severity import world_cause
from alloy.bully.stress import named_stress_phases, tile_phases
from alloy.bully.types import BullyLimits, BullySeverity, BullyTarget, EpisodeResult, Finding, SearchConfig
from alloy.core.ids import digest_of
from alloy.core.newton_compat import NEWTON_PIN

log = logging.getLogger("alloy.bully")

RESTART_STALL_GENS = 25  # SCR-WP7-6: generations without 1 % improvement of the best b before a restart
RESTART_STALL_REL = 0.01
RESTART_POP_FACTOR = 2


@dataclass
class Candidate:
    """The worst tripped world seen so far for one monitor."""

    monitor: str
    severity: int
    value: float
    threshold: float
    badness: float
    first_step: int
    generation: int
    trace: np.ndarray  # [T, A] legal SI


@dataclass
class SearchOutcome:
    """Everything a B1 run produced; ``findings`` are the replay-confirmed ones (with their traces)."""

    findings: list[Finding] = field(default_factory=list)
    traces: dict[str, np.ndarray] = field(default_factory=dict)  # finding id -> trace [T, A]
    candidates: dict[str, Candidate] = field(default_factory=dict)
    best_badness: list[float] = field(default_factory=list)  # per generation (gen -1 = B0)
    first_trip_generation: dict[int, int] = field(default_factory=dict)  # severity -> first generation
    generations_run: int = 0
    wall_s: float = 0.0
    replays: dict[str, tuple[int, int]] = field(default_factory=dict)  # monitor -> (trips, runs)
    restarts: list[int] = field(default_factory=list)  # generations at which CMA restarted (IPOP)


@dataclass
class _Ipop:
    """CMA with IPOP restarts: population = ``batches`` x worlds; ``best`` is the best b per generation since the
    last restart."""

    es: BatchedCMAES
    sigma0: float
    seed: int
    batches: int = 1
    best: list[float] = field(default_factory=list)

    def note(self, b: float) -> None:
        self.best.append(max(b, self.best[-1]) if self.best else b)

    def stalled(self) -> bool:
        h, k = self.best, RESTART_STALL_GENS
        return len(h) > k and h[-1] <= h[-1 - k] * (1.0 + RESTART_STALL_REL)

    def restart(self, worlds: int) -> None:
        mean = self.es.state.best_x if self.es.state.best_x is not None else self.es.state.mean
        self.batches *= RESTART_POP_FACTOR
        self.seed += 1
        self.es = BatchedCMAES(mean, self.sigma0, lam=self.batches * worlds, seed=self.seed)
        self.best = []


def scene_hash(target: BullyTarget) -> str:
    """Digest of the target's SceneCard, or of its name for raw-Newton targets."""
    return digest_of(target.scene) if target.scene is not None else digest_of({"bully_target": target.name})


def alloy_sha() -> str:
    """HEAD of the alloy checkout, or ``ALLOY_SHA`` (set by runners on VMs that receive the tree without .git)."""
    if os.environ.get("ALLOY_SHA"):
        return os.environ["ALLOY_SHA"]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _collect(outcome: SearchOutcome, result: EpisodeResult, traces: np.ndarray, gen: int, limits: BullyLimits
             ) -> None:
    """Fold tripped worlds of one episode into the per-monitor worst candidates."""
    for w in np.nonzero(result.severity > 0)[0]:
        name, value, thr, sev = world_cause(result.health, result.monitors, limits, int(w))
        outcome.first_trip_generation.setdefault(sev, gen)
        prev = outcome.candidates.get(name)
        if prev is None or result.badness[w] > prev.badness:
            outcome.candidates[name] = Candidate(name, sev, value, thr, float(result.badness[w]),
                                                 int(result.first_step[w]), gen, traces[w].copy())


def _b0(runner: EpisodeRunner, target: BullyTarget, cfg: SearchConfig, outcome: SearchOutcome,
        limits: BullyLimits) -> np.ndarray:
    """Run B0 in one batch; return the worst phase's trace as the B1 seed."""
    phases = [trace for _, trace in named_stress_phases(target, cfg.horizon_s)]
    batch = tile_phases(phases, cfg.worlds)
    result = runner.run(batch)
    _collect(outcome, result, batch, -1, limits)
    outcome.best_badness.append(float(np.max(result.badness)))
    return batch[int(np.argmax(result.badness))]


def search_detailed(target: BullyTarget, cfg: SearchConfig, limits: BullyLimits, device: str, *,
                    seed_from_b0: bool = True, replay_runs: int = 10) -> SearchOutcome:
    """B1 with B0 seeding, budget, candidate collection and replay gating."""
    sp = target.actions
    ticks = tick_count(cfg.horizon_s, sp.rate_hz)
    outcome = SearchOutcome()
    t0 = time.perf_counter()
    runner = EpisodeRunner(target, cfg.worlds, device, limits)
    mean0 = np.full((cfg.knots, sp.dim), 0.5)
    if seed_from_b0:
        mean0 = trace_to_knots(_b0(runner, target, cfg, outcome, limits), sp.low, sp.high, cfg.knots)
    cma = _Ipop(BatchedCMAES(mean0.ravel(), cfg.sigma0, lam=cfg.worlds, seed=cfg.seed), cfg.sigma0, cfg.seed)
    budget_s = cfg.budget_gpu_h * 3600.0
    for gen in range(cfg.generations):
        if time.perf_counter() - t0 > budget_s:
            log.warning("B1 %s: budget %.2f h reached after %d generations", target.name, cfg.budget_gpu_h, gen)
            break
        _generation(runner, cma, cfg, ticks, outcome, limits, gen)
        if cma.stalled():
            cma.restart(cfg.worlds)
            outcome.restarts.append(gen)
            log.info("B1 %s gen %d: best b stalled for %d generations; IPOP restart, population %d x %d",
                     target.name, gen, RESTART_STALL_GENS, cma.batches, cfg.worlds)
    outcome.wall_s = time.perf_counter() - t0
    _confirm(outcome, target, cfg, limits, device, replay_runs)
    return outcome


def _generation(runner: EpisodeRunner, cma: _Ipop, cfg: SearchConfig, ticks: int, outcome: SearchOutcome,
                limits: BullyLimits, gen: int) -> None:
    """One CMA generation, its population run as ``cma.batches`` batched episodes of ``cfg.worlds``."""
    sp = runner.target.actions
    knots = cma.es.ask().reshape(cma.batches, cfg.worlds, cfg.knots, sp.dim)
    fitness, best, s1, s2 = [], -np.inf, 0, 0
    for batch in knots:
        traces = legalize(knots_to_trace(batch, sp.low, sp.high, ticks), sp.low, sp.high, sp.max_delta_per_tick,
                          sp.initial).trace
        result = runner.run(traces)
        fitness.append(-result.badness)
        _collect(outcome, result, traces, gen, limits)
        best = max(best, float(np.max(result.badness)))
        s1, s2 = s1 + int(np.sum(result.severity == 1)), s2 + int(np.sum(result.severity == 2))
    cma.es.tell(np.concatenate(fitness))
    cma.note(best)
    outcome.best_badness.append(best)
    outcome.generations_run = gen + 1
    log.info("B1 %s gen %d: best b %.3g, S1 worlds %d, S2 worlds %d", runner.target.name, gen, best, s1, s2)


def _confirm(outcome: SearchOutcome, target: BullyTarget, cfg: SearchConfig, limits: BullyLimits, device: str,
             runs: int) -> None:
    """Replay the worst candidate per monitor; keep those whose replay counts, most severe first."""
    s_hash, a_sha = scene_hash(target), alloy_sha()
    ordered = sorted(outcome.candidates.values(), key=lambda c: (c.severity, -c.badness))
    for k, cand in enumerate(ordered):
        rep = replay_trace(target, cand.trace, cand.monitor, device, limits, runs)
        outcome.replays[cand.monitor] = (rep.trips, rep.runs)
        if not rep.counts:
            log.info("B1 %s: %s candidate did not replay (%d/%d)", target.name, cand.monitor, rep.trips, rep.runs)
            continue
        fid = f"NEW-{k + 1:04d}"
        repro = f"python -m alloy.bully replay <id> --device {device}"
        finding = Finding(fid, BullySeverity(cand.severity), cand.monitor, cand.value, cand.threshold,
                          cand.first_step, s_hash, a_sha, NEWTON_PIN, repro, target=target.name)
        outcome.findings.append(finding)
        outcome.traces[fid] = cand.trace


def search(target: BullyTarget, cfg: SearchConfig, limits: BullyLimits, device: str) -> list[Finding]:
    """B1 CMA-ES (SPEC §3.11 signature): replay-confirmed findings, most severe first."""
    return search_detailed(target, cfg, limits, device).findings
