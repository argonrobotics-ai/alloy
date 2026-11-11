# SPDX-License-Identifier: Apache-2.0
"""The fit search: CMA-ES generations over batched twin evaluations, IPOP restarts, wall budget (SPEC §3.12, FIT-2).

`cfg.restarts` is the number of CMA-ES runs (1 = no restart); each later run doubles lambda and starts from a
seeded uniform point in the box (IPOP-CMA-ES, Auger and Hansen, CEC 2005). A generation is skipped when the time
left is below the last generation's wall time, so the search stops inside `cfg.wall_budget_s`. lambda =
max(4 + 3 ln n, cfg.worlds // n_conditions): the world budget sets the population. The search sees a `FitSetup`
and an evaluator only, never the dataset.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from alloy.fit.batch import BatchResult
from alloy.fit.cma import CMAES
from alloy.fit.objective import Score
from alloy.fit.params import INIT_SIGMA, ParamSpace
from alloy.fit.types import FitConfig

log = logging.getLogger("alloy.fit")
Evaluate = Callable[[np.ndarray], BatchResult]


@dataclass(frozen=True)
class Generation:
    run: int
    generation: int
    popsize: int
    best: float                      # best objective this generation
    median: float
    sigma: float                     # largest unit-box axis step
    mean: tuple[float, ...]          # distribution mean, SI (refine: the linearisation point)
    wall_s: float
    phase: str = "cma"               # "cma" | "refine" (refine: `sigma` holds the LM step size, unit box)


@dataclass
class SearchOutcome:
    """Best candidate, the last population (for the spread) and the history."""

    best_x: np.ndarray
    best_f: float
    best_score: Score | None
    population: np.ndarray           # final generation, SI [lambda, n]
    population_f: np.ndarray
    history: list[Generation] = field(default_factory=list)
    evaluations: int = 0
    worlds: int = 0
    wall_s: float = 0.0
    stop: list[str] = field(default_factory=list)
    method: str = "cma"
    extra: dict = field(default_factory=dict)   # e.g. refine's gn_unc95, gn_objective

    best_run: int = -1

    def improve(self, batch: BatchResult, run: int = 0) -> None:
        k = int(np.argmin(np.where(np.isfinite(batch.objective), batch.objective, np.inf)))
        if batch.objective[k] < self.best_f:
            self.best_f = float(batch.objective[k])
            self.best_x = batch.x[k].copy()
            self.best_score = batch.scores[k]
            self.best_run = run


@dataclass
class Budget:
    """Wall-clock budget shared by every phase of one fit."""

    seconds: float
    t0: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def allows(self, next_cost_s: float) -> bool:
        return self.elapsed() + next_cost_s <= self.seconds


def popsize_for(cfg: FitConfig, n_params: int, n_conditions: int) -> int:
    return max(4 + int(3 * math.log(n_params)), cfg.worlds // max(n_conditions, 1), 2)


def _generation(es: CMAES, space: ParamSpace, evaluate: Evaluate, out: SearchOutcome, run: int,
                inject: np.ndarray | None = None) -> float:
    u = es.ask(inject)
    batch = evaluate(space.from_unit(u))
    es.tell(u, batch.objective)
    out.improve(batch, run)
    if out.best_run == run:          # Q.unc: the spread of the run that holds the best point
        out.population, out.population_f = batch.x, batch.objective
    out.evaluations += len(batch.x)
    out.worlds += batch.worlds
    finite = batch.objective[np.isfinite(batch.objective)]
    gen = Generation(run, es.generation, es.lam, float(np.min(batch.objective)),
                     float(np.median(finite)) if finite.size else math.nan, float(np.max(es.axis_steps())),
                     tuple(float(v) for v in space.from_unit(es.mean)), batch.wall_s)
    out.history.append(gen)
    log.info("fit gen %d.%d: lambda %d best %.4g median %.4g sigma %.3g mean %s (%.1f s)", run, gen.generation,
             gen.popsize, gen.best, gen.median, gen.sigma, [f"{v:.5g}" for v in gen.mean], gen.wall_s)
    return batch.wall_s


def _start(space: ParamSpace, run: int, rng: np.random.Generator, start: Mapping[str, float] | None
           ) -> tuple[np.ndarray, float]:
    if run == 0:
        return space.initial(start)
    return rng.uniform(0.0, 1.0, space.n), INIT_SIGMA      # IPOP restarts are global: wide, from a random point


def run_cma(space: ParamSpace, evaluate: Evaluate, cfg: FitConfig, n_conditions: int, budget: Budget,
            start: Mapping[str, float] | None = None) -> SearchOutcome:
    """CMA-ES with `cfg.restarts` runs (IPOP), `cfg.generations` per run, inside the wall budget.

    Run 0 starts at the priors / `start` estimates (ParamSpace.initial); restarts start uniformly in the box."""
    out = SearchOutcome(np.full(space.n, np.nan), math.inf, None, np.zeros((0, space.n)), np.zeros(0))
    rng = np.random.default_rng([cfg.seed, 7919])
    lam = popsize_for(cfg, space.n, n_conditions)
    last_cost = 0.0
    for run in range(max(1, cfg.restarts)):
        mean, sigma = _start(space, run, rng, start)
        es = CMAES(mean, sigma, lam * 2 ** run, seed=int(rng.integers(2**31)), tol_x=cfg.tol_x)
        reason = "generations"
        next_cost = last_cost * (2.0 if run > 0 else 1.0)   # a restart doubles lambda
        for _ in range(cfg.generations):
            if not budget.allows(next_cost):
                reason = "wall_budget"
                break
            inject = mean if es.generation == 0 else None   # evaluate the start point itself once per run
            last_cost = next_cost = _generation(es, space, evaluate, out, run, inject)
            stop = es.stop_reason()
            if stop is not None:
                reason = stop
                break
        out.stop.append(f"run {run}: {reason}")
        if reason == "wall_budget":
            break
    out.wall_s = budget.elapsed()
    return out


__all__ = ["run_cma", "SearchOutcome", "Generation", "Budget", "popsize_for", "Evaluate"]
