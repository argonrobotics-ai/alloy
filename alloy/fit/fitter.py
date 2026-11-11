# SPDX-License-Identifier: Apache-2.0
"""`fit`: prepare (calib only) -> search (cma | grad | hybrid) -> validate (valid only) -> FitResult (SPEC §3.12).

The result's values are the best evaluated candidate; each leaf's `unc` is the 2.5-97.5 % spread of the final
population (widened to contain the value), `source="fit"` and `note` starts with the fit id. The fit id is the digest
of everything that determines the fit (law, dataset sha256, parameters, config, solver, device, conditions).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from alloy.core.ids import digest_of
from alloy.core.quantity import Q
from alloy.fit import grad as G
from alloy.fit.batch import TwinBatch
from alloy.fit.errors import FitError
from alloy.fit.problem import OBJECTIVE_STATS, FitSetup, prepare
from alloy.fit.refine import refine
from alloy.fit.search import Budget, SearchOutcome, run_cma
from alloy.fit.types import FitConfig, FitProblem, FitResult
from alloy.fit.validate import Validation, validate

log = logging.getLogger("alloy.fit")
SPREAD_Q = (2.5, 97.5)


def fit_id(problem: FitProblem, cfg: FitConfig, solver: str, device: str) -> str:
    key = {"law": problem.law.id, "dataset": problem.data.sha256, "objective": problem.objective,
           "params": [dataclasses.asdict(p) for p in problem.params], "cfg": dataclasses.asdict(cfg),
           "solver": solver, "device": device, "conditions": list(problem.conditions)}
    return "fit-" + digest_of(key)[:12]


def search(setup: FitSetup, cfg: FitConfig, device: str, budget: Budget) -> SearchOutcome:
    """Dispatch on `cfg.method`.

    cma: CMA-ES. grad: Newton autodiff where a twin passes the gradient check (none does at this pin, grad.py), so
    CMA-ES with the reason in `stop`. hybrid: CMA-ES, then Gauss-Newton with batched finite-difference Jacobians
    (refine.py) for up to `cfg.grad_steps` iterations."""
    if cfg.method not in ("cma", "grad", "hybrid"):
        raise FitError(f"unknown fit method {cfg.method!r} (cma | grad | hybrid)")
    batch = TwinBatch(setup, device, cfg.worlds, cfg.seed)
    out = run_cma(setup.space, batch.evaluate, cfg, len(setup.targets.conditions), budget, setup.start)
    if cfg.method == "grad":
        reason = G.unsupported_reason(setup)
        log.warning("fit %s: method grad runs cma: %s", setup.spec.id, reason)
        out.stop.append(f"grad unavailable: {reason}")
    elif cfg.method == "hybrid" and math.isfinite(out.best_f):
        targets = [t for t in setup.targets.targets if t.stat in OBJECTIVE_STATS[setup.objective]]
        refine(setup.space, batch.evaluate, out, targets, cfg.grad_steps, budget)
    return out


def spread(population: np.ndarray, value: float, column: int) -> tuple[float, float]:
    col = population[:, column] if population.size else np.array([value])
    col = col[np.isfinite(col)]
    if col.size == 0:
        return value, value
    lo, hi = np.percentile(col, SPREAD_Q)
    return min(float(lo), value), max(float(hi), value)


def _params(setup: FitSetup, out: SearchOutcome, fid: str, device: str) -> dict[str, Q]:
    t = setup.targets
    step = setup.spec.dt / setup.spec.substeps
    dt_note = "" if setup.spec.production_dt else f" (fit_dt_only: solver step {step:.6g} s)"
    note = (f"{fid}: {t.law_id} on {setup.solver}/{device}{dt_note}, dataset {t.dataset_id} sha256 "
            f"{t.dataset_sha[:12]}, "
            f"{t.n_calib} calib trials, objective {setup.objective} = {out.best_f:.4g}, {out.evaluations} "
            f"evaluations; unc = {SPREAD_Q[0]}-{SPREAD_Q[1]} % spread of the final population")
    return {leaf: Q(float(out.best_x[j]), "fit", spread(out.population, float(out.best_x[j]), j), note)
            for j, leaf in enumerate(setup.space.leaves)}


def _report(setup: FitSetup, out: SearchOutcome, val: Validation, cfg: FitConfig, device: str) -> dict[str, Any]:
    return {"schema": "alloy.fit.report/1", "law": setup.spec.id, "solver": setup.solver, "device": device,
            "method": out.method, "objective": setup.objective, "best_objective": out.best_f,
            "start": dict(setup.start), "dt": setup.spec.dt, "substeps": setup.spec.substeps,
            "fit_dt_only": not setup.spec.production_dt,
            "residuals": dict(out.best_score.residuals) if out.best_score else {},
            "stop": list(out.stop), "worlds": out.worlds, "search_wall_s": out.wall_s, **out.extra,
            "history": [dataclasses.asdict(g) for g in out.history], "config": dataclasses.asdict(cfg),
            "dataset": {"id": setup.targets.dataset_id, "sha256": setup.targets.dataset_sha},
            "validation": {"verdicts": {v.name: v.verdict for v in val.result.verdicts}, "bounds": val.bounds,
                           "wall_s": val.result.wall_s, "config": dict(val.result.config)}}


def fit(problem: FitProblem, cfg: FitConfig, device: str) -> FitResult:
    """Fit the problem's leaves on calib trials, then validate once on valid trials."""
    t0 = time.perf_counter()
    budget = Budget(cfg.wall_budget_s)
    setup = prepare(problem, cfg)
    out = search(setup, cfg, device, budget)
    if not math.isfinite(out.best_f) or out.best_score is None:
        raise FitError(f"{problem.law.id}: no finite candidate in {out.evaluations} evaluations ({out.stop})")
    missing = sorted(k for k, r in out.best_score.residuals.items() if not math.isfinite(r))
    if missing:
        raise FitError(f"{problem.law.id}: even the best of {out.evaluations} candidates did not produce {missing} "
                       "(NaN or missing simulated value); the twin cannot judge these targets")
    values: Mapping[str, float] = {f: float(v) for f, v in zip(setup.fields, out.best_x)}
    val = validate(setup, problem.data, problem.conditions, values, device=device, seed=cfg.seed,
                   search_errors=out.best_score.errors)
    fid = fit_id(problem, cfg, setup.solver, device)
    log.info("fit %s %s: %s (objective %.4g, %d evaluations, %.1f s)", fid, problem.law.id,
             {k: round(v, 6) for k, v in values.items()}, out.best_f, out.evaluations, time.perf_counter() - t0)
    return FitResult(_params(setup, out, fid, device), val.calib_error, val.valid_error, val.heldout_gap,
                     tuple(val.result.verdicts), time.perf_counter() - t0, out.evaluations, fid,
                     _report(setup, out, val, cfg, device))


__all__ = ["fit", "fit_id", "search", "spread"]
