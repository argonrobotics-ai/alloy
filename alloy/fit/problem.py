# SPDX-License-Identifier: Apache-2.0
"""Fit setup: check a FitProblem, read the `calib` split once into targets, choose the solver and run settings.

The split guard (SPEC §3.12): `prepare` is the only code that reads trials before the final validation, and it reads
`data.split("calib")` only. The search receives a `FitSetup`, which holds targets (numbers) and no dataset. Fit
twins (duck-typed hooks, documented in alloy.fit.twins) declare `fit_fields` and accept `bind_values`.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.bench import compare as C
from alloy.bench.types import Condition, LawSpec, RigCard, RigTrial, Stat, Tolerance
from alloy.fit.errors import FitError
from alloy.fit.params import ParamSpace
from alloy.fit.types import FitConfig, FitProblem

log = logging.getLogger("alloy.fit")
CURVE_POINTS = 50
OBJECTIVE_STATS: Mapping[str, frozenset[str]] = {
    "law_error": frozenset({"mean", "curve", "rate", "distribution"}),
    "nrmse": frozenset({"curve"}),
    "w1": frozenset({"distribution"}),
}


@dataclass(frozen=True)
class Target:
    """What the calib trials of one condition say about one metric."""

    metric: str
    condition: str
    stat: Stat
    tol: Tolerance
    real: np.ndarray                                  # calib scalars of this (metric, condition), in trial order
    u_instr: float = 0.0
    curve: tuple[np.ndarray, np.ndarray] | None = None   # (grid, mean calib curve) for stat == "curve"

    @property
    def key(self) -> str:
        return f"{self.metric}@{self.condition}"

    @property
    def mean(self) -> float:
        return float(np.mean(self.real)) if self.real.size else float("nan")

    @property
    def bound(self) -> float:
        """Tolerance bound in law units (curves and distributions: tol.abs, as the comparators judge them)."""
        return self.tol.abs if self.stat in ("curve", "distribution") else self.tol.bound(self.mean)


@dataclass(frozen=True)
class CalibTargets:
    law_id: str
    dataset_id: str
    dataset_sha: str
    conditions: tuple[Condition, ...]
    targets: tuple[Target, ...]
    n_calib: int


@dataclass(frozen=True)
class FitSetup:
    """Everything the search needs; holds no dataset (the valid split cannot be reached from here)."""

    spec: LawSpec                     # the law with the fit's run settings (duration, worlds per condition)
    space: ParamSpace
    fields: tuple[str, ...]           # twin field per parameter (the leaf's last segment)
    targets: CalibTargets
    solver: str
    objective: str
    twin: Any                         # pristine template; deep-copied per batch
    start: dict[str, float] = field(default_factory=dict)   # leaf -> SI start estimate (twin closed form)


def _u_instr(card: RigCard, metric: str) -> float:
    inst = card.instruments.get(metric, {})
    return float(inst.get("u95", 0.0)) if isinstance(inst, Mapping) else 0.0


def _calib_curve(trials: Sequence[RigTrial], metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    usable = [t for t in trials if metric in t.series and "t" in t.series]
    if not usable:
        return None
    lo = max(float(t.series["t"][0]) for t in usable)
    hi = min(float(t.series["t"][-1]) for t in usable)
    grid = np.linspace(lo, hi, CURVE_POINTS)
    y = np.mean([np.interp(grid, t.series["t"], t.series[metric]) for t in usable], axis=0)
    return grid, y


def _target(spec: LawSpec, card: RigCard, metric: str, cond: Condition, trials: Sequence[RigTrial]) -> Target | None:
    stat: Stat = spec.stat.get(metric, "mean")
    tol = spec.metrics[metric]
    real = np.array([t.scalars[metric] for t in trials if metric in t.scalars], np.float64)
    curve = _calib_curve(trials, metric) if stat == "curve" else None
    if stat == "curve" and curve is None:
        return None
    if stat != "curve" and real.size == 0:
        return None
    return Target(metric, cond.id, stat, tol, real, _u_instr(card, metric), curve)


def calib_targets(spec: LawSpec, data: Any, only: Sequence[str] = ()) -> CalibTargets:
    """Targets per (metric, condition) from the calib split only; `only` restricts the conditions."""
    calib = data.split("calib")
    conds = tuple(c for c in data.conditions if not only or c.id in only)
    missing = sorted(set(only) - {c.id for c in conds})
    if missing:
        raise FitError(f"{spec.id}: conditions {missing} are not in dataset {data.card.dataset_id}")
    out = []
    for cond in conds:
        trials = [t for t in calib if t.condition_id == cond.id]
        for metric in spec.metrics:
            tgt = _target(spec, data.card, metric, cond, trials)
            if tgt is not None:
                out.append(tgt)
    if not out:
        raise FitError(f"{spec.id}: dataset {data.card.dataset_id} has no calib trials for the fit's conditions")
    used = {t.condition for t in out}
    n_used = sum(1 for t in calib if t.condition_id in used)
    return CalibTargets(spec.id, data.card.dataset_id, data.sha256, tuple(c for c in conds if c.id in used),
                        tuple(out), n_used)


def _fields(twin: Any, space: ParamSpace) -> tuple[str, ...]:
    can = getattr(twin, "fit_fields", None)
    hooks = [h for h in ("bind_values", "fix_values") if not callable(getattr(twin, h, None))]
    if can is None or hooks:
        raise FitError(f"twin {type(twin).__name__} is not a fit twin (needs fit_fields, bind_values and "
                       "fix_values; see alloy.fit.twins.base)")
    fields = tuple(r.field for r in space.refs)
    for ref in space.refs:
        if ref.field not in can:
            raise FitError(f"twin {type(twin).__name__} cannot realise leaf {ref.leaf} (fit_fields: {sorted(can)})")
    if len(set(fields)) != len(fields):
        raise FitError(f"two leaves map to the same twin field: {dict(zip(space.leaves, fields))}")
    if getattr(twin, "model", None) is not None or getattr(twin, "worlds", None):
        raise FitError(f"twin {type(twin).__name__} has already built a model; pass a fresh twin (the fitter "
                       "deep-copies it per batch)")
    return fields


def _check_objective(kind: str, targets: CalibTargets) -> None:
    if kind not in OBJECTIVE_STATS:
        raise FitError(f"unknown objective {kind!r} (expected one of {sorted(OBJECTIVE_STATS)})")
    usable = [t for t in targets.targets if t.stat in OBJECTIVE_STATS[kind]]
    if not usable:
        raise FitError(f"objective {kind!r} has no {sorted(OBJECTIVE_STATS[kind])} metric in {targets.law_id}")
    zero = [t.key for t in usable if not t.bound > 0.0]
    if zero:
        raise FitError(f"tolerance bound is 0 for {zero}; a fit needs a positive bound to weigh errors")


def calib_means(targets: CalibTargets) -> dict[str, float]:
    """Mean calib value per scalar metric, averaged over conditions (input of a twin's `initial_values`)."""
    by: dict[str, list[float]] = {}
    for t in targets.targets:
        if t.stat == "mean" and t.real.size:
            by.setdefault(t.metric, []).append(t.mean)
    return {m: float(np.mean(v)) for m, v in by.items()}


def start_values(twin: Any, space: ParamSpace, fields: Sequence[str], targets: CalibTargets) -> dict[str, float]:
    """leaf -> SI start from the twin's optional `initial_values(calib_means)` hook ({} without it)."""
    hook = getattr(twin, "initial_values", None)
    if not callable(hook):
        return {}
    est = hook(calib_means(targets))
    return {leaf: float(est[f]) for leaf, f in zip(space.leaves, fields) if f in est}


def run_spec(spec: LawSpec, cfg: FitConfig) -> LawSpec:
    """The law with the fit's run settings: quick duration, per-condition world count and substeps overrides.

    A substeps override changes the solver step, so the law is then marked `production_dt=False` and every result
    of the fit (search and validation) reports `fit_dt_only` (SPEC §3.10)."""
    changes: dict[str, Any] = {}
    if cfg.quick:
        changes["duration_s"] = float(spec.quick.get("duration_s", spec.duration_s))
    if cfg.per_condition is not None:
        changes["worlds_per_condition"] = int(cfg.per_condition)
    if cfg.substeps is not None and cfg.substeps != spec.substeps:
        if cfg.substeps < 1:
            raise FitError(f"FitConfig.substeps must be >= 1, got {cfg.substeps}")
        changes["substeps"] = int(cfg.substeps)
        changes["production_dt"] = False
    return dataclasses.replace(spec, **changes) if changes else spec


def choose_solver(problem: FitProblem) -> str:
    solver = problem.solver or (problem.law.solvers[0] if problem.law.solvers else "mujoco")
    if problem.law.solvers and solver not in problem.law.solvers:
        raise FitError(f"{problem.law.id} does not run on solver {solver!r} (law solvers: {problem.law.solvers})")
    return solver


def prepare(problem: FitProblem, cfg: FitConfig) -> FitSetup:
    """Validate the problem and read the calib split once (the only trial access before validation)."""
    if problem.law.suite != "rig":
        raise FitError(f"{problem.law.id}: fitting needs a rig law (suite 'rig'), got {problem.law.suite!r}")
    space = ParamSpace(problem.params)
    fields = _fields(problem.twin, space)
    targets = calib_targets(problem.law, problem.data, problem.conditions)
    _check_objective(problem.objective, targets)
    solver = choose_solver(problem)
    log.info("fit %s: %d params %s, %d targets over %d conditions, solver %s, objective %s", problem.law.id,
             space.n, list(space.leaves), len(targets.targets), len(targets.conditions), solver, problem.objective)
    start = start_values(problem.twin, space, fields, targets)
    if start:
        log.info("fit %s: start from the twin's closed form %s", problem.law.id, start)
    return FitSetup(run_spec(problem.law, cfg), space, fields, targets, solver, problem.objective, problem.twin,
                    start)


__all__ = ["Target", "CalibTargets", "FitSetup", "calib_targets", "prepare", "run_spec", "choose_solver",
           "OBJECTIVE_STATS", "calib_means", "start_values"]
