# SPDX-License-Identifier: Apache-2.0
"""Judge a twin run against a rig dataset: per (metric, condition) on the `valid` split (bench §4.3).

Fit vs validate: material values are fitted on `calib` trials only (WP11); here the `calib` split is used solely to
report the error on fitted trials (`fit_error`, BEN-2's RE(calib)), never as validation.
A calibration condition (trials in the `calib` split only, e.g. R10a's calib poses) reports its fit error and gets no
verdict; a condition with no trials at all stays `not_run`.
Instrument uncertainty for metric `m` comes from `card.instruments[m]["u95"]` (0 when absent).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from alloy.bench import compare as C
from alloy.bench.types import Condition, LawSpec, MetricVerdict, RigDataset, RigTrial

CURVE_POINTS = 50


def _trials(data: RigDataset, split: str, cond: str) -> list[RigTrial]:
    return [t for t in data.split(split) if t.condition_id == cond]  # type: ignore[arg-type]


def _u_instr(data: RigDataset, metric: str) -> float:
    inst = data.card.instruments.get(metric, {})
    return float(inst.get("u95", 0.0)) if isinstance(inst, Mapping) else 0.0


def _scalar(spec: LawSpec, metric: str, unit: str, cond: Condition, sim: np.ndarray, data: RigDataset
            ) -> MetricVerdict:
    tol = spec.metrics[metric]
    label = f"{metric}@{cond.id}"
    real_vals = [t.scalars[metric] for t in _trials(data, "valid", cond.id) if metric in t.scalars]
    if not real_vals:
        return C.placeholder_verdict(label, unit, "not_run")
    real = C.summarize(label, unit, real_vals, _u_instr(data, metric))
    return C.compare_scalar(real, C.summarize(label, unit, sim), tol, spec.min_tur)


def _rate(spec: LawSpec, metric: str, cond: Condition, sim: np.ndarray, data: RigDataset) -> MetricVerdict:
    label = f"{metric}@{cond.id}"
    real = [t.scalars[metric] for t in _trials(data, "valid", cond.id) if metric in t.scalars]
    if not real:
        return C.placeholder_verdict(label, "1", "not_run", "rate")
    real_k = int(np.sum(np.asarray(real) > 0.5))
    sim_k = int(np.sum(np.asarray(sim) > 0.5))
    return C.compare_rate(real_k, len(real), sim_k, int(sim.size), spec.metrics[metric].abs, label)


def _distribution(spec: LawSpec, metric: str, unit: str, cond: Condition, sim: np.ndarray,
                  data: RigDataset) -> MetricVerdict:
    label = f"{metric}@{cond.id}"
    real = [t.scalars[metric] for t in _trials(data, "valid", cond.id) if metric in t.scalars]
    if not real:
        return C.placeholder_verdict(label, unit, "not_run", "distribution")
    pts_r = np.asarray(real, np.float64)[:, None]
    return C.compare_distribution(pts_r, np.asarray(sim, np.float64)[:, None], spec.metrics[metric].abs, label,
                                  unit)


def _curve(spec: LawSpec, metric: str, unit: str, cond: Condition, world_mask: np.ndarray, data: RigDataset,
           log: Any) -> MetricVerdict:
    label = f"{metric}@{cond.id}"
    trials = [t for t in _trials(data, "valid", cond.id) if metric in t.series and "t" in t.series]
    if not trials:
        return C.placeholder_verdict(label, unit, "not_run", "curve")
    t_sim = log.times()
    lo = max(max(float(t.series["t"][0]) for t in trials), float(t_sim[0]))
    hi = min(min(float(t.series["t"][-1]) for t in trials), float(t_sim[-1]))
    grid = np.linspace(lo, hi, CURVE_POINTS)
    y_real = np.mean([np.interp(grid, t.series["t"], t.series[metric]) for t in trials], axis=0)
    y_sim = np.mean(np.asarray(log.series(metric))[world_mask], axis=0)
    return C.compare_curve((grid, y_real), (t_sim, y_sim), grid, spec.metrics[metric], label, unit)


def _calib_only(data: RigDataset, cond: Condition) -> bool:
    """A calibration condition: calib trials and no valid ones (judged nowhere, reported as fit error only)."""
    return not _trials(data, "valid", cond.id) and bool(_trials(data, "calib", cond.id))


def _calib_error(metric: str, cond: Condition, sim: np.ndarray, data: RigDataset) -> float | None:
    vals = [t.scalars[metric] for t in _trials(data, "calib", cond.id) if metric in t.scalars]
    return float(abs(np.mean(sim) - np.mean(vals))) if vals and sim.size else None


def judge_rig(spec: LawSpec, twin: Any, values: Mapping[str, np.ndarray], data: RigDataset,
              world_conds: Sequence[str], log: Any) -> tuple[tuple[MetricVerdict, ...], dict[str, float | None]]:
    """Verdicts per (metric, condition) on `valid` trials, plus calib errors keyed `metric@condition`."""
    units: Mapping[str, str] = getattr(twin, "units", {})
    conds = np.asarray(world_conds)
    verdicts: list[MetricVerdict] = []
    calib: dict[str, float | None] = {}
    for metric in spec.metrics:
        unit, stat = units.get(metric, "1"), spec.stat.get(metric, "mean")
        for cond in data.conditions:
            mask, calib_only = conds == cond.id, _calib_only(data, cond)
            if stat == "curve":
                verdicts += [] if calib_only else [_curve(spec, metric, unit, cond, mask, data, log)]
                continue
            if metric not in values:
                verdicts.append(C.placeholder_verdict(f"{metric}@{cond.id}", unit, "error", stat))
                continue
            sim = np.asarray(values[metric], np.float64)[mask]
            calib[f"{metric}@{cond.id}"] = _calib_error(metric, cond, sim, data)
            if calib_only:
                continue
            if stat == "rate":
                verdicts.append(_rate(spec, metric, cond, sim, data))
            elif stat == "distribution":
                # distributions default to metres + PRF-02 when the twin declares no unit (SCR-WP2-8)
                verdicts.append(_distribution(spec, metric, units.get(metric, "m"), cond, sim, data))
            else:
                verdicts.append(_scalar(spec, metric, unit, cond, sim, data))
    return tuple(verdicts), calib


__all__ = ["judge_rig"]
