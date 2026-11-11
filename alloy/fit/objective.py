# SPDX-License-Identifier: Apache-2.0
"""Fit objectives: score one candidate's simulated outcome against the calib targets (SPEC §3.12).

Each target contributes a normalised error r (law units / bound); the objective is the sum of r^2 over the targets
the objective kind selects (`law_error`: all; `nrmse`: curves; `w1`: distributions) plus the prior z^2 (params.py):
- scalar ("mean"): r = (mean sim - mean calib) / bound, bound = tol.bound(mean calib) as the comparator judges it;
- curve: r = NRMSE(sim mean curve, calib mean curve on the calib grid points inside the simulated time) / tol.abs;
- rate: r = (sim rate - calib rate) / max(tol.abs, 1/n_calib) (rates are indicator scalars > 0.5);
- distribution: r = assignment W1(calib points, sim points) / tol.abs (GPUSimBench W1, alloy.bench.compare).
A target whose simulated value is missing or not finite adds NAN_PENALTY, so CMA-ES ranks it below every finite
candidate instead of crashing; the reported error stays NaN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from alloy.bench import compare as C
from alloy.fit.problem import OBJECTIVE_STATS, Target

NAN_PENALTY = 1.0e6


@dataclass(frozen=True)
class SimOutcome:
    """One candidate x condition: per-world scalars and (for curves) the log times and per-world series."""

    scalars: Mapping[str, np.ndarray]                  # metric -> [worlds of this group]
    times: np.ndarray | None = None
    series: Mapping[str, np.ndarray] | None = None     # metric -> [worlds, T]


@dataclass(frozen=True)
class Score:
    objective: float
    errors: Mapping[str, float]        # target key -> |sim - real| in law units (NRMSE / W1 for curves / dists)
    residuals: Mapping[str, float]     # target key -> signed normalised error r


def _scalar(t: Target, out: SimOutcome) -> tuple[float, float]:
    v = out.scalars.get(t.metric)
    if v is None or np.size(v) == 0:
        return math.nan, math.nan
    sim = float(np.mean(v))
    diff = sim - t.mean
    return abs(diff), diff / t.bound


def _rate(t: Target, out: SimOutcome) -> tuple[float, float]:
    v = out.scalars.get(t.metric)
    if v is None or np.size(v) == 0 or not np.all(np.isfinite(v)):
        return math.nan, math.nan
    real = float(np.mean(t.real > 0.5))
    sim = float(np.mean(np.asarray(v) > 0.5))
    scale = max(t.tol.abs, 1.0 / max(t.real.size, 1))
    return abs(sim - real), (sim - real) / scale


def _distribution(t: Target, out: SimOutcome) -> tuple[float, float]:
    v = out.scalars.get(t.metric)
    if v is None or np.size(v) == 0 or not np.all(np.isfinite(v)):
        return math.nan, math.nan
    try:
        w1, _ = C.assignment_w1(t.real[:, None], np.asarray(v, np.float64)[:, None])
    except ValueError:      # unequal sizes whose least common multiple exceeds the comparator's cap
        return math.nan, math.nan
    return w1, w1 / t.bound


def _curve(t: Target, out: SimOutcome) -> tuple[float, float]:
    if t.curve is None or out.series is None or out.times is None or t.metric not in out.series:
        return math.nan, math.nan
    grid, y_real = t.curve
    y_sim = np.mean(np.asarray(out.series[t.metric], np.float64), axis=0)
    inside = (grid >= out.times[0] - 1e-12) & (grid <= out.times[-1] + 1e-12)   # never extrapolate the sim; the
    if inside.sum() < 2:                                                          # rig record may run longer
        return math.nan, math.nan
    err = C.nrmse(y_real[inside], np.interp(grid[inside], out.times, y_sim))
    return err, err / t.bound


_TERMS = {"mean": _scalar, "rate": _rate, "distribution": _distribution, "curve": _curve}


def score(targets: Sequence[Target], outcomes: Mapping[str, SimOutcome], kind: str, prior: float = 0.0) -> Score:
    """Objective of one candidate; `outcomes` maps condition id -> that condition's SimOutcome."""
    stats = OBJECTIVE_STATS[kind]
    total, errors, resid = float(prior), {}, {}
    for t in targets:
        if t.stat not in stats:
            continue
        out = outcomes.get(t.condition)
        err, r = _TERMS[t.stat](t, out) if out is not None else (math.nan, math.nan)
        errors[t.key], resid[t.key] = err, r
        total += r * r if math.isfinite(r) else NAN_PENALTY
    return Score(total, errors, resid)


__all__ = ["SimOutcome", "Score", "score", "NAN_PENALTY"]
