# SPDX-License-Identifier: Apache-2.0
"""Final validation: the one call that reads `valid` trials (SPEC §3.12, BEN-2).

The fitted values are fixed on a fresh copy of the twin (`fix_values`), and WP6's `run_law` judges the law on the
`valid` split (`judge_rig`), so the verdicts are exactly what the bench would report for the fitted card. Errors
per `metric@condition` in law units, the same definition on both sides: scalars |sim mean - real mean| (valid: the
verdict error; calib: `judge_rig`'s calib error); rates |sim rate - real rate| (valid: from the verdict's rate
quantities, not its distance outside the Wilson interval); curves NRMSE and distributions W1 (calib: the search's
error for the same target). held-out gap = valid - calib. A judged verdict whose error is not finite is kept as NaN
so `write_back` refuses it.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from alloy.bench.runner import run_law
from alloy.bench.types import LawResult, MetricVerdict, RigDataset
from alloy.fit.errors import FitError, SplitError
from alloy.fit.problem import FitSetup

log = logging.getLogger("alloy.fit")


@dataclass(frozen=True)
class Validation:
    result: LawResult
    calib_error: dict[str, float]
    valid_error: dict[str, float]
    heldout_gap: dict[str, float]
    bounds: dict[str, float]


def restrict(data: RigDataset, conditions: Sequence[str]) -> RigDataset:
    """The dataset limited to `conditions` (all when empty); same card and sha256 (the data on disk is unchanged)."""
    if not conditions:
        return data
    keep = set(conditions)
    return dataclasses.replace(data, conditions=tuple(c for c in data.conditions if c.id in keep),
                               trials=tuple(t for t in data.trials if t.condition_id in keep))


def _valid_error(v: MetricVerdict) -> float:
    if v.stat == "rate":
        return abs(v.sim.mean - v.real.mean) if v.real is not None else math.nan
    return float(v.error)


def _calib_error(v: MetricVerdict, calib: Mapping[str, float | None], search_errors: Mapping[str, float]) -> float:
    c = calib.get(v.name) if v.stat in ("mean", "rate") else None   # judge_rig's calib error is |delta mean|
    c = search_errors.get(v.name) if c is None else c
    return float(c) if c is not None else math.nan


def held_out(verdicts: Sequence[MetricVerdict], calib: Mapping[str, float | None],
             search_errors: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """(calib error, valid error, gap) per judged verdict name (not_run excluded); NaN where not computable."""
    judged = [v for v in verdicts if v.verdict != "not_run"]
    valid = {v.name: _valid_error(v) for v in judged}
    cal = {v.name: _calib_error(v, calib, search_errors) for v in judged}
    gap = {n: valid[n] - cal[n] for n in valid}
    return cal, valid, gap


def validate(setup: FitSetup, data: RigDataset, conditions: Sequence[str], values: Mapping[str, float], *,
             device: str, seed: int, search_errors: Mapping[str, float]) -> Validation:
    """Run the law with the fitted `values` (twin field -> SI) on the valid split; raise FitError on a run error."""
    twin = copy.deepcopy(setup.twin)
    twin.fix_values(dict(values))
    res = run_law(setup.spec, twin, restrict(data, conditions), device=device, seed=seed, solver=setup.solver)
    if any(v.verdict == "error" for v in res.verdicts):
        msg = str(res.monitors.get("error", "error verdicts"))
        # run_law turns exceptions into an `error` verdict with "<Type>: <text>"; keep a split violation's type
        cls = SplitError if msg.startswith(f"{SplitError.__name__}:") else FitError
        raise cls(f"{setup.spec.id}: validation run failed: {msg}")
    cal, valid, gap = held_out(res.verdicts, res.monitors.get("calib_error", {}), search_errors)
    bounds = {v.name: float(v.bound) for v in res.verdicts}
    log.info("fit validation %s: %s", setup.spec.id, {k: (valid[k], gap[k]) for k in valid})
    return Validation(res, cal, valid, gap, bounds)


__all__ = ["Validation", "validate", "restrict", "held_out"]
