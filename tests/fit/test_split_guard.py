# SPDX-License-Identifier: Apache-2.0
"""The fitter never reads `valid` trials before the final validation call (SPEC §7 WP11 acceptance).

The dataset's valid trials raise SplitError on any field but `split` while the guard is armed; the guard disarms
only when `alloy.fit.fitter.validate` is entered. A fit that completes therefore touched no valid data during
preparation and search.
"""

from __future__ import annotations

import pytest

from alloy.bench.twins.bounce import LAW as R4
from alloy.bench.types import Condition
from alloy.fit import fitter
from alloy.fit.errors import SplitError
from alloy.fit.problem import prepare, run_spec
from alloy.fit.search import Budget
from alloy.fit.twins import FitBounceTwin
from alloy.fit.types import FitConfig, FitParam, FitProblem
from alloy.testing.fit import SyntheticSpec, disarm_on, guarded, twin_dataset

CFG = FitConfig(worlds=8, generations=2, quick=True, per_condition=1)
PARAMS = (FitParam("material:steel_carbon.surface.restitution", 0.05, 0.99),)


def _guarded_problem():
    spec = run_spec(R4, CFG)
    data = twin_dataset(spec, FitBounceTwin(path="native"), {"restitution": 0.6},
                        (Condition("drop10cm", {"drop_height": 0.10}),),
                        SyntheticSpec({"restitution": 0.005}, trials_per_condition=6, calib_per_condition=3),
                        solver="xpbd", device="cpu")
    data, guard = guarded(data)
    return FitProblem(R4, FitBounceTwin(path="native"), data, PARAMS, solver="xpbd"), guard


def test_search_never_touches_valid_trials():
    problem, guard = _guarded_problem()
    setup = prepare(problem, CFG)
    out = fitter.search(setup, CFG, "cpu", Budget(600.0))
    assert out.evaluations > 0 and guard.touched == [] and guard.armed


def test_fit_reads_valid_only_inside_validate(monkeypatch, record_row):
    problem, guard = _guarded_problem()
    monkeypatch.setattr(fitter, "validate", disarm_on(fitter.validate, guard))
    res = fitter.fit(problem, CFG, "cpu")
    assert guard.touched == [] and not guard.armed          # disarmed by validate, nothing read while armed
    assert res.valid_error and all(v.verdict != "not_run" for v in res.verdicts)
    record_row("WP11-split-guard", 0.0, "pass", touched_before_validate=len(guard.touched))


def test_armed_guard_fails_the_validation_not_the_search():
    problem, guard = _guarded_problem()
    with pytest.raises(SplitError):
        fitter.fit(problem, CFG, "cpu")                     # the first valid read is in run_law's judge
    assert guard.touched and all(t.startswith("drop10cm#") for t in guard.touched)
