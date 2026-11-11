# SPDX-License-Identifier: Apache-2.0
"""alloy.fit.search: CMA-ES generations over an evaluator, restarts, wall budget, best tracking."""

from __future__ import annotations

import time

import numpy as np
import pytest

from alloy.fit.batch import BatchResult
from alloy.fit.objective import Score
from alloy.fit.params import ParamSpace
from alloy.fit.search import Budget, popsize_for, run_cma
from alloy.fit.types import FitConfig, FitParam

SPACE = ParamSpace((FitParam("pair:a__b.mu_static", 0.0, 2.0), FitParam("pair:a__b.mu_dynamic", 0.0, 2.0)))
TRUTH = np.array([0.42, 0.36])


def _evaluate(x: np.ndarray, sleep: float = 0.0) -> BatchResult:
    if sleep:
        time.sleep(sleep)
    f = np.sum(((x - TRUTH) / 0.01) ** 2, axis=1)
    scores = tuple(Score(float(v), {}, {}) for v in f)
    return BatchResult(x, f, scores, len(x), sleep)


def test_popsize_follows_the_world_budget():
    assert popsize_for(FitConfig(worlds=1024), 2, 2) == 512
    assert popsize_for(FitConfig(worlds=2), 2, 1) == 6           # never below 4 + 3 ln n


def test_recovers_truth_and_records_history():
    out = run_cma(SPACE, _evaluate, FitConfig(worlds=16, generations=60), 1, Budget(60.0))
    np.testing.assert_allclose(out.best_x, TRUTH, rtol=2e-3)
    assert out.evaluations == sum(g.popsize for g in out.history)
    assert out.history[0].run == 0 and out.stop and out.population.shape[1] == 2


def test_start_point_is_evaluated_first():
    seen = []

    def ev(x):
        seen.append(x.copy())
        return _evaluate(x)

    out = run_cma(SPACE, ev, FitConfig(worlds=8, generations=1), 1, Budget(60.0),
                  start={"pair:a__b.mu_static": 0.42, "pair:a__b.mu_dynamic": 0.36})
    np.testing.assert_allclose(seen[0][0], TRUTH)
    assert out.best_f == pytest.approx(0.0, abs=1e-18)


def test_restarts_double_lambda_and_unc_comes_from_the_best_run():
    out = run_cma(SPACE, _evaluate, FitConfig(worlds=8, generations=3, restarts=2), 1, Budget(60.0))
    assert {g.popsize for g in out.history if g.run == 0} == {8}
    assert {g.popsize for g in out.history if g.run == 1} == {16}
    assert len(out.population) == (8 if out.best_run == 0 else 16)


def test_wall_budget_stops_the_search():
    out = run_cma(SPACE, lambda x: _evaluate(x, 0.05), FitConfig(worlds=8, generations=50), 1, Budget(0.12))
    assert len(out.history) < 5 and any("wall_budget" in s for s in out.stop)
