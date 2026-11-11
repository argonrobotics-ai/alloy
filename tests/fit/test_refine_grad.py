# SPDX-License-Identifier: Apache-2.0
"""alloy.fit.refine (hybrid: CMA-ES + finite-difference Gauss-Newton) and alloy.fit.grad (autodiff admission)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench.types import Tolerance
from alloy.fit import grad as G
from alloy.fit.batch import BatchResult
from alloy.fit.objective import Score
from alloy.fit.params import ParamSpace
from alloy.fit.problem import Target
from alloy.fit.refine import lm_step, linearise, refine, stencil, unc95
from alloy.fit.search import Budget, SearchOutcome
from alloy.fit.types import FitParam

SPACE = ParamSpace((FitParam("pair:a__b.mu_static", 0.0, 2.0), FitParam("pair:a__b.mu_dynamic", 0.0, 2.0)))
TRUTH = np.array([0.42, 0.36])
BOUND = 0.01
# two targets: sim_1 = mu_s, sim_2 = mu_s + 2 mu_d (a coupled linear model; Gauss-Newton solves it in one step)
TARGETS = (Target("a", "c", "mean", Tolerance(abs=BOUND, source="t"), np.full(10, TRUTH[0]), 0.0),
           Target("b", "c", "mean", Tolerance(abs=BOUND, source="t"), np.full(10, TRUTH[0] + 2 * TRUTH[1]), 0.0))


def _sim(x: np.ndarray) -> np.ndarray:
    return np.stack([x[:, 0], x[:, 0] + 2 * x[:, 1]], axis=1)


def _evaluate(x: np.ndarray) -> BatchResult:
    x = np.atleast_2d(x)
    r = (_sim(x) - np.array([t.mean for t in TARGETS])) / BOUND
    scores = tuple(Score(float(np.sum(ri ** 2)), {}, {"a@c": float(ri[0]), "b@c": float(ri[1])}) for ri in r)
    return BatchResult(x, np.array([s.objective for s in scores]), scores, len(x), 0.0)


def _outcome(x0) -> SearchOutcome:
    out = SearchOutcome(np.asarray(x0, float), math.inf, None, np.zeros((0, 2)), np.zeros(0))
    out.improve(_evaluate(np.asarray(x0, float)[None, :]))
    return out


def test_stencil_shape_and_reflection_at_the_box():
    s = stencil(np.array([0.0, 0.5]), h=0.1)
    assert s.shape == (5, 2)
    np.testing.assert_allclose(s[1], [0.1, 0.5]) and np.testing.assert_allclose(s[2], [0.0, 0.5])  # clipped


def test_linearise_gives_the_exact_jacobian_of_a_linear_model():
    lin, batch = linearise(SPACE, _evaluate, SPACE.to_unit(TRUTH[None, :])[0] + 0.05, ["a@c", "b@c"])
    # r = A x / BOUND, x = 2 u  ->  dr/du = A * 2 / BOUND
    np.testing.assert_allclose(lin.jac, np.array([[1, 0], [1, 2]]) * 2.0 / BOUND, rtol=1e-6)
    assert len(batch.x) == 5


def test_refine_converges_from_a_cma_estimate():
    out = refine(SPACE, _evaluate, _outcome([0.5, 0.3]), TARGETS, steps=5, budget=Budget(60.0))
    np.testing.assert_allclose(out.best_x, TRUTH, atol=1e-9)
    assert out.method == "hybrid" and out.best_f < 1e-12
    assert any(g.phase == "refine" for g in out.history) and out.stop[-1].startswith("refine")


def test_lm_rejects_a_bad_step_and_damps():
    lin, _ = linearise(SPACE, _evaluate, np.array([0.3, 0.2]), ["a@c", "b@c"])
    small = lm_step(lin, 1e6)
    big = lm_step(lin, 1e-9)
    assert np.linalg.norm(small) < 1e-3 * np.linalg.norm(big)


def test_unc95_from_the_jacobian():
    noisy = (Target("a", "c", "mean", Tolerance(abs=BOUND, source="t"), TRUTH[0] + np.array([-1, 1] * 5) * 0.003, 0.0),
             Target("b", "c", "mean", Tolerance(abs=BOUND, source="t"),
                    TRUTH[0] + 2 * TRUTH[1] + np.array([-1, 1] * 5) * 0.003, 0.0))
    lin, _ = linearise(SPACE, _evaluate, SPACE.to_unit(TRUTH[None, :])[0], ["a@c", "b@c"])
    u = unc95(SPACE, lin, noisy)
    lo, hi = u["pair:a__b.mu_static"]
    # sigma_t = U95/1.96 with U95 = t975(10) * s / sqrt(10); mu_s is observed directly by target a
    from alloy.bench.compare import summarize
    sig = summarize("a", "1", noisy[0].real).u95 / 1.96
    assert hi - lo == pytest.approx(2 * 1.96 * sig, rel=0.05)
    assert lo < TRUTH[0] < hi


def test_grad_falls_back_with_the_measured_reason():
    class Setup:
        solver, twin = "mujoco", object()
    assert "not differentiable" in G.unsupported_reason(Setup())
    Setup.solver = "xpbd"
    assert "fail the finite-difference check" in G.unsupported_reason(Setup())


def test_gradient_check():
    ok, tape, fd = G.gradient_check(lambda x: (x ** 3, 3 * x ** 2), 0.7, 1e-4)
    assert ok and tape == pytest.approx(fd, rel=1e-6)
    ok, _, _ = G.gradient_check(lambda x: (x ** 3, 0.2 * x ** 2), 0.7, 1e-4)   # a 5x-too-small tape gradient
    assert not ok
