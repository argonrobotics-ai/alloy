# SPDX-License-Identifier: Apache-2.0
"""Batched CMA-ES (alloy.bully.cma) against closed-form optima."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.errors import BullyError
from alloy.bully.cma import BatchedCMAES, CMAParams


def _run(es: BatchedCMAES, f, gens: int) -> None:
    for _ in range(gens):
        x = es.ask()
        es.tell(f(x))


def test_params_match_hansen_table():
    # Hansen 2016 Table 1 at n = 10, lam = 10: mu = 5, mu_eff = 1/sum(w^2) ~= 3.17
    p = CMAParams.default(10, 10)
    assert p.mu == 5
    assert p.weights.sum() == pytest.approx(1.0)
    assert p.mu_eff == pytest.approx(3.1672, abs=1e-3)
    # chi_n = sqrt(n)(1 - 1/(4n) + 1/(21 n^2)) = 3.0847 at n = 10
    assert p.chi_n == pytest.approx(3.0847, abs=1e-4)
    assert 0.0 < p.c_1 + p.c_mu < 1.0


def test_sphere_converges_inside_box():
    centre = np.linspace(0.2, 0.8, 8)
    es = BatchedCMAES(np.full(8, 0.5), 0.3, lam=16, seed=1)
    _run(es, lambda x: np.sum((x - centre) ** 2, axis=1), 150)
    assert es.state.best_f < 1e-8
    np.testing.assert_allclose(es.state.best_x, centre, atol=1e-4)


def test_box_corner_optimum_is_reached():
    # minimise -sum(x) on [0, 1]^n: optimum at the all-ones corner, value -n
    es = BatchedCMAES(np.full(6, 0.5), 0.3, lam=12, seed=2)
    _run(es, lambda x: -np.sum(x, axis=1), 60)
    assert es.state.best_f == pytest.approx(-6.0, abs=1e-3)
    assert np.all(es.ask() <= 1.0) and np.all(es.ask() >= 0.0)


def test_same_seed_same_search_and_nan_is_worst():
    def f(x):
        out = np.sum((x - 0.3) ** 2, axis=1)
        out[0] = np.nan
        return out

    runs = []
    for _ in range(2):
        es = BatchedCMAES(np.full(4, 0.5), 0.2, lam=8, seed=7)
        _run(es, f, 10)
        runs.append((es.state.mean.copy(), es.state.best_f))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    assert runs[0][1] == runs[1][1] and np.isfinite(runs[0][1])


def test_tell_before_ask_raises():
    es = BatchedCMAES(np.zeros(2), 0.1, lam=4)
    with pytest.raises(BullyError):
        es.tell(np.zeros(4))
