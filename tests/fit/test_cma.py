# SPDX-License-Identifier: Apache-2.0
"""alloy.fit.cma: CMA-ES in the unit box."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.fit.cma import CMAES, CMAConstants, default_popsize, mirror


def _run(f, mean, sigma=0.3, lam=10, gens=80, seed=0):
    es = CMAES(np.asarray(mean, float), sigma, lam, seed)
    for _ in range(gens):
        x = es.ask()
        es.tell(x, np.array([f(v) for v in x]))
        if es.stop_reason():
            break
    return es


def test_mirror_reflects_into_box():
    np.testing.assert_allclose(mirror(np.array([-0.25, 0.3, 1.25, 2.5, -1.5])), [0.25, 0.3, 0.75, 0.5, 0.5])


def test_constants_match_hansen_defaults():
    k = CMAConstants.make(2, 6)          # n = 2, lambda = 6 -> mu = 3
    w = np.log(3.5) - np.log([1, 2, 3])
    np.testing.assert_allclose(k.weights, w / w.sum())
    assert k.mueff == pytest.approx(1 / np.sum((w / w.sum()) ** 2))
    assert k.cs == pytest.approx((k.mueff + 2) / (2 + k.mueff + 5))
    assert default_popsize(2) == 6 and default_popsize(1) == 4


def test_converges_on_shifted_sphere_inside_box():
    target = np.array([0.31, 0.77])
    es = _run(lambda v: float(np.sum((v - target) ** 2)), [0.5, 0.5])
    np.testing.assert_allclose(es.mean, target, atol=1e-4)


def test_converges_on_rotated_ellipsoid():
    rot = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    scale = np.array([1.0, 100.0])
    target = np.array([0.4, 0.6])
    es = _run(lambda v: float(np.sum(scale * (rot @ (v - target)) ** 2)), [0.2, 0.2], gens=150)
    np.testing.assert_allclose(es.mean, target, atol=1e-3)


def test_optimum_on_the_boundary_is_reached():
    es = _run(lambda v: float(np.sum((v - np.array([1.2, -0.3])) ** 2)), [0.5, 0.5], gens=120)
    np.testing.assert_allclose(es.mean, [1.0, 0.0], atol=2e-2)


def test_same_seed_same_path_and_inject_keeps_draw():
    a, b = CMAES(np.array([0.5]), 0.2, 6, 3), CMAES(np.array([0.5]), 0.2, 6, 3)
    xa, xb = a.ask(), b.ask(inject=np.array([0.1]))
    assert xb[0, 0] == 0.1
    np.testing.assert_array_equal(xa[1:], xb[1:])


def test_nan_fitness_ranks_last():
    es = CMAES(np.array([0.5, 0.5]), 0.2, 8, 0)
    x = es.ask()
    f = np.arange(8.0)
    f[0] = np.nan
    es.tell(x, f)                         # must not raise; NaN treated as worst
    assert np.all(np.isfinite(es.mean))


def test_stops_on_tolx():
    es = _run(lambda v: float(np.sum((v - 0.5) ** 2)), [0.4], gens=400)
    assert es.stop_reason() in ("tolx", "tolfun")
