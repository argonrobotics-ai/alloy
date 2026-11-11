# SPDX-License-Identifier: Apache-2.0
"""alloy.fit.objective: normalised error terms per stat, NaN penalty, objective kinds."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench.types import Tolerance
from alloy.fit.objective import NAN_PENALTY, SimOutcome, score
from alloy.fit.problem import Target


def _t(metric, stat, real, tol, curve=None, cond="c0"):
    return Target(metric, cond, stat, tol, np.asarray(real, float), 0.0, curve)


def test_scalar_term_uses_comparator_bound():
    t = _t("e", "mean", [0.5, 0.7], Tolerance(abs=0.01, rel=0.1))      # mean 0.6, bound 0.01 + 0.06 = 0.07
    s = score([t], {"c0": SimOutcome({"e": np.array([0.74, 0.74])})}, "law_error")
    assert s.residuals["e@c0"] == pytest.approx(0.14 / 0.07)
    assert s.objective == pytest.approx(4.0)
    assert s.errors["e@c0"] == pytest.approx(0.14)


def test_rate_term():
    t = _t("jam", "rate", [1, 0, 0, 0], Tolerance(abs=0.1))           # real rate 0.25, scale max(0.1, 1/4)
    s = score([t], {"c0": SimOutcome({"jam": np.array([1.0, 1.0, 0.0, 0.0])})}, "law_error")
    assert s.residuals["jam@c0"] == pytest.approx((0.5 - 0.25) / 0.25)


def test_distribution_term_is_w1_over_tol():
    t = _t("x", "distribution", [0.0, 1.0], Tolerance(abs=0.5))
    s = score([t], {"c0": SimOutcome({"x": np.array([0.2, 1.2])})}, "law_error")
    assert s.errors["x@c0"] == pytest.approx(0.2)                     # matched shift 0.2
    assert s.residuals["x@c0"] == pytest.approx(0.4)


def test_curve_term_is_nrmse_over_tol():
    grid = np.linspace(0.0, 1.0, 5)
    t = _t("f", "curve", [], Tolerance(abs=0.1), curve=(grid, 2.0 * np.ones(5)))
    times = np.linspace(0.0, 1.0, 11)
    out = SimOutcome({}, times, {"f": np.full((2, 11), 2.2)})
    s = score([t], {"c0": out}, "law_error")
    assert s.errors["f@c0"] == pytest.approx(0.1)                     # RMS 0.2 / max|real| 2
    assert s.residuals["f@c0"] == pytest.approx(1.0)
    short = SimOutcome({}, np.linspace(0.0, 0.5, 6), {"f": np.full((1, 6), 2.0)})   # rig record runs longer:
    assert score([t], {"c0": short}, "law_error").errors["f@c0"] == pytest.approx(0.0)   # grid points <= 0.5 only
    tiny = SimOutcome({}, np.linspace(0.0, 0.1, 3), {"f": np.full((1, 3), 2.0)})     # < 2 grid points inside
    assert math.isnan(score([t], {"c0": tiny}, "law_error").errors["f@c0"])


def test_missing_or_nan_adds_penalty_and_kind_filters():
    ts = [_t("e", "mean", [0.5], Tolerance(abs=0.1)), _t("x", "distribution", [0.0], Tolerance(abs=1.0))]
    s = score(ts, {"c0": SimOutcome({"e": np.array([np.nan]), "x": np.array([0.5])})}, "law_error")
    assert s.objective == pytest.approx(NAN_PENALTY + 0.25)
    w1 = score(ts, {"c0": SimOutcome({"e": np.array([np.nan]), "x": np.array([0.5])})}, "w1", prior=2.0)
    assert set(w1.errors) == {"x@c0"} and w1.objective == pytest.approx(2.25)
    none = score(ts, {}, "law_error")                                  # condition not simulated
    assert none.objective == pytest.approx(2 * NAN_PENALTY)
