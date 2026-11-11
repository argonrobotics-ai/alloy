# SPDX-License-Identifier: Apache-2.0
"""Comparators (SPEC §3.10, bench §4.3) against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.bench import compare as C
from alloy.bench.types import Quantity, Tolerance

TOL = Tolerance(abs=0.1, rel=0.05, source="test")  # bound(10) = 0.1 + 0.5 = 0.6


def q(mean: float, std: float = 0.0, n: int = 20, u95: float = 0.0) -> Quantity:
    return Quantity("f", "N", mean, std, n, u95)


def test_summarize_u95_formula() -> None:
    # U95 = sqrt(u_instr^2 + (t_0.975(n-1) s / sqrt n)^2); n = 4, s = std([1,2,3,4], ddof=1) = 1.29099
    got = C.summarize("f", "N", [1.0, 2.0, 3.0, 4.0], u_instr=0.5)
    s = np.std([1, 2, 3, 4], ddof=1)
    assert got.mean == 2.5 and got.n == 4 and got.std == pytest.approx(s)
    assert got.u95 == pytest.approx(math.hypot(0.5, 3.182446305284263 * s / 2.0))
    assert C.summarize("f", "N", [3.0]).u95 == 0.0 and C.summarize("f", "N", []).n == 0


def test_scalar_pass_fail_edges() -> None:
    real = q(10.0, u95=0.06)  # TUR = 10
    assert C.compare_scalar(real, q(10.599), TOL).verdict == "pass"
    assert C.compare_scalar(real, q(10.601), TOL).verdict == "fail"
    assert C.compare_scalar(real, q(9.399), TOL).verdict == "fail"
    v = C.compare_scalar(real, q(10.3), TOL)
    assert v.error == pytest.approx(0.3) and v.bound == pytest.approx(0.6) and v.tur == pytest.approx(10.0)


def test_scalar_spread_rule() -> None:
    real = q(10.0, std=0.1, u95=0.06)  # max(2 * 0.1, 0.6 / 2) = 0.3
    assert C.compare_scalar(real, q(10.0, std=0.299), TOL).verdict == "pass"
    assert C.compare_scalar(real, q(10.0, std=0.301), TOL).verdict == "fail"


def test_tur_below_three_is_inconclusive() -> None:
    assert C.compare_scalar(q(10.0, u95=0.6 / 2.9), q(10.0), TOL).verdict == "inconclusive"
    assert C.compare_scalar(q(10.0, u95=0.6 / 3.1), q(10.0), TOL).verdict == "pass"
    assert C.compare_scalar(q(10.0, u95=0.6 / 2.9), q(10.0), TOL, min_tur=2.0).verdict == "pass"


def test_scalar_nonfinite_fails() -> None:
    assert C.compare_scalar(q(10.0, u95=0.01), q(math.nan), TOL).verdict == "fail"


def test_curve_nrmse() -> None:
    x = np.linspace(0, 1, 11)
    y = 2.0 * x + 1.0  # max|y| = 3; offset 0.15 -> NRMSE 0.05
    tol = Tolerance(abs=0.05, source="t")
    assert C.compare_curve((x, y), (x, y + 0.149), x, tol, "c", "N").verdict == "pass"
    assert C.compare_curve((x, y), (x, y + 0.151), x, tol, "c", "N").verdict == "fail"
    assert C.nrmse(y, y + 0.15) == pytest.approx(0.05)
    short = C.compare_curve((x, y), (x[:5], y[:5]), x, tol, "c", "N")
    assert short.verdict == "fail" and math.isnan(short.error)


def test_distribution_w1_and_prf02() -> None:
    a = np.array([[0.0, 0.0], [1e-3, 0.0], [2e-3, 0.0]])
    w1, d = C.assignment_w1(a, a[::-1])
    assert w1 == 0.0 and d.size == 3
    w1, _ = C.assignment_w1(a, a + [5e-4, 0.0])
    assert w1 == pytest.approx(5e-4)
    assert C.assignment_w1(a, a[:1])[0] == pytest.approx(1e-3)  # unequal sizes: replicated to lcm
    assert C.compare_distribution(a, a + [5e-4, 0.0], 6e-4, "d").verdict == "pass"
    assert C.compare_distribution(a, a + [5e-4, 0.0], 4e-4, "d").verdict == "fail"
    assert not C.prf02_ok(np.array([0.0, 0.0, 0.03]))  # one pose beyond 25 mm


def test_distribution_unit_skips_prf02_off_metres() -> None:
    # SCR-WP2-8: a node count shifted by one node passes W1 <= 1 in unit "1" but fails PRF-02 when read as metres
    n = np.array([[6.0], [7.0], [7.0], [8.0]])
    v = C.compare_distribution(n, n + 1.0, 1.0, "nodes", "1")
    assert v.verdict == "pass" and v.unit == "1" and v.real.unit == "1" and v.sim.unit == "1" and v.error == 1.0
    assert C.compare_distribution(n, n + 1.0, 1.0, "nodes").verdict == "fail"  # default metres: PRF-02 median 1 m
    assert C.compare_distribution(n, n + 1.0, 0.999, "nodes", "1").verdict == "fail"
    assert C.compare_distribution(n, np.full_like(n, np.nan), 1.0, "nodes", "1").verdict == "fail"


def test_wilson_and_rate() -> None:
    lo, hi = C.wilson_interval(5, 20)
    assert lo == pytest.approx(0.1118610, abs=1e-6) and hi == pytest.approx(0.4687, abs=1e-4)
    assert C.wilson_interval(0, 10)[0] == 0.0
    assert C.compare_rate(5, 20, 9, 20, 0.0, "r").verdict == "pass"   # 0.45 inside
    assert C.compare_rate(5, 20, 10, 20, 0.0, "r").verdict == "fail"  # 0.5 above 0.4687
    assert C.compare_rate(5, 20, 10, 20, 0.05, "r").verdict == "pass"


def test_compare_errors_worst_world() -> None:
    tol = Tolerance(abs=1.0, rel=0.1, source="t")
    v = C.compare_errors("e", "m", np.array([0.5, 1.5]), tol, ref=np.array([1.0, 10.0]))  # bounds 1.1, 2.0
    # ratios 0.5/1.1 and 1.5/2.0: the second world is the worst
    assert v.verdict == "pass" and v.error == 1.5 and v.bound == pytest.approx(2.0) and v.real is None
    v = C.compare_errors("e", "m", np.array([0.5, 2.5]), tol, ref=np.array([1.0, 10.0]))
    assert v.verdict == "fail" and v.error == 2.5
    assert C.compare_errors("e", "m", np.array([math.nan]), tol).verdict == "fail"
    assert C.compare_errors("e", "m", np.zeros(0), tol).verdict == "error"


def test_placeholder_never_passes() -> None:
    assert C.placeholder_verdict("x", "1", "not_run").verdict == "not_run"
    with pytest.raises(ValueError):
        C.placeholder_verdict("x", "1", "pass")
