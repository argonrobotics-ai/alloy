# SPDX-License-Identifier: Apache-2.0
"""A16 harness self-test (VAL-02, BEN-1): every comparator on synthetic answers just inside and just outside its
tolerance must give the expected verdict; 100 % or every other suite is blocked (bench §4.5 A16).

Each case is (name, thunk returning a MetricVerdict, expected verdict). "Just inside" = the bound shrunk or the
error placed at (1 - EPS) of the bound, "just outside" = (1 + EPS). The metric is the share of cases that match.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from alloy.bench import compare as C
from alloy.bench.types import Condition, LawSpec, MetricVerdict, Quantity, Tolerance

EPS = 1e-3
_SRC = "bench §4.5 A16 / VAL-02: every comparator right on synthetic answers, 100 %"

LAW = LawSpec(id="A16.selftest", suite="analytic",
              metrics={"selftest_pass_rate": Tolerance(abs=0.0, source=_SRC)}, stat={"selftest_pass_rate": "mean"},
              dt=1.0, substeps=1, duration_s=0.0, worlds_per_condition=1, solvers=("none",))

Case = tuple[str, Callable[[], MetricVerdict], str]


def _real(mean: float, std: float = 0.0, n: int = 20, u95: float = 0.0) -> Quantity:
    return Quantity("q", "N", mean, std, n, u95)


def _sim(mean: float, std: float = 0.0) -> Quantity:
    return Quantity("q", "N", mean, std, 16, 0.0)


def _scalar_cases() -> list[Case]:
    tol = Tolerance(abs=0.1, rel=0.05, source="selftest")  # bound at 10 N = 0.6 N
    b = tol.bound(10.0)
    small_u = b / 10.0  # TUR = 10
    return [
        ("scalar_inside", lambda: C.compare_scalar(_real(10.0, u95=small_u), _sim(10.0 + b * (1 - EPS)), tol), "pass"),
        ("scalar_outside", lambda: C.compare_scalar(_real(10.0, u95=small_u), _sim(10.0 + b * (1 + EPS)), tol), "fail"),
        ("scalar_below", lambda: C.compare_scalar(_real(10.0, u95=small_u), _sim(10.0 - b * (1 + EPS)), tol), "fail"),
        ("scalar_spread_inside", lambda: C.compare_scalar(_real(10.0, 0.1, u95=small_u), _sim(10.0, b / 2 * (1 - EPS)),
                                                          tol), "pass"),
        ("scalar_spread_outside", lambda: C.compare_scalar(_real(10.0, 0.1, u95=small_u),
                                                           _sim(10.0, b / 2 * (1 + EPS)), tol), "fail"),
        ("scalar_tur_inside", lambda: C.compare_scalar(_real(10.0, u95=b / 3 * (1 - EPS)), _sim(10.0), tol), "pass"),
        ("scalar_tur_outside", lambda: C.compare_scalar(_real(10.0, u95=b / 3 * (1 + EPS)), _sim(10.0), tol),
         "inconclusive"),
        ("scalar_nan", lambda: C.compare_scalar(_real(10.0, u95=small_u), _sim(math.nan), tol), "fail"),
    ]


def _curve_cases() -> list[Case]:
    x = np.linspace(0.0, 1.0, 101)
    y = np.sin(3 * x) + 1.0
    off = 0.05 * float(np.max(np.abs(y)))  # constant offset -> NRMSE = off / max|y| = 0.05
    tol = Tolerance(abs=0.05, source="selftest")
    tol_in = Tolerance(abs=0.05 * (1 + EPS), source="selftest")
    return [
        ("curve_inside", lambda: C.compare_curve((x, y), (x, y + off), x, tol_in, "c", "N"), "pass"),
        ("curve_outside", lambda: C.compare_curve((x, y), (x, y + off * (1 + 10 * EPS)), x, tol, "c", "N"), "fail"),
        ("curve_short_support", lambda: C.compare_curve((x, y), (x[:50], y[:50]), x, tol, "c", "N"), "fail"),
    ]


def _distribution_cases() -> list[Case]:
    rng = np.random.default_rng(7)
    real = rng.normal(0.0, 0.004, size=(16, 2))
    perm = real[rng.permutation(16)]
    shift = np.array([0.002, 0.0])
    nodes = rng.integers(5, 10, size=(16, 1)).astype(np.float64)
    return [
        ("dist_permutation_zero", lambda: C.compare_distribution(real, perm, 1e-9, "d"), "pass"),
        ("dist_shift_inside", lambda: C.compare_distribution(real, real + shift, 0.002 * (1 + EPS), "d"), "pass"),
        ("dist_shift_outside", lambda: C.compare_distribution(real, real + shift, 0.002 * (1 - EPS), "d"), "fail"),
        ("dist_prf02_outlier", lambda: C.compare_distribution(real, np.vstack([real[:-1], [[0.03, 0.0]]]), 1.0, "d"),
         "fail"),
        # SCR-WP2-8: a count (unit "1") is judged by W1 alone; a one-node shift sits exactly on tol 1
        ("dist_count_shift_inside", lambda: C.compare_distribution(nodes, nodes + 1.0, 1.0, "n", "1"), "pass"),
        ("dist_count_shift_outside", lambda: C.compare_distribution(nodes, nodes + 1.0, 1.0 - EPS, "n", "1"), "fail"),
    ]


def _rate_cases() -> list[Case]:
    lo, hi = C.wilson_interval(5, 20)
    k_in, k_out = _rate_counts(hi, 1000)
    return [
        ("rate_inside", lambda: C.compare_rate(5, 20, 5, 20, 0.0, "r"), "pass"),
        ("rate_edge_inside", lambda: C.compare_rate(5, 20, k_in, 1000, 0.0, "r"), "pass"),
        ("rate_edge_outside", lambda: C.compare_rate(5, 20, k_out, 1000, 0.0, "r"), "fail"),
        ("rate_tol_widens", lambda: C.compare_rate(5, 20, k_out + 20, 1000, 0.05, "r"), "pass"),
        ("rate_zero_real", lambda: C.compare_rate(0, 20, 0, 1000, 0.0, "r"), "pass"),
    ]


def _rate_counts(hi: float, n: int) -> tuple[int, int]:
    """Largest count with k/n <= hi and the next one up (just inside / just outside the Wilson upper edge)."""
    k = math.floor(hi * n)
    return k, k + 1


def _analytic_cases() -> list[Case]:
    tol = Tolerance(abs=1e-3, rel=0.01, source="selftest")
    ref = np.array([1.0, 2.0])
    b = np.array([tol.bound(1.0), tol.bound(2.0)])
    return [
        ("errors_inside", lambda: C.compare_errors("e", "m", b * (1 - EPS), tol, ref), "pass"),
        ("errors_outside", lambda: C.compare_errors("e", "m", b * np.array([1 - EPS, 1 + EPS]), tol, ref), "fail"),
        ("errors_nonfinite", lambda: C.compare_errors("e", "m", np.array([0.0, math.inf]), tol, ref), "fail"),
        ("errors_empty", lambda: C.compare_errors("e", "m", np.zeros(0), tol), "error"),
    ]


def cases() -> list[Case]:
    return _scalar_cases() + _curve_cases() + _distribution_cases() + _rate_cases() + _analytic_cases()


def run_cases() -> tuple[float, dict[str, str]]:
    """(pass rate, {case: "ok" | "got <verdict> want <verdict>" | "raised ..."})."""
    out: dict[str, str] = {}
    for name, thunk, want in cases():
        try:
            got = thunk().verdict
            out[name] = "ok" if got == want else f"got {got} want {want}"
        except Exception as exc:  # noqa: BLE001 - a comparator crash is a self-test failure, reported per case
            out[name] = f"raised {type(exc).__name__}: {exc}"
    ok = sum(v == "ok" for v in out.values())
    return ok / len(out), out


class SelfTestTwin:
    """Direct law: no simulation; `evaluate` runs the comparator cases."""

    law_id = LAW.id

    def __init__(self) -> None:
        self.monitors: dict[str, object] = {}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return (Condition("comparators", {}),)

    def evaluate(self, ctx: object) -> tuple[MetricVerdict, ...]:
        rate, detail = run_cases()
        self.monitors = {"cases": detail, "n_cases": len(detail)}
        return (C.compare_errors("selftest_pass_rate", "1", np.array([1.0 - rate]), LAW.metrics["selftest_pass_rate"]),)


__all__ = ["LAW", "SelfTestTwin", "run_cases", "cases"]
