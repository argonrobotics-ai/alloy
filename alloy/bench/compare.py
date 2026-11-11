# SPDX-License-Identifier: Apache-2.0
"""Comparators: judge a simulated quantity against a real (rig) or analytic reference (SPEC §3.10, bench §4.3).

Every comparator returns a `MetricVerdict`. Rules:
- scalar: pass iff |sim mean - real mean| <= bound and sim std <= max(2 real std, bound/2); TUR = bound / U_real,
  with U_real = sqrt(u_instr^2 + (t_0.975 s / sqrt(n))^2) carried as `real.u95`; TUR < min_tur -> inconclusive.
- curve: NRMSE = RMS(y_sim - y_real) / max|y_real| on the spec grid, pass iff <= tol.abs.
- distribution: assignment W1 (GPUSimBench, arXiv:2607.13059) plus PRF-02 pose limits on matched distances.
- rate: sim rate inside the real Wilson 95 % interval +/- tol_abs.
- analytic (`compare_errors`): per-world errors against per-world bounds; `real` is None (closed form, no noise).
Non-finite simulated values always fail; they never pass silently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

from alloy.bench.types import MetricVerdict, Quantity, Stat, Tolerance, Verdict

Z95 = 1.959963984540054  # two-sided 95 % normal quantile (Wilson interval)
PRF02_MEDIAN_M = 5e-3  # PRF-02 pose limits: median <= 5 mm, >= 90 % within 10 mm, none beyond 25 mm
PRF02_P90_M = 10e-3
PRF02_MAX_M = 25e-3
MAX_ASSIGNMENT = 4096  # largest replicated set for unequal-size W1


def t975(n: int) -> float:
    """Two-sided 95 % Student-t quantile for n samples (n - 1 degrees of freedom); inf for n < 2."""
    return float(stats.t.ppf(0.975, n - 1)) if n >= 2 else math.inf


def summarize(name: str, unit: str, values: Sequence[float] | np.ndarray, u_instr: float = 0.0) -> Quantity:
    """Quantity of a sample: mean, sample std (ddof=1), n and U95 = sqrt(u_instr^2 + (t s / sqrt n)^2).

    With one sample the sampling term is dropped (std 0) and U95 = u_instr: a single trial cannot measure its own
    scatter, which callers must account for (rig laws require n >= 10 per condition, bench §4.9).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    n = int(v.size)
    if n == 0:
        return Quantity(name, unit, math.nan, math.nan, 0, math.nan)
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    sampling = t975(n) * std / math.sqrt(n) if n > 1 else 0.0
    return Quantity(name, unit, mean, std, n, math.hypot(u_instr, sampling))


def _finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def compare_scalar(real: Quantity, sim: Quantity, tol: Tolerance, min_tur: float = 3.0) -> MetricVerdict:
    """Judge a scalar: |sim - real| <= tol.bound(real.mean), sim spread bounded, rig able to judge (TUR >= min_tur)."""
    bound = tol.bound(real.mean)
    error = abs(sim.mean - real.mean)
    tur = bound / real.u95 if real.u95 > 0 else None
    verdict: Verdict
    if not _finite(sim.mean, sim.std, error):
        verdict = "fail"
    elif tur is not None and tur < min_tur:
        verdict = "inconclusive"
    else:
        spread_ok = sim.std <= max(2.0 * real.std, bound / 2.0)
        verdict = "pass" if (error <= bound and spread_ok) else "fail"
    return MetricVerdict(sim.name, sim.unit, real, sim, error, bound, tur, "mean", verdict)


def _resample(xy: tuple[np.ndarray, np.ndarray], grid: np.ndarray) -> np.ndarray:
    x = np.asarray(xy[0], dtype=np.float64)
    y = np.asarray(xy[1], dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if grid.min() < x[0] - 1e-12 or grid.max() > x[-1] + 1e-12:
        return np.full(grid.shape, np.nan)  # never extrapolate a curve: outside its support it is missing
    return np.interp(grid, x, y)


def nrmse(real: np.ndarray, sim: np.ndarray) -> float:
    """RMS(sim - real) / max|real| (bench §4.3); inf when the reference is identically zero."""
    scale = float(np.max(np.abs(real)))
    rms = float(np.sqrt(np.mean((sim - real) ** 2)))
    return rms / scale if scale > 0 else (0.0 if rms == 0 else math.inf)


def _curve_quantity(name: str, unit: str, y: np.ndarray) -> Quantity:
    f = y[np.isfinite(y)]
    if f.size == 0:
        return Quantity(name, unit, math.nan, math.nan, 0, math.nan)
    return Quantity(name, unit, float(np.mean(f)), float(np.std(f)), int(f.size), 0.0)


def compare_curve(real: tuple[np.ndarray, np.ndarray], sim: tuple[np.ndarray, np.ndarray], grid: np.ndarray,
                  tol: Tolerance, name: str, unit: str) -> MetricVerdict:
    """NRMSE of the simulated curve against the (mean) real curve on `grid`; pass iff NRMSE <= tol.abs."""
    g = np.asarray(grid, dtype=np.float64)
    y_r, y_s = _resample(real, g), _resample(sim, g)
    error = nrmse(y_r, y_s) if np.all(np.isfinite(y_r)) and np.all(np.isfinite(y_s)) else math.nan
    bound = tol.abs
    real_q, sim_q = _curve_quantity(name, unit, y_r), _curve_quantity(name, unit, y_s)
    verdict: Verdict = "pass" if (math.isfinite(error) and error <= bound) else "fail"
    return MetricVerdict(name, "1", real_q, sim_q, error, bound, None, "curve", verdict)


def _replicate_to_common(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Repeat both point sets to their least common size so the assignment W1 is the exact discrete W1."""
    n = math.lcm(len(a), len(b))
    if n > MAX_ASSIGNMENT:
        raise ValueError(f"distribution sizes {len(a)} and {len(b)} need {n} > {MAX_ASSIGNMENT} assignment rows")
    return np.repeat(a, n // len(a), axis=0), np.repeat(b, n // len(b), axis=0)


def assignment_w1(real_xy: np.ndarray, sim_xy: np.ndarray) -> tuple[float, np.ndarray]:
    """W1 = min over matchings of the mean Euclidean distance (GPUSimBench Eq. W1); returns (W1, matched distances)."""
    a = np.atleast_2d(np.asarray(real_xy, dtype=np.float64))
    b = np.atleast_2d(np.asarray(sim_xy, dtype=np.float64))
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"point dimensions differ: {a.shape[1]} vs {b.shape[1]}")
    if len(a) != len(b):
        a, b = _replicate_to_common(a, b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    d = cost[rows, cols]
    return float(np.mean(d)), d


def prf02_ok(distances: np.ndarray) -> bool:
    """PRF-02 pose limits on matched distances [m]: median <= 5 mm, >= 90 % <= 10 mm, none > 25 mm."""
    d = np.asarray(distances, dtype=np.float64)
    return bool(np.median(d) <= PRF02_MEDIAN_M and np.mean(d <= PRF02_P90_M) >= 0.9 and np.max(d) <= PRF02_MAX_M)


def compare_distribution(real_xy: np.ndarray, sim_xy: np.ndarray, tol: float, name: str,
                         unit: str = "m") -> MetricVerdict:
    """Assignment W1 between real and simulated outcomes in `unit`; pass iff W1 <= tol (SCR-WP2-8).

    The PRF-02 pose limits apply only to metres; other units (e.g. a node count, unit "1") judge W1 alone.
    """
    real_xy, sim_xy = np.asarray(real_xy, np.float64), np.asarray(sim_xy, np.float64)
    finite = bool(np.all(np.isfinite(real_xy)) and np.all(np.isfinite(sim_xy)))
    w1, d = assignment_w1(real_xy, sim_xy) if finite else (math.nan, np.array([math.inf]))
    real_q = summarize(name, unit, np.linalg.norm(np.atleast_2d(real_xy), axis=-1))
    sim_q = summarize(name, unit, d)
    ok = finite and w1 <= tol and (unit != "m" or prf02_ok(d))
    return MetricVerdict(name, unit, real_q, sim_q, w1, tol, None, "distribution", "pass" if ok else "fail")


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (95 % by default)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compare_rate(real_k: int, real_n: int, sim_k: int, sim_n: int, tol_abs: float, name: str) -> MetricVerdict:
    """Sim rate must lie inside the real Wilson 95 % interval widened by tol_abs; error = distance outside it."""
    lo, hi = wilson_interval(real_k, real_n)
    rate = sim_k / sim_n if sim_n > 0 else math.nan
    error = max(lo - rate, rate - hi, 0.0) if math.isfinite(rate) else math.nan
    p = real_k / real_n if real_n > 0 else math.nan
    real_q = Quantity(name, "1", p, math.sqrt(p * (1 - p)) if math.isfinite(p) else math.nan, real_n, (hi - lo) / 2)
    sim_q = Quantity(name, "1", rate, math.sqrt(rate * (1 - rate)) if math.isfinite(rate) else math.nan, sim_n, 0.0)
    verdict: Verdict = "pass" if (math.isfinite(error) and error <= tol_abs) else "fail"
    return MetricVerdict(name, "1", real_q, sim_q, error, tol_abs, None, "rate", verdict)


def compare_errors(name: str, unit: str, errors: np.ndarray, tol: Tolerance, ref: np.ndarray | None = None,
                   stat: Stat = "mean") -> MetricVerdict:
    """Analytic law: per-world |sim - law| errors against per-world bounds tol.bound(ref_w) (ref = 0 if None).

    Reports the worst world (largest error/bound ratio): `error` and `bound` are that world's values; `sim` summarises
    the per-world errors. Any non-finite error fails the metric.
    """
    e = np.asarray(errors, dtype=np.float64).ravel()
    r = np.zeros_like(e) if ref is None else np.broadcast_to(np.asarray(ref, np.float64).ravel(), e.shape)
    bounds = np.array([tol.bound(float(x)) for x in r]) if e.size else np.zeros(0)
    sim_q = summarize(name, unit, e)
    if e.size == 0:
        return MetricVerdict(name, unit, None, sim_q, math.nan, tol.bound(0.0), None, stat, "error")
    if not np.all(np.isfinite(e)):
        worst = int(np.argmax(~np.isfinite(e)))
        return MetricVerdict(name, unit, None, sim_q, math.inf, float(bounds[worst]), None, stat, "fail")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bounds > 0, e / bounds, np.where(e > 0, math.inf, 0.0))
    worst = int(np.argmax(ratio))
    verdict: Verdict = "pass" if bool(np.all(e <= bounds)) else "fail"
    return MetricVerdict(name, unit, None, sim_q, float(e[worst]), float(bounds[worst]), None, stat, verdict)


def placeholder_verdict(name: str, unit: str, verdict: Verdict, stat: Stat = "mean") -> MetricVerdict:
    """A metric that produced no value (not_run or error): NaN quantities, never a pass."""
    if verdict == "pass":
        raise ValueError("a placeholder verdict cannot pass")
    q = Quantity(name, unit, math.nan, math.nan, 0, math.nan)
    return MetricVerdict(name, unit, None, q, math.nan, math.nan, None, stat, verdict)


__all__ = ["compare_scalar", "compare_curve", "compare_distribution", "compare_rate", "compare_errors",
           "placeholder_verdict", "summarize", "wilson_interval", "assignment_w1", "prf02_ok", "nrmse", "t975"]
