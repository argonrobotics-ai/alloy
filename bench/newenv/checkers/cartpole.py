# SPDX-License-Identifier: Apache-2.0
"""E06 cart-pole: free fall time and growth rate at rest, state-feedback hold, per the request's numbers.

Derivation (cart M = 1 kg, uniform pole m = 0.1 kg, L = 0.5 m, hinge at the cart, COM l = L/2, g = 9.81):
  pole about its hinge: I = m L^2 / 3;  linearized, frictionless rail, no force:
  (M + m) x'' + m l th'' = 0;   I th'' + m l x'' = m g l th
  => th'' = lam^2 th with lam^2 = m g l / (I - (m l)^2 / (M + m)) = 31.6 1/s^2, lam = 5.62 1/s
  (the bare pole sqrt(3 g / (2 L)) = 5.43 times the cart's correction 1 / sqrt(1 - 3 m / (4 (M + m))) = 1.036).
  From a start of theta0 (|theta0| <= 0.05, any rate), theta(t) = a cosh(lam t) + b sinh(lam t); the measured rows with
  |theta| < 0.3 are fitted for lam (least squares over a, b, lam) and must be within 25 % of 5.62.
  Fall time: from 0.05 rad with zero rate, cosh(lam t) = 8 gives t = 0.50 s; from a tiny start up to ~0.9 s: the
  request's 0.4-3 s holds with room. The rest phase is 2 s: a world that has not fallen by then started within
  0.4 / cosh(5.62 * 2) = 3e-6 rad of upright and is not counted against the check (allowed: 1 of 16).
  Hold: the harness law is LQR on the linearized model (scripts/cartpole.py); with the right force scale (10 N x
  action) it keeps |theta| < 0.2 rad for 5 s; fails with another scale or sign.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

LAM = 5.6199
MAX_ANGLE = 0.4
FIT_ANGLE = 0.3
MIN_FIT_ROWS = 6
LAM_TOL = 0.25
HOLD_ANGLE = 0.2
HOLD_WORLDS = 14


def first_done_rows(done: np.ndarray) -> np.ndarray:
    """(N,) index of each world's first done row, or -1."""
    any_done = done.any(axis=0)
    return np.where(any_done, done.argmax(axis=0), -1)


def fit_growth(t: np.ndarray, theta: np.ndarray) -> float:
    """Best lam for theta(t) = a cosh(lam t) + b sinh(lam t) over a grid, least squares in a, b; NaN if unfit."""
    if len(t) < MIN_FIT_ROWS:
        return float("nan")
    best, best_err = float("nan"), np.inf
    for lam in np.linspace(1.0, 14.0, 261):
        basis = np.stack([np.cosh(lam * t), np.sinh(lam * t)], axis=1)
        coef, *_ = np.linalg.lstsq(basis, theta, rcond=None)
        err = float(np.sum((basis @ coef - theta) ** 2))
        if err < best_err:
            best, best_err = float(lam), err
    return best


def world_growth(t: np.ndarray, theta: np.ndarray, end: int) -> float:
    """Fit over a world's first-episode rows [0, end) while |theta| < FIT_ANGLE."""
    rows = np.arange(end if end >= 0 else len(t))
    small = np.abs(theta[rows]) < FIT_ANGLE
    stop = int(np.argmin(small)) if not small.all() else len(rows)
    return fit_growth(t[:stop], theta[:stop])


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    first = first_done_rows(rest.done)
    fell = first >= 0
    t_fall = np.where(fell, rest.t[np.maximum(first, 0)], np.nan)
    before = np.maximum(first - 1, 0)
    idx = np.arange(rest.probe["pole_angle"].shape[1])
    angle_at_end = np.abs(rest.probe["pole_angle"][before, idx])
    lam = np.array([world_growth(rest.t, rest.probe["pole_angle"][:, i], first[i]) for i in idx])
    return [c.within("rest_fall_time", t_fall, 0.4, 3.0, 2, min_worlds=len(idx) - 1),
            c.at_least("rest_worlds_fell", float(fell.sum()), len(idx) - 1, 2),
            c.at_least("rest_done_at_tilt", float(np.min(angle_at_end[fell], initial=1.0)), 0.3, 2),
            c.within("growth_rate", float(np.nanmedian(lam)) if np.isfinite(lam).any() else np.nan,
                     LAM * (1 - LAM_TOL), LAM * (1 + LAM_TOL), 2)]


def _hold_checks(r: c.Rollout) -> list[c.Check]:
    hold = r.phase("act_hold")
    worst = np.abs(hold.probe["pole_angle"]).max(axis=0)
    no_done = ~hold.done[:-1].any(axis=0)                  # a done on the very last row is the episode's time limit
    ok = (worst < HOLD_ANGLE) & no_done
    return [c.at_least("hold_worlds", float(ok.sum()), HOLD_WORLDS, 2),
            c.at_most("hold_max_angle_median", float(np.median(worst)), HOLD_ANGLE, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + _hold_checks(rollout)
    return c.finish(checks)
