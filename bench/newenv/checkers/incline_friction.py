# SPDX-License-Identifier: Apache-2.0
"""E04 puck on a 15 degree incline with per-world friction 0.1-0.6.

Derivations (m = 0.2 kg, theta = 15 deg, sin = 0.2588, cos = 0.9659, tan = 0.2679, g = 9.81):
- Rest, free slide: a puck on an incline with Coulomb friction mu < tan(theta) accelerates down at
  a = g (sin - mu cos); mu = 0.1 gives 1.59 m/s2, mu = 0.24 gives 0.265. The incline is 1.5 m long, so the fit window is
  t <= 0.6 s (0.29 m at the fastest). The distance d(t) from the start is fitted as 0.5 a t^2 (start at rest, least squares
  through the origin) and a must be within 20 % of the formula using the probe's own friction. The displacement must
  point down the slope: dz / |d| = -sin(theta) within 0.05 (engine- and frame-independent).
- Rest, held: mu = 0.30 > tan(theta) = 0.268 so the puck sticks; displacement < 5 mm over the first second (or the
  rest phase if shorter).
- Friction spread: the sentence says the coefficient is drawn per world from 0.1 to 0.6. Every value must lie in
  [0.1, 0.6] (1e-3 slack) and the 16 values must span at least 0.25. (CHECK-CHANGE: "span at least 0.15-0.55" would fail
  an honest uniform draw in 34 % of seeds: P(min > 0.15) = P(max < 0.55) = 0.9^16 = 0.185.)
- Act (+1 = 1 N up the slope, 0.5 s from a fresh reset): net acceleration up the slope is a = F/m - g (sin + mu cos) while
  positive, i.e. for mu < (1 - 0.2 g sin)/(0.2 g cos) = 0.26. Worlds with mu < 0.2 (a >= 0.56 m/s2) must move up
  (dz > 0) with a within 25 % of the formula; worlds with mu > 0.3 must not move up (dz < 5 mm).
"""
from __future__ import annotations

import math

import numpy as np

from bench.newenv.checkers import common as c

G = 9.81
THETA = math.radians(15.0)
MASS = 0.2
FORCE = 1.0
FIT_S = 0.6


def _disp(p: c.Phase) -> np.ndarray:
    pos = p.probe["puck_pos"].astype(np.float64)
    return pos - pos[0]


def _group(name: str, values: np.ndarray, lo: float, hi: float) -> c.Check:
    """``within`` for a subset of worlds; an empty subset (nothing to test) fails instead of passing vacuously."""
    if values.size == 0:
        return c.Check(name, False, "no world in the group", f"[{lo:g}, {hi:g}]", 2)
    return c.within(name, values, lo, hi, 2)


def _accel_and_slope(p: c.Phase, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    k = int(np.searchsorted(p.t, window_s + 1e-6, side="right"))
    d = _disp(p)[:k]
    dist = np.linalg.norm(d, axis=-1)
    t = p.t[:k].astype(np.float64)
    a = (0.5 * t[:, None] ** 2 * dist).sum(0) / (0.25 * t ** 4).sum()
    slope = d[-1, :, 2] / np.maximum(dist[-1], 1e-9)
    return a, slope


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("rest")
    mu = p.probe["friction"][0].astype(np.float64)
    a, slope = _accel_and_slope(p, FIT_S)
    free, held = mu < 0.24, mu > 0.30
    expected = G * (math.sin(THETA) - mu * math.cos(THETA))
    k = int(np.searchsorted(p.t, 1.0 + 1e-6, side="right"))
    moved = np.linalg.norm(_disp(p)[:k], axis=-1).max(axis=0)
    return [_group("slide_accel_ratio", (a / expected)[free], 0.8, 1.2),
            _group("slide_direction", slope[free], -math.sin(THETA) - 0.05, -math.sin(THETA) + 0.05),
            _group("held_still", moved[held], 0.0, 0.005)]


def _spread_checks(r: c.Rollout) -> list[c.Check]:
    mu = r.phase("random").probe["friction"][0].astype(np.float64)
    return [c.within("friction_bounds", mu, 0.1 - 1e-3, 0.6 + 1e-3, 2),
            c.at_least("friction_spread", float(np.ptp(mu)), 0.25, 2)]


def _act_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("act_up")
    mu = p.probe["friction"][0].astype(np.float64)
    a, slope = _accel_and_slope(p, 0.5)
    dz = _disp(p)[-1, :, 2]
    low, high = mu < 0.2, mu > 0.3
    expected = FORCE / MASS - G * (math.sin(THETA) + mu * math.cos(THETA))
    return [_group("push_up_accel_ratio", (a / expected)[low], 0.75, 1.25),
            _group("push_up_direction", slope[low], math.sin(THETA) - 0.05, math.sin(THETA) + 0.05),
            _group("push_held_below_threshold", dz[high], -np.inf, 0.005)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _rest_checks(rollout) + _spread_checks(rollout) + _act_checks(rollout)
    return c.finish(checks)
