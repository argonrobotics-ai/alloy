# SPDX-License-Identifier: Apache-2.0
"""E17 tap and glass: no flow at rest, 3 s of 50 ml/s fills 150 ml (5.3 cm), none lost, the amount stays after closing.

Derivations (from the request sentence only; no engine's frames, particle order or counts; ``glass_pos`` gives the
glass's axis, its height reference is the agent's choice):
- Glass: inner radius 3 cm, 12 cm tall: area pi 0.03^2 = 28.27 cm2. 150 ml fills it to 5.30 cm.
- The glass's floor height is read from the water itself: the lowest point inside the glass column (horizontal distance
  from glass_pos <= 3.5 cm) at the end of the act phase, minus half a particle spacing (spacing = particle_volume^(1/3)).
  Water "in the glass" = points in the column between that floor -5 mm and floor + 12 cm + 5 mm (the 5 mm margins
  cover wall softness; the stream falling toward the glass above the rim is not in it).
- Rest (zero action, rest_s): water in the glass <= 2 ml at every row.
- Act: full flow (action +1 = 50 ml/s) for 3 s then zero (the law, scripts/fill_level.py). At the end of the phase the
  glass holds 150 ml +- 20 % (120-180 ml), its level (0.5 % to 99.5 % height of its points + one spacing) is 5.3 cm
  +- 1 cm, and its water is >= 90 % of the water the action commanded: sum over the rows of
  clip(action, 0, 1) x 50 ml/s x dt (the sentence's own numbers: "emitted" cannot be read from a probe when the engine
  cannot stop a nozzle, see CHECK-CHANGE E17). Also <= 110 % of it (nothing is added that nobody asked for).
- After the tap closes (the last row whose action is > 0.05) the amount keeps constant: the glass volume 0.5 s after
  closing and at the end of the phase differ by <= 3 %.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

RADIUS, HEIGHT = 0.03, 0.12
AREA = np.pi * RADIUS ** 2
COLUMN_MARGIN = 0.005
Z_MARGIN = 0.005
Q_MAX = 50e-6
FILL_ML = (120.0, 180.0)
LEVEL_M, LEVEL_TOL = 150e-6 / AREA, 0.01
COMMANDED_SHARE = (0.9, 1.1)
REST_MAX_ML = 2.0
KEEP_TOL = 0.03
SETTLE_S = 0.5


def _column(points: np.ndarray, glass_xy: np.ndarray) -> np.ndarray:
    """Bool (..., P): finite points within the glass's radius (+ margin) of its axis."""
    d = np.linalg.norm(points[..., :2] - glass_xy[..., None, :], axis=-1)
    return np.isfinite(points).all(axis=-1) & (d <= RADIUS + COLUMN_MARGIN)


def _floor(ph: c.Phase) -> np.ndarray:
    """(N,) glass floor height per world from the last row of ``ph`` (see module doc); NaN with no water in the column."""
    pts = ph.probe["liquid_points"][-1].astype(np.float64)
    glass_xy = ph.probe["glass_pos"][-1][:, :2].astype(np.float64)
    col = _column(pts, glass_xy)
    half = 0.5 * ph.probe["particle_volume"][-1].astype(np.float64) ** (1 / 3)
    z = np.where(col, pts[..., 2], np.inf).min(axis=-1)
    return np.where(np.isfinite(z), z - half, np.nan)


def _in_glass(ph: c.Phase, floor: np.ndarray) -> np.ndarray:
    """Bool (T, N, P): points in the glass, given the floor height (N,)."""
    pts = ph.probe["liquid_points"].astype(np.float64)
    glass_xy = ph.probe["glass_pos"][..., :2].astype(np.float64)
    z = pts[..., 2]
    lo, hi = floor[None, :, None] - Z_MARGIN, floor[None, :, None] + HEIGHT + Z_MARGIN
    with np.errstate(invalid="ignore"):
        return _column(pts, glass_xy) & (z >= lo) & (z <= hi)


def _glass_ml(ph: c.Phase, floor: np.ndarray) -> np.ndarray:
    """(T, N) millilitres of water in the glass."""
    return _in_glass(ph, floor).sum(axis=-1) * ph.probe["particle_volume"].astype(np.float64) * 1e6


def _level(ph: c.Phase, floor: np.ndarray) -> np.ndarray:
    pts = ph.probe["liquid_points"][-1].astype(np.float64)
    inside = _in_glass(ph, floor)[-1]
    spacing = ph.probe["particle_volume"][-1].astype(np.float64) ** (1 / 3)
    out = np.full(ph.num_envs, np.nan)
    for w in range(ph.num_envs):
        z = pts[w, inside[w], 2]
        if len(z) > 20:
            out[w] = np.percentile(z, 99.5) - np.percentile(z, 0.5) + spacing[w]
    return out


def _commanded_ml(ph: c.Phase) -> np.ndarray:
    dt = np.diff(ph.t)[:, None]
    return (np.clip(ph.action[1:, :, 0], 0.0, 1.0) * Q_MAX * dt).sum(axis=0) * 1e6


def _close_row(ph: c.Phase) -> int:
    open_rows = np.flatnonzero((ph.action[:, :, 0] > 0.05).any(axis=1))
    return int(open_rows[-1]) if len(open_rows) else 0


def _act_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("act_fill")
    floor = _floor(ph)
    ml = _glass_ml(ph, floor)
    final, cmd = ml[-1], _commanded_ml(ph)
    k_close = _close_row(ph)
    k_settle = min(int(np.argmin(np.abs(ph.t - (ph.t[k_close] + SETTLE_S)))), len(ph.t) - 1)
    drift = np.abs(final - ml[k_settle]) / np.maximum(ml[k_settle], 1e-9)
    share = final / np.maximum(cmd, 1e-9)
    return [c.within("glass_ml", final, *FILL_ML, 2),
            c.within("glass_level_m", _level(ph, floor), LEVEL_M - LEVEL_TOL, LEVEL_M + LEVEL_TOL, 2),
            c.within("glass_share_of_commanded", share, *COMMANDED_SHARE, 2),
            c.at_most("amount_drift_after_close", float(np.nanmax(drift)), KEEP_TOL, 2)]


def _rest_check(r: c.Rollout) -> c.Check:
    act, rest = r.phase("act_fill"), r.phase("rest")
    floor = _floor(act)
    ml = _glass_ml(rest, floor)
    return c.at_most("rest_adds_no_water_ml", float(np.nanmax(ml)), REST_MAX_ML, 2)


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + [_rest_check(rollout)] + _act_checks(rollout)
    return c.finish(checks)
