# SPDX-License-Identifier: Apache-2.0
"""E18 sand scoop: bed depth, mass, settled at rest; a 3 cm deep, 15 cm long pass leaves a furrow with sloped walls.

Derivations (from the request sentence only; no engine's frames, particle order or counts):
- Particle spacing s = (particle_mass / 1550 kg/m3)^(1/3): the mid bulk density of dry sand (1,400-1,700); a 10 %
  density error is 3 % in s.
- Bed depth: (0.5 % to 99.5 % height of the points) + s over the points of the first rest row; 4-6 cm (5 cm +- 1).
- Mass: live points x particle_mass = 5-9 kg: a 30 x 30 x 5 cm bed is 4.5 l at 1,400-1,700 kg/m3 = 6.3-7.7 kg, +- 20 %.
- Inside the tray: over all rest rows the points' x and y extents are <= 0.30 + 0.02 m (the tray is 30 cm square; a
  grain's radius and the wall's softness are inside 1 cm per side) and no point is lower than the first row's lowest
  point - 1 cm (nothing below the tray's floor). Settled: the RMS point displacement over the last 1 s of the rest
  phase, as a speed, is <= 5 mm/s (a heap still sliding moves at cm/s).
- Dig (the law, scripts/sand_scoop.py): the scoop is lowered to 3 cm below the bed's top at 0.1 m/s, pushed 15 cm at
  0.1 m/s (action +1), then held for 2 s. The pass is the rows where the commanded x velocity is >= 0.5 of the maximum in size;
  its x-y path is the scoop's ``scoop_pos`` in those rows. The surface is the 95th percentile height of the points in 1 cm
  cells of the tray, read in the last row. Untouched bed = cells >= 7 cm from the path (the scoop is 8 cm wide: 4 cm each
  side) of the path's line; the corridor = cells within 3 cm of the path and 3 cm inside its ends.
  * Furrow: the untouched surface's median minus the corridor's mean >= 5 mm (3 cm of sand removed across 8 cm; water
    would level to 0, and a 34 deg repose slumps the walls but leaves > 1 cm in the middle).
  * Walls hold a slope: the steepest drop between neighbouring 1 cm cells across the path (the y profile averaged over
    the corridor's x range) is >= 2.7 mm per cm (15 deg; sand rests at 22-34 deg, water at 0).
  * Sand ahead is displaced: the highest cell within 6 cm beyond the path's end, within 3 cm of its line, is >= 1 cm above
    the untouched median (360 ml of sand pushed ahead piles up).
  * The scoop follows the command: its mean x speed over the push rows (the first excluded) is 0.1 m/s +- 10 %, and its
    mean z speed in the descent rows (action z <= -0.5) is -0.1 m/s +- 10 %.
"""
from __future__ import annotations

import warnings

import numpy as np

from bench.newenv.checkers import common as c

BULK = 1550.0
DEPTH_M = (0.04, 0.06)
MASS_KG = (5.0, 9.0)
TRAY = 0.30
EXTENT_TOL = 0.02
FLOOR_TOL = 0.01
SETTLED_MPS = 0.005
CELL = 0.01
UNTOUCHED_MIN = 0.07
CORRIDOR_HALF, END_MARGIN, AHEAD = 0.03, 0.03, 0.06
FURROW_M, WALL_SLOPE, HEAP_M = 0.005, 0.0027 / 0.01, 0.01
SPEED, SPEED_TOL = 0.1, 0.10


def _safe(fn, a: np.ndarray) -> float:
    """``fn`` (a nan-reduction) of ``a``, NaN when ``a`` has no finite value (no warning)."""
    return float(fn(a)) if np.isfinite(a).any() else np.nan


def _finite_points(points: np.ndarray) -> np.ndarray:
    return points[np.isfinite(points).all(axis=-1)]


def _spacing(pm: np.ndarray) -> np.ndarray:
    return (pm.astype(np.float64) / BULK) ** (1 / 3)


def _depth(points: np.ndarray, s: float) -> float:
    z = _finite_points(points)[:, 2]
    return float(np.percentile(z, 99.5) - np.percentile(z, 0.5) + s)


def _rest_speed(rest: c.Phase, w: int) -> float:
    """RMS point speed over the last second of the rest phase (world w)."""
    k0 = int(np.argmin(np.abs(rest.t - (rest.t[-1] - 1.0))))
    a, b = rest.probe["sand_points"][k0, w], rest.probe["sand_points"][-1, w]
    ok = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    d = np.linalg.norm((b - a)[ok].astype(np.float64), axis=-1)
    return float(np.sqrt(np.mean(d ** 2)) / max(rest.t[-1] - rest.t[k0], 1e-9))


def _extent_and_floor(rest: c.Phase) -> tuple[np.ndarray, np.ndarray]:
    """Worst x-y extent (N,) over rest rows and the lowest drop below row 0's lowest point (N,)."""
    pts = rest.probe["sand_points"].astype(np.float64)
    ext = np.nanmax(np.nanmax(pts[..., :2], axis=2) - np.nanmin(pts[..., :2], axis=2), axis=-1).max(axis=0)
    drop = np.nanmin(pts[0, :, :, 2], axis=1) - np.nanmin(pts[..., 2], axis=(0, 2))
    return ext, drop


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    pm = rest.probe["particle_mass"][0]
    s = _spacing(pm)
    p0 = rest.probe["sand_points"][0].astype(np.float64)
    depth = np.array([_depth(p0[w], s[w]) for w in range(rest.num_envs)])
    mass = np.isfinite(p0).all(axis=-1).sum(axis=-1) * pm
    ext, drop = _extent_and_floor(rest)
    speed = np.array([_rest_speed(rest, w) for w in range(rest.num_envs)])
    return [c.within("bed_depth_m", depth, *DEPTH_M, 2), c.within("total_mass_kg", mass, *MASS_KG, 2),
            c.at_most("tray_extent_m", float(ext.max()), TRAY + EXTENT_TOL, 2),
            c.at_most("below_floor_m", float(drop.max()), FLOOR_TOL, 2),
            c.at_most("rest_speed_mps", float(speed.max()), SETTLED_MPS, 2)]


def _surface(points: np.ndarray, lo: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """95th percentile height per 1 cm cell (NaN for empty cells); points (P, 3) finite."""
    ij = np.floor((points[:, :2] - lo) / CELL).astype(int)
    ok = (ij >= 0).all(axis=1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    h = np.full(shape, np.nan)
    flat = ij[ok, 0] * shape[1] + ij[ok, 1]
    order = np.argsort(flat, kind="stable")
    cells, starts = np.unique(flat[order], return_index=True)
    z = points[ok, 2][order]
    for cell, a, b in zip(cells, starts, list(starts[1:]) + [len(z)]):
        h[cell // shape[1], cell % shape[1]] = np.percentile(z[a:b], 95)
    return h


def _push_rows(ph: c.Phase, w: int) -> np.ndarray:
    return np.flatnonzero(np.abs(ph.action[:, w, 0]) >= 0.5)


def _furrow_metrics(ph: c.Phase, w: int, lo: np.ndarray, shape: tuple[int, int]) -> dict[str, float]:
    rows = _push_rows(ph, w)
    if len(rows) < 3:
        return {}
    path = ph.probe["scoop_pos"][rows, w, :2].astype(np.float64)
    y0, x_lo, x_hi = float(path[:, 1].mean()), float(path[:, 0].min()), float(path[:, 0].max())
    sign = 1.0 if path[-1, 0] >= path[0, 0] else -1.0
    h = _surface(_finite_points(ph.probe["sand_points"][-1, w].astype(np.float64)), lo, shape)
    cx = lo[0] + (np.arange(shape[0]) + 0.5) * CELL
    cy = lo[1] + (np.arange(shape[1]) + 0.5) * CELL
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    untouched = np.abs(Y - y0) >= UNTOUCHED_MIN
    corridor = (np.abs(Y - y0) <= CORRIDOR_HALF) & (X >= x_lo + END_MARGIN) & (X <= x_hi - END_MARGIN)
    ahead_lo, ahead_hi = (x_hi, x_hi + AHEAD) if sign > 0 else (x_lo - AHEAD, x_lo)
    ahead = (np.abs(Y - y0) <= CORRIDOR_HALF) & (X >= ahead_lo) & (X <= ahead_hi)
    base = _safe(np.nanmedian, h[untouched])
    band = np.where((X >= x_lo + END_MARGIN) & (X <= x_hi - END_MARGIN), h, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmean(band, axis=0)
    steps = np.abs(np.diff(profile[np.abs(cy - y0) <= UNTOUCHED_MIN]))
    return {"furrow": base - _safe(np.nanmean, h[corridor]), "slope": _safe(np.nanmax, steps) / CELL,
            "heap": _safe(np.nanmax, h[ahead]) - base}


def _dig_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("act_dig")
    rest0 = _finite_points(r.phase("rest").probe["sand_points"][0, 0].astype(np.float64))
    lo = rest0[:, :2].min(axis=0) - 0.01
    shape = tuple(int(np.ceil((rest0[:, :2].max(axis=0) + 0.01 - lo)[i] / CELL)) + 1 for i in range(2))
    ms = [_furrow_metrics(ph, w, lo, shape) for w in range(ph.num_envs)]
    get = lambda k: np.array([m.get(k, np.nan) for m in ms])  # noqa: E731
    return [c.at_least("furrow_depth_m", float(np.nanmin(get("furrow"))) if np.isfinite(get("furrow")).any() else np.nan, FURROW_M, 2),
            c.at_least("wall_slope", float(np.nanmin(get("slope"))) if np.isfinite(get("slope")).any() else np.nan, WALL_SLOPE, 2),
            c.at_least("heap_ahead_m", float(np.nanmin(get("heap"))) if np.isfinite(get("heap")).any() else np.nan, HEAP_M, 2),
            *_speed_checks(ph)]


def _mean_speed(ph: c.Phase, rows: np.ndarray, axis: int) -> np.ndarray:
    """(N,) mean speed of scoop_pos along ``axis`` over the rows (contiguous run; the first row of the run is excluded)."""
    out = np.full(ph.num_envs, np.nan)
    for w in range(ph.num_envs):
        k = rows[w]
        if len(k) >= 3:
            out[w] = (ph.probe["scoop_pos"][k[-1], w, axis] - ph.probe["scoop_pos"][k[1] - 1, w, axis]) / (ph.t[k[-1]] - ph.t[k[1] - 1])
    return out


def _speed_checks(ph: c.Phase) -> list[c.Check]:
    push = [_push_rows(ph, w) for w in range(ph.num_envs)]
    down = [np.flatnonzero(ph.action[:, w, 2] <= -0.5) for w in range(ph.num_envs)]
    sign = np.array([np.sign(ph.probe["scoop_pos"][k[-1], w, 0] - ph.probe["scoop_pos"][k[0], w, 0]) if len(k) > 1 else 1.0
                     for w, k in enumerate(push)])
    vx, vz = _mean_speed(ph, push, 0) * sign, _mean_speed(ph, down, 2)
    lo, hi = SPEED * (1 - SPEED_TOL), SPEED * (1 + SPEED_TOL)
    return [c.within("scoop_push_speed", vx, lo, hi, 2), c.within("scoop_descent_speed", -vz, lo, hi, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + _dig_checks(rollout)
    return c.finish(checks)
