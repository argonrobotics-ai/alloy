# SPDX-License-Identifier: Apache-2.0
"""E16 tank slosh: volume, rest level, surface tilt under 2 m/s2, the tank follows the command, nothing leaves the tank.

Derivations (from the request sentence only; no engine's frames, particle order or counts; the tank's reference point
in ``tank_pos`` is the agent's choice, so only its CHANGE is used):
- Volume: half of a 20 x 10 x 15 cm tank is 0.2 x 0.1 x 0.075 = 1.5 l; points x particle_volume at the first row is
  1.35-1.65 l (10 %).
- Rest level: the water's depth is (0.5 % to 99.5 % height of the points) + one particle spacing, the spacing being
  particle_volume^(1/3) (a slab of points spans its depth minus one spacing); 7.5 cm +- 1 cm. All water inside: in the
  tank's frame (points minus the tank_pos change since the first row) every point of every rest row lies in the first
  row's x-y extent +- 1 cm and between its lowest point -5 mm and its highest +1 cm; >= 99.9 % of the points.
- Surface tilt under a = 2 m/s2 for 0.5 s: the tank accelerates along +x, so the water rises at the back (-x): the
  surface slope dz/dx is about -a/g = -0.204. The sloshing mode of a 20 cm wide, 7.5 cm deep tank has
  omega^2 = (pi g / L) tanh(pi h / L) = 127 /s2, a period of 0.56 s, so the slope does not sit at a/g but oscillates
  about it, as (a/g)(1 - cos(omega t)) when undamped; its mean over the 0.5 s of acceleration is 1.10 a/g (0.22) and
  damping only pulls that toward a/g. The check is the mean over t in [0, 0.5] s of -dz/dx (a line through the top
  surface of 10 bins along x, the 98th percentile height of each bin), 0.204 +- 50 % (0.102-0.306), in every world.
- The tank follows the command: its x displacement at t = 0.5 s is a t^2 / 2 = 0.25 m and its mean speed over
  t in [0.4, 0.5] s is a * 0.45 = 0.9 m/s, each +- 10 %.
- Contained: during the act phase (0.5 s: the sentence's own acceleration time) no point (tank frame, as above) lies outside the
  first row's x-y extent +- 1 cm or below its lowest point -5 mm (the floor): >= 99.9 % of the points. No height cap: water
  rising over the rim is a splash, which the sentence's "walls or floor" does not cover.
- Volume kept: the live point count at the end is within 5 % of the start.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

VOLUME_L = (1.35, 1.65)
LEVEL_M, LEVEL_TOL = 0.075, 0.01
MARGIN, FLOOR_MARGIN, RIM_MARGIN = 0.01, 0.005, 0.005
INSIDE_FRACTION = 0.999
GRAVITY, ACCEL, T_ACCEL = 9.81, 2.0, 0.5
SLOPE = ACCEL / GRAVITY
SLOPE_TOL = 0.5
FOLLOW_TOL = 0.10
TANK_HEIGHT = 0.15
BINS = 10


def _volume_l(points: np.ndarray, pv: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1).sum(axis=-1) * pv * 1e3


def _tank_frame(ph: c.Phase) -> np.ndarray:
    """Points (T, N, P, 3) minus the tank's displacement since the phase's first row."""
    pos = ph.probe["tank_pos"].astype(np.float64)
    return ph.probe["liquid_points"].astype(np.float64) - (pos - pos[:1])[:, :, None, :]


def _reference_box(points0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per world (N, 3) lower and upper bounds of the first row's points."""
    return np.nanmin(points0, axis=1), np.nanmax(points0, axis=1)


def _outside_share(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(N,) worst share over rows of points outside [lo, hi] (N, 3); points (T, N, P, 3)."""
    bad = ((points < lo[None, :, None, :]) | (points > hi[None, :, None, :])).any(axis=-1)
    live = np.isfinite(points).all(axis=-1)
    return (bad & live).sum(axis=-1).max(axis=0) / np.maximum(live[0].sum(axis=-1), 1)


def _containment_bounds(p0: np.ndarray, rim: bool) -> tuple[np.ndarray, np.ndarray]:
    lo0, hi0 = _reference_box(p0)
    lo = lo0 - np.array([MARGIN, MARGIN, FLOOR_MARGIN])
    hi = hi0 + np.array([MARGIN, MARGIN, MARGIN if not rim else 0.0])
    if rim:
        hi[:, 2] = np.inf           # over the rim is a splash, not a wall failure (CHECK-CHANGE E16)
    return lo, hi


def _depth(points: np.ndarray, pv: float) -> float:
    z = points[np.isfinite(points).all(axis=-1), 2]
    return float(np.percentile(z, 99.5) - np.percentile(z, 0.5) + pv ** (1 / 3))


def _surface_slope(points: np.ndarray) -> float:
    """dz/dx of the top surface: line through the 98th percentile height of BINS bins along x."""
    p = points[np.isfinite(points).all(axis=-1)]
    edges = np.linspace(p[:, 0].min(), p[:, 0].max() + 1e-9, BINS + 1)
    idx = np.digitize(p[:, 0], edges) - 1
    xs, zs = [], []
    for b in range(BINS):
        sel = p[idx == b]
        if len(sel) >= 20:
            xs.append(sel[:, 0].mean())
            zs.append(np.percentile(sel[:, 2], 98))
    return float(np.polyfit(xs, zs, 1)[0]) if len(xs) >= 4 else np.nan


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    pts, pv = rest.probe["liquid_points"].astype(np.float64), rest.probe["particle_volume"]
    vol = _volume_l(pts[:1], pv[:1])[0]
    levels = np.array([_depth(pts[-1, w], pv[-1, w]) for w in range(rest.num_envs)])
    lo, hi = _containment_bounds(pts[0], rim=False)
    outside = _outside_share(_tank_frame(rest), lo, hi)
    return [c.within("volume_l", vol, *VOLUME_L, 2),
            c.within("rest_level_m", levels, LEVEL_M - LEVEL_TOL, LEVEL_M + LEVEL_TOL, 2),
            c.at_most("rest_inside_share_outside", float(outside.max()), 1 - INSIDE_FRACTION, 2)]


def _tilt_check(ph: c.Phase) -> c.Check:
    rows = np.flatnonzero(ph.t <= T_ACCEL + 1e-9)
    pts = ph.probe["liquid_points"].astype(np.float64)
    mean = np.array([np.nanmean([-_surface_slope(pts[k, w]) for k in rows]) for w in range(ph.num_envs)])
    return c.within("surface_slope", mean, SLOPE * (1 - SLOPE_TOL), SLOPE * (1 + SLOPE_TOL), 2)


def _follow_checks(ph: c.Phase) -> list[c.Check]:
    x = ph.probe["tank_pos"][..., 0].astype(np.float64)
    x = x - x[:1]
    k5 = int(np.argmin(np.abs(ph.t - T_ACCEL)))
    k4 = int(np.argmin(np.abs(ph.t - (T_ACCEL - 0.1))))
    speed = (x[k5] - x[k4]) / (ph.t[k5] - ph.t[k4])
    d, v = 0.5 * ACCEL * T_ACCEL ** 2, ACCEL * (T_ACCEL - 0.05)
    return [c.within("tank_displacement_m", x[k5], d * (1 - FOLLOW_TOL), d * (1 + FOLLOW_TOL), 2),
            c.within("tank_speed_mps", speed, v * (1 - FOLLOW_TOL), v * (1 + FOLLOW_TOL), 2)]


def _act_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("act_slosh")
    pts = ph.probe["liquid_points"].astype(np.float64)
    lo, hi = _containment_bounds(pts[0], rim=True)
    outside = _outside_share(_tank_frame(ph), lo, hi)
    pv = ph.probe["particle_volume"]
    kept = _volume_l(pts[-1:], pv[-1:])[0] / _volume_l(pts[:1], pv[:1])[0]
    return [_tilt_check(ph), *_follow_checks(ph),
            c.at_most("contained_share_outside", float(outside.max()), 1 - INSIDE_FRACTION, 2),
            c.within("volume_kept", kept, 0.95, 1.05, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + _act_checks(rollout)
    return c.finish(checks)
