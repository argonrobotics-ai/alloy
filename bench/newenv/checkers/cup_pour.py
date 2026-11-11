# SPDX-License-Identifier: Apache-2.0
"""E15 cup pour: water volume, water stays in the cup at rest, a 0.5 rad/s tilt pours it into the bowl.

Derivations (from the request sentence only; no engine's frames, particle order or counts):
- Volume: points x particle_volume at the first row is 200 ml +- 10 % (180-220 ml).
- Rest (2 s, zero action): the water's start region is read from the first row of the same run: a cylinder around the
  start centroid with the start points' largest horizontal radius + 1 cm, spanning the start height range -1 cm to
  +2 cm. >= 95 % of the points inside it in every frame, in every world (all but one world).
- Pour: tilting at 0.5 rad/s (action 0.5 x 1 rad/s) to 2.2 rad takes 4.4 s; the law (scripts/cup_pour.py) then holds.
  At the end of the phase >= 70 % of the points are within bowl_radius of bowl_pos (horizontally). Ground is z = 0
  (z up, world frame): no point below -5 mm. A point count that changed (leak, deletion) shows as a volume drift:
  the end volume stays within 10 % of the start volume.
- Tilt rate: while the commanded rate is 0.5 rad/s (rows with action ~ 0.5 before the tilt reaches 2.2 rad), the mean
  rate of cup_tilt is 0.5 rad/s +- 10 %.
Where the bowl sits relative to the cup is the agent's choice, so the pour's success is the one thing this file does
not fix; it decides only that the water reaches the bowl the agent itself reports.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

VOLUME_ML = (180.0, 220.0)
REST_FRACTION = 0.95
POUR_FRACTION = 0.7
GROUND_Z = -0.005
RATE = 0.5
RATE_TOL = 0.10


def _volumes_ml(points: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """Per world volume (ml) of the rows (T, N, P, 3) given pv (T, N); NaN points (not yet emitted) do not count."""
    finite = np.isfinite(points).all(axis=-1)
    return finite.sum(axis=-1) * pv * 1e6


def _in_start_region(points: np.ndarray) -> np.ndarray:
    """(T, N) share of points inside the cylinder fixed by row 0 (see the module doc)."""
    p0 = points[0]
    centre = np.nanmean(p0[..., :2], axis=1, keepdims=True)
    radius = np.nanmax(np.linalg.norm(p0[..., :2] - centre, axis=-1), axis=1) + 0.01
    zlo = np.nanmin(p0[..., 2], axis=1) - 0.01
    zhi = np.nanmax(p0[..., 2], axis=1) + 0.02
    d = np.linalg.norm(points[..., :2] - centre[None], axis=-1)
    inside = (d <= radius[None, :, None]) & (points[..., 2] >= zlo[None, :, None]) & (points[..., 2] <= zhi[None, :, None])
    return inside.mean(axis=-1)


def _volume_and_rest(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    vol = _volumes_ml(rest.probe["liquid_points"], rest.probe["particle_volume"])
    stays = _in_start_region(rest.probe["liquid_points"]).min(axis=0)
    return [c.within("volume_ml", vol[0], *VOLUME_ML, 2),
            c.within("rest_in_cup", stays, REST_FRACTION, 1.0, 2, min_worlds=rest.num_envs - 1)]


def _pour_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("act_pour")
    pts = ph.probe["liquid_points"][-1]
    bowl, rad = ph.probe["bowl_pos"][-1], ph.probe["bowl_radius"][-1]
    finite = np.isfinite(pts).all(axis=-1)
    in_bowl = (np.linalg.norm(pts[..., :2] - bowl[:, None, :2], axis=-1) < rad[:, None]) & finite
    share = in_bowl.sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
    below = ((pts[..., 2] < GROUND_Z) & finite).sum(axis=1)
    v0 = _volumes_ml(ph.probe["liquid_points"][:1], ph.probe["particle_volume"][:1])[0]
    v1 = _volumes_ml(ph.probe["liquid_points"][-1:], ph.probe["particle_volume"][-1:])[0]
    return [c.within("pour_in_bowl", share, POUR_FRACTION, 1.0, 2, min_worlds=ph.num_envs - 1),
            c.at_most("pour_below_ground", float(below.max()), 0.0, 2),
            c.within("pour_volume_kept", v1 / v0, 0.9, 1.1, 2)]


def _rate_check(r: c.Rollout) -> c.Check:
    ph = r.phase("act_pour")
    tilt, act = ph.probe["cup_tilt"], ph.action[..., 0]
    rate = np.full(tilt.shape[1], np.nan)
    for w in range(tilt.shape[1]):
        rows = np.flatnonzero((np.abs(act[:, w] - 0.5) < 0.05) & (tilt[:, w] < 2.1))
        if len(rows) >= 3:
            rate[w] = (tilt[rows[-1], w] - tilt[rows[0] - 1, w]) / (ph.t[rows[-1]] - ph.t[rows[0] - 1])
    return c.within("tilt_rate", rate, RATE * (1 - RATE_TOL), RATE * (1 + RATE_TOL), 2)


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _volume_and_rest(rollout) + _pour_checks(rollout)
    checks.append(_rate_check(rollout))
    return c.finish(checks)
