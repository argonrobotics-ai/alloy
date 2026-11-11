# SPDX-License-Identifier: Apache-2.0
"""E03 bin clutter: ten objects settle inside a 30 cm x 30 cm x 15 cm bin; the bin follows the action at 0.2 m/s.

Derivations (from the request sentence only):
- Rest: the sentence gives 3 s; the driver's "rest" phase is 2 s, so the script adds "rest_more" (1 s, zero actions,
  no reset) and the rest checks read its last rows (CHECK-CHANGE: rest_s cannot be edited in requests.json).
- Inside the footprint: object centre within +-0.15 m of ``bin_pos`` in x and y (the bin is 0.30 m square and bin_pos
  is its position). Above the floor: z >= 0.015 (a 2.5 cm sphere on the ground has its centre at 0.025, a 4 cm box
  at 0.02; allow floor-wall slop); below the rim: z <= 0.15 (the bin stands on the ground and is 15 cm deep).
- Speed under 5 cm/s over the last 10 rows.
- No two object centres closer than 3.0 cm in the last 3 rows of "rest" and every row of "rest_more". CHECK-CHANGE: the
  request says 3.5 cm, but two touching objects are at least r1 + r2 apart (box inscribed radius 2 cm, sphere 2.5 cm,
  cylinder its own radius, which the sentence leaves open): two 3 cm wide cylinders lying side by side are 3.0 cm apart
  with no overlap, so 3.5 cm would fail an honest env. 3.0 cm still catches any interpenetration of boxes (4 cm) and
  spheres (5 cm).
- Act: action +1 on axis 0 must move the bin at 0.2 m/s along +x (within 10 %, mean velocity over t in [0.3, 0.7] s,
  ie after any start-up transient), no y drift (< 1 cm/s); -1 on axis 1 likewise along -y.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

HALF = 0.15
RIM = 0.15
FLOOR = 0.015
SPEED = 0.2
MIN_GAP = 0.03


def _inside(objects: np.ndarray, bin_pos: np.ndarray) -> np.ndarray:
    """(N,) bool: every object centre inside the footprint, above the floor, below the rim (last row)."""
    rel = objects.astype(np.float64) - bin_pos[:, None, :]
    flat = np.all(np.abs(rel[..., :2]) <= HALF, axis=(1, 2))
    height = np.all((objects[..., 2] >= FLOOR) & (objects[..., 2] <= RIM), axis=1)
    return flat & height


def _min_pair_distance(objects: np.ndarray) -> np.ndarray:
    """(N,) smallest centre distance over all pairs and rows; objects (T, N, K, 3)."""
    d = np.linalg.norm(objects[:, :, :, None, :] - objects[:, :, None, :, :], axis=-1)
    k = d.shape[-1]
    d[..., np.arange(k), np.arange(k)] = np.inf
    return d.min(axis=(0, 2, 3))


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("rest_more")
    obj = p.probe["object_pos"]
    speed = c.speed(obj, p.t)
    rows = np.concatenate([r.phase("rest").probe["object_pos"][-3:], obj])
    return [c.at_least("rest_all_inside", float(_inside(obj[-1], p.probe["bin_pos"][-1]).sum()), obj.shape[1], 2),
            c.within("rest_speed", speed[-10:].max(axis=(0, 2)), 0.0, 0.05, 2),
            c.within("pair_distance", _min_pair_distance(rows), MIN_GAP, np.inf, 2)]


def _bin_velocity(p: c.Phase) -> np.ndarray:
    pos = p.probe["bin_pos"].astype(np.float64)
    t = p.t.astype(np.float64)
    i0 = int(np.searchsorted(t, 0.3 - 1e-6))
    return (pos[-1] - pos[i0]) / (t[-1] - t[i0])


def _move_checks(r: c.Rollout) -> list[c.Check]:
    vx, vy = _bin_velocity(r.phase("act_x")), _bin_velocity(r.phase("act_y"))
    return [c.within("bin_speed_x", vx[:, 0], 0.9 * SPEED, 1.1 * SPEED, 2),
            c.within("bin_drift_y", vx[:, 1], -0.01, 0.01, 2),
            c.within("bin_speed_y", vy[:, 1], -1.1 * SPEED, -0.9 * SPEED, 2),
            c.within("bin_drift_x", vy[:, 0], -0.01, 0.01, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _rest_checks(rollout) + _move_checks(rollout)
    return c.finish(checks)
