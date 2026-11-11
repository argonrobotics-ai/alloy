# SPDX-License-Identifier: Apache-2.0
"""E24 arm cloth: a 30 cm, 50 g cloth covers a 6 cm cube; the Panda pulls it off without moving the cube.

Derivations (request sentence only; no engine's frames, vertex order or counts):
- Cube: 6 cm edge, ``cube_pos`` its centre, so its top face is at cube_pos z + 0.03 and its footprint the square
  |dx|, |dy| < 0.03 about cube_pos.
- Rest (zero action): the cloth covers the cube: at the last rest row >= 1 cloth point over the top face (footprint,
  height top - 5 mm to top + 3 cm) and the sheet's horizontal bounding box spans the whole footprint. No penetration
  > 5 mm in any rest row: no point inside the cube shrunk by 5 mm on every side, none more than 5 mm below table_z.
- Stretch: a 30 cm square sheet has a 30 cm x sqrt 2 = 42.4 cm diagonal and no larger extent; extent over the 13
  directions (common.extent) <= 1.15 x 42.4 cm in every row of every phase (a draped or crumpled sheet reads smaller).
- act_joint (harness law: +1 on joint 1 for 10 steps = 10 x 0.05 rad): joint 1 turns 0.3-0.6 rad.
- act_gripper (close, then open): gripper_width range >= 5 cm, closed width >= 5 cm below the start, final width
  >= 5 cm above the closed one (as E19: a Panda hand opens 8 cm).
- Policy (level 3): in the last state of the first episode (the row before the first done) no cloth point is over the
  footprint above top - 1 cm (the success sentence), and the cube centre is < 2 cm from its start (row 0 of the phase) at that same last state
  ("success is ... while the cube has moved less than 2 cm" is a condition on the success state, not on the path), and
  never > 6 cm from it in the episode (a cube knocked away and back is not "still"; a transient tug of 3-4 cm is not); the cloth is still a sheet: finite, stretch <= 1.15, nothing > 5 mm below table_z nor > 1 m above it
  (a sheet teleported away or deleted is not a pull). All in
  >= 80 % of worlds.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

CUBE = 0.06
SIDE = 0.30
STRETCH_MAX = 1.15
PENETRATION = 0.005
MOVED_MAX = 0.02
MOVED_PEAK = 0.06
GRIPPER_TRAVEL = 0.05
JOINT_TURN = (0.3, 0.6)
WORLD_RATE = 0.8
NEAR_TABLE_M = 1.0
REST_EXTENT = SIDE * np.sqrt(2.0)


def _footprint(points: np.ndarray, cube: np.ndarray, half: float) -> np.ndarray:
    """(..., V) points inside the cube's footprint square of half-width ``half``; cube (..., 3) broadcasts."""
    d = np.abs(points[..., :2] - cube[..., None, :2])
    return (d[..., 0] < half) & (d[..., 1] < half)


def count_over_top(points: np.ndarray, cube: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """(...) points over the footprint with height between cube top + lo and cube top + hi."""
    z = points[..., 2] - (cube[..., None, 2] + CUBE / 2)
    return (_footprint(points, cube, CUBE / 2) & (z >= lo) & (z <= hi)).sum(axis=-1)


def count_penetrating(points: np.ndarray, cube: np.ndarray, table: np.ndarray) -> np.ndarray:
    """(...) points inside the cube shrunk by 5 mm on each side, or more than 5 mm below the table."""
    z = points[..., 2]
    inside = (_footprint(points, cube, CUBE / 2 - PENETRATION) & (z > table[..., None] + PENETRATION)
              & (z < cube[..., None, 2] + CUBE / 2 - PENETRATION))
    return inside.sum(axis=-1) + (z < table[..., None] - PENETRATION).sum(axis=-1)


def covers_footprint(points: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """(N,) the sheet's horizontal bounding box spans the cube's footprint (points (N, V, 3), cube (N, 3))."""
    lo, hi = points[..., :2].min(axis=1), points[..., :2].max(axis=1)
    half = CUBE / 2
    return np.all((lo < cube[:, :2] - half) & (hi > cube[:, :2] + half), axis=-1)


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("rest")
    pts, cube, table = p.probe["cloth_points"], p.probe["cube_pos"], p.probe["table_z"]
    over = count_over_top(pts[-1], cube[-1], -PENETRATION, 0.03)
    deep = count_penetrating(pts, cube[:, :, :], table).max(axis=0)
    return [c.within("rest_covers_cube", ((over >= 1) & covers_footprint(pts[-1], cube[-1])).astype(float), 1.0, 1.0, 2),
            c.within("rest_no_penetration", deep, 0.0, 0.0, 2)]


def _stretch_check(r: c.Rollout) -> c.Check:
    worst = 0.0
    for name in r.phases:
        pts = r.phase(name).probe.get("cloth_points")
        if pts is not None:
            worst = max(worst, float(np.nanmax(c.stretch(pts, REST_EXTENT))))
    return c.at_most("cloth_stretch", worst, STRETCH_MAX, 2)


def _response_checks(r: c.Rollout) -> list[c.Check]:
    q = r.phase("act_joint").probe["joint_pos"]
    w = r.phase("act_gripper").probe["gripper_width"]
    closed = w.min(axis=0)
    return [c.within("joint1_turn", q[-1, :, 0] - q[0, :, 0], *JOINT_TURN, 2),
            c.within("gripper_close", w[0] - closed, GRIPPER_TRAVEL, np.inf, 2),
            c.within("gripper_open", w[-1] - closed, GRIPPER_TRAVEL, np.inf, 2)]


def _first_episode_end(p: c.Phase, w: int) -> int:
    hits = np.flatnonzero(p.done[:, w])
    return int(hits[0]) - 1 if len(hits) and hits[0] > 0 else len(p.t) - 1


def _intact(pts: np.ndarray, table: np.ndarray) -> bool:
    """Finite, within the stretch bound in every row, and nowhere more than 5 mm below the table."""
    if not np.isfinite(pts).all():
        return False
    stretched = np.nanmax(c.stretch(pts[:, None], REST_EXTENT))
    near = (pts[..., 2] <= table[:, None] + NEAR_TABLE_M).all()
    return bool(stretched <= STRETCH_MAX and near and (pts[..., 2] >= table[:, None] - PENETRATION).all())


def _policy_worlds(p: c.Phase) -> dict[str, np.ndarray]:
    """Per world booleans for the policy rows of the first episode."""
    n = p.num_envs
    out = {k: np.zeros(n, bool) for k in ("clear", "cube_still", "sheet")}
    for w in range(n):
        end = _first_episode_end(p, w)
        pts, cube = p.probe["cloth_points"][: end + 1, w], p.probe["cube_pos"][: end + 1, w]
        table = p.probe["table_z"][: end + 1, w]
        away = np.linalg.norm(cube - cube[0], axis=-1)
        out["clear"][w] = count_over_top(pts[end], cube[end], -0.01, np.inf) == 0
        out["cube_still"][w] = away[end] < MOVED_MAX and away.max() < MOVED_PEAK
        out["sheet"][w] = _intact(pts, table)
    return out


def _policy_checks(r: c.Rollout) -> list[c.Check]:
    w = _policy_worlds(r.phase("policy"))
    both = w["clear"] & w["cube_still"] & w["sheet"]
    return [c.at_least("policy_cloth_clear", float(w["clear"].mean()), WORLD_RATE, 3),
            c.at_least("policy_cube_still", float(w["cube_still"].mean()), WORLD_RATE, 3),
            c.at_least("policy_sheet_intact", float(w["sheet"].mean()), WORLD_RATE, 3),
            c.at_least("policy_all_conditions", float(both.mean()), WORLD_RATE, 3)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + [_stretch_check(rollout)]
    checks += _response_checks(rollout) + _policy_checks(rollout)
    return c.finish(checks)
