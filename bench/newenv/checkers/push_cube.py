# SPDX-License-Identifier: Apache-2.0
"""E01 push cube: rest height and stillness, response to a constant +x force, per-world starts and targets.

Derivations (all from the request sentence, nothing from an engine's frames):
- Rest: a 5 cm cube on a table whose top is at 0.5 m has its centre at 0.525 m (+-3 mm); speed < 1 cm/s.
- Push: the force is at most 1 N per axis on 0.1 kg, so the acceleration is at most F/m = 10 m/s^2 and in 0.2 s the
  cube moves at most 0.5 * 10 * 0.2^2 = 0.2 m (we allow 5 % for the first control period's discretisation). The
  floor is the weakest bound: friction is not given in the sentence, so the cube must merely move more than 5 mm in
  +x (it would not if the scale were off by 50x or the sign flipped). Sideways drift stays under 2 cm (no y force).
  The request says 1 s; at 1 N on 0.1 kg that pushes the cube 5 m, off a 1 m table (CHECK-CHANGE): the push is 0.2 s.
  "Not below the table": the cube centre stays above 0.525 - 3 mm in >= 12 of 16 worlds (a start within 0.2 m of an
  edge may legitimately leave the table within 0.2 s; the request lets starts be anywhere on the table).
- Starts and targets: any point on a 1 m square table lies within 1 m of any other, so the per-axis range of all starts
  and targets in "random" must be <= 1.0 m minus the cube edge (0.95) for the cube, <= 1.0 for targets; they must
  vary between worlds (std > 1 cm) and targets must sit on the table surface (z within 0.495 - 0.53).
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

CUBE_Z = 0.525
TABLE_TOP = 0.5
PUSH_MAX_M = 0.2 * 1.05
MIN_WORLDS_ON_TABLE = 12


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    pos = rest.probe["cube_pos"]
    speed = c.speed(pos, rest.t)
    return [c.within("rest_height", pos[-1, :, 2], CUBE_Z - 0.003, CUBE_Z + 0.003, 2),
            c.within("rest_speed", np.max(speed[-10:], axis=0), 0.0, 0.01, 2)]


def _push_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("act_push").probe["cube_pos"]
    d = p[-1] - p[0]
    zmin = p[:, :, 2].min(axis=0)
    return [c.within("push_dx", d[:, 0], 0.005, PUSH_MAX_M, 2),
            c.within("push_dy", d[:, 1], -0.02, 0.02, 2),
            c.within("push_on_table", zmin, CUBE_Z - 0.003, np.inf, 2, min_worlds=MIN_WORLDS_ON_TABLE)]


def _spread_checks(r: c.Rollout) -> list[c.Check]:
    rows = r.phase("random")
    cube = rows.probe["cube_pos"][0]
    target = rows.probe["target_pos"][0]
    checks = [c.at_least("start_spread", float(np.std(cube[:, :2], axis=0).min()), 0.01, 2),
              c.at_least("target_spread", float(np.std(target[:, :2], axis=0).min()), 0.01, 2),
              c.within("target_height", target[:, 2], TABLE_TOP - 0.005, TABLE_TOP + 0.03, 2)]
    both = np.concatenate([cube[:, :2], target[:, :2]])
    checks.append(c.at_most("on_table_extent", float((both.max(0) - both.min(0)).max()), 1.0, 2))
    return checks


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _rest_checks(rollout) + _push_checks(rollout) + _spread_checks(rollout)
    return c.finish(checks)
