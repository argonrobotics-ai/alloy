# SPDX-License-Identifier: Apache-2.0
"""E20 arm push: rest, joint response, start/target geometry, the policy pushes the cube to the target, and the cube
moves only when the hand is near it.

Derivations (request sentence only, no engine frames):
- Rest: a 6 cm cube on a table: centre at table_z + 0.03 +- 3 mm; it stays put (< 2 mm over the 2 s rest, speed under
  1 cm/s at the end); every arm joint moves < 0.01 rad.
- Joints: +1 is 0.05 rad per step; 10 steps = 0.5 rad, accepted 0.3-0.6 rad (servo lag below, overshoot above), for
  joint 1 at +1 and joint 4 at -1 (the sign check), measured first row to last row of each 1 s phase.
- Targets: "15-25 cm away on the table": the horizontal distance cube start -> target in [0.149, 0.251] (1 mm slack)
  at every start row (rest, each phase's row 0, each post-done row in "random"); targets differ between worlds
  (std > 1 cm per axis) and lie on the table (z within table_z - 0.005 .. table_z + 0.035: a surface point, or the
  cube centre's height).
- Policy (level 3): on the last row of the first episode the cube centre is within 3 cm (horizontal) of the target,
  "still on the table and upright": centre height <= table_z + 0.034. A 6 cm cube resting on a face has its centre at
  0.03; tilted by t about an edge it is at 0.03 (cos t + sin t), which exceeds 0.034 from t = 8 degrees, and off the
  table it is below table_z + 0.02; so the window [table_z + 0.026, table_z + 0.034] means flat on the table
  (the probes carry no orientation). In >= 80 % of worlds.
- Contact law: the cube moves only while the arm is within 10 cm of it. At every row (any phase, not the row of a
  reset, not while the cube is off the table) where the cube centre moved more than 1.5 mm since the previous row,
  some part of the arm must have been within 0.12 m of the cube centre at that row or in the previous 0.5 s.
  "The arm" is the Panda's centre line (joint frames, flange, grasp point, every 2 cm) plus the hand's 0.2 m width
  bar, built from joint_pos by Franka's published kinematics and anchored on ee_pos (checkers/panda_points.py): the
  probes give the grasp point only, but the forearm and the 0.2 m wide hand touch a cube 13+ cm from it (seen in the
  reference's random phase). CHECK-CHANGE: 0.10 m (the sentence) + 0.02 m because the arm's surface is thicker than
  that centre line (contacts were seen at 0.117 m), and 0.5 s of coasting because a struck cube slides or tumbles on
  after the arm has left (a 1 m/s strike stops in about 0.2 s at mu 0.5). No violation allowed in any row of any
  world: nothing else moves a cube on a table (a teleport or a stray force breaks it).
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import panda_points as pp
from bench.newenv.checkers.arm_common import coast_rows, episode_end, recent

CUBE_HALF = 0.03
DIST_LO, DIST_HI, DIST_SLACK = 0.15, 0.25, 0.001
SUCCESS_M = 0.03
FLAT_TOL = 0.004
NEAR_M = 0.12
MOVE_M = 0.0015
POLICY_RATE = 0.8
ARM_PHASES = ("rest", "act_joint", "act_joint4", "random", "policy")


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    cube, q = rest.probe["cube_pos"], rest.probe["joint_pos"]
    z = cube[-1, :, 2] - rest.probe["table_z"][-1]
    moved = np.linalg.norm(cube[-1] - cube[0], axis=-1)
    drift = np.abs(q[-1] - q[0]).max(axis=-1)
    return [c.within("rest_cube_height", z, CUBE_HALF - 0.003, CUBE_HALF + 0.003, 2),
            c.within("rest_cube_still", moved, 0.0, 0.002, 2),
            c.within("rest_speed", c.speed(cube, rest.t)[-10:].max(axis=0), 0.0, 0.01, 2),
            c.within("rest_arm_still", drift, 0.0, 0.01, 2)]


def _response_checks(r: c.Rollout) -> list[c.Check]:
    q1, q4 = r.phase("act_joint").probe["joint_pos"], r.phase("act_joint4").probe["joint_pos"]
    return [c.within("joint1_turn", q1[-1, :, 0] - q1[0, :, 0], 0.3, 0.6, 2),
            c.within("joint4_turn", q4[-1, :, 3] - q4[0, :, 3], -0.6, -0.3, 2)]


def start_rows(p: c.Phase) -> list[tuple[np.ndarray, np.ndarray]]:
    """(cube_pos, target_pos) pairs (N, 3) at row 0 and at every row where some world just reset."""
    rows = [0] + [k for k in range(1, len(p.t)) if p.done[k].any()]
    return [(p.probe["cube_pos"][k][m], p.probe["target_pos"][k][m])
            for k in rows for m in [np.ones(p.num_envs, bool) if k == 0 else p.done[k]]]


def _target_checks(r: c.Rollout) -> list[c.Check]:
    pairs = [x for name in ARM_PHASES if r.has(name) for x in start_rows(r.phase(name))]
    cube = np.concatenate([a for a, _ in pairs])
    target = np.concatenate([b for _, b in pairs])
    dist = np.linalg.norm((target - cube)[:, :2], axis=-1)
    tz = target[:, 2] - r.phase("rest").probe["table_z"][0, 0]
    spread = float(np.std(r.phase("rest").probe["target_pos"][0, :, :2], axis=0).min())
    return [c.within("target_distance", dist, DIST_LO - DIST_SLACK, DIST_HI + DIST_SLACK, 2),
            c.within("target_on_table", tz, -0.005, 0.035, 2),
            c.at_least("target_spread", spread, 0.01, 2),
            c.at_least("cube_spread", float(np.std(r.phase("rest").probe["cube_pos"][0, :, :2], axis=0).min()), 0.005, 2)]


def _policy_check(r: c.Rollout) -> list[c.Check]:
    p = r.phase("policy")
    idx, w = episode_end(p), np.arange(p.num_envs)
    cube, target = p.probe["cube_pos"][idx, w], p.probe["target_pos"][idx, w]
    miss = np.linalg.norm((cube - target)[:, :2], axis=-1)
    z = cube[:, 2] - p.probe["table_z"][idx, w]
    good = (miss <= SUCCESS_M) & (np.abs(z - CUBE_HALF) <= FLAT_TOL)
    return [c.at_least("policy_push_to_target", float(good.mean()), POLICY_RATE, 3)]


def contact_violations(p: c.Phase) -> int:
    """Rows where the cube moved with the whole arm farther than NEAR_M for the last COAST_S."""
    cube, ee = p.probe["cube_pos"].astype(np.float64), p.probe["ee_pos"].astype(np.float64)
    moved = np.linalg.norm(np.diff(cube, axis=0), axis=-1) > MOVE_M
    near = pp.distance_to_arm(p.probe["joint_pos"].astype(np.float64), ee, cube) <= NEAR_M
    near_recent = recent(near, coast_rows(p.t))[1:]
    on_table = cube[1:, :, 2] - p.probe["table_z"][1:] > 0.02
    return int((moved & ~near_recent & on_table & ~p.done[1:]).sum())


def _contact_check(r: c.Rollout) -> c.Check:
    total = sum(contact_violations(r.phase(n)) for n in ARM_PHASES if r.has(n))
    return c.at_most("cube_moves_only_near_hand", float(total), 0.0, 2)


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + _response_checks(rollout)
    checks += _target_checks(rollout) + _policy_check(rollout) + [_contact_check(rollout)]
    return c.finish(checks)
