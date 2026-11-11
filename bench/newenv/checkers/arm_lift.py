# SPDX-License-Identifier: Apache-2.0
"""E19 arm lift: rest pose, joint and gripper response, scripted policy lifts the cube; cube starts differ.

Derivations (request sentence only):
- Rest: a 4 cm cube on the table: centre at table_z + 0.02 +- 3 mm; the arm holds still: every joint moves < 0.01 rad
  between the first and last rest row.
- Joint: +1 on joint 1 is 0.05 rad per step; 10 steps = 0.5 rad, accepted 0.3-0.6 rad (servo lag below, overshoot
  above), measured as the change of joint_pos[0] between the first and the last row of the 1 s phase.
- Gripper: a close (-1) then open (+1) command moves gripper_width by >= 5 cm: the width range over the phase is
  >= 0.05 m, the closed width is >= 0.05 below the start, and the final width is >= 0.05 above the closed one.
- Policy (level 3): "cube held 15 cm above the table for 1 s": the cube centre is >= table_z + 0.15 and within
  3 cm of ee_pos for rows spanning >= 1 s (one unbroken run, inside the first episode), in >= 80 % of worlds. (The
  cube's centre rests 2 cm up, so a lift of its centre by 15 cm would be table_z + 0.17; the looser reading is
  accepted: the sentence does not say which point of the cube.)
- Starts: the cube starts lie in a 10 cm square: per axis range of all starts (rest + random reset rows) <= 0.12 m
  and the spread across worlds is > 5 mm.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

CUBE_HALF = 0.02
LIFT = 0.15
HOLD_S = 1.0
NEAR_M = 0.03
POLICY_RATE = 0.8


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    z = rest.probe["cube_pos"][-1, :, 2] - rest.probe["table_z"][-1]
    drift = np.abs(rest.probe["joint_pos"][-1] - rest.probe["joint_pos"][0]).max(axis=-1)
    return [c.within("rest_cube_height", z, CUBE_HALF - 0.003, CUBE_HALF + 0.003, 2),
            c.within("rest_arm_still", drift, 0.0, 0.01, 2)]


def _response_checks(r: c.Rollout) -> list[c.Check]:
    q = r.phase("act_joint").probe["joint_pos"]
    turn = q[-1, :, 0] - q[0, :, 0]
    w = r.phase("act_gripper").probe["gripper_width"]
    closed = w.min(axis=0)
    return [c.within("joint1_turn", turn, 0.3, 0.6, 2),
            c.within("gripper_close", w[0] - closed, 0.05, np.inf, 2),
            c.within("gripper_open", w[-1] - closed, 0.05, np.inf, 2)]


def longest_run_s(ok: np.ndarray, t: np.ndarray) -> float:
    """Longest time span (s) of consecutive True rows: (last t - first t) of the run."""
    best, start = 0.0, None
    for k, flag in enumerate(ok):
        if flag and start is None:
            start = k
        if start is not None and (not flag or k == len(ok) - 1):
            end = k if flag else k - 1
            best, start = max(best, float(t[end] - t[start])), None
    return best


def held_seconds(p: c.Phase) -> np.ndarray:
    """(N,) longest hold of the lifted-and-near condition inside each world's first episode."""
    cube, ee, table = p.probe["cube_pos"], p.probe["ee_pos"], p.probe["table_z"]
    out = np.zeros(cube.shape[1])
    for w in range(cube.shape[1]):
        hits = np.flatnonzero(p.done[:, w])
        end = hits[0] + 1 if len(hits) else len(p.t)           # a done row already shows the next episode's start
        up = cube[:end, w, 2] - table[:end, w] >= LIFT
        near = np.linalg.norm(cube[:end, w] - ee[:end, w], axis=-1) <= NEAR_M
        out[w] = longest_run_s(up & near, p.t[:end])
    return out


def _policy_check(r: c.Rollout) -> c.Check:
    held = held_seconds(r.phase("policy"))
    share = float((held >= HOLD_S).mean())
    return c.at_least("policy_lift_hold", share, POLICY_RATE, 3)


def _start_checks(r: c.Rollout) -> list[c.Check]:
    starts = np.concatenate([r.phase("rest").probe["cube_pos"][0], r.phase("random").probe["cube_pos"][0]])
    spread = float(np.std(r.phase("rest").probe["cube_pos"][0, :, :2], axis=0).max())
    return [c.at_least("cube_starts_differ", spread, 0.005, 2),
            c.at_most("cube_starts_in_square", float((starts[:, :2].max(0) - starts[:, :2].min(0)).max()), 0.12, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + _response_checks(rollout)
    checks += _start_checks(rollout) + [_policy_check(rollout)]
    return c.finish(checks)
