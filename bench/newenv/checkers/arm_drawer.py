# SPDX-License-Identifier: Apache-2.0
"""E21 arm drawer: rest, joint and gripper response, handle and drawer move together, the policy opens the drawer,
and the drawer moves only while the arm is at it.

Derivations (request sentence only):
- Rest: the drawer starts closed: drawer_open stays within [-2 mm, +2 mm] over the 2 s rest; no arm joint moves
  more than 0.01 rad.
- Joint / gripper: as E19: +1 on joint 1 is 0.05 rad per step, 10 steps = 0.5 rad, accepted 0.3-0.6 rad; a close
  then open command changes gripper_width by >= 5 cm each way.
- Handle: "handle_pos moves with drawer_open one to one along the rail axis": in every phase and world where
  drawer_open changes by more than 5 mm, the handle displacement h(t) = handle_pos(t) - handle_pos(0) equals u d(t)
  with d = drawer_open(t) - drawer_open(0) for one fixed vector u: |u| in [0.98, 1.02] (one to one) and the largest
  residual |h - u d| <= 3 mm (it moves along a straight rail). u is fitted by least squares, no engine frame needed.
  Where nothing moves the check holds trivially (level 2); the level-3 check "handle_moves_with_drawer" needs the
  relation seen in the policy phase.
- Travel: "the drawer slides 30 cm": every row of every phase has drawer_open in [-2 mm, 0.302 m].
- Policy (level 3): on the last row of the first episode drawer_open >= 0.20 m, in >= 80 % of worlds (the limit 0.30 m
  is the travel check above).
- Contact law: the drawer moves only while the hand is at the handle. Two checks, both "no row of any world breaks
  it", on rows where drawer_open changed by more than 1 mm since the previous row (reset rows excluded), with the
  arm taken as the Panda's centre line plus the 0.2 m wide hand bar built from joint_pos (checkers/panda_points.py,
  anchored on ee_pos) and 0.5 s of coasting after the arm was last near:
  * policy phase: some part of the arm within 0.12 m of handle_pos (the sentence's 10 cm, plus 2 cm for the surface
    the centre line misses, as in E20): the scripted policy must open the drawer by its handle.
  * every phase: some part of the arm within 0.40 m of handle_pos. CHECK-CHANGE: in "random" the arm legitimately
    pushes the drawer at its front panel, which is 0.54 m wide: the reference's random phase moved the drawer with the
    arm 0.137 m from the handle. A drawer's panel corner is 0.30 m from the handle and the arm's surface adds 0.06 m,
    so 0.40 m still catches a drawer that moves with the arm elsewhere (a stray force, a teleport, a slide that runs
    on after the arm left: friction 2 N stops a 1 kg drawer within a few mm).
- Travel CHECK-CHANGE: drawer_open in [-0.015, 0.315]: the rail's end stops are soft under the arm's push (the
  reference's random phase pressed it 10 mm into the closed stop).
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import panda_points as pp
from bench.newenv.checkers.arm_common import coast_rows, episode_end, recent

OPEN_GOAL, TRAVEL = 0.20, 0.30
MOVE_M, NEAR_M, NEAR_ANY_M = 0.001, 0.12, 0.40
SOFT_STOP_M = 0.015
MOVED_FOR_FIT_M = 0.005
POLICY_RATE = 0.8
PHASES = ("rest", "act_joint", "act_gripper", "random", "policy")


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    drift = np.abs(rest.probe["joint_pos"][-1] - rest.probe["joint_pos"][0]).max(axis=-1)
    return [c.within("rest_drawer_closed", np.abs(rest.probe["drawer_open"]).max(axis=0), 0.0, 0.002, 2),
            c.within("rest_arm_still", drift, 0.0, 0.01, 2)]


def _response_checks(r: c.Rollout) -> list[c.Check]:
    q = r.phase("act_joint").probe["joint_pos"]
    w = r.phase("act_gripper").probe["gripper_width"]
    closed = w.min(axis=0)
    return [c.within("joint1_turn", q[-1, :, 0] - q[0, :, 0], 0.3, 0.6, 2),
            c.within("gripper_close", w[0] - closed, 0.05, np.inf, 2),
            c.within("gripper_open", w[-1] - closed, 0.05, np.inf, 2)]


def rail_fit(p: c.Phase, world: int, end: int) -> tuple[float, float] | None:
    """(|u|, largest residual m) of handle displacement against drawer displacement for one world, rows 0..end;
    None when the drawer moved less than MOVED_FOR_FIT_M."""
    d = p.probe["drawer_open"][:end + 1, world].astype(np.float64)
    d = d - d[0]
    if np.abs(d).max() < MOVED_FOR_FIT_M:
        return None
    h = p.probe["handle_pos"][:end + 1, world].astype(np.float64)
    h = h - h[0]
    u = (h * d[:, None]).sum(axis=0) / float((d * d).sum())
    return float(np.linalg.norm(u)), float(np.abs(h - u * d[:, None]).max())


def _fits(r: c.Rollout, names: tuple[str, ...]) -> list[tuple[float, float]]:
    out = []
    for name in names:
        if not r.has(name):
            continue
        p = r.phase(name)
        ends = episode_end(p)
        out += [f for w in range(p.num_envs) if (f := rail_fit(p, w, int(ends[w]))) is not None]
    return out


def _handle_checks(r: c.Rollout) -> list[c.Check]:
    fits = _fits(r, PHASES)
    scale = np.array([f[0] for f in fits]) if fits else np.ones(1)
    resid = np.array([f[1] for f in fits]) if fits else np.zeros(1)
    seen = _fits(r, ("policy",))
    return [c.within("handle_one_to_one", scale, 0.98, 1.02, 2), c.within("handle_on_straight_rail", resid, 0.0, 0.003, 2),
            c.at_least("handle_moves_with_drawer", float(len(seen)), 1.0, 3)]


def _travel_check(r: c.Rollout) -> c.Check:
    rows = np.concatenate([r.phase(n).probe["drawer_open"].ravel() for n in PHASES if r.has(n)])
    return c.within("drawer_travel", rows, -SOFT_STOP_M, TRAVEL + SOFT_STOP_M, 2)


def _policy_check(r: c.Rollout) -> c.Check:
    p = r.phase("policy")
    final = p.probe["drawer_open"][episode_end(p), np.arange(p.num_envs)]
    return c.at_least("policy_drawer_open", float((final >= OPEN_GOAL).mean()), POLICY_RATE, 3)


def contact_violations(p: c.Phase, near_m: float) -> int:
    """Rows where the drawer moved with the whole arm farther than ``near_m`` from the handle for the last 0.5 s."""
    drawer = p.probe["drawer_open"].astype(np.float64)
    moved = np.abs(np.diff(drawer, axis=0)) > MOVE_M
    dist = pp.distance_to_arm(p.probe["joint_pos"].astype(np.float64), p.probe["ee_pos"].astype(np.float64),
                              p.probe["handle_pos"].astype(np.float64))
    near = recent(dist <= near_m, coast_rows(p.t))[1:]
    return int((moved & ~near & ~p.done[1:]).sum())


def _contact_checks(r: c.Rollout) -> list[c.Check]:
    anywhere = sum(contact_violations(r.phase(n), NEAR_ANY_M) for n in PHASES if r.has(n))
    return [c.at_most("drawer_moves_only_near_hand", float(contact_violations(r.phase("policy"), NEAR_M)), 0.0, 2),
            c.at_most("drawer_moves_only_with_arm_near", float(anywhere), 0.0, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + _response_checks(rollout)
    checks += _handle_checks(rollout) + [_travel_check(rollout), _policy_check(rollout)] + _contact_checks(rollout)
    return c.finish(checks)
