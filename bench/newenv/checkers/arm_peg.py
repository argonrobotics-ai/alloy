# SPDX-License-Identifier: Apache-2.0
"""E22 arm peg: the held peg, joint response, hole positions, no pass through the block, the policy inserts the peg.

Derivations (request sentence only; nothing from an engine's frames):
- hole_pos is read as the centre of the hole's opening in the block's top face, hole axis vertical (a block on a
  table with a hole in its top): the sentence says "3 cm deep in the hole" and the check list "peg tip >= 3 cm below
  the hole's top"; the probe list gives one hole point, and the opening is the point depth is measured from.
- Peg axis: peg_quat (wxyz) rotates the peg's frame; the request does not say which local axis is the length, so the
  length axis is the local axis most nearly vertical (an inserted peg is vertical to within 1 mm / 3 cm = 2 degrees).
  Tip = centre - 0.04 m * axis * sign(axis_z): the lower end of an 8 cm peg (the end that enters the hole).
- Rest: "the peg stays in the hand": peg_pos - ee_pos changes by <= 3 mm between the first and last row of the 2 s
  rest; arm joints move < 0.01 rad.
- Joint: +1 on joint 1 is 0.02 rad per step (the request); 10 steps = 0.2 rad, accepted 0.12-0.24 rad (servo lag
  below, overshoot above, the ratios of E19's 0.3-0.6 for 0.5); the peg keeps its place in the hand through it
  (peg - ee changes <= 10 mm).
- Holes: "a random spot within 2 cm": the hole positions over all start rows span <= 4.2 cm per axis (2 cm either way,
  2 mm slack), differ between worlds (std > 3 mm), share one height (spread < 1 mm), and the block is fixed: hole_pos
  moves < 1 mm within the rest.
- No pass through the block: whenever the peg tip is more than 15 mm below the hole's opening (half the 3 cm insertion)
  while it is over the block (tip within 3 cm of the axis horizontally, the block being a few cm wide) it is inside
  the hole: tip within 3 mm of the axis. A tip that deep and 3 mm-3 cm off the axis is the peg through the block's
  top. CHECK-CHANGE: 15 mm, not the 5 mm first written: contact is soft, and a random-action push by the arm
  (hundreds of N through the position servos) pressed the tip 7 mm into the block's top in the reference.
- Policy (level 3): at the last row of the first episode the tip is >= 3 cm below the opening, the tip and the peg
  centre are both within 3 mm of the hole axis, in >= 80 % of worlds.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers.arm_common import episode_end

HALF_LEN = 0.04
DEPTH, AXIS_TOL = 0.03, 0.003
OVER_BLOCK_M, DEEP_M = 0.03, 0.015
POLICY_RATE = 0.8
PHASES = ("rest", "act_joint", "random", "policy")


def rotation(q: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices of unit quaternions (..., 4) in wxyz order (columns: the body's axes)."""
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def peg_axis(quat: np.ndarray) -> np.ndarray:
    """(..., 3) the peg's length axis (the most vertical local axis), pointing up."""
    r = rotation(quat)
    k = np.abs(r[..., 2, :]).argmax(axis=-1)
    axis = np.take_along_axis(r, k[..., None, None].repeat(3, axis=-2), axis=-1)[..., 0]
    return axis * np.where(axis[..., 2:3] < 0, -1.0, 1.0)


def peg_geometry(p: c.Phase) -> dict[str, np.ndarray]:
    """Per row and world: tip position (T, N, 3), depth of the tip below the opening, lateral offsets of the tip and of
    the centre from the hole axis, and the tilt of the peg from vertical (rad)."""
    centre = p.probe["peg_pos"].astype(np.float64)
    axis = peg_axis(p.probe["peg_quat"])
    tip = centre - HALF_LEN * axis
    hole = p.probe["hole_pos"].astype(np.float64)
    return {"tip": tip, "depth": hole[..., 2] - tip[..., 2],
            "tip_lat": np.linalg.norm((tip - hole)[..., :2], axis=-1),
            "centre_lat": np.linalg.norm((centre - hole)[..., :2], axis=-1),
            "tilt": np.arccos(np.clip(axis[..., 2], -1.0, 1.0))}


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    rel = rest.probe["peg_pos"].astype(np.float64) - rest.probe["ee_pos"]
    slip = np.linalg.norm(rel[-1] - rel[0], axis=-1)
    drift = np.abs(rest.probe["joint_pos"][-1] - rest.probe["joint_pos"][0]).max(axis=-1)
    moved = np.linalg.norm(rest.probe["hole_pos"][-1] - rest.probe["hole_pos"][0], axis=-1)
    norm = np.linalg.norm(rest.probe["peg_quat"][-1], axis=-1)
    return [c.within("rest_peg_in_hand", slip, 0.0, 0.003, 2), c.within("rest_arm_still", drift, 0.0, 0.01, 2),
            c.within("hole_fixed", moved, 0.0, 0.001, 2), c.within("peg_quat_unit", norm, 0.99, 1.01, 2)]


def _response_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("act_joint")
    q = p.probe["joint_pos"]
    rel = p.probe["peg_pos"].astype(np.float64) - p.probe["ee_pos"]
    return [c.within("joint1_turn", q[-1, :, 0] - q[0, :, 0], 0.12, 0.24, 2),
            c.within("peg_follows_hand", np.linalg.norm(rel[-1] - rel[0], axis=-1), 0.0, 0.01, 2)]


def _hole_checks(r: c.Rollout) -> list[c.Check]:
    rows = np.concatenate([r.phase(n).probe["hole_pos"][0] for n in ("rest", "act_joint", "random") if r.has(n)])
    first = r.phase("rest").probe["hole_pos"][0]
    return [c.at_most("hole_range", float((rows.max(0) - rows.min(0))[:2].max()), 0.042, 2),
            c.at_least("hole_spread", float(np.std(first[:, :2], axis=0).max()), 0.003, 2),
            c.at_most("hole_height_spread", float((rows[:, 2].max() - rows[:, 2].min())), 0.001, 2)]


def block_violations(p: c.Phase) -> int:
    """Rows (not reset rows) with the tip deep below the opening, over the block, yet off the hole axis."""
    g = peg_geometry(p)
    bad = (g["depth"] > DEEP_M) & (g["tip_lat"] > AXIS_TOL) & (g["tip_lat"] < OVER_BLOCK_M)
    return int((bad[1:] & ~p.done[1:]).sum())


def _block_check(r: c.Rollout) -> c.Check:
    total = sum(block_violations(r.phase(n)) for n in PHASES if r.has(n))
    return c.at_most("peg_not_through_block", float(total), 0.0, 2)


def _policy_check(r: c.Rollout) -> c.Check:
    p = r.phase("policy")
    idx, w = episode_end(p), np.arange(p.num_envs)
    g = peg_geometry(p)
    good = (g["depth"][idx, w] >= DEPTH) & (g["tip_lat"][idx, w] <= AXIS_TOL) & (g["centre_lat"][idx, w] <= AXIS_TOL)
    return c.at_least("policy_peg_inserted", float(good.mean()), POLICY_RATE, 3)


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + _response_checks(rollout)
    checks += _hole_checks(rollout) + [_block_check(rollout), _policy_check(rollout)]
    return c.finish(checks)
