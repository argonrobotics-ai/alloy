# SPDX-License-Identifier: Apache-2.0
"""E23 arm pour: 150 ml of water in a cup the Panda holds; a pour into a bowl 40 cm away; the cup stays in the hand.

Derivations (request sentence only; no engine's frames, particle order or counts):
- Volume: finite points x particle_volume at the first rest row is 150 ml +-10 % (135-165 ml, the sentence's own band).
- Rest (zero action, ``rest_s``): the water's start region is read from the first row of the same run: a cylinder around
  the start centroid with the start points' largest horizontal radius + 1 cm, from the start's lowest point - 1 cm to
  its highest + 2 cm. >= 95 % of the points inside it in every row, in every world (all but one).
- Cup in the hand: "the cup moves with ee_pos (within 5 cm) throughout": |cup_pos - ee_pos| <= 5 cm in every row of
  rest, act_joint and policy; and the LENGTH of the offset cup - ee changes by <= 2 cm over a phase (a held cup does not
  slide in the hand). The offset vector itself turns with the hand (a 3.7 cm offset under a 2 rad tip changes by
  up to 6 cm with no slip), so only its length is compared; no hand orientation is needed.
- act_joint (harness law: +1 on joint 1 for 10 steps = 10 x 0.05 rad): joint 1 turns 0.3-0.6 rad (servo lag below,
  overshoot above) and the water volume is kept (end / start 0.9-1.1).
- Policy (level 3): in the last state of the first episode (the row before the first done: a done row already shows
  the next episode's start) >= 80 % of the water is inside the bowl: horizontal distance to bowl_pos < bowl_radius and
  height within 10 cm of bowl_pos z (bowl_pos may be the bowl's base or its centre; 10 cm covers either and excludes
  water still held in the cup, which is >= 15 cm above the table). Nothing "below the table": no point more than
  10 cm below bowl_pos z in any row of the episode (water that missed the bowl rests on the table, at most one bowl
  height below a centre reading). The count of water points is kept (finite points end / start 0.9-1.1). All these in
  >= 80 % of worlds.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

VOLUME_ML = (135.0, 165.0)
REST_FRACTION = 0.95
CUP_NEAR_M = 0.05
CUP_SLIDE_M = 0.02
JOINT_TURN = (0.3, 0.6)
KEEP = (0.9, 1.1)
BOWL_FRACTION = 0.8
HEIGHT_BAND = 0.10
WORLD_RATE = 0.8


def _live(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def _in_start_region(points: np.ndarray) -> np.ndarray:
    """(T, N) share of points inside the cylinder fixed by row 0 (see the module doc)."""
    p0 = points[0]
    centre = np.nanmean(p0[..., :2], axis=1, keepdims=True)
    radius = np.nanmax(np.linalg.norm(p0[..., :2] - centre, axis=-1), axis=1) + 0.01
    zlo, zhi = np.nanmin(p0[..., 2], axis=1) - 0.01, np.nanmax(p0[..., 2], axis=1) + 0.02
    d = np.linalg.norm(points[..., :2] - centre[None], axis=-1)
    inside = (d <= radius[None, :, None]) & (points[..., 2] >= zlo[None, :, None]) & (points[..., 2] <= zhi[None, :, None])
    return inside.mean(axis=-1)


def cup_offsets(p: c.Phase) -> tuple[np.ndarray, np.ndarray]:
    """(worst |cup - ee| over the rows, largest change of that length from row 0), both (N,)."""
    length = np.linalg.norm(p.probe["cup_pos"] - p.probe["ee_pos"], axis=-1)
    dist = length.max(axis=0)
    slide = np.abs(length - length[0]).max(axis=0)
    return dist, slide


def _cup_checks(p: c.Phase, tag: str, level: int) -> list[c.Check]:
    dist, slide = cup_offsets(p)
    return [c.within(f"{tag}_cup_near_ee", dist, 0.0, CUP_NEAR_M, level),
            c.within(f"{tag}_cup_no_slide", slide, 0.0, CUP_SLIDE_M, level)]


def _volume_and_rest(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    pts, pv = rest.probe["liquid_points"], rest.probe["particle_volume"]
    vol_ml = _live(pts[:1]).sum(axis=-1)[0] * pv[0] * 1e6
    stays = _in_start_region(pts).min(axis=0)
    return [c.within("volume_ml", vol_ml, *VOLUME_ML, 2),
            c.within("rest_in_cup", stays, REST_FRACTION, 1.0, 2, min_worlds=rest.num_envs - 1)] + _cup_checks(rest, "rest", 2)


def _joint_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("act_joint")
    q = p.probe["joint_pos"]
    keep = _live(p.probe["liquid_points"][-1]).sum(axis=-1) / np.maximum(_live(p.probe["liquid_points"][0]).sum(axis=-1), 1)
    return [c.within("joint1_turn", q[-1, :, 0] - q[0, :, 0], *JOINT_TURN, 2),
            c.within("act_water_kept", keep, *KEEP, 2)] + _cup_checks(p, "act", 2)


def _first_episode_end(p: c.Phase, w: int) -> int:
    """Row of the last state of world w's first episode (the row before its first done, else the last row)."""
    hits = np.flatnonzero(p.done[:, w])
    return int(hits[0]) - 1 if len(hits) and hits[0] > 0 else len(p.t) - 1


def _pour_shares(p: c.Phase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per world: share of water in the bowl at the episode's end, lowest height below the bowl row, count kept."""
    n = p.num_envs
    share, lowest, kept = np.zeros(n), np.zeros(n), np.zeros(n)
    for w in range(n):
        end = _first_episode_end(p, w)
        pts = p.probe["liquid_points"][: end + 1, w]
        bowl, rad = p.probe["bowl_pos"][end, w], p.probe["bowl_radius"][end, w]
        live = _live(pts[end])
        near = np.linalg.norm(pts[end][:, :2] - bowl[:2], axis=-1) < rad
        height = np.abs(pts[end][:, 2] - bowl[2]) <= HEIGHT_BAND
        share[w] = (near & height & live).sum() / max(live.sum(), 1)
        lowest[w] = np.nanmin(np.where(_live(pts), pts[..., 2], np.inf)) - bowl[2]
        kept[w] = live.sum() / max(_live(pts[0]).sum(), 1)
    return share, lowest, kept


def _policy_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("policy")
    share, lowest, kept = _pour_shares(p)
    cup_dist, cup_slide = cup_offsets(p)
    ok_world = (share >= BOWL_FRACTION) & (lowest >= -HEIGHT_BAND) & (kept >= KEEP[0]) & (kept <= KEEP[1])
    return [c.at_least("policy_water_in_bowl", float((share >= BOWL_FRACTION).mean()), WORLD_RATE, 3),
            c.at_least("policy_nothing_below_table", float((lowest >= -HEIGHT_BAND).mean()), WORLD_RATE, 3),
            c.at_least("policy_water_count_kept", float(((kept >= KEEP[0]) & (kept <= KEEP[1])).mean()), WORLD_RATE, 3),
            c.at_least("policy_all_conditions", float(ok_world.mean()), WORLD_RATE, 3),
            c.at_least("policy_cup_near_ee", float((cup_dist <= CUP_NEAR_M).mean()), WORLD_RATE, 3),
            c.at_least("policy_cup_no_slide", float((cup_slide <= CUP_SLIDE_M).mean()), WORLD_RATE, 3)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _volume_and_rest(rollout) + _joint_checks(rollout)
    checks += _policy_checks(rollout)
    return c.finish(checks)
