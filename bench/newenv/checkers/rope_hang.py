# SPDX-License-Identifier: Apache-2.0
"""E14 rope hang: a 1 m, 50 g rope hangs from a fixed point 1.5 m up; a gripper moves its free end at up to 0.3 m/s.

Derivations (from the sentence only):
- Length: a rope does not stretch more than 5 %: the polyline through its points is 0.95-1.05 m in every phase.
- Hang at rest: straight down from the anchor at z = 1.5 m (+-1 cm); the free end is 1 m below it (0.47-0.55 m high).
- Anchor: the rope end that is not the gripper's stays within 5 mm of its start in every phase.
- Speed: at action 1 the free end moves 0.3 m/s: 0.3 m in 1 s (+-10 %) along +z, within 2 cm off-axis.
- Catenary: the free end raised to the anchor's height, 0.6 m away. A flexible chain of length L = 1 m between two points
  at the same height 0.6 m apart: L = 2 a sinh(s / 2a), s = 0.6, so sinh(x) / x = 1 / 0.6 with x = 0.3 / a: x = 1.838,
  a = 0.1632 m, sag = a (cosh x - 1) = 0.1632 x 2.221 = 0.3625 m. A bending stiffness lifts it a little, a span 0.03 m off
  moves it by 0.02 m: so the lowest point is 0.33-0.45 m below the ends (the request's band), measured as the mean over
  the last 10 rows before the world's first done; the end must be within 3 cm of the point 0.6 m from the anchor.
- Targets: the midpoint of a 1 m rope fixed at the anchor can reach only points within 0.5 m of it: every target is at
  most 0.52 m from the anchor; targets spread by more than 1 cm per axis across worlds.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

ROPE_M = 1.0
SPAN_M = 0.6
SPEED = 0.3


def _anchor_index(pts: np.ndarray, end: np.ndarray) -> np.ndarray:
    """(N,) 0 or -1: the rope end that is not at the gripper, from row 0 of a phase."""
    first = np.linalg.norm(pts[0, :, 0] - end[0], axis=-1) > np.linalg.norm(pts[0, :, -1] - end[0], axis=-1)
    return np.where(first, 0, -1)


def _anchor_track(pts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return pts[:, np.arange(pts.shape[1]), idx]                     # (T, N, 3)


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("rest").probe
    idx = _anchor_index(p["rope_points"], p["end_pos"])
    a = _anchor_track(p["rope_points"], idx)[-1]
    return [c.within("rest_anchor_height", a[:, 2], 1.49, 1.51, 2),
            c.within("rest_end_height", p["end_pos"][-1][:, 2], 0.47, 0.55, 2),
            c.at_least("rope_points", float(p["rope_points"].shape[2]), 10, 2)]


def _all_phase_checks(r: c.Rollout) -> list[c.Check]:
    lengths, drift = [], 0.0
    for ph in r.phases.values():
        pts = ph.probe.get("rope_points")
        if pts is None:
            continue
        lengths.append(c.polyline_length(pts))
        a = _anchor_track(pts, _anchor_index(pts, ph.probe["end_pos"]))
        drift = max(drift, float(np.linalg.norm(a - a[0], axis=-1).max()))
    ln = np.concatenate([x.ravel() for x in lengths])
    return [c.within("rope_length", ln, 0.95 * ROPE_M, 1.05 * ROPE_M, 2),
            c.at_most("anchor_holds", drift, 0.005, 2)]


def _speed_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("act_speed")
    d = ph.probe["end_pos"][-1] - ph.probe["end_pos"][0]
    secs = float(ph.t[-1] - ph.t[0])
    return [c.within("speed_travel", d[:, 2], 0.9 * SPEED * secs, 1.1 * SPEED * secs, 2),
            c.at_most("speed_off_axis", float(np.abs(d[:, :2]).max()), 0.02, 2)]


def _last_valid_row(done: np.ndarray) -> np.ndarray:
    """(N,) row index just before each world's first done (the last row if none)."""
    first = np.where(done.any(axis=0), done.argmax(axis=0), done.shape[0])
    return np.clip(first - 1, 0, done.shape[0] - 1)


def _catenary_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("act_raise")
    pts, end = ph.probe["rope_points"], ph.probe["end_pos"]
    row = _last_valid_row(ph.done)
    n = pts.shape[1]
    idx = _anchor_index(pts, end)
    anchor = _anchor_track(pts, idx)
    sag, miss = np.zeros(n), np.zeros(n)
    for w in range(n):
        rows = slice(max(row[w] - 9, 0), row[w] + 1)
        ends = np.minimum(anchor[rows, w, 2], end[rows, w, 2])
        sag[w] = float(np.mean(ends - pts[rows, w, :, 2].min(axis=-1)))
        goal = anchor[row[w], w] + np.array([SPAN_M, 0.0, 0.0])
        miss[w] = float(np.linalg.norm(end[row[w], w] - goal))
    return [c.at_most("raise_reaches_span", float(miss.max()), 0.03, 2),
            c.within("catenary_sag", sag, 0.33, 0.45, 2)]


def _target_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("random").probe
    target, pts, end = p["target_pos"][0], p["rope_points"][0], p["end_pos"][0]
    anchor = pts[np.arange(len(pts)), np.where(np.linalg.norm(pts[:, 0] - end, axis=-1) >
                                               np.linalg.norm(pts[:, -1] - end, axis=-1), 0, -1)]
    reach = np.linalg.norm(target - anchor, axis=-1)
    return [c.at_most("target_reachable", float(reach.max()), 0.52, 2),
            c.at_least("target_spread", float(np.std(target, axis=0).min()), 0.01, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _rest_checks(rollout) + _all_phase_checks(rollout) + _speed_checks(rollout)
    checks += _catenary_checks(rollout) + _target_checks(rollout)
    return c.finish(checks)
