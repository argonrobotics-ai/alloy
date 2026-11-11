# SPDX-License-Identifier: Apache-2.0
"""E05 ten dominoes (8 x 24 x 48 mm, 10 g) 25 mm apart in a row, pushed on the first one.

Derivations (from the request sentence only):
- Layout at rest: a standing domino's centre is 0.024 m up (+-2 mm). Faces are 25 mm apart and each is 8 mm thick, so
  centres sit 0.033 m apart along the row (+-3 mm); the row is straight (lateral offset from the end-to-end line
  < 3 mm). The row axis is the axis along which the start positions spread most; no frame order is assumed.
- Upright: tilt (angle of the body z axis to world z, wxyz quaternion) < 2 deg in every rest row.
- Push: +1 for 0.5 s then zero; a 0.5 N push on a 10 g domino at 4.8 cm tips it at once (torque 0.024 Nm against a
  restoring 0.0004 Nm). "Toppled" = tilt > 60 deg at some row; all ten must topple within 4 s of the push start.
- Order: sorted by position along the row, the toppling times must be monotone (all rising or all falling, one control
  step of slack for ties): a chain, so the first to fall is an end domino and the wave runs down the row.
- success(): in each world it is true at some row of the push phase exactly when all ten toppled, and never at rest.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

HALF_HEIGHT = 0.024
PITCH = 0.033
TOPPLED = np.radians(60.0)
UPRIGHT = np.radians(2.0)
TOPPLE_WITHIN_S = 4.0


def _row_axis(pos0: np.ndarray) -> np.ndarray:
    """Unit horizontal direction of the row from start positions (N, K, 3): the longest-extent axis, per world."""
    span = pos0[..., :2].max(axis=1) - pos0[..., :2].min(axis=1)
    axis = np.zeros_like(span)
    axis[np.arange(len(span)), np.argmax(span, axis=1)] = 1.0
    return axis


def _layout_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    pos = rest.probe["domino_pos"][-1].astype(np.float64)
    axis = _row_axis(pos)
    along = np.einsum("nkd,nd->nk", pos[..., :2], axis)
    order = np.argsort(along, axis=1)
    sorted_along = np.take_along_axis(along, order, axis=1)
    cross = np.stack([axis[:, 1], -axis[:, 0]], axis=1)
    lateral = np.einsum("nkd,nd->nk", pos[..., :2], cross)
    tilt = c.tilt_from_quat_wxyz(rest.probe["domino_quat"])
    return [c.within("rest_height", np.abs(pos[..., 2] - HALF_HEIGHT).max(axis=1), 0.0, 0.002, 2),
            c.within("rest_upright", tilt.max(axis=(0, 2)), 0.0, UPRIGHT, 2),
            c.within("row_pitch", np.abs(np.diff(sorted_along, axis=1) - PITCH).max(axis=1), 0.0, 0.003, 2),
            c.within("row_straight", np.ptp(lateral, axis=1), 0.0, 0.003, 2)]


def _topple_times(p: c.Phase) -> np.ndarray:
    """(N, K) time of the first row with tilt > 60 deg; inf when never."""
    tilt = c.tilt_from_quat_wxyz(p.probe["domino_quat"])
    hit = tilt > TOPPLED
    first = np.argmax(hit, axis=0)
    return np.where(hit.any(axis=0), p.t[first].astype(np.float64), np.inf)


def _monotone(times: np.ndarray, along: np.ndarray, slack: float) -> np.ndarray:
    ordered = np.take_along_axis(times, np.argsort(along, axis=1), axis=1)
    d = np.diff(ordered, axis=1)
    return np.all(d >= -slack, axis=1) | np.all(d <= slack, axis=1)


def _push_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("act_push")
    times = _topple_times(p)
    pos0 = p.probe["domino_pos"][0].astype(np.float64)
    along = np.einsum("nkd,nd->nk", pos0[..., :2], _row_axis(pos0))
    dt = float(r.meta.get("dt", 0.02))
    toppled = np.isfinite(times).all(axis=1)
    agrees = p.success.any(axis=0) == toppled
    return [c.within("all_toppled_in_time", times.max(axis=1), 0.0, TOPPLE_WITHIN_S, 2),
            c.Check("topple_in_order", bool(_monotone(np.where(np.isfinite(times), times, 1e9), along, dt).all()),
                    times[0].tolist(), "monotone along the row", 2),
            c.Check("success_agrees", bool(agrees.all()), agrees.tolist(), "success any-row == all ten toppled", 2),
            c.Check("success_never_at_rest", not bool(r.phase("rest").success.any()), None, "no success at rest", 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _layout_checks(rollout) + _push_checks(rollout)
    return c.finish(checks)
