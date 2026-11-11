# SPDX-License-Identifier: Apache-2.0
"""Geometry helpers of the ported forklift criteria (F/evaluator.py:85-142, 683-765): box frames from corners, yaw and
angle differences, plan hulls, clipping and signed plan gaps of convex polygons."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.tasks.forklift_site import ForkliftSite, quat_xyzw_to_matrix
from alloy.tasks.forklift_ga_types import BOX_SIGNS, Episode


@dataclass(frozen=True)
class BoxPose:
    """A box load at one sample: its eight corners (world), centre and rotation (columns = its x, y, z axes)."""

    corners: np.ndarray
    centre: np.ndarray
    R: np.ndarray

    @property
    def bottom(self) -> np.ndarray:
        """The four corners of the face at the box frame's -z (a pallet: its bottom face, GA's pallet_corners at z =
        0)."""
        return self.corners[BOX_SIGNS[:, 2] < 0]

    @property
    def bottom_origin(self) -> np.ndarray:
        """Centre of the bottom face (a pallet's body origin)."""
        return self.bottom.mean(axis=0)


def box_pose(corners) -> BoxPose:
    """Frame of a box from its eight corners in BOX_SIGNS order."""
    c = np.asarray(corners, float)
    ex, ey, ez = c[4] - c[0], c[2] - c[0], c[1] - c[0]
    R = np.stack([ex / np.linalg.norm(ex), ey / np.linalg.norm(ey), ez / np.linalg.norm(ez)], axis=1)
    return BoxPose(c, c.mean(axis=0), R)


def load_pose(ep: Episode, name: str, i: int = -1) -> BoxPose:
    return box_pose(ep.loads[name][i])


def truck_frame(ep: Episode, i: int = -1) -> tuple[np.ndarray, np.ndarray]:
    return ep.truck_pos[i], quat_xyzw_to_matrix(ep.truck_quat_xyzw[i])


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def ang_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def tilt_deg(R, axis=2) -> float:
    """Angle (deg) between a frame's axis and +z."""
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, axis]))))


def square_deg(yaw_err: float) -> float:
    """Yaw error (deg) folded to a line: 0..90."""
    e = abs(math.degrees(yaw_err))
    return min(e, 180.0 - e)


def ground_at_tips(ep: Episode, site: ForkliftSite, i: int = -1) -> np.ndarray:
    return np.array([site.ground_height(t[0], t[1]) for t in ep.fork_tips[i]])


def tip_heights(ep: Episode, site: ForkliftSite, i: int = -1) -> np.ndarray:
    """Each blade tip's height above the ground under it."""
    return ep.fork_tips[i][:, 2] - ground_at_tips(ep, site, i)


def forks_height(ep: Episode, site: ForkliftSite) -> float:
    """GA's _forks_low value: the higher tip over the ground under the first tip (F/evaluator.py:284-287)."""
    return float(ep.fork_tips[-1][:, 2].max() - site.ground_height(*ep.fork_tips[-1][0][:2]))


def tips_to_origin(ep: Episode, origin_xy) -> float:
    """Distance in plan from the nearer blade tip to a point."""
    return min(float(np.linalg.norm(t[:2] - origin_xy[:2])) for t in ep.fork_tips[-1])


# ---------------------------------------------------------------------------------------------- convex polygons
def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(pts) -> np.ndarray:
    """Convex hull of plan points, counter-clockwise (monotone chain); fewer than 3 distinct points are returned as
    is."""
    pts = sorted(set(map(tuple, np.round(np.asarray(pts, float).reshape(-1, 2), 9))))
    if len(pts) < 3:
        return np.array(pts, float).reshape(-1, 2)

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and _cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(pts) + half(pts[::-1]), float)


def clip(poly, by) -> np.ndarray:
    """The part of convex polygon poly inside convex counter-clockwise polygon by (Sutherland-Hodgman)."""
    out = list(np.asarray(poly, float))
    for i in range(len(by)):
        a, b = by[i], by[(i + 1) % len(by)]

        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        inp, out = out, []
        for j, p in enumerate(inp):
            q = inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if sp * sq < 0:
                out.append(p + (q - p) * sp / (sp - sq))
        if not out:
            break
    return np.array(out, float).reshape(-1, 2)


def in_convex(p, poly) -> bool:
    n = len(poly)
    return n >= 3 and all((poly[(i + 1) % n][0] - poly[i][0]) * (p[1] - poly[i][1])
                          - (poly[(i + 1) % n][1] - poly[i][1]) * (p[0] - poly[i][0]) >= -1e-9 for i in range(n))


def plan_gap(A, B) -> float:
    """Signed plan distance between convex polygons A and B: their clearance (> 0) or minus their penetration depth."""
    sep = []
    for Q in (A, B):
        for i in range(len(Q)):
            e = Q[(i + 1) % len(Q)] - Q[i]
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            sep.append(max((B @ n).min() - (A @ n).max(), (A @ n).min() - (B @ n).max()))
    if max(sep) <= 0:
        return float(max(sep))

    def seg(p, a, b):
        t = max(0.0, min(1.0, float((p - a) @ (b - a) / ((b - a) @ (b - a)))))
        return float(np.linalg.norm(p - a - t * (b - a)))
    return min(seg(p, Q[i], Q[(i + 1) % len(Q)]) for S, Q in ((A, B), (B, A)) for p in S for i in range(len(Q)))
