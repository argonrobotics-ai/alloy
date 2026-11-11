# SPDX-License-Identifier: Apache-2.0
"""Hook paths that open a carton flap (gift_carton.py) by pushing its inner face along an arc about the hinge.

The hook (gift_gantry.py, tool "hook") is a vertical shaft with a tine at its foot. To open a flap the wrist turns the
tine toward the flap's hinge; the tine comes in beyond the flap's free edge, below it, slides under it to ``reach``
from the hinge and then follows the arc as the flap turns: its tip stays on the flap's inner face, one board and one
tine thickness off the flap's mid-plane. The shaft stays beyond the free edge while the flap is low and inside the
flap's lean once it is past upright, so it never crosses the board.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gift_carton import CartonSpec, _hinges
from .gift_gantry import FINGER_LEN, HOOK_T, HOOK_TINE


@dataclass(frozen=True)
class Push:
    reach: float = 0.10                 # contact from the hinge along the flap [m]
    clear: float = 0.001                # gap between the tine and the board [m]
    beyond: float = 0.008               # the tip's start beyond the free edge [m]
    target_deg: float = 200.0           # pushed this far, a flap keeps about target - 45 deg
    steps: int = 20                     # waypoints along the arc (the same in every world)
    sweep_s: float = 2.4                # time for the whole sweep [s] (about 1.4 rad/s)
    under_deg: float = 1.5              # the tine starts this far below the flap


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def tip_offset(yaw: float) -> np.ndarray:
    """The tine's tip from the palm (world frame) at wrist yaw ``yaw``."""
    return np.array([HOOK_TINE * math.cos(yaw), HOOK_TINE * math.sin(yaw), -FINGER_LEN + HOOK_T])


def flap_frame(s: CartonSpec, flap: str, pose: tuple[np.ndarray, np.ndarray]):
    """(hinge point, axis, inward direction, closed underside normal) in world, for a carton at ``pose`` (pos, R)."""
    h = _hinges(s)[flap]
    pos, R = pose
    p = pos + R @ np.asarray(h["p"], float)
    return p, R @ np.asarray(h["axis"], float), R @ np.asarray(h["in"], float), R @ np.array([0.0, 0.0, -1.0])


def tool_yaw(inward: np.ndarray) -> float:
    """Wrist yaw that points the tine toward the hinge (against the flap's inward direction)."""
    return math.atan2(-inward[1], -inward[0])


def contact(s: CartonSpec, flap: str, pose, phi: float, push: Push, reach: float | None = None) -> np.ndarray:
    """Tine tip target with the flap at ``phi`` [rad] from closed."""
    p, ax, d0, n0 = flap_frame(s, flap, pose)
    Rf = _rot(ax, phi)
    off = 0.5 * s.caliper + HOOK_T + push.clear
    return p + (push.reach if reach is None else reach) * (Rf @ d0) + off * (Rf @ n0)


def waypoints(s: CartonSpec, flap: str, pose, phi0: float, push: Push) -> list[tuple[str, np.ndarray, float]]:
    """(phase, tine-tip target, seconds) for one flap at angle ``phi0`` [rad]: above, down beyond the free edge, in
    under the flap, the sweep, a hold, back off along the board's normal, and up."""
    p, ax, d0, n0 = flap_frame(s, flap, pose)
    lo = phi0 - math.radians(push.under_deg)                 # a little under the flap
    edge = contact(s, flap, pose, lo, push, reach=s.flap + push.beyond)
    up = np.array([0.0, 0.0, 1.0])
    travel = pose[0][2] + s.height + s.flap + 0.06 - edge[2]  # over an upright flap, whatever the others' state
    pts = [("over", edge + travel * up, 0.8), ("down", edge, 0.8), ("in", contact(s, flap, pose, lo, push), 0.4)]
    target = math.radians(push.target_deg)
    n = push.steps
    for k in range(1, n + 1):
        phi = lo + (target - lo) * k / n
        pts.append(("sweep", contact(s, flap, pose, phi, push), push.sweep_s / n))
    end = contact(s, flap, pose, target, push)
    pts.append(("hold", end, 0.3))
    back = end + 0.03 * (_rot(ax, target) @ n0)
    pts += [("back", back, 0.4), ("up", back + (pose[0][2] + s.height + s.flap + 0.06 - back[2]) * up, 0.5)]
    return pts
