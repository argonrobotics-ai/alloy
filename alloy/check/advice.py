# SPDX-License-Identifier: Apache-2.0
"""Wording and arithmetic shared by the checks' fixes: lengths in mm, which object a pose edit should move, and
the new pose that clears an overlap.

Example::

    >>> mm(0.0032)
    '3.2 mm'
    >>> clearance(0.0032)
    0.004
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from alloy.check.objects import ObjectIndex

STEP_M = 5e-4          # fixes round up to half millimetres
EXTRA_M = 5e-4         # and clear by at least this much beyond the measured depth
FINE_STEP_M = 1e-4     # overlaps under 1 mm: 0.1 mm steps
FINE_EXTRA_M = 5e-5
VERTICAL_COS = 0.7     # a normal within ~45 degrees of z reads as "lift" / "lower"


def mm(x: float) -> str:
    """Metres as a short mm (or um) string.

    Example::

        >>> mm(0.0032)
        '3.2 mm'
        >>> mm(5e-5)
        '50 um'
    """
    if not math.isfinite(x):
        return "nan"
    if abs(x) < 1e-4:
        return f"{x * 1e6:.3g} um"
    return f"{x * 1e3:.3g} mm"


def clearance(depth: float) -> float:
    """How far to move to clear ``depth`` [m]: depth + 0.5 mm rounded up to the next 0.5 mm; under 1 mm, depth +
    0.05 mm rounded up to the next 0.1 mm (tight fits stay tight).

    Example::

        >>> clearance(0.0032)
        0.004
        >>> clearance(0.0004)
        0.0005
    """
    if depth < 1e-3:
        return math.ceil((depth + FINE_EXTRA_M) / FINE_STEP_M - 1e-9) * FINE_STEP_M
    return math.ceil((depth + EXTRA_M) / STEP_M - 1e-9) * STEP_M


def worlds_text(worlds: Sequence[int], total: int) -> str:
    """"in every world", "in 5 of 64 worlds (e.g. 3, 7, 12)".

    Example::

        >>> worlds_text([3, 7, 12, 1, 2], 64)
        'in 5 of 64 worlds (e.g. 1, 2, 3)'
        >>> worlds_text([0, 1], 2)
        'in every world'
    """
    if total <= 1 or len(worlds) >= total:
        return "in every world" if total > 1 else ""
    eg = ", ".join(str(w) for w in sorted(worlds)[:3])
    return f"in {len(worlds)} of {total} worlds (e.g. {eg})"


@dataclass(frozen=True)
class Move:
    """Move ``name`` by ``distance`` [m] along unit ``direction``; ``pose`` is its new card pose (None: robots or
    objects without a placement pose).

    Example::

        >>> Move("mug", (0.0, 0.0, 1.0), 0.004, None).verb()
        'lift mug by 4 mm'
    """

    name: str
    direction: tuple[float, float, float]
    distance: float
    pose: tuple[float, ...] | None

    def verb(self) -> str:
        """"lift mug by 4 mm" / "lower ..." / "move mug 4 mm along (0.71, 0.71, 0)".

        Example::

            >>> Move("mug", (0.707, 0.707, 0.0), 0.004, None).verb()
            'move mug 4 mm along (0.71, 0.71, 0.00)'
        """
        dz = self.direction[2]
        if dz >= VERTICAL_COS:
            return f"lift {self.name} by {mm(self.distance)}"
        if dz <= -VERTICAL_COS:
            return f"lower {self.name} by {mm(self.distance)}"
        d = ", ".join(f"{c:.2f}" for c in self.direction)
        return f"move {self.name} {mm(self.distance)} along ({d})"

    def suggestion(self, index: ObjectIndex) -> dict[str, Any] | None:
        """The card edit that applies this move (None when ``pose`` is None): which object and which card field to set.

        Example::

            >>> move.suggestion(index)  # doctest: +SKIP
            {'object': 'mug', 'field': 'objects.1.pose', 'pose': [0.0, 0.0, 0.104, 0, 0, 0, 1]}
        """
        if self.pose is None:
            return None
        where = index.placement.get(self.name)
        field = f"objects.{where}.pose" if where is not None else "robots.base_pose"
        return {"object": self.name, "field": field, "pose": [round(v, 6) for v in self.pose]}


def separate(index: ObjectIndex, a: str, b: str, normal: Sequence[float], depth: float) -> Move | None:
    """The pose edit that clears an overlap of ``depth`` between ``a`` and ``b`` (``normal`` from a toward b).

    Moves the dynamic one (b along +normal, or a along -normal); with both dynamic, the one the move lifts (so
    stacks grow upward); None when neither can move (static, kinematic, ground).

    Example::

        >>> move = separate(index, "table", "mug", (0.0, 0.0, 1.0), 0.0032)  # doctest: +SKIP
        >>> move.verb()
        'lift mug by 4 mm'
    """
    n = tuple(float(v) for v in normal)
    up_b, up_a = n, (0.0 - n[0], 0.0 - n[1], 0.0 - n[2])     # 0.0 - x: no negative zeros in text
    cands = [(b, up_b), (a, up_a)]
    movable = [(name, d) for name, d in cands if index.movable(name)]
    if not movable:
        movable = [(name, d) for name, d in cands if index.motion.get(name) == "robot"]
    if not movable:
        return None
    name, direction = max(movable, key=lambda c: c[1][2])     # prefer the move that goes up
    dist = clearance(depth)
    return Move(name, direction, dist, _shifted(index.poses.get(name), direction, dist))


def _shifted(pose: tuple[float, ...] | None, direction: Sequence[float], dist: float) -> tuple[float, ...] | None:
    if pose is None or len(pose) < 3:
        return None
    xyz = tuple(pose[i] + direction[i] * dist for i in range(3))
    return xyz + tuple(pose[3:])


__all__ = ["Move", "clearance", "mm", "separate", "worlds_text"]
