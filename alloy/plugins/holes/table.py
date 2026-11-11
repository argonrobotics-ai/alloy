# SPDX-License-Identifier: Apache-2.0
"""Host-side hole records: what a drill has cut in each workpiece layer, read from the drill's hole table.

The drill keeps, per world and layer, up to ``max_holes`` holes of up to ``max_stages`` stages (state arrays
``hole_n``, ``hole_entry``, ``hole_axis``, ``stage_n``, ``stage_d``, ``stage_b``; slot ``layer * H + h``, stage slot
``slot * S + k``). ``holes_of`` turns one world's rows into ``Hole`` records in the layer's face frame (z = 0 on the
face, +z out of the material); ``Hole.in_world`` moves one into the world with the layer's face transform.

Example::

    >>> h = Hole(0, np.zeros(3), np.array([0.0, 0.0, -1.0]), (Stage(0.006, 0.012),), 0.01, 1.664)
    >>> h.through_at(0.006), round(h.tilt_deg, 6)
    (True, 0.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.plugins.npgeom import quat_rotate

SAME_DIAMETER = 1.0e-5          # m: a re-entry with a drill this close to a stage's diameter continues that stage


@dataclass(frozen=True)
class Stage:
    """One drill's cut in a hole: its diameter and how far its point tip went along the axis from the entry [m]."""

    diameter: float
    depth: float


@dataclass(frozen=True)
class Hole:
    """A hole in one layer (face frame): entry on the face, unit axis into the material, its stages."""

    layer: int
    entry: np.ndarray
    axis: np.ndarray
    stages: tuple[Stage, ...]
    thickness: float               # the layer's thickness along its face normal [m]
    tan_half: float                # tan of the drills' point half-angle

    @property
    def span(self) -> float:
        """The layer's length along the hole axis [m]."""
        return self.thickness / max(-float(np.float32(self.axis[2])), 1.0e-6)        # as the drill's ledger

    @property
    def diameter(self) -> float:
        return max(s.diameter for s in self.stages)

    @property
    def tilt_deg(self) -> float:
        """Angle between the hole axis and the face normal [deg]."""
        return math.degrees(math.acos(min(max(-float(self.axis[2]), -1.0), 1.0)))

    def _stage(self, diameter: float | None) -> Stage | None:
        if diameter is None:
            return max(self.stages, key=lambda s: s.diameter)
        hits = [s for s in self.stages if abs(s.diameter - diameter) <= SAME_DIAMETER]
        return hits[0] if hits else None

    def full_depth(self, diameter: float | None = None) -> float:
        """Depth of the full diameter along the axis (the point tip's depth minus the point height) [m]."""
        s = self._stage(diameter)
        return 0.0 if s is None else max(s.depth - 0.5 * s.diameter / self.tan_half, 0.0)

    def through_at(self, diameter: float) -> bool:
        """True when a stage at least ``diameter`` wide has its full diameter out of the layer's back face."""
        return any(s.diameter >= diameter - SAME_DIAMETER and s.depth >= self.span + 0.5 * s.diameter / self.tan_half
                   for s in self.stages)

    @property
    def through(self) -> bool:
        return self.through_at(min(s.diameter for s in self.stages))

    def in_world(self, face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(entry, axis) in the world, given the layer's face transform (x, y, z, qx, qy, qz, qw)."""
        f = np.asarray(face, np.float64)
        return f[:3] + quat_rotate(f[3:], self.entry), quat_rotate(f[3:], self.axis)


def holes_of(read: dict[str, np.ndarray], world: int, layer: int, max_holes: int, max_stages: int,
             thickness: float, tan_half: float) -> list[Hole]:
    """The holes of ``layer`` in ``world`` from a drill's ``read()``."""
    out = []
    for h in range(int(read["hole_n"][world, layer])):
        slot = layer * max_holes + h
        stages = tuple(Stage(float(read["stage_d"][world, slot * max_stages + k]),
                             float(read["stage_b"][world, slot * max_stages + k]))
                       for k in range(int(read["stage_n"][world, slot])))
        entry = np.array([*read["hole_entry"][world, slot].astype(np.float64), 0.0])
        out.append(Hole(layer, entry, read["hole_axis"][world, slot].astype(np.float64), stages, thickness, tan_half))
    return out


__all__ = ["Hole", "SAME_DIAMETER", "Stage", "holes_of"]
