# SPDX-License-Identifier: Apache-2.0
"""Scene and motion plan of task rinse_plates (Gym-Anything's pre-rinse-plates): the peg rack on the pre-rinse sink,
six plates standing in it, the spray valve on its balancer, and the hand's moves.

Layout (world, z up; sources in ``rinse_ware``): the rack floor is level with the dish table (34 in) over the
20 x 20 x 5 in pre-rinse sink; seven rows of pegs 35 mm apart make six slots, and each plate stands in its slot facing
-y, leaning back onto the next row (their 6-12°, ``scene.py:433-451``; here 15.1°, where a released plate comes to
rest in a slot). The rows are static boxes 66 mm tall (the pegs' envelope). A backsplash stands behind the rack.
The spray station is in front of the rack over the sink: a plate is held there upright, face to -y, while the valve
rasters 0.20 m in front of it (their calibration raster, ``params.py:143-147``: 0.30 m/s, rows 2 cm apart).

Moves are segments between poses with a smooth-step profile (zero speed at both ends); the hand gets the pose and
its velocity as feed-forward. A plate leaves its slot along its own lean (parallel to its neighbours), clears their
tops, turns upright on the way to the station and comes back the same way.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.tasks.rinse_ware import INCH, Plate, SprayValve

TABLE_Z = 34.0 * INCH                 # m, dish table top (their params.py:176)
RACK_Z = TABLE_Z + 0.0125             # m, the rack floor's top
SINK_FLOOR = TABLE_Z - 5.0 * INCH     # m
ROW_PITCH = 0.035                     # m
ROW_HALF = (0.24, 0.008, 0.033)       # m, a row of pegs as a 16 mm box: against an 8 mm one a 6 mm disc's rim sank
                                      # 6 mm into the edge and rocked (MuJoCo convex contact, both CPU routes)
SLOTS = 6


def quat_x(angle: float) -> np.ndarray:
    """(x, y, z, w) of a rotation by ``angle`` about +x."""
    return np.array([math.sin(0.5 * angle), 0.0, 0.0, math.cos(0.5 * angle)])


def slerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + s * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1.0 - s) * th) * a + math.sin(s * th) * b) / math.sin(th)


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray
    quat: np.ndarray                  # x, y, z, w


@dataclass(frozen=True)
class Layout:
    """Where everything is. ``lean`` [rad] of the plates in the rack; ``station`` the held plate's centre."""

    plate: Plate = Plate()
    valve: SprayValve = SprayValve(sink_center=(0.0, 0.0, SINK_FLOOR))
    lean: float = math.radians(15.1)     # the measured rest of a released plate in a slot (16 mm rows)
    station: tuple[float, float, float] = (0.0, -0.22, RACK_Z + 0.22)
    standoff: float = 0.20            # m, valve tip to the plate's face
    park: tuple[float, float, float] = (0.25, -0.45, RACK_Z + 0.30)
    clear: float = 0.25               # m, lift along the lean before leaving a slot

    def row_y(self, k: int) -> float:
        return (k - 3) * ROW_PITCH

    def slot(self, k: int) -> Pose:
        """Plate ``k`` at rest in slot ``k``, leaning back on the next row (its centre 6.7 mm ahead of the slot's
        centre plus its lean; measured, as ``lean``)."""
        y_b = 0.5 * (self.row_y(k) + self.row_y(k + 1)) - 0.0067
        r, a, t = self.plate.radius, self.lean, self.plate.thickness
        c = np.array([0.0, y_b + r * math.sin(a), RACK_Z + r * math.cos(a) + 0.5 * t * math.sin(a) + 2e-4])
        return Pose(c, quat_x(0.5 * math.pi - a))

    def lifted(self, k: int) -> Pose:
        s = self.slot(k)
        up = np.array([0.0, math.sin(self.lean), math.cos(self.lean)])
        return Pose(s.pos + self.clear * up, s.quat)

    def above_station(self, k: int) -> Pose:
        return Pose(np.array([self.station[0], self.station[1], self.lifted(k).pos[2]]), quat_x(0.5 * math.pi))

    def at_station(self) -> Pose:
        return Pose(np.array(self.station, np.float64), quat_x(0.5 * math.pi))

    def valve_pose(self, x: float, z: float) -> Pose:
        """The valve's tip at (x, station_y - standoff, z), spraying +y."""
        return Pose(np.array([x, self.station[1] - self.standoff, z]), quat_x(-0.5 * math.pi))

    def valve_park(self) -> Pose:
        return Pose(np.array(self.park, np.float64), quat_x(-0.5 * math.pi))

    def build(self, b: newton.ModelBuilder, plates: int) -> None:
        """Static rack and backsplash, ``plates`` plates in their slots, the valve at its park pose."""
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.30)
        b.add_shape_box(-1, xform=_xf(np.array([0.0, 0.0, RACK_Z - 0.0055]), np.array([0, 0, 0, 1.0])), hx=0.25,
                        hy=0.25, hz=0.0055, cfg=cfg, label="rack_floor")
        for k in range(SLOTS + 1):
            b.add_shape_box(-1, xform=_xf(np.array([0.0, self.row_y(k), RACK_Z + ROW_HALF[2]]),
                                          np.array([0, 0, 0, 1.0])),
                            hx=ROW_HALF[0], hy=ROW_HALF[1], hz=ROW_HALF[2], cfg=cfg, label=f"rack_row{k}")
        b.add_shape_box(-1, xform=_xf(np.array([0.0, 0.40, TABLE_Z + 0.20]), np.array([0, 0, 0, 1.0])), hx=0.40,
                        hy=0.01, hz=0.20, label="backsplash")
        for k in range(plates):
            s = self.slot(k)
            self.plate.build(b, f"plate{k}", _xf(s.pos, s.quat))
        p = self.valve_park()
        self.valve.build(b, "valve", _xf(p.pos, p.quat))


def _xf(pos: np.ndarray, quat: np.ndarray) -> wp.transform:
    return wp.transform(wp.vec3(*(float(v) for v in pos)), wp.quat(*(float(v) for v in quat)))


@dataclass(frozen=True)
class Step:
    """One control period of a move: pose and feed-forward velocity."""

    pose: Pose
    vel: np.ndarray


def move(a: Pose, b: Pose, seconds: float, dt: float) -> Iterator[Step]:
    """A smooth-step move from ``a`` to ``b`` (position lerp, orientation slerp), one Step per period."""
    n = max(int(round(seconds / dt)), 1)
    for i in range(1, n + 1):
        s = i / n
        ease = s * s * (3.0 - 2.0 * s)
        rate = 6.0 * s * (1.0 - s) / seconds
        yield Step(Pose(a.pos + ease * (b.pos - a.pos), slerp(a.quat, b.quat, ease)), rate * (b.pos - a.pos))


def _row_xs(speed: float, half: float, dt: float) -> np.ndarray:
    return np.arange(-half, half + 1e-9, speed * dt)


def row_steps(speed: float, half: float, dt: float) -> int:
    """Control periods in one raster row."""
    return len(_row_xs(speed, half, dt))


def raster(layout: Layout, speed: float, spacing: float, half: float, dt: float) -> Iterator[Step]:
    """The valve's serpentine over the station plate's face: rows along x, ``spacing`` apart in z, ±``half``."""
    cz = layout.station[2]
    rows = np.arange(-half, half + 1e-9, spacing)
    xs = _row_xs(speed, half, dt)
    for k, z in enumerate(rows):
        sign = 1.0 if k % 2 == 0 else -1.0
        for x in xs if k % 2 == 0 else xs[::-1]:
            yield Step(layout.valve_pose(layout.station[0] + x, cz + z), np.array([sign * speed, 0.0, 0.0]))


__all__ = ["Layout", "Pose", "RACK_Z", "SINK_FLOOR", "SLOTS", "Step", "TABLE_Z", "move", "quat_x", "raster", "row_steps",
           "slerp"]
