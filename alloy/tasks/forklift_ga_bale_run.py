# SPDX-License-Identifier: Apache-2.0
"""GA's load-bales on Alloy (SCR-WP8-12 F5): three bales stand on dunnage strips in the bale row; the truck forks two
of them, one at a time, carries each over the leveler into the trailer at door3 and sets it on the strips laid at the
nose (left, then right), then backs out onto the dock. Scored by ``forklift_ga_bales.load_bales``.

The rig is F4's dock (``forklift_ga_dock``) widened to hold the bale row, mapped to GA's facility frame by
``forklift_ga_dock.frame``. GA's numbers (F/task.py:285-311): bales 60 x 30 x 48 in, 450-590 kg (here 520 kg), mu 0.4,
each on two 4 x 4 x 48 in strips 0.13 m either side of its centre; the bale row's three bales 1.8 m apart at x = 20.5;
two pairs of strips at the nose; the truck at (16, 6) facing the row, forks 0.10 m up. GA spots a second trailer at
door4 that a misplaced bale could end in; it is not built here (a bale is only ever carried to door3).

Out of scope (SCR-WP8-12 ruling): the LPG engine and torque converter (an electric traction motor), K·d^1.5 tyres,
wheel width and curbs, the humanoid operator (scripted; the seat and head criteria are not judged).
"""

from __future__ import annotations

import math

import numpy as np

from alloy._vendor import newton
from alloy.tasks import forklift_dock as fd
from alloy.tasks import forklift_ga_bales as gb
from alloy.tasks import forklift_ga_dock as gd
from alloy.tasks import forklift_trailer as tr
from alloy.tasks import forklift_truck as ft
from alloy.tasks.forklift_ga_props import FORK_GAP, POCKET, REST_GAP, add_box_load, yaw_xform
from alloy.tasks.forklift_ga_run import Box, Task, arc, truck_at
from alloy.tasks.forklift_ga_truck import Extras
from alloy.tasks.forklift_path import segment
from alloy.tasks.forklift_plans import Plan
from alloy.tasks.forklift_site import SITE

DOOR = "door3"
FRAME = gd.frame(DOOR)
BALE = SITE.bale
BALE_MASS = 520.0
STRIP = BALE.dunnage_size                         # (4 in, 4 in, 48 in): along the bale's 60 in
ROW_X, ROW_Y = BALE.row_x[0] + 1.5, 0.5 * (BALE.row_y[0] + BALE.row_y[1])     # F/task.py:291-292
BALES_Y = tuple(ROW_Y + 1.8 * (k - 1) for k in range(3))
START = (16.0, 6.0, 0.0)                          # F/task.py:310
START_POSE = ft.Pose(free=0.10 - ft.FORK_BOTTOM)
AHEAD = ft.FORK_FACE + FORK_GAP + BALE.size[0] / 2                            # bale centre ahead of the front axle
TURN = 2.5
TRIPS = ((1, "L"), (2, "R"))                      # (bale, nose side)


def _rig(x: float, y: float, z: float, yaw: float) -> tuple[float, float, float, float]:
    """A GA pose (z above the dock floor) in the rig's frame."""
    p = FRAME.to_rig(np.array([x, y, z]))
    return float(p[0]), float(p[1]), float(p[2]), yaw - FRAME.yaw


def _nose() -> dict[str, tuple[float, float]]:
    """GA x, y of each nose bale's centre (F/task.py:302-308: strips centred 30 in apart, 30 mm off the nose)."""
    door_x, face = SITE.dock.door(DOOR)
    nose_y = face + fd.BUMPER + tr.REAR_FRAME + 0.006 + tr.INSIDE_L
    return {side: (door_x + sx * (BALE.size[1] / 2 + 0.03), nose_y - 0.03 - BALE.size[0] / 2)
            for side, sx in (("L", -1.0), ("R", 1.0))}


def _strips(b: newton.ModelBuilder, prefix: str, x: float, y: float, z: float, yaw: float) -> None:
    """Two strips along ``yaw`` either side of (x, y), GA frame."""
    for tag, sy in (("a", -1.0), ("b", 1.0)):
        ox, oy = -sy * BALE.dunnage_y * math.sin(yaw), sy * BALE.dunnage_y * math.cos(yaw)
        rx, ry, rz, ryaw = _rig(x + ox, y + oy, z, yaw)
        add_box_load(b, f"{prefix}{tag}", yaw_xform(rx, ry, rz + REST_GAP, ryaw),
                     (STRIP[2], STRIP[0], STRIP[1]), BALE.dunnage_mass, 0.5)


def _build(b: newton.ModelBuilder) -> None:
    fd.add_dock(b, depth=21.0, half_width=13.0)                    # GA's dock section: y 0-20 inside (F/facility.py:257)
    gd._wall(b, half_width=13.0)
    fd.add_leveler(b, gd.SPOT.bed)
    tr.add_trailer(b, gd.SPOT)
    for k, y in enumerate(BALES_Y):
        _strips(b, f"dunnage_{k}", ROW_X, y, 0.0, 0.0)
        rx, ry, rz, ryaw = _rig(ROW_X, y, STRIP[1], 0.0)
        add_box_load(b, f"bale_{k}", yaw_xform(rx, ry, rz + 2 * REST_GAP, ryaw), BALE.size, BALE_MASS,
                     BALE.mu)
    for side, (x, y) in _nose().items():
        _strips(b, f"nose_dunnage_{side}", x, y, gd.SPOT.bed - fd.DOCK_H, math.pi / 2)
    rx, ry, _, ryaw = _rig(*START[:2], 0.0, START[2])
    ft.build(START_POSE, None, locked=False, at=truck_at(rx, ry, ryaw, START_POSE, z=fd.DOCK_H))(b)


def _trip(n: int, bale: int, side: str, start: tuple[float, float, float]) -> dict:
    """One bale's segments (GA frame), suffixed ``n``."""
    door_x, face = SITE.dock.door(DOOR)
    y_b = BALES_Y[bale]
    x_pick = ROW_X - AHEAD
    x_arc = door_x + TURN                              # backing away, the tail swings south to face the door
    y_dock = face - fd.DECK_L - (ft.FORK_FACE + FORK_GAP + BALE.size[0]) - 0.3
    nx, ny = _nose()[side]
    y_set = ny - AHEAD
    k = str(n)
    return {
        "pick" + k: segment([start, (x_pick, y_b, 0.0, 1)], 3.0),
        "reverse" + k: segment([(x_pick, y_b, 0.0, -1), (x_arc, y_b, 0.0, -1),
                                *arc(x_arc, y_b, 0.0, math.pi / 2, TURN, direction=-1)], 4.0),
        "dock" + k: segment([(door_x, y_b - TURN, math.pi / 2, 1), (door_x, y_dock, math.pi / 2, 1)], 6.0),
        "in" + k: segment([(door_x, y_dock, math.pi / 2, 1), (door_x, face + 2.5, math.pi / 2, 1),
                           (nx, y_set, math.pi / 2, 1)], 4.0),
        "back" + k: segment([(nx, y_set, math.pi / 2, -1), (door_x, face + 2.5, math.pi / 2, -1),
                             (door_x, face - 4.0, math.pi / 2, -1)], 4.0)}


def _segments() -> dict:
    """Both trips; the second starts where the first's back leg left the truck, turning east toward the row."""
    door_x, face = SITE.dock.door(DOOR)
    segs = _trip(1, *TRIPS[0], (*START[:2], START[2], 1))
    y2 = BALES_Y[TRIPS[1][0]]
    y_turn = y2 + TURN
    segs["back1"] = segment([*segs["back1"].points[:2], (door_x, y_turn, math.pi / 2, -1),
                             *arc(door_x, y_turn, math.pi / 2, -math.pi / 2, TURN, direction=-1)], 4.0)
    x2, y2_ = segs["back1"].end[:2]
    segs.update(_trip(2, *TRIPS[1], (x2, y2_, 0.0, 1)))
    return segs


def _box(body: str, size: tuple[float, float, float], mass: float) -> Box:
    return Box(body=body, half=tuple(0.5 * v for v in size), centre=(0.0, 0.0, 0.5 * size[2]), mass=mass)


def task() -> Task:
    levels = {"approach": 0.04, "carry": 0.15, "set": POCKET, "rest": 0.0, "bales": 2.0, "side": 0.0}
    plan = Plan("load_bales", _segments(), levels)
    strip = (STRIP[2], STRIP[0], STRIP[1])
    loads = {f"bale_{k}": _box(f"bale_{k}", BALE.size, BALE_MASS) for k in range(3)}
    loads.update({f"dunnage_{k}{t}": _box(f"dunnage_{k}{t}", strip, BALE.dunnage_mass) for k in range(3)
                  for t in "ab"})
    loads.update({f"nose_dunnage_{s}{t}": _box(f"nose_dunnage_{s}{t}", strip, BALE.dunnage_mass) for s in "LR"
                  for t in "ab"})
    return Task(name="load-bales", build=_build, frame=FRAME, plan=plan, score=gb.load_bales,
                extras=Extras(restraint_green=lambda io: True), surfaces=(*fd.DRIVABLE, *tr.DRIVABLE),
                resting=fd.RESTING, ground_plane=False, companions=lambda: tr.supports(gd.SPOT, fd.dock_rest(gd.SPOT)),
                loads=loads, trailer=tr.LABEL, watch=gd.WATCH, floors=("floor", "trailer_floor"),
                rename={"trailer_nose": f"{DOOR}:nose"})


__all__ = ["DOOR", "FRAME", "task"]
