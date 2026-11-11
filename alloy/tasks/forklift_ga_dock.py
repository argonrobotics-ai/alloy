# SPDX-License-Identifier: Apache-2.0
"""GA's load-trailer on Alloy (SCR-WP8-12 F5): the truck waits in the staging lane in front of a dock door with a
loaded pallet on its forks, drives to the leveler, checks the restraint light, crosses the leveler into the trailer,
sets the pallet down behind the nose pallet on its side, against the wall, and backs out onto the dock. With the
restraint red (``secured=False``: no hook, no chocks) the job is to stay off the leveler and wait. Scored by
``forklift_ga.load_trailer``.

The rig is F4's (``forklift_dock``, ``forklift_trailer``): the dock face at x = 0, the trailer's nose toward +x, the
dock floor 48 in up. GA's dock wall stands on the dock face with the door's opening (F/facility.py:277-292), the
leveler deploys onto the trailer floor, and the trailer stands on its support vehicles (the tandem braked, the
landing gear on sand shoes). ``FRAME`` maps it to GA's facility frame: rig +x is GA's +y at the door, the dock floor
is GA's z = 0. The start is GA's (F/task.py:279-283) at door1, side L, the trailer floor level with the dock (GA draws
44-52 in), without the random jitter: lift 0.15 m tilted back 3°, a wrapped load of 600-1100 kg and 1.0-1.4 m (here
900 kg, 1.2 m); one row of two pallets at the nose (500-900 kg, 1.0-1.4 m; here 700 kg, 1.2 m).

Out of scope (SCR-WP8-12 ruling): the LPG engine and torque converter (an electric traction motor), K·d^1.5 tyres,
wheel width and curbs, the humanoid operator (scripted; the seat and head criteria are not judged).
"""

from __future__ import annotations

import functools
import math

import numpy as np

from alloy._vendor import newton
from alloy.plugins.harness.scene import body_xform
from alloy.tasks import forklift_dock as fd
from alloy.tasks import forklift_ga as ga
from alloy.tasks import forklift_trailer as tr
from alloy.tasks import forklift_truck as ft
from alloy.tasks.forklift_ga_props import FORK_GAP, POCKET, REST_GAP, UnitLoad, add_pallet, carried_at, yaw_xform
from alloy.tasks.forklift_ga_rack import _box
from alloy.tasks.forklift_ga_run import Task, truck_at
from alloy.tasks.forklift_ga_truck import Extras, Frame
from alloy.tasks.forklift_path import segment
from alloy.tasks.forklift_plans import Plan
from alloy.tasks.forklift_site import SITE

DOOR, SIDE = "door1", "L"
SPOT = tr.Spot(x=fd.BUMPER + tr.REAR_FRAME + 0.006, bed=fd.DOCK_H)          # F/facility.py:567
CARRIED = UnitLoad(900.0, (SITE.pallet.L - 0.04, SITE.pallet.W - 0.04, 1.2))
NOSE = UnitLoad(700.0, (SITE.pallet.L - 0.04, SITE.pallet.W - 0.04, 1.2))
START_POSE = ft.Pose(tilt=math.radians(-3.0), free=0.15 - ft.FORK_BOTTOM)
AHEAD = ft.FORK_FACE + FORK_GAP + SITE.pallet.L / 2
WALL_T, DOOR_W, DOOR_H, WALL_H = 0.30, 9 * 0.3048, 10 * 0.3048, 9.0         # F/facility.py:279; F/params.py:869
SIDE_Y = tr.INSIDE_W / 2 - SITE.pallet.W / 2 - 0.03                         # nose pallets' y (F/task.py:274)
SHIFT = 0.10                                                                # side shift used inside (its limit)
GAP = 0.025                                                                 # m behind the nose pallet
WATCH = {"dockwall": "wall", "dockheader": "wall", "dock_bumper": "bumper", "trailer_wall": "trailer_wall",
         "trailer_nose": "trailer_wall", "trailer_post": "trailer_wall", "trailer_header": "trailer_wall",
         "trailer_roof": "trailer_wall", "trailer_icc": "trailer_wall", "trailer_floor": "trailer_floor",
         "trailer_sill": "trailer_floor", "dock_floor": "floor", "dock_side": "floor", "leveler": "floor",
         "nose_": "trailer_wall"}


def frame(door: str = DOOR) -> Frame:
    x, face = SITE.dock.door(door)
    return Frame(yaw=math.pi / 2, shift=(x, face, -fd.DOCK_H))


def _nose_x() -> float:
    return SPOT.x + tr.INSIDE_L - 0.03 - SITE.pallet.L / 2


def _wall(b: newton.ModelBuilder, half_width: float = 4.0) -> None:
    """GA's dock wall on the dock face: two piers beside the door and the header over it."""
    x, z0 = -WALL_T / 2, fd.DOCK_H
    for name, y0, y1 in (("dockwall_l", DOOR_W / 2, half_width), ("dockwall_r", -half_width, -DOOR_W / 2)):
        b.add_shape_box(-1, xform=body_xform(x, 0.5 * (y0 + y1), z0 + WALL_H / 2), hx=WALL_T / 2, hy=0.5 * (y1 - y0),
                        hz=WALL_H / 2, label=name)
    b.add_shape_box(-1, xform=body_xform(x, 0.0, z0 + 0.5 * (DOOR_H + WALL_H)), hx=WALL_T / 2, hy=DOOR_W / 2,
                    hz=0.5 * (WALL_H - DOOR_H), label="dockheader")


def _build(start_x: float, b: newton.ModelBuilder) -> None:
    fd.add_dock(b, depth=16.0)
    _wall(b)
    fd.add_leveler(b, SPOT.bed)
    tr.add_trailer(b, SPOT)
    for k, sign in ((1, 1.0), (2, -1.0)):
        add_pallet(b, f"nose_{k}", yaw_xform(_nose_x(), sign * SIDE_Y, SPOT.bed + REST_GAP, 0.0), NOSE)
    at = truck_at(start_x, 0.0, 0.0, START_POSE, z=fd.DOCK_H)
    ft.build(START_POSE, None, locked=False, at=at)(b)
    add_pallet(b, "carried", carried_at(at, START_POSE), CARRIED)


def _rest_loads() -> tr.Loads:
    """The trailer's extra static loads: the leveler's lip tip and the two nose pallets (trailer frame)."""
    weight = (SITE.pallet.mass + NOSE.mass) * 9.81
    return (*fd.dock_rest(SPOT), (2.0 * weight, _nose_x() - SPOT.x))


def _segments(secured: bool) -> dict:
    """The trip in the rig frame: up to the leveler, in along the centre, across to the wall side, and back out."""
    pallet_x = _nose_x() - SITE.pallet.L - GAP
    x_set, y_set = pallet_x - AHEAD, (SIDE_Y - 0.045) - SHIFT
    x_dock = -fd.DECK_L - AHEAD - SITE.pallet.L / 2 - 0.3                 # load 0.3 m short of the leveler
    rig = {"dock": [(-11.0, 0.0, 0.0, 1), (x_dock, 0.0, 0.0, 1)],
           "in": [(x_dock, 0.0, 0.0, 1), (2.0, 0.0, 0.0, 1), (x_set, y_set, 0.0, 1)],
           "back": [(x_set, y_set, 0.0, -1), (2.0, 0.0, 0.0, -1), (-4.0, 0.0, 0.0, -1)]}
    caps = {"dock": 6.0, "in": 4.0, "back": 4.0}
    f = frame()
    out = {}
    for key, points in rig.items():
        xy = f.point(np.array([[p[0], p[1], 0.0] for p in points]))
        out[key] = segment([(float(q[0]), float(q[1]), p[2] + f.yaw, p[3]) for q, p in zip(xy, points, strict=True)],
                           caps[key])
    return out


def task(secured: bool = True) -> Task:
    plan = Plan("load_trailer", _segments(secured), {"carry": 0.15, "set": POCKET, "rest": 0.0, "side": SHIFT})
    side = "L"
    loads = {"carried": _box("carried", CARRIED), "trailer_1L": _box("nose_1", NOSE),
             "trailer_1R": _box("nose_2", NOSE)}
    return Task(name="load-trailer", build=functools.partial(_build, -11.0), frame=frame(), plan=plan,
                score=functools.partial(ga.load_trailer, door=DOOR, side=side, secured=secured),
                extras=Extras(restraint_green=lambda io: secured),
                surfaces=(*fd.DRIVABLE, *tr.DRIVABLE), resting=fd.RESTING, ground_plane=False,
                companions=lambda: tr.supports(SPOT, _rest_loads()), loads=loads, trailer=tr.LABEL,
                watch=WATCH, floors=("floor", "trailer_floor"), waits_at=None if secured else "restraint_check")


__all__ = ["CARRIED", "DOOR", "SIDE", "SPOT", "frame", "task"]
