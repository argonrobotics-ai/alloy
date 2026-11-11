# SPDX-License-Identifier: Apache-2.0
"""GA's ramp-down-loaded on Alloy (SCR-WP8-12 F5): the truck stands inside the yard trailer with a loaded pallet on
its forks, backs out over the yard ramp's lip plates and down its incline (the load uphill), turns, drives into the
painted drop zone, sets the pallet down on the asphalt and backs away. Scored by ``forklift_ga.ramp_down_loaded``.

The rig is F4's yard (``forklift_dock.add_yard_ramp``, ``forklift_trailer``): the trailer's nose toward +x, the ramp
toward -x from its rear sill, the ground at z = 0. ``FRAME`` maps it to GA's yard (F/facility.py:340-408): the trailer's
nose toward GA's -y, its rear sill at (10, -8), the ramp's foot toward +y. The start is GA's (F/task.py:324-331) 5 m
in from the rear sill (GA draws 3-9 m), forks toward the nose, lift 0.15 m tilted back 3°, a wrapped load of 900-1300
kg and 1.0-1.3 m (here 1100 kg, 1.15 m); the east drop zone.

Out of scope (SCR-WP8-12 ruling): the LPG engine and torque converter (an electric traction motor), K·d^1.5 tyres,
wheel width and curbs, the humanoid operator (scripted; the seat and head criteria are not judged).
"""

from __future__ import annotations

import functools
import math


from alloy._vendor import newton
from alloy.plugins.vehicle import laws_grip as lg
from alloy.tasks import forklift_dock as fd
from alloy.tasks import forklift_ga as ga
from alloy.tasks import forklift_mast as fm
from alloy.tasks import forklift_trailer as tr
from alloy.tasks import forklift_truck as ft
from alloy.tasks.forklift_ga_dock import WATCH
from alloy.tasks.forklift_ga_props import FORK_GAP, POCKET, UnitLoad, add_pallet, carried_at
from alloy.tasks.forklift_ga_rack import _box
from alloy.tasks.forklift_ga_run import Task, arc, truck_at
from alloy.tasks.forklift_ga_truck import Extras, Frame
from alloy.tasks.forklift_path import segment
from alloy.tasks.forklift_plans import Plan
from alloy.tasks.forklift_site import SITE, ramp_speed_zone

SPOT = tr.Spot(x=0.0, bed=fd.DOCK_H)                       # GA's yard trailer: bed 48 in (F/facility.py:376)
REAR_GA = (10.0, -8.0)                                     # F/facility.py:344-345
FRAME = Frame(yaw=-math.pi / 2, shift=(REAR_GA[0], REAR_GA[1] + SPOT.x, 0.0))
IN_FROM_REAR = 5.0                                         # m, the front axle inside the rear sill (GA: 3-9)
CARRIED = UnitLoad(1100.0, (SITE.pallet.L - 0.04, SITE.pallet.W - 0.04, 1.15))
START_POSE = ft.Pose(tilt=math.radians(-3.0), free=0.15 - ft.FORK_BOTTOM)
AHEAD = ft.FORK_FACE + FORK_GAP + SITE.pallet.L / 2
DROP = "east"
TURN = 3.0                                                 # m, the reversing turn off the ramp's foot
CARRY = 0.30            # m: the forks reach 1.8 m up the incline at its foot; at 0.15 the pallet's lip meets it


def _plates() -> list[str]:
    return [f"ramp_lip_{k}" for k in range(fd.PLATES)]


def _build(b: newton.ModelBuilder) -> None:
    _, labels = fd.add_yard_ramp(b, SPOT.x, SPOT.bed)
    tr.add_trailer(b, SPOT)
    at = truck_at(SPOT.x + IN_FROM_REAR, 0.0, 0.0, START_POSE, z=SPOT.bed)
    ft.build(START_POSE, None, locked=False, at=at)(b)
    add_pallet(b, "carried", carried_at(at, START_POSE), CARRIED)


def _rest_loads() -> tr.Loads:
    """The yard trailer's extra static loads: the ramp's lip plates and the laden truck on its two axles."""
    load = ft.Load(SITE.pallet.mass + CARRIED.mass)
    front, rear = fm.axle_loads(START_POSE, load, lg.TRUCK["radial_stiffness"][0])
    x = IN_FROM_REAR
    return (*fd.ramp_rest(SPOT.x), (front, x), (rear, x - 1.649))


def _segments() -> dict:
    """In GA's frame: reverse out and down the ramp, swing the nose east, drive into the drop zone, back away."""
    r, dz = SITE.ramp, SITE.drop_zone(DROP)
    x0, y_in = REAR_GA[0], REAR_GA[1] - IN_FROM_REAR
    y_off = r.y_foot + 0.5                                  # the front axle just past the foot: forks still above it
    swing = arc(x0, y_off, -math.pi / 2, math.pi / 2, TURN, direction=-1)
    cx, cy = 0.5 * (dz.x[0] + dz.x[1]), 0.5 * (dz.y[0] + dz.y[1])
    x_set = cx - AHEAD
    xe, ye = swing[-1][:2]
    return {"out": segment([(x0, y_in, -math.pi / 2, -1), (x0, r.y_rear + 1.0, -math.pi / 2, -1)], 4.0),
            "ramp": segment([(x0, r.y_rear + 1.0, -math.pi / 2, -1), (x0, y_off, -math.pi / 2, -1), *swing], 4.0),
            "to_drop": segment([(xe, ye, 0.0, 1), (x_set - 2.0, cy, 0.0, 1), (x_set, cy, 0.0, 1)], 6.0),
            "back_out": segment([(x_set, cy, 0.0, -1), (x_set - 1.5, cy, 0.0, -1)], 3.0)}


def _on_ramp(io) -> bool:
    p, _ = io.pose()
    zone = ramp_speed_zone(SITE)
    return zone.x[0] <= p[0] <= zone.x[1] and zone.y[0] <= p[1] <= zone.y[1]


def task() -> Task:
    plan = Plan("ramp_down", _segments(), {"carry": CARRY, "set": POCKET, "rest": 0.0})
    plates = _plates()
    surfaces = ("ramp_incline", "ramp_level", *plates, *tr.DRIVABLE)
    watch = {**WATCH, "ramp_": "floor", "ground_plane": "floor"}            # GA: floor/apron/ramp geoms
    return Task(name="ramp-down-loaded", build=_build, frame=FRAME, plan=plan,
                score=functools.partial(ga.ramp_down_loaded, drop_zone=DROP),
                extras=Extras(on_ramp=_on_ramp, floor_z=lambda io: 0.0), surfaces=surfaces,
                resting=tuple((p, "trailer_floor") for p in plates), ground_plane=True,
                companions=lambda: tr.supports(SPOT, _rest_loads()), loads={"carried": _box("carried", CARRIED)},
                watch=watch, floors=("floor", "trailer_floor"))


__all__ = ["DROP", "FRAME", "SPOT", "task"]
