# SPDX-License-Identifier: Apache-2.0
"""GA's put-away-rack on Alloy (SCR-WP8-12 F5): the truck enters a rack aisle carrying a loaded pallet, turns to face
the named location, raises the load over the beams, creeps in, lowers it onto both beams, backs out and lowers its
forks. Scored by ``forklift_ga.put_away_rack``.

GA's warehouse frame is the rig's frame. The scene holds the floor (ground plane), rack row A (the target's frame)
and row B across aisle 1 for bays 1-3 as GA builds them (``forklift_ga_props.add_rack_row``), a stored unit load on
the target's neighbouring slot (a body, as GA's around a put-away slot, F/facility.py:108-118), and the carried
pallet seated on the forks. The start is GA's (F/task.py:236-243): the aisle's entry, heading down it, forks at
0.15 m tilted back 3°, a wrapped load of 600-1000 kg and 0.9-1.2 m (here 800 kg, 1.05 m).

Out of scope (SCR-WP8-12 ruling): the LPG engine and torque converter (an electric traction motor), K·d^1.5 tyres,
wheel width and curbs, the humanoid operator (scripted; the seat and head criteria are not judged).
"""

from __future__ import annotations

import functools
import math

from alloy._vendor import newton
from alloy.tasks import forklift_ga as ga
from alloy.tasks import forklift_truck as ft
from alloy.tasks.forklift_ga_props import (FORK_GAP, POCKET, REST_GAP, UnitLoad, add_pallet, add_rack_row, carried_at,
                                          yaw_xform)
from alloy.tasks.forklift_ga_run import Box, Task, arc, truck_at
from alloy.tasks.forklift_ga_truck import Frame
from alloy.tasks.forklift_path import segment
from alloy.tasks.forklift_plans import Plan
from alloy.tasks.forklift_site import SITE, rack_aisles, rack_location

LOCATION = "A-e-01-1-0"
NEIGHBOUR = "A-e-01-1-1"
CARRIED = UnitLoad(800.0, (SITE.pallet.L - 0.04, SITE.pallet.W - 0.04, 1.05))
STORED = UnitLoad(700.0, (SITE.pallet.L - 0.04, SITE.pallet.W - 0.04, 1.00))
START_POSE = ft.Pose(tilt=math.radians(-3.0), free=0.15 - ft.FORK_BOTTOM)     # GA's lift 0.15 is the blade bottom
AHEAD = ft.FORK_FACE + FORK_GAP + SITE.pallet.L / 2                          # pallet centre ahead of the front axle
CLEAR = 0.16                                                                 # m, pallet front to the beam face
TURN = 0.95             # m, the front axle's radius into the face (min 0.84 at full lock; the tail swings 2.6 m)


def _geometry(location: str) -> dict[str, float]:
    loc = rack_location(SITE, location)
    xa, xb = rack_aisles(SITE)["1"]
    over = SITE.rack.stored_overhang(SITE.pallet.L)
    sign = 1.0 if loc.face == "e" else -1.0                       # the aisle is on +x of an east face
    centre = loc.aisle_x + sign * (over - SITE.pallet.L / 2)
    return {"x_start": 0.5 * (xa + xb), "y": loc.y, "z": loc.z, "x_set": centre + sign * AHEAD,
            "x_face": loc.aisle_x + sign * (CLEAR + SITE.pallet.L + AHEAD - SITE.pallet.L / 2),
            "yaw": math.pi if loc.face == "e" else 0.0, "centre": centre}


def _build(location: str, b: newton.ModelBuilder) -> None:
    geo = _geometry(location)
    add_rack_row(b, "A", range(1, 4))
    add_rack_row(b, "B", range(1, 4))
    nb = rack_location(SITE, NEIGHBOUR)
    add_pallet(b, "stored", yaw_xform(geo["centre"], nb.y, nb.z + REST_GAP, 0.0), STORED)
    at = truck_at(geo["x_start"], 24.6, -math.pi / 2, START_POSE)
    ft.build(START_POSE, None, locked=False, at=at)(b)
    add_pallet(b, "carried", carried_at(at, START_POSE), CARRIED)


def task(location: str = LOCATION) -> Task:
    geo = _geometry(location)
    x0, y, yaw = geo["x_start"], geo["y"], geo["yaw"]
    sign = 1.0 if yaw == math.pi else -1.0                     # an east face: turn right (west) from heading south
    x_in = geo["x_face"] + sign * TURN
    turn = arc(x_in, y + TURN, -math.pi / 2, -sign * math.pi / 2, TURN)
    segs = {"to_face": segment([(x0, 24.6, -math.pi / 2, 1), (x_in, y + TURN + 3.0, -math.pi / 2, 1),
                                (x_in, y + TURN, -math.pi / 2, 1), *turn], 6.0),
            "creep": segment([(geo["x_face"], y, yaw, 1), (geo["x_set"], y, yaw, 1)], 2.0),
            "back_out": segment([(geo["x_set"], y, yaw, -1), (geo["x_face"] + 0.3, y, yaw, -1)], 3.0)}
    plan = Plan("put_away", segs, {"beam": geo["z"], "drop": -POCKET})          # blades mid-pocket on the beams
    loads = {"carried": _box("carried", CARRIED), f"stored_{NEIGHBOUR}": _box("stored", STORED)}
    watch = {"A0_": "rack", "B0_": "rack", "B1_": "rack", "stored": "rack", "ground_plane": "ground"}
    return Task(name="put-away-rack", build=functools.partial(_build, location), frame=Frame(), plan=plan,
                score=functools.partial(ga.put_away_rack, location=location), loads=loads, watch=watch)


def _box(body: str, load: UnitLoad) -> Box:
    g = SITE.pallet
    return Box(body=body, half=(g.L / 2, g.W / 2, g.H / 2), centre=(0.0, 0.0, g.H / 2), mass=g.mass + load.mass)


__all__ = ["CARRIED", "LOCATION", "NEIGHBOUR", "task"]
