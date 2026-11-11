# SPDX-License-Identifier: Apache-2.0
"""GA's start-and-park on Alloy (SCR-WP8-12 F5): the truck stands in the warehouse's main aisle, key and engine off;
the operator starts it, drives west into the painted parking bay, lowers the forks and tilts their tips to the floor,
parks, and switches off. Scored by ``forklift_ga.start_and_park``.

GA's warehouse frame is the rig's frame (F/facility.py:149: 50 x 36 m inside its walls). The scene holds the floor
(the ground plane), the west and north walls and the LPG cage beside the bay: the structures near the route that GA
watches. The start is GA's "main aisle west" (F/task.py:205-207) without its random jitter.

Out of scope (SCR-WP8-12 ruling): the LPG engine and torque converter (an electric traction motor), K·d^1.5 tyres,
wheel width and curbs, the humanoid operator (scripted; the seat and head criteria are not judged).
"""

from __future__ import annotations

import math


from alloy._vendor import newton
from alloy.plugins.harness.scene import body_xform
from alloy.tasks import forklift_ga as ga
from alloy.tasks import forklift_truck as ft
from alloy.tasks.forklift_ga_run import Task, truck_at
from alloy.tasks.forklift_ga_truck import Frame
from alloy.tasks.forklift_path import segment
from alloy.tasks.forklift_plans import Plan

START = (14.5, 28.25, math.pi)                     # F/task.py:206-207, main aisle west
PARK = (2.8, 28.25, math.pi)                       # footprint (-2.14 .. +1.62 m) centred in the bay's x 0.5-5.6
WALL_H = 9.0                                       # F/params.py:869
STRUCTURES = (("wall_w", (-0.15, 18.0, WALL_H / 2), (0.15, 18.0, WALL_H / 2), "wall"),    # F/facility.py:152-154
              ("wall_n", (25.0, 36.15, WALL_H / 2), (25.3, 0.15, WALL_H / 2), "wall"),
              ("lpg_cage", (1.0, 32.5, 0.9), (0.6, 0.9, 0.9), "structure"))               # F/facility.py:224
CAP_KMH = 7.0                                      # under the posted 8 km/h


def _build(b: newton.ModelBuilder) -> None:
    for label, centre, half, _ in STRUCTURES:
        b.add_shape_box(-1, xform=body_xform(*centre), hx=half[0], hy=half[1], hz=half[2], label=label)
    ft.build(ft.Pose(), None, locked=False, at=truck_at(*START))(b)


def task() -> Task:
    to_bay = segment([(*START[:2], START[2], 1), (*PARK[:2], PARK[2], 1)], CAP_KMH)
    plan = Plan("start_park", {"to_bay": to_bay})
    return Task(name="start-and-park", build=_build, frame=Frame(), plan=plan, score=ga.start_and_park,
                watch={label: cls for label, *_, cls in STRUCTURES})


__all__ = ["PARK", "START", "STRUCTURES", "task"]
