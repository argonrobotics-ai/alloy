# SPDX-License-Identifier: Apache-2.0
"""tool-joints: the jointer's timing against the mortar's thumbprint window (5-20 kPa, ``concrete_materials``).

The verdict wants 90 % of the joints' centre cells last worked inside the window (``tooled_thumbprint_hard``), and the
process layer records a cell's yield whenever it lies under an advancing tool's bearing sole (``tau_w``). On the whole
wall two things put the oldest course outside it (the 100k pass on rc3: 0.873 in 4 of 4 worlds, while windows of the
wall, which clip the schedule, read 1.0):
- its mortar starts at 18.0 kPa and passes 20 kPa 102 s into the run; at 0.10 m/s its bed got its last pass at
  102-114 s (``course_speed`` raises the speed so a course ends ``hard_margin`` before its mortar leaves the window);
- the infill's side joints run through every course, and the rod, 150 mm long, rides 69 mm past a stroke's end: the
  next course's side strokes worked the oldest side joints again at 228-240 s, at 22.7-22.9 kPa (``off_older`` stops
  a side stroke short of the older course below, and runs it downwards only).

``ripe_share`` is the same bookkeeping without a simulation: the share of joint length whose last working by a
program lies inside the window. It predicted 0.859 for the program that scored 0.873, and is the whole wall's fast
check (the whole-wall run takes 4.4 h on an L4).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from alloy.tasks import concrete_joints_start as JS
from alloy.tasks import concrete_materials as CM
from alloy.tasks.concrete_tools import FORCE, Stroke


def hard_time(course: int, temperature: float, wait: float = 0.0) -> float:
    """Program time [s] at which ``course``'s mortar leaves the thumbprint window (reaches its upper yield), for a
    course tooled after ``wait`` s of parked ageing. Example: ``hard_time(0, 21.1)`` -> 102 (18.0 kPa at the start)."""
    age0 = JS.AGES_MIN[course] * 60.0 * CM.masco_factor(temperature)
    return CM.MasonryMortar().time_to_tau(CM.THUMBPRINT[1], temperature) - age0 - wait


def course_speed(end_at: Callable[[float], float], deadline: float, speed: float, speed_max: float) -> float:
    """The lowest jointer speed [m/s] from ``speed`` up to ``speed_max`` (5 % steps) at which a course's last stroke
    ends by ``deadline``; ``end_at(v)`` is that end time at speed v. ``speed_max`` if none does.
    Example: ``course_speed(lambda v: 33.0 + 8.08 / v, 92.0, 0.10, 0.20)`` -> 0.141 (the oldest course's 8.08 m)."""
    v = speed
    while end_at(v) > deadline and v < speed_max:
        v = min(v * 1.05, speed_max)
    return v


def off_older(lane: Any, stop: float | None) -> Any:
    """A side joint's stroke for a course above the oldest: its lower end ``stop`` above the lane's, and one way
    (downwards), so the rod never bears on the older course's side joint below. Other lanes, and every lane when
    ``stop`` is None, come back unchanged. ``lane``: a ``concrete_joints.Lane`` (vertical lanes run top to bottom)."""
    j = JS.JOINT_W
    side = lane.p0[0] == lane.p1[0] and min(abs(lane.p0[0] - (JS.INFILL_X[0] + j / 2)),
                                           abs(lane.p0[0] - (JS.INFILL_X[1] - j / 2))) < 1.0e-9
    if stop is None or not side or lane.course == 0:
        return lane
    return dataclasses.replace(lane, p1=(lane.p1[0], min(lane.p1[1] + stop, lane.p0[1])), one_way=True)


def working(keys: Sequence[Stroke]) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """A program's working segments: (t0, t1, from, to) for every keyframe pair pressed at a force (world x, y)."""
    return [(keys[i].t, keys[i + 1].t, np.array(keys[i].pos[:2]), np.array(keys[i + 1].pos[:2]))
            for i in range(len(keys) - 1) if keys[i].mode == FORCE and keys[i].force > 0.0]


@dataclasses.dataclass(frozen=True)
class Ripe:
    """``share``: joint length last worked inside the window / all of it; ``hard``, ``soft``, ``never``: the shares
    last worked above it, below it, and never worked."""

    share: float
    hard: float
    soft: float
    never: float


def ripe_share(keys: Sequence[Stroke], lanes: Sequence[Any], world: Callable[[float, float], tuple[float, float]],
               sole: tuple[float, float], t_wait: float, wait: float, temperature: float,
               step: float = 0.002) -> Ripe:
    """Where the jointer's program ``keys`` leaves each joint's ``tau_w``: points every ``step`` m along the full
    ``lanes`` (wall coordinates; ``world`` maps them to the program's), each last under the rod's bearing ``sole``
    (half length along the stroke, half width) at some time; its course's mortar then has that age (plus ``wait``
    after ``t_wait``)."""
    pts, crs = [], []
    for ln in lanes:
        a, b = np.array(world(*ln.p0)), np.array(world(*ln.p1))
        n = max(int(np.linalg.norm(b - a) / step), 2)
        pts += [a + f * (b - a) for f in (np.arange(n) + 0.5) / n]
        crs += [ln.course] * n
    pts, crs = np.array(pts), np.array(crs)
    last = np.full(len(pts), -1.0)
    for t0, t1, a, b in working(keys):
        d = float(np.linalg.norm(b - a))
        u = (b - a) / d
        rel = pts - a
        along, across = rel @ u, np.abs(rel @ np.array([-u[1], u[0]]))
        under = (across <= sole[1]) & (along >= -sole[0]) & (along <= d + sole[0])
        left = t0 + np.clip(along + sole[0], 0.0, d) / d * (t1 - t0)      # the rod's trailing end leaves the point
        last = np.where(under, np.maximum(last, left), last)
    age = np.array(JS.AGES_MIN)[crs] * 60.0 * CM.masco_factor(temperature) + last + np.where(last > t_wait, wait, 0.0)
    tau = CM.MasonryMortar().tau_bulk(age, temperature)
    lo, hi = CM.THUMBPRINT
    done = last >= 0.0
    return Ripe(share=float((done & (tau >= lo) & (tau <= hi)).mean()), hard=float((done & (tau > hi)).mean()),
                soft=float((done & (tau < lo)).mean()), never=float((~done).mean()))
