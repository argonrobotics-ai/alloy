# SPDX-License-Identifier: Apache-2.0
"""The five Gym-Anything forklift task phase lists, built from a ``Plan``.

A ``Plan`` is the task geometry the scene supplies: named ``segments`` (waypoint paths with speed caps) and named
``levels`` (fork heights in m, sideshifts in m, a bale count). Required names per ``kind``:

- ``start_park``: segments ``to_bay``.
- ``ramp_down``: segments ``out``, ``ramp``, ``to_drop``, ``back_out``; levels ``carry`` 0.15, ``set`` 0.0,
  ``rest`` 0.0.
- ``put_away``: segments ``to_face``, ``creep``, ``back_out``; levels ``beam`` (1.60 or 3.20), ``clearance`` 0.10,
  ``carry`` 0.2, ``drop`` 0.05 (below the beam top), ``final`` 0.2.
- ``load_trailer``: segments ``pick`` (left out when the load starts on the forks), ``reverse`` (optional: backing
  away from the pick before driving to the dock), ``dock``, ``in``, ``back``; levels ``approach`` 0.05, ``carry`` 0.15,
  ``set`` 0.0, ``rest`` 0.0, ``side`` 0.0 (sideshift toward the wall).
- ``load_bales``: ``load_trailer``'s per bale ``i`` (names ``pick1``, ``dock1``, ``in1``, ``back1``, ``side1``),
  for ``i`` in 1..``levels["bales"]`` (default 2).

Defaults for missing levels are INFERRED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from alloy.tasks.forklift_control import TruckSpec
from alloy.tasks.forklift_path import Segment
from alloy.tasks.forklift_phases import DriveSegment, SetDown, Stationary, Targets, TipsDown, restraint_not_red

Phase = DriveSegment | Stationary | TipsDown | SetDown


@dataclass(frozen=True)
class Plan:
    kind: str
    segments: Mapping[str, Segment]
    levels: Mapping[str, float] = field(default_factory=dict)

    def level(self, key: str, default: float) -> float:
        return float(self.levels.get(key, default))

    def seg(self, key: str) -> Segment:
        if key not in self.segments:
            raise KeyError(f"plan '{self.kind}' needs a segment named '{key}'")
        return self.segments[key]


def drive(plan: Plan, key: str, targets: Targets, stop: bool = True, next_key: str | None = None) -> DriveSegment:
    """A drive phase for segment ``key``; a non-stopping one slows to the next segment's cap by its end."""
    nxt = plan.seg(next_key).cap_kmh if next_key else 0.0
    return DriveSegment(key, plan.seg(key), targets, stop=stop, next_cap_kmh=nxt)


def start_park_phases(plan: Plan, spec: TruckSpec) -> list[Phase]:
    tips_down = Targets(lift=0.0)                       # the tilt stays where TipsDown left it
    return [Stationary("key_on", key_on=True, engine_on=False, dwell=0.5),
            Stationary("engine_start", dwell=1.0),
            Stationary("release_brake", hold_brake=False, dwell=0.3),
            drive(plan, "to_bay", Targets()),
            TipsDown("forks_down_tilt_forward"),
            Stationary("neutral", tips_down, dwell=0.3),
            Stationary("park", tips_down, park=True, dwell=0.5),
            Stationary("engine_off", tips_down, engine_on=False, park=True, dwell=0.5),
            Stationary("key_off", tips_down, key_on=False, engine_on=False, park=True, dwell=0.3)]


def _turns(plan: Plan, key: str, next_key: str) -> bool:
    """The next segment drives the other way: stop first (direction changes happen at rest)."""
    return plan.seg(key).direction != plan.seg(next_key).direction


def ramp_down_phases(plan: Plan, spec: TruckSpec) -> list[Phase]:
    carry = Targets(lift=plan.level("carry", 0.15), tilt=-spec.tilt_limit)
    down = Targets(lift=plan.level("set", 0.0), tilt=0.0)
    return [Stationary("carry_pose", carry),
            drive(plan, "out", carry, stop=_turns(plan, "out", "ramp"), next_key="ramp"),
            drive(plan, "ramp", carry, stop=_turns(plan, "ramp", "to_drop"), next_key="to_drop"),
            drive(plan, "to_drop", carry),
            SetDown("lower_load", down),
            drive(plan, "back_out", down),
            Stationary("lower_forks", Targets(lift=plan.level("rest", 0.0), tilt=0.0))]


def put_away_phases(plan: Plan, spec: TruckSpec) -> list[Phase]:
    beam = plan.level("beam", 1.60)
    high = beam + plan.level("clearance", 0.10)
    low = beam - plan.level("drop", 0.05)
    carry = Targets(lift=plan.level("carry", 0.2), tilt=-spec.tilt_limit)
    return [Stationary("carry_pose", carry),
            drive(plan, "to_face", carry),
            Stationary("mast_vertical", Targets(lift=plan.level("carry", 0.2), tilt=0.0)),
            Stationary("raise_above_beams", Targets(lift=high, tilt=0.0)),
            drive(plan, "creep", Targets(lift=high, tilt=0.0)),
            SetDown("lower_onto_beams", Targets(lift=low, tilt=0.0)),
            drive(plan, "back_out", Targets(lift=low, tilt=0.0)),
            Stationary("lower_forks", Targets(lift=plan.level("final", 0.2), tilt=0.0))]


def trip_phases(plan: Plan, spec: TruckSpec, n: str = "", wait: Callable = restraint_not_red) -> list[Phase]:
    """Pick a load up from the floor, carry it over the leveler into the trailer, set it down, back out."""
    side = plan.level("side" + n, plan.level("side", 0.0))
    approach = Targets(lift=plan.level("approach", 0.05), tilt=0.0, sideshift=0.0)
    carry = Targets(lift=plan.level("carry", 0.15), tilt=-spec.tilt_limit, sideshift=0.0)
    inside = Targets(lift=plan.level("carry", 0.15), tilt=-spec.tilt_limit, sideshift=side)
    put = Targets(lift=plan.level("set", 0.0), tilt=0.0, sideshift=side)
    picks = "pick" + n in plan.segments                             # no pick segment: the load starts on the forks
    pick = [Stationary("forks_to_pick" + n, approach), drive(plan, "pick" + n, approach)] if picks else []
    return [*pick,
            Stationary("lift_load" + n, carry),
            *([drive(plan, "reverse" + n, carry)] if "reverse" + n in plan.segments else []),   # back out of the pick
            drive(plan, "dock" + n, carry),
            Stationary("restraint_check" + n, carry, park=True, until=wait, dwell=0.5),
            Stationary("release_park" + n, carry, park=False, dwell=0.3),
            drive(plan, "in" + n, inside),
            SetDown("set_down" + n, put),
            drive(plan, "back" + n, put),
            Stationary("recentre_lower" + n, Targets(lift=plan.level("rest", 0.0), tilt=0.0, sideshift=0.0))]


def load_trailer_phases(plan: Plan, spec: TruckSpec) -> list[Phase]:
    return trip_phases(plan, spec)


def load_bales_phases(plan: Plan, spec: TruckSpec) -> list[Phase]:
    phases: list[Phase] = []
    for i in range(1, int(plan.level("bales", 2)) + 1):
        phases += trip_phases(plan, spec, str(i))
    return phases


BUILDERS: dict[str, Callable[[Plan, TruckSpec], list[Phase]]] = {
    "start_park": start_park_phases, "ramp_down": ramp_down_phases, "put_away": put_away_phases,
    "load_trailer": load_trailer_phases, "load_bales": load_bales_phases}


def phases_for(plan: Plan, spec: TruckSpec) -> list[Phase]:
    if plan.kind not in BUILDERS:
        raise ValueError(f"unknown plan kind '{plan.kind}'; one of {sorted(BUILDERS)}")
    return BUILDERS[plan.kind](plan, spec)
