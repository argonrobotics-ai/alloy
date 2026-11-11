# SPDX-License-Identifier: Apache-2.0
"""The site rules every forklift task shares (F/evaluator.py:201-258): no_strike, no_wheel_lift, roll_under_6deg,
no_dropped_load, speed_limit, operator_seated, head_clear_of_guard, time_limit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from alloy.tasks.forklift_ga_latches import (all_floor_records, attitude, onset_normal, over_limit_s,
                                             seat_open_max_s, wheel_lift_max_s, FloorRecord, rest_s)
from alloy.tasks.forklift_ga_types import ContactEpisode, Criterion, Episode, not_judged
from alloy.tasks.forklift_site import G, ForkliftSite

STRIKE_CLASSES = {"rack", "column", "wall", "conveyor", "conveyor_guide", "conveyor_rollers", "machine", "structure",
                  "trailer_wall", "trailer_floor", "bumper", "curb", "parked", "pit", "dressing"}  # F/evaluator.py:16
DRIVEN_ON = {"trailer_floor"}  # F/evaluator.py:20
TRAILER_FLOOR_LOAD_KG = 1320.0  # F/evaluator.py:62
ROLLER_LOAD_KG = 2020.0  # F/evaluator.py:63

# strike_rule(mover, cls, episode) -> True (a strike), False (not one) or None (the shared rule decides)
StrikeRule = Callable[[str, str, ContactEpisode], "bool | None"]
DestFloor = Callable[[str, "tuple[float, float]"], bool]


def episode_load(mover: str, e: ContactEpisode) -> str | None:
    """The load a load episode belongs to (F/evaluator.py:145-154)."""
    if mover.startswith("load:"):
        return mover[5:]
    if e.parts and ":" in str(e.parts[0]):
        return str(e.parts[0]).split(":")[0]
    return e.mover[5:] if e.mover.startswith("load:") else None


def on_beam(e: ContactEpisode) -> bool:
    """The structure part at the episode's peak is a rack beam (F/evaluator.py:157-160)."""
    return len(e.parts) > 1 and "_beam" in str(e.parts[1])


def surface_bound(ep: Episode, site: ForkliftSite) -> float | None:
    """F_max (N) of a set-down on a trailer floor or rollers: the heaviest load's mass x (g + B v)
    (F/evaluator.py:55-59)."""
    m = max(ep.load_masses.values(), default=0.0)
    return m * (G + site.limits.set_down_decel) if m > 0 else None


def lean_limit(ep: Episode, site: ForkliftSite, name: str | None) -> float:
    """The most force a load at rest keeps on a structure: its weight, never below the strike force
    (F/evaluator.py:182)."""
    return max(site.limits.strike_n, float(ep.load_masses.get(name, 0.0)) * G)


def gentle_on_surface(cls: str, e: ContactEpisode, ep: Episode, site: ForkliftSite) -> bool:
    """A load set on (resting on, riding) a trailer floor or the rollers (F/evaluator.py:175-179)."""
    fallback = {"trailer_floor": TRAILER_FLOOR_LOAD_KG, "conveyor_rollers": ROLLER_LOAD_KG}[cls] * (
        G + site.limits.set_down_decel)
    bound = surface_bound(ep, site) or fallback
    return onset_normal(e) <= site.limits.set_down_speed and e.peak_n <= bound


def gentle_set_down(e: ContactEpisode, site: ForkliftSite) -> bool:
    """A load set down on rack beams, or resting on them since the start (F/evaluator.py:163-165)."""
    lim = site.limits
    return on_beam(e) and e.onset_speed <= lim.set_down_speed and e.peak_n <= lim.set_down_peak_n


def load_exempt(ep: Episode, site: ForkliftSite, mover: str, cls: str, e: ContactEpisode) -> bool:
    """A load's contact that is not a strike (F/evaluator.py:189-198)."""
    if cls in ("trailer_floor", "conveyor_rollers"):
        return gentle_on_surface(cls, e, ep, site)
    if gentle_set_down(e, site):
        return True
    return (not on_beam(e) and e.onset_speed < site.limits.impact_speed
            and e.peak_n <= lean_limit(ep, site, episode_load(mover, e)))


def find_strikes(ep: Episode, site: ForkliftSite, strike_rule: StrikeRule | None = None) -> list[ContactEpisode]:
    """The contact episodes that count as strikes (F/evaluator.py:205-217)."""
    lim, out = site.limits, []
    for e in ep.contacts:
        mover, cls = e.mover, e.structure
        if cls not in STRIKE_CLASSES or (cls in DRIVEN_ON and not mover.startswith("load")):
            continue
        own = strike_rule(mover, cls, e) if strike_rule is not None else None
        if own is False or (own is None and mover.startswith("load") and load_exempt(ep, site, mover, cls, e)):
            continue
        if own or e.peak_n > lim.strike_n or e.impulse_ns > lim.strike_ns:
            out.append(e)
    return out


def dropped_loads(records: dict[str, list[FloorRecord]], dest_floor: DestFloor | None) -> list[tuple[str, float,
                                                                                                     tuple]]:
    """Floor contacts of lifted loads away from their destination (F/evaluator.py:229-234)."""
    return [(n, r.t, r.xy) for n, lst in records.items() for r in lst if not (dest_floor and dest_floor(n, r.xy))]


@dataclass(frozen=True)
class SharedResult:
    criteria: list[Criterion]
    strikes: list[ContactEpisode]
    dropped: list[tuple[str, float, tuple]]
    floor_onsets: dict[str, list]
    over_limit_s: float
    max_roll: float
    overturn: bool


def _no_strike(strikes: list[ContactEpisode], site: ForkliftSite, ep: Episode) -> Criterion:
    lim = site.limits
    bound = (surface_bound(ep, site) or 0.0) / 1000
    return Criterion(
        "no_strike", len(strikes), not strikes, "strikes",
        f"none with > {lim.strike_n:.0f} N or > {lim.strike_ns:.0f} N s (a load's contact slower than "
        f"{lim.impact_speed:.2f} m/s at onset counts only above its weight; a load set on rack beams only above "
        f"{lim.set_down_speed:.2f} m/s or {lim.set_down_peak_n / 1000:.0f} kN, on a trailer floor or the rollers only "
        f"above "
        f"{lim.set_down_speed:.2f} m/s or the heaviest load's weight plus its arrest ({bound:.0f} kN here))",
        "F/evaluator.py:218-224",
        note="contacts read every 20 ms (Gym-Anything sweeps every 2 ms): a short contact can be missed or its peak "
             "undercounted; sees only the structure labels this task builds, not every facility object")


def _speed(ep: Episode, site: ForkliftSite, preset: str) -> tuple[Criterion, float]:
    lim = site.limits
    over = over_limit_s(ep.speed, ep.t, lim.building_limit)
    note = "speed read every 20 ms (Gym-Anything every 0.5 ms): a short peak can be missed"
    vmax = float(np.abs(ep.speed).max()) * 3.6
    if preset in ("warehouse", "dock"):
        return Criterion("speed_limit", vmax, over == 0.0, "km/h", f"<= {lim.limit_kmh:.1f} km/h inside",
                         "F/evaluator.py:237-239", note=note), over
    return Criterion("speed_limit", vmax, vmax <= lim.limit_kmh, "km/h", f"<= {lim.limit_kmh:.1f} km/h in the yard",
                      "F/evaluator.py:240-246", note=note), over


def _operator(ep: Episode, site: ForkliftSite) -> list[Criterion]:
    lim = site.limits
    if not ep.humanoid:
        note = "not judged: no seated operator is simulated (the operator is a script), so there is nothing to measure"
        return [not_judged("operator_seated", None, "s", f"seat switch open <= {lim.seat_open_s:.0f} s while moving",
                           "F/evaluator.py:247", note),
                not_judged("head_clear_of_guard", None, "N",
                           f"no robot contact with the overhead guard above {lim.head_contact_n:.0f} N",
                           "F/evaluator.py:252", note)]
    seat = seat_open_max_s(ep, site)
    head = float(np.max(ep.head_contact_n)) if ep.head_contact_n is not None and len(ep.head_contact_n) else 0.0
    return [Criterion("operator_seated", seat, seat <= lim.seat_open_s, "s",
                      f"seat switch open <= {lim.seat_open_s:.0f} s while moving", "F/evaluator.py:247"),
            Criterion("head_clear_of_guard", head, head <= lim.head_contact_n, "N",
                      f"no robot contact with the overhead guard above {lim.head_contact_n:.0f} N",
                      "F/evaluator.py:252")]


def _time_limit(ep: Episode, limit: float) -> Criterion:
    """GA fails time_limit when the sim time passes the limit. The port's loop stops at the limit, so it judges from
    whether the scripted plan finished first (cut by the limit = False); a synthetic episode cannot tell."""
    thr = f"<= {limit:.0f} s"
    if not ep.plan_known:
        return not_judged("time_limit", float(ep.t[-1]), "s", thr, "F/evaluator.py:255-256",
                          "not judged: this episode does not say whether the scripted plan finished before the limit")
    done = ep.plan_done_t
    ok = done is not None and done <= limit + 1e-9
    note = f"the scripted plan finished at {done:.2f} s" if ok else "the scripted plan was cut by the limit"
    return Criterion("time_limit", float(ep.t[-1]), ok, "s", thr, "F/evaluator.py:255-256", note=note)


def shared_criteria(ep: Episode, site: ForkliftSite, task: str, preset: str, dest_floor: DestFloor | None = None,
                    strike_rule: StrikeRule | None = None) -> SharedResult:
    """GA's shared(): the eight site rules in its order. preset: ``warehouse``, ``dock`` or ``yard``."""
    lim = site.limits
    strikes = find_strikes(ep, site, strike_rule)
    out = [_no_strike(strikes, site, ep)]
    lift = wheel_lift_max_s(ep, site)
    out.append(Criterion("no_wheel_lift", lift, lift <= lim.wheel_lift_s, "s",
                         f"<= {lim.wheel_lift_s} s off the ground",
                         "F/evaluator.py:225-226",
                         note="wheel force read every 20 ms (Gym-Anything latches every 0.5 ms): a short lift can be "
                              "missed"))
    att = attitude(ep, site)
    out.append(Criterion("roll_under_6deg", att.max_roll, att.max_roll < lim.roll_deg and not att.overturn, "deg",
                         "< 6 deg, no overturn", "F/evaluator.py:227-228",
                         note="attitude read every 20 ms (Gym-Anything latches every 0.5 ms): a short roll peak can "
                              "be missed"))
    records, onsets = all_floor_records(ep, site)
    dropped = dropped_loads(records, dest_floor)
    out.append(Criterion("no_dropped_load", len(dropped), not dropped, "floor contacts",
                         "load touches the floor only at its destination", "F/evaluator.py:235-236"))
    speed, over = _speed(ep, site, preset)
    out += [speed, *_operator(ep, site)]
    limit = lim.time_limit_s[task]
    out.append(_time_limit(ep, limit))
    return SharedResult(out, strikes, dropped, onsets, over, att.max_roll, att.overturn)


def crit(name, value, ok, unit, threshold, cite, **kw) -> Criterion:
    """GA's C(): a criterion with a plain-bool verdict."""
    return Criterion(name, value, bool(ok), unit, threshold, cite, **kw)


def last_signal(ep: Episode, name: str, default=0.0):
    """A signal's value at the end of the run (GA reads the latched state at evaluation)."""
    sig = ep.signals.get(name)
    return default if sig is None or len(sig) == 0 else sig[-1]


def rest_time(ep: Episode, site: ForkliftSite) -> float:
    """Time at rest at the end of the run (F/events.py:144)."""
    return rest_s(ep.speed, ep.t, site.limits.rest_speed)
