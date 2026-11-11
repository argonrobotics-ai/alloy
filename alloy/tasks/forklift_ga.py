# SPDX-License-Identifier: Apache-2.0
"""Success criteria of Gym-Anything's forklift tasks, ported to Alloy episodes: start-and-park, ramp-down-loaded,
put-away-rack, load-trailer (with the unsecured/red-restraint variant) and load-bales.

Each public function takes one world's ``Episode`` and the ``ForkliftSite`` and returns a ``Verdict`` with GA's
criterion
names, thresholds and success rule (every judged criterion met, no fault; some are not judged, see
Verdict.summary()). The thresholds are authored by GA unless a cite says
otherwise; each criterion's ``cite`` is its definition in ``F/evaluator.py`` (F = GA's forklift environment folder, see
``forklift_site``). load-bales is in ``forklift_ga_bales``.
"""

from __future__ import annotations

import math

import numpy as np

from alloy.tasks.forklift_ga_bales import load_bales  # noqa: F401  (public, defined with its reward)
from alloy.tasks.forklift_ga_geom import (BoxPose, forks_height, load_pose, tilt_deg, tip_heights,
                                          tips_to_origin, truck_frame, yaw_of)
from alloy.tasks.forklift_ga_latches import ramp_max_speed, tilt_between, traj, trailer_max_move
from alloy.tasks.forklift_ga_shared import crit, last_signal, rest_time, shared_criteria
from alloy.tasks.forklift_ga_types import Criterion, Episode, Verdict, make_verdict, not_judged, partly
from alloy.tasks.forklift_site import (SITE, SLOT_NAMES, ForkliftSite, front_beam_xs, rack_location, ramp_speed_zone,
                                       trailer_frame_from_pose, trailer_rear)

__all__ = ["start_and_park", "ramp_down_loaded", "put_away_rack", "load_trailer", "load_bales"]


_last, _rest, _crit = last_signal, rest_time, crit


def _stationary(ep: Episode, site: ForkliftSite, need: float, name: str = "stationary") -> Criterion:
    r = _rest(ep, site)
    return _crit(name, r, r >= need, "s", f">= {need:g} s at rest", "F/evaluator.py:301,441,1144")


def _forks_lowered(ep: Episode, site: ForkliftSite, limit: float, text: str, cite: str,
                   name="forks_lowered") -> Criterion:
    h = forks_height(ep, site)
    return _crit(name, h, h <= limit, "m", text, cite)


def _backed_out(ep: Episode, site: ForkliftSite, pallet: BoxPose) -> Criterion:
    gap = tips_to_origin(ep, pallet.bottom_origin) - site.pallet.L / 2
    return _crit("backed_out", gap, gap >= 0.5, "m", "fork tips >= 0.5 m from the pallet", "F/evaluator.py:1139-1141")


_ECHO = ("not judged: read back from the operator's last command (forklift_ga_truck.py TruckIO.apply), not from a "
         "quantity the truck model computes; it can fail, never pass")


def _echo(name: str, value, wrong: bool, threshold: str, cite: str) -> Criterion:
    """A gate read from the script's own command: False when the command is wrong, else not judged."""
    return Criterion(name, value, False if wrong else None, "-", threshold, cite, False, _ECHO)


def _command_criteria(ep: Episode) -> list[Criterion]:
    """GA's parking_brake_set, direction_neutral, engine_off, key_off (F/evaluator.py:302-308) from the operator's
    last command only (see ``_ECHO``)."""
    park, d = float(_last(ep, "park")), int(_last(ep, "direction"))
    run, key_on = bool(_last(ep, "engine_running", False)), bool(_last(ep, "key_on", False))
    return [_echo("parking_brake_set", park, not park > 0.5, "applied", "F/evaluator.py:302"),
            _echo("direction_neutral", d, d != 0, "neutral", "F/evaluator.py:304"),
            _echo("engine_off", run, run, "off", "F/evaluator.py:306"),
            _echo("key_off", "on" if key_on else "off", key_on, "key in the OFF position", "F/evaluator.py:308")]


# ------------------------------------------------------------------------------------------------ start-and-park
def start_and_park(ep: Episode, site: ForkliftSite = SITE) -> Verdict:
    """start-and-park (F/evaluator.py:290-323): the truck parked in the painted bay, stopped, brake set, neutral, engine
    and key off, forks lowered and tilted to the floor. Preset: warehouse."""
    sr = shared_criteria(ep, site, "start-and-park", "warehouse")
    crit = list(sr.criteria)
    p, R = truck_frame(ep)
    fp = site.truck.corners(float(p[0]), float(p[1]), yaw_of(R))
    worst = max(site.parking.outside_by(c) for c in fp)
    crit.append(_crit("in_parking_bay", dict(outside_by_m=worst, footprint=np.round(fp, 3).tolist()), worst <= 0.0, "m",
                      "the truck's footprint (counterweight to fork tips) inside the parking bay's lines",
                      "F/evaluator.py:294-298"))
    crit.append(_stationary(ep, site, 2.0, "stationary_2s"))
    crit += _command_criteria(ep)
    crit.append(_forks_lowered(ep, site, 0.020, "fork tips <= 20 mm above the floor", "F/evaluator.py:310"))
    pitch = math.degrees(float(ep.blade_pitch[-1]))
    tips = tip_heights(ep, site)
    crit.append(_crit("forks_tilted_to_floor", dict(blade_pitch_deg=pitch, tip_heights_m=tips.tolist()),
                      pitch <= 0.0 and float(tips.max()) <= 0.005, "-",
                      "both fork tips <= 5 mm above the floor, the blades sloping down to them (pitch <= 0)",
                      "F/evaluator.py:313-319"))
    return make_verdict("start-and-park", crit, ep.fault)


# ------------------------------------------------------------------------------------------------ ramp-down-loaded
def ramp_down_loaded(ep: Episode, site: ForkliftSite = SITE, drop_zone: str = "east") -> Verdict:
    """ramp-down-loaded (F/evaluator.py:1125-1147): the pallet set upright in the painted yard drop zone ('east' or
    'west'), backed out, forks lowered, stopped, and never faster than 6 km/h on the ramp. Preset: yard."""
    dz = site.drop_zone(drop_zone)

    def dest(name: str, xy) -> bool:
        return name == "carried" and dz.contains(xy)
    sr = shared_criteria(ep, site, "ramp-down-loaded", "yard", dest_floor=dest)
    crit = list(sr.criteria)
    pal = load_pose(ep, "carried")
    inside = all(dz.contains(c) for c in pal.bottom)
    tilt = tilt_deg(pal.R)
    crit.append(_crit("in_drop_zone", inside, inside and tilt < 5.0, "-", "inside the drop zone, upright",
                      "F/evaluator.py:1131-1134"))
    fz = [float(c[2] - site.ground_height(c[0], c[1])) for c in pal.bottom]
    crit.append(_crit("resting_on_floor", fz, all(abs(z) <= 0.010 for z in fz), "m",
                      "corners within 10 mm of the ground",
                      "F/evaluator.py:1136-1137"))
    crit.append(_backed_out(ep, site, pal))
    crit.append(_forks_lowered(ep, site, 0.15, "fork tips at or below travel height (0.15 m)", "F/evaluator.py:1142"))
    crit.append(_stationary(ep, site, 1.0))
    v = ramp_max_speed(ep, ramp_speed_zone(site)) * 3.6
    lim = site.ramp.speed_limit_kmh
    crit.append(_crit("slow_on_ramp", v, v <= lim, "km/h", f"<= {lim:g} km/h on the ramp", "F/evaluator.py:1145-1146"))
    return make_verdict("ramp-down-loaded", crit, ep.fault)


# ------------------------------------------------------------------------------------------------ put-away-rack
def _beam_contacts(ep: Episode, site: ForkliftSite, location: str) -> tuple[set, set]:
    """(front/rear beam x of the target location touched, any other beam's x touched), F/evaluator.py:476-489."""
    loc = rack_location(site, location)
    beam_x = front_beam_xs(site, loc)
    mine, other = set(), set()
    for label in ep.beam_contacts:
        parts = label.split(":")[0].split("_")  # <row><frame>_beam<bay-1>_<level>_<x>
        try:
            bay, lv, bx = int(parts[1][4:]) + 1, int(parts[2]), float(parts[-1])
        except (IndexError, ValueError):
            continue
        ok = bay == loc.bay and lv == loc.level and min(abs(bx - x) for x in beam_x) < site.rack.beam_x_tol
        (mine if ok else other).add(parts[-1])
    return mine, other


def _side_clearance(site: ForkliftSite, name: str, ys: np.ndarray) -> float:
    """Least gap (m) from the pallet's sides to the upright and to the neighbouring pallet position
    (F/evaluator.py:502-513)."""
    k, loc = site.rack, rack_location(site, name)
    yc = loc.y + (k.beam / 4 if loc.slot == 0 else -k.beam / 4)
    lo_face, hi_face = yc - k.beam / 2, yc + k.beam / 2
    other = rack_location(site, name[:-1] + str(1 - loc.slot))  # GA: never reserved, so the neighbour always counts
    gaps = [float(ys.min() - lo_face) if loc.slot == 0 else float(hi_face - ys.max())]
    w = site.pallet.W
    gaps.append(float((other.y - w / 2) - ys.max()) if loc.slot == 0 else float(ys.min() - (other.y + w / 2)))
    return min(gaps)


def put_away_rack(ep: Episode, site: ForkliftSite = SITE, location: str = "A-e-01-1-0") -> Verdict:
    """put-away-rack (F/evaluator.py:468-529): the carried pallet on both beams of the named location
    ('row-face-bay-level-slot'), at its beam level, 51-101 mm past the front beam, clear of the neighbours, level, the
    stored loads undisturbed and the forks withdrawn and lowered. Preset: warehouse."""
    sr = shared_criteria(ep, site, "put-away-rack", "warehouse")
    crit = list(sr.criteria)
    loc = rack_location(site, location)
    pal = load_pose(ep, "carried")
    mine, other = _beam_contacts(ep, site, location)
    crit.append(_crit("on_both_beams", dict(target_location=sorted(mine), elsewhere=sorted(other)),
                      len(mine) >= 2 and not other, "-",
                      f"contact on the front and the rear beam of bay {loc.bay} at beam level {loc.level} "
                      f"({loc.z:.2f} m), on no other beam", "F/evaluator.py:490-493"))
    fz = [float(z) for z in pal.bottom[:, 2]]
    tol = site.rack.beam_level_tol
    crit.append(_crit("at_beam_level", fz, all(abs(z - loc.z) <= tol for z in fz), "m",
                      f"pallet bottom corners within 30 mm of the beam tops at {loc.z:.2f} m",
                      "F/evaluator.py:494-496"))
    xs = pal.bottom[:, 0]
    over = float(loc.aisle_x - xs.min()) if loc.face == "w" else float(xs.max() - loc.aisle_x)
    lo, hi = site.rack.overhang_band
    crit.append(_crit("front_overhang", over, lo <= over <= hi, "m", "51-101 mm past the front beam",
                      "F/evaluator.py:499-501"))
    clear = _side_clearance(site, location, pal.bottom[:, 1])
    crit.append(_crit("side_clearance", clear, clear >= site.rack.side_clearance, "m",
                      ">= 50 mm to the upright and the neighbouring pallet", "F/evaluator.py:502-515"))
    tilt = tilt_deg(pal.R)
    crit.append(_crit("level", tilt, tilt <= site.rack.level_tol_deg, "deg", "within 2 deg", "F/evaluator.py:516-517"))
    moved = {n: float(np.linalg.norm(load_pose(ep, n).bottom_origin - load_pose(ep, n, 0).bottom_origin))
             for n in ep.loads if n != "carried" and not n.startswith("carried_")}
    worst = max(moved.values(), default=0.0)
    crit.append(_crit("stored_loads_undisturbed", dict(max_m=worst, moved={k: v for k,
                                                                           v in moved.items() if v > 0.005}),
                      worst <= site.rack.stored_moved_max, "m", "every stored load within 30 mm of where it stood",
                      "F/evaluator.py:519-523"))
    out_ = tips_to_origin(ep, pal.bottom_origin)
    h = forks_height(ep, site)
    crit.append(_crit("forks_withdrawn_lowered", dict(tip_to_pallet_m=out_, tip_height_m=h), out_ > 1.0 and h <= 0.30,
                      "-",
                      "tips clear of the location and at travel height", "F/evaluator.py:524-528"))
    return make_verdict("put-away-rack", crit, ep.fault)


# ------------------------------------------------------------------------------------------------ load-trailer
def load_trailer(ep: Episode, site: ForkliftSite = SITE, door: str = "door1", side: str = "L",
                 secured: bool = True, behind: str | None = None) -> Verdict:
    """load-trailer and load-trailer-chocked (F/evaluator.py:573-620), or with ``secured=False`` the red-restraint
    variant
    (unsecured trailer: stay off the leveler, wait at the door; F/evaluator.py:626-658). door: 'door1'..'door4';
    side: 'L'
    or 'R' as seen facing the nose; behind: the nose pallet the load goes behind (default ``trailer_1<side>``).
    Preset: dock."""
    if not secured:
        return _unsecured_trailer(ep, site, door)
    behind = behind or f"trailer_1{side}"
    frame = trailer_frame_from_pose(site, ep.trailer_pos[-1], ep.trailer_quat_xyzw[-1])
    dest = _slot_dest(ep, site, frame, side, behind)
    sr = shared_criteria(ep, site, "load-trailer", "dock", dest_floor=dest)
    crit = list(sr.criteria)
    pal, ahead = load_pose(ep, "carried"), load_pose(ep, behind)
    loc, a_loc = frame.to_local(pal.bottom), frame.to_local(ahead.bottom)
    tr, dock = site.trailer, site.dock
    inside = bool(np.all((loc[:, 0] > 0) & (loc[:, 0] < tr.L) & (np.abs(loc[:, 1]) < tr.W / 2)))
    on_floor = bool(np.all(np.abs(loc[:, 2]) < dock.on_floor_tol))
    crit.append(_crit("in_trailer_on_floor", dict(inside=inside, on_floor=on_floor), inside and on_floor, "-",
                      "inside the trailer, resting on its floor", "F/evaluator.py:594-600"))
    gap = float(a_loc[:, 0].min() - loc[:, 0].max())
    lo, hi = dock.gap_ahead
    crit.append(_crit("gap_to_pallets_ahead", gap, lo <= gap <= hi, "m", "<= 50 mm behind the pallets at the nose",
                      "F/evaluator.py:601-606"))
    wall = float(tr.W / 2 - loc[:, 1].max()) if side == "L" else float(tr.W / 2 + loc[:, 1].min())
    crit.append(_crit("gap_to_side_wall", wall, 0.0 <= wall <= dock.wall_gap_max, "m",
                      f"<= {dock.wall_gap_max * 1000:.0f} mm to the {'left' if side == 'L' else 'right'} wall",
                      "F/evaluator.py:607-612"))
    tilt = tilt_between(frame.R, pal.R)
    crit.append(_crit("level", tilt, tilt <= dock.level_tol_deg, "deg", "within 2 deg of the trailer floor",
                      "F/evaluator.py:613-614"))
    back = float(ep.truck_pos[-1][1])
    face = dock.door(door)[1]
    crit.append(_crit("truck_back_on_dock", back, back < face - dock.stop_short_of_door, "m",
                      "chassis back inside the building", "F/evaluator.py:615-617"))
    mv = trailer_max_move(ep, trailer_rear(site, door))
    crit.append(_crit("trailer_movement", mv, mv <= tr.max_move, "m", "<= 25 mm from its spotted position",
                      "F/evaluator.py:618-619"))
    return make_verdict("load-trailer", crit, ep.fault)


def _slot_dest(ep: Episode, site: ForkliftSite, frame, side: str, behind: str):
    """dest_floor of load-trailer: the carried pallet may touch the floor only in its slot (F/evaluator.py:580-591)."""
    g, tr, dock = site.pallet, site.trailer, site.dock

    def dest(name: str, xy) -> bool:
        if name != "carried":
            return False
        x_ahead = float(frame.to_local(load_pose(ep, behind).bottom)[:, 0].min())
        sg = SLOT_NAMES[side]
        ys = sorted((sg * tr.W / 2, sg * (tr.W / 2 - dock.wall_gap_max - g.W)))
        rel = frame.to_local(np.array([xy[0], xy[1], frame.origin[2]]))
        m = dock.slot_margin
        return (x_ahead - g.L - m <= rel[0] <= x_ahead + 0.05) and (ys[0] - m <= rel[1] <= ys[1] + m)
    return dest


def _unsecured_trailer(ep: Episode, site: ForkliftSite, door: str) -> Verdict:
    sr = shared_criteria(ep, site, "load-trailer", "dock")
    crit = list(sr.criteria)
    dock, reach_l = site.dock, site.truck.reach_loaded
    edge = dock.door(door)[1] - site.leveler.length
    tj = traj(ep)
    reach = float(np.max(tj[:, 2] + reach_l * np.sin(tj[:, 3]))) if len(tj) else -99.0
    crit.append(_crit("stayed_off_the_leveler", reach - edge, reach < edge, "m",
                      "the truck's loaded front never past the leveler's inner edge (OSHA 1910.178(m)(7))",
                      "F/evaluator.py:626-637"))
    lane = dock.staging_lane(door)
    front = float(tj[-1, 2] + reach_l * math.sin(tj[-1, 3])) if len(tj) else -99.0
    crit.append(_crit("waiting_at_the_door", front - lane.y[1], front >= lane.y[1], "m",
                      "its loaded front past the end of the staging lane's lines", "F/evaluator.py:638-648"))
    r, park, d = _rest(ep, site), float(_last(ep, "park")), int(_last(ep, "direction"))
    ok = partly(bool(r >= 2.0), None if park > 0.5 else False, None if d == 0 else False)
    crit.append(Criterion("stopped_short", dict(rest_s=r, park=park, direction=d), ok, "-",
                          ">= 2 s at rest, parking brake set, direction neutral", "F/evaluator.py:649-653",
                          note="rest time is judged from the chassis speed; the brake and direction are the "
                               "operator's last command, not a truck-model quantity: not judged (can only fail)"))
    pal = load_pose(ep, "carried")
    on_forks = tips_to_origin(ep, pal.bottom_origin) < site.pallet.L
    crit.append(_crit("load_on_forks", on_forks, on_forks, "-", "the pallet still on the forks",
                      "F/evaluator.py:654-657"))
    return make_verdict("load-trailer", crit, ep.fault)

