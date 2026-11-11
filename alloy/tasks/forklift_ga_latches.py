# SPDX-License-Identifier: Apache-2.0
"""GA's per-tick event latches (F/events.py:128-203), recomputed from an Episode's sampled arrays.

GA latches at the 2 ms physics tick; here each control-period sample stands for the interval since the previous one
(the first sample takes the second's period), so a time is a count of samples times their period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.tasks.forklift_ga_geom import yaw_of
from alloy.tasks.forklift_ga_types import ContactEpisode, Episode, LoadFloorTrack
from alloy.tasks.forklift_site import ForkliftSite, Rect, quat_xyzw_to_matrix


def periods(t) -> np.ndarray:
    """Per-sample period (s)."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return np.zeros(len(t))
    d = np.diff(t)
    return np.concatenate([[d[0]], d])


def longest_run_s(flags, t) -> float:
    """Longest continuous stretch (s) of samples with flags set."""
    dt, cur, best = periods(t), 0.0, 0.0
    for f, d in zip(flags, dt):
        cur = cur + d if f else 0.0
        best = max(best, cur)
    return best


def rest_s(speed, t, limit: float) -> float:
    """Time since the chassis last moved faster than limit (F/events.py:144)."""
    dt, r = periods(t), 0.0
    for v, d in zip(speed, dt):
        r = r + d if abs(v) < limit else 0.0
    return r


def over_limit_s(speed, t, limit: float) -> float:
    return float(periods(t)[np.abs(np.asarray(speed)) > limit].sum())


def wheel_lift_max_s(ep: Episode, site: ForkliftSite) -> float:
    """Longest interval any wheel carried no more than 1 N (F/events.py:160-164)."""
    fz = np.asarray(ep.wheel_fz, float).reshape(ep.n, -1)
    return max([longest_run_s(fz[:, k] <= site.limits.wheel_lift_fz, ep.t) for k in range(fz.shape[1])] or [0.0])


@dataclass(frozen=True)
class Attitude:
    max_roll: float
    max_pitch: float
    overturn: bool


def attitude(ep: Episode, site: ForkliftSite) -> Attitude:
    """Largest chassis roll and pitch (deg) and whether either passed 30 deg (F/events.py:152-158)."""
    roll = pitch = 0.0
    for q in ep.truck_quat_xyzw:
        R = quat_xyzw_to_matrix(q)
        roll = max(roll, abs(math.degrees(math.atan2(R[2, 1], R[2, 2]))))
        pitch = max(pitch, abs(math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))))
    lim = site.limits.overturn_deg
    return Attitude(roll, pitch, roll > lim or pitch > lim)


def seat_open_max_s(ep: Episode, site: ForkliftSite) -> float:
    """Longest run with the seat switch open while moving faster than 0.05 m/s (F/events.py:170-174)."""
    closed = ep.signals.get("seat_switch_closed")
    if closed is None:
        return 0.0
    moving_open = (~np.asarray(closed, bool)) & (np.abs(ep.speed) > site.limits.seat_moving_speed)
    return longest_run_s(moving_open, ep.t)


def ramp_max_speed(ep: Episode, zone: Rect) -> float:
    """Largest speed while the chassis is in the ramp's speed zone (F/events.py:185-189)."""
    on = [zone.x[0] < p[0] < zone.x[1] and zone.y[0] < p[1] < zone.y[1] for p in ep.truck_pos]
    v = np.abs(ep.speed)[np.array(on, bool)]
    return float(v.max()) if len(v) else 0.0


def trailer_max_move(ep: Episode, rear_xy) -> float:
    """Largest travel of the target trailer from its spotted position along its own axis (F/dock.py:300 x_from_spot)."""
    if ep.trailer_pos is None:
        return 0.0
    out = 0.0
    for p, q in zip(ep.trailer_pos, ep.trailer_quat_xyzw):
        fx = quat_xyzw_to_matrix(q)[:, 0]
        out = max(out, abs(float((np.asarray(p[:2]) - np.asarray(rear_xy)) @ fx[:2])))
    return out


def traj(ep: Episode) -> np.ndarray:
    """GA's trajectory rows (t, x, y, yaw, v_forward) per sample (F/events.py:145-148)."""
    yaw = np.array([yaw_of(quat_xyzw_to_matrix(q)) for q in ep.truck_quat_xyzw])
    return np.column_stack([ep.t, ep.truck_pos[:, 0], ep.truck_pos[:, 1], yaw, ep.speed])


@dataclass(frozen=True)
class FloorRecord:
    """A lifted load's floor contact: sample time and contact point (F/events.py: load_floor)."""

    t: float
    xy: tuple[float, float]


def floor_records(track: LoadFloorTrack, t, limits_lifted_s: float, gap_s: float) -> tuple[list[FloorRecord], list]:
    """GA's load_floor and load_floor_onset for one load (F/events.py:268-284): a load counts as lifted once it has been
    off every floor for 0.25 s; from then on each floor contact is recorded (at most one per 0.25 s) with its position,
    and each new contact's approach speed is kept. Returns (records, onsets as (t, speed))."""
    dt = periods(t)
    off, lifted = 0.0, False
    recs: list[FloorRecord] = []
    onsets: list[tuple[float, float]] = []
    for i, d in enumerate(dt):
        if track.touching[i]:
            new = off > 0.0
            off = 0.0
            if lifted:
                if not recs or t[i] - recs[-1].t > gap_s:
                    recs.append(FloorRecord(float(t[i]), (float(track.xy[i][0]), float(track.xy[i][1]))))
                if new:
                    onsets.append((float(t[i]), float(track.onset_speed[i])))
        else:
            off += d
            lifted = lifted or off >= limits_lifted_s
    return recs, onsets


def all_floor_records(ep: Episode, site: ForkliftSite) -> tuple[dict[str, list[FloorRecord]], dict[str, list]]:
    lim = site.limits
    recs, ons = {}, {}
    for name, tr in ep.floor.items():
        r, o = floor_records(tr, ep.t, lim.lifted_s, lim.floor_record_gap_s)
        if r:
            recs[name] = r
        if o:
            ons[name] = o
    return recs, ons


def onset_normal(e: ContactEpisode) -> float:
    """The approach speed along the contact normal; an episode latched without it reads its onset speed."""
    return float(e.onset_speed if e.onset_normal is None else e.onset_normal)


def tilt_between(Ra, Rb) -> float:
    """Angle (deg) between two frames' z axes."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(Ra[:, 2] @ Rb[:, 2])))))

