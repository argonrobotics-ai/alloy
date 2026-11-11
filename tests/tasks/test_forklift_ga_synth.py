# SPDX-License-Identifier: Apache-2.0
"""Synthetic forklift episodes for the GA criteria tests: a passing episode per task and small mutators."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from alloy.tasks.forklift_ga_types import BOX_SIGNS, ContactEpisode, Episode, LoadFloorTrack, TrailerTrack
from alloy.tasks.forklift_site import (SITE, beam_label, docked_trailer_frame, front_beam_xs, rack_location,
                                       yaw_quat_xyzw)

DT = 0.125  # exact in binary: sample counts times DT are exact times
N = 48
IN = 0.0254


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def quat_of(R) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, float)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2], np.trace(R)]))
    if i == 3:
        w = math.sqrt(1 + np.trace(R)) / 2
        return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w),
                         w])
    j, k = (i + 1) % 3, (i + 2) % 3
    q = np.zeros(4)
    q[i] = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) / 2
    q[j] = (R[j, i] + R[i, j]) / (4 * q[i])
    q[k] = (R[k, i] + R[i, k]) / (4 * q[i])
    q[3] = (R[k, j] - R[j, k]) / (4 * q[i])
    return q


def box_corners(centre, R, size) -> np.ndarray:
    """Eight corners (BOX_SIGNS order) of a box of full edges size at centre with rotation R."""
    return np.asarray(centre, float) + (BOX_SIGNS * np.asarray(size, float) / 2) @ np.asarray(R).T


def pallet_corners(bottom_centre, R, h=SITE.pallet.H) -> np.ndarray:
    """A GMA pallet's box with its bottom face centre at bottom_centre."""
    g = SITE.pallet
    return box_corners(np.asarray(bottom_centre, float) + np.asarray(R)[:, 2] * h / 2, R, (g.L, g.W, h))


def const(x, n=N) -> np.ndarray:
    return np.repeat(np.asarray(x, float)[None], n, axis=0)


def base_episode(pos, yaw, tip_z=0.0, n=N) -> Episode:
    """A truck at rest at pos with yaw, brake set, neutral, engine and key off, forks at the floor, no loads."""
    R = rot_z(yaw)
    tips = np.array([R @ np.array([1.601, y, 0.0]) + np.asarray(pos, float) for y in (0.293, -0.293)])
    tips[:, 2] = tip_z
    return Episode(
        t=np.arange(n) * DT, truck_pos=const(pos, n), truck_quat_xyzw=const(yaw_quat_xyzw(yaw), n), speed=np.zeros(n),
        wheel_fz=np.full((n, 4), 5000.0), fork_tips=const(tips, n), blade_pitch=np.zeros(n),
        signals=dict(park=np.ones(n), direction=np.zeros(n, int), engine_running=np.zeros(n, bool),
                     key_on=np.zeros(n, bool), seat_switch_closed=np.ones(n, bool)))


def set_tips(ep: Episode, xy_left, xy_right, z) -> Episode:
    tips = np.array([[*xy_left, z], [*xy_right, z]])
    return replace(ep, fork_tips=const(tips, ep.n))


def static_load(corners, n=N) -> np.ndarray:
    return const(corners, n)


def with_load(ep: Episode, name: str, corners, mass: float | None = None) -> Episode:
    loads = dict(ep.loads, **{name: static_load(corners, ep.n) if np.ndim(corners) == 2 else corners})
    masses = dict(ep.load_masses, **({} if mass is None else {name: mass}))
    return replace(ep, loads=loads, load_masses=masses)


def with_signal(ep: Episode, name: str, value, at=None) -> Episode:
    arr = np.array(ep.signals[name])
    arr[slice(None) if at is None else at] = value
    return replace(ep, signals=dict(ep.signals, **{name: arr}))


def with_speed(ep: Episode, idx, v) -> Episode:
    s = np.array(ep.speed)
    s[idx] = v
    return replace(ep, speed=s)


def with_roll(ep: Episode, idx: int, deg: float) -> Episode:
    q = np.array(ep.truck_quat_xyzw)
    yaw = math.atan2(2 * (q[0, 3] * q[0, 2]), 1 - 2 * q[0, 2] ** 2)
    q[idx] = quat_of(rot_z(yaw) @ rot_x(math.radians(deg)))
    return replace(ep, truck_quat_xyzw=q)


def with_contact(ep: Episode, **kw) -> Episode:
    base = dict(mover="truck", structure="wall", t=1.0, onset_speed=1.0, peak_n=10.0, impulse_ns=1.0)
    return replace(ep, contacts=ep.contacts + (ContactEpisode(**{**base, **kw}),))


def with_floor(ep: Episode, name: str, off_samples: int, xy=(50.0, 50.0), start: int = 4, onset=0.0) -> Episode:
    """A load on a floor, off every floor for off_samples samples from `start`, then touching at xy (0: never
    lifted)."""
    touching = np.ones(ep.n, bool)
    touching[start:start + off_samples] = False
    pts = const(np.asarray(xy, float), ep.n)
    ons = np.zeros(ep.n)
    ons[start + off_samples:] = onset
    return replace(ep, floor=dict(ep.floor, **{name: LoadFloorTrack(touching, pts, ons)}))


def with_trailer(ep: Episode, door: str, bed_z: float, shift=0.0) -> Episode:
    fr = docked_trailer_frame(SITE, door, bed_z)
    pos = const(fr.origin, ep.n)
    pos[-1] = pos[-1] + fr.R[:, 0] * shift
    return replace(ep, trailer_pos=pos, trailer_quat_xyzw=const(quat_of(fr.R), ep.n))


# ------------------------------------------------------------------------------------------------ good episodes
def good_park() -> Episode:
    ep = base_episode((3.0, 28.25, 0.0), math.pi, tip_z=0.003)
    return replace(ep, blade_pitch=np.full(ep.n, -0.01))


def good_ramp_down(drop="east") -> Episode:
    dz = SITE.drop_zone(drop)
    cx, cy = (dz.x[0] + dz.x[1]) / 2, (dz.y[0] + dz.y[1]) / 2
    ep = base_episode((cx - 3.5, cy, 0.0), 0.0, tip_z=0.05)
    ep = set_tips(ep, (cx - 2.0, cy + 0.293), (cx - 2.0, cy - 0.293), 0.05)
    ep = with_speed(ep, 5, 1.0)
    return with_load(ep, "carried", pallet_corners((cx, cy, 0.0), np.eye(3)), 1000.0)


def rack_pallet_centre(loc, overhang=0.076):
    """Bottom-face centre of the carried pallet set on a location with the given front overhang."""
    half = SITE.pallet.L / 2
    x = loc.aisle_x + overhang - half if loc.face == "e" else loc.aisle_x - overhang + half
    return np.array([x, loc.y, loc.z])


def good_put_away(location="A-e-01-1-0") -> Episode:
    loc = rack_location(SITE, location)
    centre = rack_pallet_centre(loc)
    ep = base_episode((centre[0] + 3.0, centre[1], 0.0), math.pi, tip_z=0.05)
    ep = set_tips(ep, (centre[0] + 1.6, centre[1] + 0.3), (centre[0] + 1.6, centre[1] - 0.3), 0.05)
    ep = with_load(ep, "carried", pallet_corners(centre, np.eye(3)), 1000.0)
    other = rack_location(SITE, location[:-1] + "1")
    ep = with_load(ep, "stored_" + other.name, pallet_corners(rack_pallet_centre(other), np.eye(3)), 900.0)
    labels = tuple(beam_label(SITE, loc, x) for x in front_beam_xs(SITE, loc))
    return replace(ep, beam_contacts=labels)


def trailer_pallet(frame, lx, ly, tilt_deg=0.0, dz=0.0):
    """Corners of a pallet at trailer-frame (lx, ly) lying along the trailer, optionally tilted about its long axis."""
    R = frame.R @ rot_x(math.radians(tilt_deg))
    return pallet_corners(frame.origin + frame.R @ np.array([lx, ly, dz]), R)


def good_load_trailer(door="door1", side="L", bed_z=0.2) -> Episode:
    fr = docked_trailer_frame(SITE, door, bed_z)
    tr, g = SITE.trailer, SITE.pallet
    sg = 1.0 if side == "L" else -1.0
    ahead_x = tr.L - 0.03 - g.L / 2
    ahead = trailer_pallet(fr, ahead_x, sg * (tr.W / 2 - 0.03 - g.W / 2))
    ly = sg * (tr.W / 2 - 0.05 - g.W / 2)
    carried = trailer_pallet(fr, ahead_x - g.L - 0.02, ly)
    x, face = SITE.dock.door(door)
    ep = base_episode((x, face - 3.0, 0.0), math.pi / 2, tip_z=0.4)
    ep = set_tips(ep, (x, face - 1.5), (x, face - 1.5), 0.4)
    ep = with_load(with_load(ep, "carried", carried, 800.0), f"trailer_1{side}", ahead, 700.0)
    return with_trailer(ep, door, bed_z)


def bale_corners(frame, lx, ly, tilt_deg=0.0, yaw=0.0):
    size = SITE.bale.size
    R = frame.R @ rot_z(yaw) @ rot_x(math.radians(tilt_deg))
    bottom = frame.origin + frame.R @ np.array([lx, ly, 0.1016])
    return box_corners(bottom + R[:, 2] * size[2] / 2, R, size)


def strip_corners(frame, lx, ly):
    d = SITE.bale
    return box_corners(frame.origin + frame.R @ np.array([lx, ly, d.dunnage_size[0] / 2]), frame.R,
                       (d.dunnage_size[2], d.dunnage_size[0], d.dunnage_size[1]))


def good_load_bales(door="door3", bed_z=0.2) -> Episode:
    fr = docked_trailer_frame(SITE, door, bed_z)
    tr, bs = SITE.trailer, SITE.bale
    x, face = SITE.dock.door(door)
    ep = base_episode((x, face - 3.0, 0.0), math.pi / 2, tip_z=0.1)
    ep = set_tips(ep, (x, face - 1.5), (x, face - 1.5), 0.1)
    bx = tr.L - 0.02 - bs.size[0] / 2
    for k, ly in enumerate((0.411, -0.411)):
        ep = with_load(ep, f"bale_{k}", bale_corners(fr, bx, ly), 500.0)
        for sy in (-1, 1):
            ep = with_load(ep, f"nose_dunnage_{k}{'ab'[sy > 0]}", strip_corners(fr, tr.L - 0.03 - 0.6096,
                                                                                ly + sy * bs.dunnage_y))
    ep = with_trailer(ep, door, bed_z)
    other = "door4" if door == "door3" else "door3"
    fo = docked_trailer_frame(SITE, other, bed_z)
    return replace(ep, other_trailers={other: TrailerTrack(const(fo.origin, ep.n), const(quat_of(fo.R), ep.n))})


# ------------------------------------------------------------------------------------------------ mutators
def identity(ep):
    return ep


def truck_xy(x=None, y=None):
    def f(ep):
        pos = np.array(ep.truck_pos)
        pos[:, 0] = pos[0, 0] if x is None else x
        pos[:, 1] = pos[0, 1] if y is None else y
        return replace(ep, truck_pos=pos)
    return f


def tips_z(z):
    def f(ep):
        tips = np.array(ep.fork_tips)
        tips[:, :, 2] = z
        return replace(ep, fork_tips=tips)
    return f


def rest_for(n):
    return lambda ep: with_speed(ep, ep.n - 1 - n, 1.0)


def last_signal(name, value):
    return lambda ep: with_signal(ep, name, value, -1)
