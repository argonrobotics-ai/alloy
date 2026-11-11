# SPDX-License-Identifier: Apache-2.0
"""Site geometry and per-task data of Gym-Anything's forklift environment, for the ported success criteria.

Frame: GA's facility frame, copied as is so that its numbers carry over. Metres, z up (ground at z = 0), x and y
horizontal in the plan; yaw is a rotation about +z from +x (counter-clockwise seen from above), so ``yaw = atan2(R[1,
0],
R[0, 0])``. Warehouse and dock share one building frame (the dock section: interior x 0-26, y 0-20, dock wall at
y = 20, the trailers outside it at y > 20 with their noses toward +y); the yard is a separate outdoor frame (the ramp's
foot toward +y, the trailer's rear at y = -8, the trailer's nose toward +y at x = 10). The truck's own frame: x forward
(fork tips at +1.62, counterweight at -2.14), y left. A trailer's frame: origin on the floor top at the rear sill
centre, x toward the nose, y left, z up.

Every number carries its source as a trailing ``# F/<file>:<line>``, F being
``benchmarks/roboworld/environments/forklift/`` of Gym-Anything's robotics repo (MIT, commit febb260). A line with no
cite is derived from cited ones (the arithmetic is in the code).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

IN, FT, LB = 0.0254, 0.3048, 0.45359237  # F/params.py:25
G = 9.81  # F/params.py:26
RAMP_X = 10.0  # F/facility.py:344 (the yard ramp's centreline x)


@dataclass(frozen=True)
class Rect:
    """An axis-aligned plan rectangle in the facility frame."""

    x: tuple[float, float]
    y: tuple[float, float]

    def contains(self, xy, margin: float = 0.0) -> bool:
        """GA's inside_rect: inclusive edges, shrunk by margin."""
        return (self.x[0] + margin <= xy[0] <= self.x[1] - margin
                and self.y[0] + margin <= xy[1] <= self.y[1] - margin)

    def outside_by(self, xy) -> float:
        """How far (m) a point is outside the rectangle on its worst axis (<= 0 inside)."""
        return max(self.x[0] - xy[0], xy[0] - self.x[1], self.y[0] - xy[1], xy[1] - self.y[1])

    def polygon(self) -> np.ndarray:
        return np.array([[self.x[0], self.y[0]], [self.x[1], self.y[0]], [self.x[1], self.y[1]], [self.x[0],
                                                                                                  self.y[1]]])


@dataclass(frozen=True)
class TruckFootprint:
    """The truck's footprint in its own frame, counterweight to fork tips."""

    x: tuple[float, float] = (-2.14, 1.62)  # F/evaluator.py:67
    half_y: float = 0.66  # F/evaluator.py:67
    fork_half_w: float = 0.051  # F/evaluator.py:75
    reach_loaded: float = 1.80  # F/evaluator.py:623

    def corners(self, x: float, y: float, yaw: float) -> np.ndarray:
        """GA's footprint(): the four plan corners (world) of a truck at (x, y, yaw)."""
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([(x + c * fx - s * fy, y + s * fx + c * fy) for fx in self.x for fy in (-self.half_y,
                                                                                                self.half_y)])


@dataclass(frozen=True)
class Pallet:
    """The 48 x 40 in GMA pallet; its body frame has the origin at the bottom face centre (x along the 48 in)."""

    L: float = 48 * IN  # F/params.py:598
    W: float = 40 * IN  # F/params.py:598
    H: float = 4.76 * IN  # F/params.py:598
    deck: float = 0.630 * IN  # F/params.py:598
    stringer_w: float = 1.375 * IN  # F/params.py:598
    mass: float = 20.0  # F/params.py:598


@dataclass(frozen=True)
class Leveler:
    width: float = 78 * IN  # F/params.py:612
    length: float = 8 * FT  # F/params.py:612
    lip: float = 16 * IN  # F/params.py:612
    capacity: float = 35000 * LB  # F/params.py:612
    deck_travel_deg: float = 10.0  # F/params.py:914
    lip_pendant_deg: float = -95.0  # F/params.py:914
    lip_stowed_up_deg: float = 15.0  # F/params.py:914
    lip_plate: float = 5 / 8 * IN  # F/params.py:833
    extended_stop_deg: float = -3.0  # F/params.py:833
    deck_mass: float = 650.0  # F/params.py:746
    lip_mass: float = 90.0  # F/params.py:746
    deck_thickness: float = 0.10  # F/dock.py:96 (collider thickness)


@dataclass(frozen=True)
class TrailerSpec:
    """The 53 ft dry van: interior (floor top at z = 0 of the trailer frame), mass, supports."""

    L: float = 52.5 * FT  # F/params.py:619
    W: float = 101 * IN  # F/params.py:619
    H: float = 110.25 * IN  # F/params.py:619
    floor: float = 1.375 * IN  # F/params.py:619
    mass: float = 6800.0  # F/params.py:719
    tandem_from_rear: float = 2.6  # F/params.py:719
    gear_from_rear: float = 12.6  # F/params.py:719
    wall: float = 0.025  # F/params.py:719
    subframe: float = 0.22  # F/params.py:719
    track: float = 1.83  # F/params.py:719
    outside_W: float = 102 * IN  # F/params.py:719
    spring_k: float = 4.11e5  # F/params.py:722
    damper_c: float = 2.0e4  # F/params.py:733
    sag_in: tuple[float, float] = (3.5 * IN, 5.0 * IN)  # F/params.py:686
    chock_k: float = 2.0e6  # F/params.py:742
    chock_capacity: float = 60e3  # F/params.py:742
    restraint_k: float = 5.0e6  # F/params.py:744
    restraint_capacity: float = 142e3  # F/params.py:744
    bed_in: tuple[float, float] = (44.0, 52.0)  # F/task.py:255 (floor height above the apron, in)
    rear_frame: float = 0.10  # F/params.py:869
    max_move: float = 0.025  # F/evaluator.py:619


@dataclass(frozen=True)
class RackSpec:
    """Selective pallet rack (Steel King), 7 bays per row from y = 4.0."""

    frame_depth: float = 42 * IN  # F/params.py:691
    beam: float = 96 * IN  # F/params.py:691
    levels: tuple[float, ...] = (0.0, 1.60, 3.20)  # F/params.py:691
    upright: float = 3 * IN  # F/params.py:691
    beam_h: float = 5 * IN  # F/params.py:691
    beam_w: float = 0.05  # F/params.py:691
    frame_gap: float = 12 * IN  # F/params.py:691
    lift_off: float = 6 * IN  # F/params.py:702
    aisle: float = 4.50  # F/params.py:712
    upright_h: float = 16 * FT  # F/params.py:869
    y0: float = 4.0  # F/facility.py:160
    n_bays: int = 7  # F/facility.py:160
    x0: float = 0.6  # F/facility.py:161
    stored_load_mass: tuple[float, float] = (500.0, 1000.0)  # F/params.py:688
    overhang_band: tuple[float, float] = (0.051, 0.101)  # F/evaluator.py:500
    side_clearance: float = 0.05  # F/evaluator.py:514
    level_tol_deg: float = 2.0  # F/evaluator.py:517
    beam_level_tol: float = 0.03  # F/evaluator.py:495
    stored_moved_max: float = 0.030  # F/evaluator.py:523
    beam_x_tol: float = 0.01  # F/evaluator.py:488

    @property
    def bay_pitch(self) -> float:
        return self.beam + self.upright

    @property
    def beam_levels(self) -> tuple[float, ...]:
        return self.levels[1:]

    def stored_overhang(self, pallet_length: float) -> float:
        """A stored 48 in pallet overhangs the beams by (L - frame depth) / 2 each side (F/facility.py:61)."""
        return (pallet_length - self.frame_depth) / 2


@dataclass(frozen=True)
class BaleSpec:
    size: tuple[float, float, float] = (60 * IN, 30 * IN, 48 * IN)  # F/params.py:747
    mass: tuple[float, float] = (450.0, 590.0)  # F/params.py:747
    mu: float = 0.40  # F/params.py:747
    dunnage_size: tuple[float, float, float] = (4 * IN, 4 * IN, 48 * IN)  # F/params.py:748
    dunnage_mass: float = 4.0  # F/params.py:748
    dunnage_y: float = 0.13  # F/task.py:66
    nose_gap: tuple[float, float] = (-0.005, 0.075)  # F/evaluator.py:676
    upright_deg: float = 5.0  # F/evaluator.py:678
    rest_tol: float = 0.03  # F/evaluator.py:679
    overlap_tol: float = 0.005  # F/evaluator.py:680
    row_x: tuple[float, float] = (19.0, 25.0)  # F/facility.py:315
    row_y: tuple[float, float] = (3.0, 9.0)  # F/facility.py:315


@dataclass(frozen=True)
class YardRampSpec:
    """Bluff MobilePRO yard ramp (30 ft incline, 6 ft level-off) at the yard trailer's rear."""

    incline: float = 30 * FT  # F/params.py:749
    level: float = 6 * FT  # F/params.py:749
    width: float = 84 * IN  # F/params.py:749
    rise: float = 48 * IN  # F/params.py:749
    x: float = RAMP_X  # F/facility.py:344
    y_rear: float = -8.0  # F/facility.py:345
    lip_length: float = 16 * IN  # F/params.py:768
    lip_travel: float = 3 * IN  # F/params.py:768
    lip_pieces: int = 4  # F/params.py:768
    lip_piece_gap: float = 0.006  # F/params.py:768
    lip_plate: float = 5 / 8 * IN  # F/params.py:768
    lip_hinge_gap: float = 0.010  # F/params.py:768
    bumper: float = 4 * IN  # F/params.py:869
    speed_limit_kmh: float = 6.0  # F/evaluator.py:1146
    rollback_max: float = 0.3  # F/evaluator.py:1118
    zone_margin: float = 0.5  # F/events.py:187

    @property
    def deck(self) -> float:
        """Level-off height: the trailer floor plus one lip plate (F/facility.py:343)."""
        return self.rise + self.lip_plate

    @property
    def run(self) -> float:
        return math.sqrt(self.incline ** 2 - self.deck ** 2)

    @property
    def y_top(self) -> float:
        return self.y_rear + self.level

    @property
    def y_foot(self) -> float:
        return self.y_top + self.run

    @property
    def grade(self) -> float:
        return self.deck / self.run

    def level_end(self, trailer_rear_frame: float) -> float:
        """Where the level-off ends at its bumpers (F/facility.py:354)."""
        return self.y_rear + trailer_rear_frame + self.bumper


@dataclass(frozen=True)
class DockSpec:
    """Shipping dock: four 9 x 10 ft doors on 12 ft centres in the wall at y = 20, levelers in pits, staging lanes."""

    face_y: float = 20.0  # F/params.py:869
    door_pitch: float = 12 * FT  # F/params.py:869
    door_x0: float = 5.0  # F/facility.py:245
    door_w: float = 9 * FT  # F/params.py:869
    door_h: float = 10 * FT  # F/params.py:869
    bumper: float = 4.5 * IN  # F/params.py:869
    interior_x: float = 26.0  # F/facility.py:257
    lane_half_w: float = 1.4  # F/facility.py:311
    lane_y: tuple[float, float] = (6.0, 15.5)  # F/facility.py:311
    stop_short_of_door: float = 1.0  # F/evaluator.py:616
    slot_margin: float = 0.30  # F/evaluator.py:570
    wall_gap_max: float = 0.125  # F/evaluator.py:72
    gap_ahead: tuple[float, float] = (-0.005, 0.050)  # F/evaluator.py:605
    level_tol_deg: float = 2.0  # F/evaluator.py:614
    on_floor_tol: float = 0.015  # F/evaluator.py:598

    def door_x(self, k: int) -> float:
        return self.door_x0 + k * self.door_pitch

    def door_names(self) -> tuple[str, ...]:
        return ("door1", "door2", "door3", "door4")  # F/task.py:85

    def door(self, name: str) -> tuple[float, float]:
        """(x, face_y) of a door by name ('door1'..'door4')."""
        return self.door_x(self.door_names().index(name)), self.face_y

    def staging_lane(self, door: str) -> Rect:
        x = self.door(door)[0]
        return Rect(x=(x - self.lane_half_w, x + self.lane_half_w), y=self.lane_y)


@dataclass(frozen=True)
class TaskLimits:
    """Speed, time and set-down thresholds shared by the five tasks."""

    posted_speed: float = 8.0 / 3.6  # F/params.py:714
    over_posted: float = 1.1  # F/evaluator.py:65
    time_limit_s: dict[str, float] = field(default_factory=lambda: {
        "start-and-park": 90.0, "put-away-rack": 240.0, "load-trailer": 240.0,  # F/task.py:56
        "load-trailer-chocked": 240.0, "load-bales": 480.0, "ramp-down-loaded": 240.0})  # F/task.py:58
    strike_n: float = 2000.0  # F/events.py:33
    strike_ns: float = 50.0  # F/events.py:33
    impact_speed: float = 0.10  # F/evaluator.py:15
    set_down_speed: float = 0.10  # F/evaluator.py:28
    set_down_peak_n: float = 25000.0  # F/evaluator.py:34
    contact_timeconst: float = 0.004  # F/params.py:30
    contact_dmax: float = 0.99  # F/params.py:31
    wheel_lift_s: float = 0.3  # F/evaluator.py:226
    wheel_lift_fz: float = 1.0  # F/events.py:162
    roll_deg: float = 6.0  # F/evaluator.py:227
    overturn_deg: float = 30.0  # F/events.py:157
    seat_open_s: float = 1.0  # F/evaluator.py:247
    seat_moving_speed: float = 0.05  # F/events.py:170
    head_contact_n: float = 130.0  # F/params.py:534
    rest_speed: float = 0.02  # F/events.py:144
    lifted_s: float = 0.25  # F/events.py:28
    floor_record_gap_s: float = 0.25  # F/events.py:277

    @property
    def building_limit(self) -> float:
        """m/s: posted 8 km/h + 10 % (F/events.py:30)."""
        return self.posted_speed * self.over_posted

    @property
    def limit_kmh(self) -> float:
        return self.posted_speed * self.over_posted * 3.6

    @property
    def set_down_decel(self) -> float:
        """B v at the speed rule (F/evaluator.py:52), m/s^2."""
        return 2.0 * self.set_down_speed / (self.contact_dmax * self.contact_timeconst)


@dataclass(frozen=True)
class LoadRange:
    """A drawn unit load on a GMA pallet: mass (kg, pallet excluded) and height (m)."""

    mass: tuple[float, float]
    height: tuple[float, float]


# per-task carried-load ranges
LOAD_RANGES = {
    "put-away-rack": LoadRange(mass=(600.0, 1000.0), height=(0.9, 1.2)),  # F/task.py:241
    "ramp-down-loaded": LoadRange(mass=(900.0, 1300.0), height=(1.0, 1.3)),  # F/task.py:329
    "load-trailer": LoadRange(mass=(600.0, 1100.0), height=(1.0, 1.4)),  # F/task.py:282
    "load-trailer-nose": LoadRange(mass=(500.0, 900.0), height=(1.0, 1.4)),  # F/task.py:277
}

# fork blade tip / heel points in the side-shift frame (blades() of F/evaluator.py:118-121)
BLADE_TIP_X = 1.601  # F/evaluator.py:119
BLADE_HEEL_X = 0.55  # F/evaluator.py:120
BLADE_Y = 0.293  # F/evaluator.py:118
BLADE_Z = 0.104  # F/evaluator.py:119

SLOT_NAMES = {"L": 1.0, "R": -1.0}  # the trailer's left is +y (F/evaluator.py:607)


@dataclass(frozen=True)
class ForkliftSite:
    """Everything the five criteria read, in GA's facility frame."""

    truck: TruckFootprint = TruckFootprint()
    pallet: Pallet = Pallet()
    leveler: Leveler = Leveler()
    trailer: TrailerSpec = TrailerSpec()
    rack: RackSpec = RackSpec()
    bale: BaleSpec = BaleSpec()
    ramp: YardRampSpec = YardRampSpec()
    dock: DockSpec = DockSpec()
    limits: TaskLimits = TaskLimits()
    parking: Rect = Rect(x=(0.5, 5.6), y=(26.5, 30.0))  # F/facility.py:221
    drop_east: Rect = Rect(x=(14.0, 17.0), y=(5.0, 8.0))  # F/facility.py:403
    drop_west: Rect = Rect(x=(2 * RAMP_X - 17.0, 2 * RAMP_X - 14.0), y=(5.0, 8.0))  # F/facility.py:407

    def drop_zone(self, which: str) -> Rect:
        """'east' or 'west' yard drop zone of ramp-down-loaded (F/evaluator.py:1122)."""
        return {"east": self.drop_east, "west": self.drop_west}[which]

    def ground_height(self, x: float, y: float) -> float:
        """GA's ground_height: highest static surface at (x, y); the yard ramp rises to its deck, the floor is 0."""
        r = self.ramp
        if abs(x - r.x) > r.width / 2:
            return 0.0
        if r.y_top <= y <= r.y_foot:
            return max(0.0, r.deck * (r.y_foot - y) / r.run)
        if r.level_end(self.trailer.rear_frame) <= y < r.y_top:
            return r.deck
        return 0.0


SITE = ForkliftSite()


# ------------------------------------------------------------------------------------------------ derived geometry
@dataclass(frozen=True)
class TrailerFrame:
    """A trailer's pose: origin on the floor top at the rear sill centre, rotation columns = its x (nose), y (left),
    z."""

    origin: np.ndarray
    R: np.ndarray
    spec: TrailerSpec = TrailerSpec()

    def to_local(self, points) -> np.ndarray:
        """World points (..., 3) in the trailer frame (x from the rear sill, y left, z above the floor)."""
        return (np.asarray(points, float) - self.origin) @ self.R

    def interior_contains(self, points, margin: float = 0.0) -> np.ndarray:
        """Per point: inside the interior box (floor to roof, walls, rear sill to nose wall)."""
        p = self.to_local(points)
        s = self.spec
        return ((p[..., 0] > margin) & (p[..., 0] < s.L - margin) & (np.abs(p[..., 1]) < s.W / 2 - margin)
                & (p[..., 2] > -margin) & (p[..., 2] < s.H))

    def interior_corners(self) -> np.ndarray:
        """The interior box's eight corners (world), in (sx, sy, sz) order with sz fastest, x 0..L, y +-W/2, z 0..H."""
        s = self.spec
        loc = np.array([(x, y, z) for x in (0.0, s.L) for y in (-s.W / 2, s.W / 2) for z in (0.0, s.H)])
        return self.origin + loc @ self.R.T


def quat_xyzw_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_quat_xyzw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def trailer_rear(site: ForkliftSite, door: str) -> tuple[float, float]:
    """A docked trailer's rear sill centre (x, y), nose toward +y (F/facility.py:567: face + bumper + rear frame + 6
    mm)."""
    x, face = site.dock.door(door)
    return x, face + site.dock.bumper + site.trailer.rear_frame + 0.006  # F/facility.py:567


def docked_trailer_frame(site: ForkliftSite, door: str, bed_z: float) -> TrailerFrame:
    """The spotted pose of the trailer at a dock door (yaw +pi/2) with its floor top at bed_z."""
    rx, ry = trailer_rear(site, door)
    return TrailerFrame(np.array([rx, ry, bed_z]), quat_xyzw_to_matrix(yaw_quat_xyzw(math.pi / 2)), site.trailer)


def trailer_frame_from_pose(site: ForkliftSite, pos, quat_xyzw) -> TrailerFrame:
    return TrailerFrame(np.asarray(pos, float), quat_xyzw_to_matrix(quat_xyzw), site.trailer)


def ramp_footprint_polygon(site: ForkliftSite = SITE) -> np.ndarray:
    """The ramp's plan footprint (incline plus level-off), counter-clockwise (4, 2): x +- width / 2, foot to the
    bumpers."""
    r = site.ramp
    y0, y1 = r.level_end(site.trailer.rear_frame), r.y_foot
    return Rect(x=(r.x - r.width / 2, r.x + r.width / 2), y=(y0, y1)).polygon()


def ramp_speed_zone(site: ForkliftSite = SITE) -> Rect:
    """The plan zone in which GA latches the ramp speed (F/events.py:187): ``abs(x - ramp.x) < ramp.width`` (the full
    width as a half-extent: reproduced as written) and y from 0.5 m behind the rear sill to 0.5 m past the foot."""
    r = site.ramp
    m = r.zone_margin
    return Rect(x=(r.x - r.width, r.x + r.width), y=(r.y_rear - m, r.y_foot + m))


def rack_aisles(site: ForkliftSite = SITE) -> dict[str, tuple[float, float]]:
    """Working aisles '1' and '2' (x ranges between the beam faces, F/facility.py:170-173)."""
    k = site.rack
    a = k.aisle + 2 * k.stored_overhang(site.pallet.L)
    xa1 = k.x0 + k.frame_depth
    xb1 = xa1 + a + 2 * k.frame_depth + k.frame_gap
    return {"1": (xa1, xa1 + a), "2": (xb1, xb1 + a)}


@dataclass(frozen=True)
class RackLocation:
    """One pallet position on a beam pair (GA's L.zones['rack_locations'] entry)."""

    name: str
    row: str
    frame: int
    face: str
    bay: int
    level: int
    slot: int
    x0: float
    x1: float
    y: float
    z: float
    aisle_x: float

    @property
    def beams(self) -> tuple[float, float]:
        return (self.x0, self.x1)


def _row_frames(site: ForkliftSite) -> dict[str, tuple[float, bool, str | None]]:
    k = site.rack
    a = k.aisle + 2 * k.stored_overhang(site.pallet.L)
    xa1 = k.x0 + k.frame_depth
    xb0 = xa1 + a
    xb1 = xb0 + 2 * k.frame_depth + k.frame_gap
    return {"A": (k.x0, False, "e"), "B": (xb0, True, None), "C": (xb1 + a, False, "w")}  # F/facility.py:162-171


def rack_location(site: ForkliftSite, name: str) -> RackLocation:
    """Parse 'row-face-bay-level-slot' (e.g. 'B-w-03-1-0'; level 0 = floor, 1-2 beam levels; slot 0 = lower y)."""
    row, face, bay, level, slot = name.split("-")
    x0, double, faces = _row_frames(site)[row]
    k = site.rack
    frames = [(x0, x0 + k.frame_depth)]
    if double:
        frames.append((x0 + k.frame_depth + k.frame_gap, x0 + 2 * k.frame_depth + k.frame_gap))
    fi = next(i for i, _ in enumerate(frames) if (("w" if i == 0 else "e") if double else faces) == face)
    fa, fb = frames[fi]
    yc = k.y0 + (int(bay) - 0.5) * k.bay_pitch
    ys = yc + (-k.beam / 4 if int(slot) == 0 else k.beam / 4)
    lv = int(level)
    z = 0.0 if lv == 0 else k.beam_levels[lv - 1]
    return RackLocation(name, row, fi, face, int(bay), lv, int(slot), fa, fb, ys, z, fa if face == "w" else fb)


def beam_label(site: ForkliftSite, loc: RackLocation, x: float) -> str:
    """The rack beam geom label '<row><frame>_beam<bay-1>_<level>_<x>' (F/facility.py:86) with x to 2 decimals."""
    return f"{loc.row}{loc.frame}_beam{loc.bay - 1}_{loc.level}_{x:.2f}"


def front_beam_xs(site: ForkliftSite, loc: RackLocation) -> tuple[float, float]:
    """x of the location's two beams' centres (F/evaluator.py:473)."""
    return (loc.x0 + site.rack.beam_w / 2, loc.x1 - site.rack.beam_w / 2)
