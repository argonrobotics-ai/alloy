# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's dock, dock leveler, yard ramp and pallet rack for the forklift tasks (SCR-WP8-12 F4), from GA's
numbers (``F/`` = gym-anything-robotics benchmarks/roboworld/environments/forklift/; F/params.py, F/dock.py,
F/facility.py). The trailer is ``forklift_trailer``.

World frame: the dock face at x = 0, the building (its floor 48 in up, F/params.py:606) at x < 0, the trailer's rear
toward the dock with its nose toward +x. Every top face a truck drives on is a box listed in ``DRIVABLE`` (or returned
by the builder), for the vehicle plug-in's ``surfaces``; ``RESTING`` (the leveler, lying on the trailer floor) goes in
its ``resting``. ``CAPACITY`` sizes MuJoCo Warp's contacts for these rigs (``TestScene.solver_kwargs``).
- **Dock floor** with a pit for the leveler, and two rubber bumpers (4.5 in) under its edge that the trailer's rear
  sill backs against (F/params.py:869-870).
- **Leveler** (F/dock.py:88-146; F/params.py:612, :746, :833, :912-916): a 78 in x 8 ft deck (650 kg) on a revolute
  hinge at its rear (±10°, 200 N·m of hinge friction), and a 16 in lip (90 kg, 5/8 in plate, 20 N·m) on a hinge at
  the deck's front. The lip can hang to pendant but cannot rise past its extended stop 3° below the deck. Deployed, it
  rests on the trailer floor by native contact and floats with the trailer. Deck and lip are drivable.
- **Yard ramp** (F/facility.py:337-370; F/params.py:749, :768-772): a 30 ft incline and a 6 ft level-off, 84 in wide,
  the level-off one lip plate above the trailer floor, and 4 hinged lip plates (16 in, 5/8 in, 5 N·m) resting on it.
- **Rack** (F/params.py:691-693): beams 96 in long, 5 in deep, at 1.60 and 3.20 m (their tops), front and back of a
  42 in frame, on 3 in uprights 16 ft tall.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.plugins.harness.scene import body_xform
from alloy.tasks import forklift_trailer as tr

IN, FT, G = tr.IN, tr.FT, tr.G
DOCK_H = 48 * IN                                           # F/params.py:606
BUMPER = 4.5 * IN                                          # :869
DECK_W, DECK_L, LIP_L = 78 * IN, 8 * FT, 16 * IN           # :612
DECK_M, LIP_M = 650.0, 90.0                                # :746
DECK_T, LIP_T = 0.10, 5 / 8 * IN                           # F/dock.py:95; F/params.py:833
DECK_TRAVEL, LIP_STOP, LIP_PENDANT = math.radians(10.0), math.radians(3.0), math.radians(95.0)   # :833, :914
DECK_FRICTION, LIP_FRICTION = 200.0, 20.0                  # N·m, :912
RAMP_INCLINE, RAMP_LEVEL, RAMP_W = 30 * FT, 6 * FT, 84 * IN                                      # :749
RAMP_BUMPER = 4 * IN                                       # :870
PLATE_L, PLATE_TRAVEL, PLATES, PLATE_GAP, PLATE_T = 16 * IN, 3 * IN, 4, 0.006, 5 / 8 * IN       # :768
PLATE_FRICTION, STEEL = 5.0, 7850.0                        # :769, :774
RACK_DEPTH, RACK_BEAM, RACK_UPRIGHT, RACK_BEAM_H, RACK_BEAM_W = 42 * IN, 96 * IN, 3 * IN, 5 * IN, 0.05   # :691
RACK_LEVELS, RACK_H = (1.60, 3.20), 16 * FT                # :691, :869
STOP_KE, STOP_KD = 1.0e7, 1.0e5            # N·m/rad, N·m·s/rad: hinge stops (Newton's default 1e4 let the lip
                                           # ride 5° past its stop under the deck's 1.4 kN·m)
STATIC = newton.ModelBuilder.ShapeConfig(mu=tr.MU_STRUCTURE)
DRIVABLE = ("dock_floor", "dock_side_l", "dock_side_r", "leveler_deck", "leveler_lip")
RESTING = (("leveler_deck", "trailer_floor"),          # the vehicle's ``resting`` (left out of its step check):
           ("leveler_lip", "trailer_floor"))           # both lie on the trailer floor, the deck through its lip
# MuJoCo Warp's contact and row capacity per world for these rigs (its default 64 rows dropped a ramp plate's contacts)
CAPACITY = {"nconmax": 512, "njmax": 3072}


def _static_box(b: newton.ModelBuilder, label: str, lo, hi, yaw: float = 0.0, about=(0.0, 0.0)) -> None:
    """A static box spanning ``lo`` to ``hi`` (x, y, z), turned by ``yaw`` about the vertical through ``about``."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    c, s = math.cos(yaw), math.sin(yaw)
    rel = centre[:2] - np.asarray(about)
    centre[:2] = np.asarray(about) + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(*centre), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)),
                    hx=float(half[0]), hy=float(half[1]), hz=float(half[2]), cfg=STATIC, label=label)


def add_dock(b: newton.ModelBuilder, depth: float = 20.0, half_width: float = 4.0) -> None:
    """The dock floor (a slab from the ground to 48 in, the pit left open for the leveler) and its two bumpers."""
    pit = 0.5 * DECK_W + 0.01
    _static_box(b, "dock_floor", (-depth, -half_width, 0.0), (-DECK_L - 0.02, half_width, DOCK_H))   # hinge gap
    _static_box(b, "dock_side_l", (-DECK_L, pit, 0.0), (0.0, half_width, DOCK_H))
    _static_box(b, "dock_side_r", (-DECK_L, -half_width, 0.0), (0.0, -pit, DOCK_H))
    for side, y in (("l", 1.2), ("r", -1.2)):
        _static_box(b, f"dock_bumper_{side}", (0.0, y - 0.127, DOCK_H - 0.284), (BUMPER, y + 0.127, DOCK_H - 0.03))


def _plate_inertia(mass: float, length: float, width: float, thick: float) -> np.ndarray:
    """A box's inertia about its centre: length along x, width along y, thickness along z."""
    return mass / 12.0 * np.diag([width**2 + thick**2, length**2 + thick**2, length**2 + width**2])


def deployed(bed: float) -> tuple[float, float]:
    """(deck, lip) joint angles of the leveler deployed onto a floor at ``bed`` [m], as GA (F/dock.py:96-103): the lip
    at its extended stop, the deck at the grade that puts the lip's underside 3 mm above the floor at its tip. A
    positive angle turns the front down (about +y)."""
    grade = 0.0
    for _ in range(20):
        edge = (bed - DOCK_H) + LIP_T + 0.003 - LIP_L * math.sin(grade - LIP_STOP)
        grade = math.asin(max(-0.2, min(0.2, edge / DECK_L)))
    return -grade, LIP_STOP


def lip_tip_load() -> float:
    """The deck and lip, as one beam pinned at the deck's hinge, rest on the floor at the lip's tip with this load [N]
    (GA's trailer rest share, F/dock.py:333-340)."""
    span = DECK_L + LIP_L
    return G * (DECK_M * DECK_L / 2.0 + LIP_M * (DECK_L + LIP_L / 2.0)) / span


def add_leveler(b: newton.ModelBuilder, bed: float, y: float = 0.0) -> tuple[int, int]:
    """The deployed leveler at the door centred on ``y``; returns the (deck, lip) joints."""
    q_deck, q_lip = deployed(bed)
    hinge = wp.transform(wp.vec3(-DECK_L, y, DOCK_H), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), q_deck))
    i_deck = _plate_inertia(DECK_M, DECK_L, DECK_W, DECK_T)
    deck = b.add_link(xform=hinge, mass=DECK_M, com=wp.vec3(DECK_L / 2.0, 0.0, -DECK_T / 2.0),
                      inertia=wp.mat33(*i_deck.ravel()), lock_inertia=True, label="leveler_deck")
    front = wp.transform_multiply(hinge, wp.transform(wp.vec3(DECK_L, 0.0, 0.0),
                                                      wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), q_lip)))
    i_lip = _plate_inertia(LIP_M, LIP_L, DECK_W, LIP_T)
    lip = b.add_link(xform=front, mass=LIP_M, com=wp.vec3(LIP_L / 2.0, 0.0, -0.02), inertia=wp.mat33(*i_lip.ravel()),
                     lock_inertia=True, label="leveler_lip")
    massless = newton.ModelBuilder.ShapeConfig(density=0.0, mu=tr.MU_STRUCTURE)
    b.add_shape_box(deck, xform=body_xform(DECK_L / 2.0, 0.0, -DECK_T / 2.0), hx=DECK_L / 2.0, hy=DECK_W / 2.0,
                    hz=DECK_T / 2.0, cfg=massless, label="leveler_deck")
    b.add_shape_box(lip, xform=body_xform(LIP_L / 2.0, 0.0, -LIP_T / 2.0), hx=LIP_L / 2.0, hy=DECK_W / 2.0 - 0.02,
                    hz=LIP_T / 2.0, cfg=massless, label="leveler_lip")
    j_deck = b.add_joint_revolute(-1, deck, parent_xform=wp.transform(wp.vec3(-DECK_L, y, DOCK_H), wp.quat_identity()),
                                  axis=(0.0, 1.0, 0.0), limit_lower=-DECK_TRAVEL, limit_upper=DECK_TRAVEL,
                                  limit_ke=STOP_KE, limit_kd=STOP_KD, friction=DECK_FRICTION, label="leveler_deck")
    j_lip = b.add_joint_revolute(deck, lip, parent_xform=body_xform(DECK_L, 0.0, 0.0), axis=(0.0, 1.0, 0.0),
                                 limit_lower=LIP_STOP, limit_upper=LIP_PENDANT, limit_ke=STOP_KE, limit_kd=STOP_KD,
                                 friction=LIP_FRICTION, label="leveler_lip")
    b.joint_q[b.joint_q_start[j_deck]] = q_deck
    b.joint_q[b.joint_q_start[j_lip]] = q_lip
    b.add_articulation([j_deck, j_lip], label="leveler")
    return j_deck, j_lip


def dock_rest(spot: tr.Spot) -> tr.Loads:
    """The trailer's extra static loads at a dock: the leveler's lip tip, in the trailer frame."""
    return ((lip_tip_load(), LIP_L * math.cos(LIP_STOP) - spot.x),)


def dock_spot(bed: float = DOCK_H) -> tr.Spot:
    """A trailer backed against the bumpers: its rear frame touching them, its floor at ``bed``."""
    return tr.Spot(x=BUMPER + tr.REAR_FRAME, bed=bed)


def add_yard_ramp(b: newton.ModelBuilder, rear_x: float, bed: float, y: float = 0.0) -> tuple[list[int], list[str]]:
    """The ramp toward -x from a trailer whose floor's rear edge is at ``rear_x``, floor at ``bed``: the level-off one
    plate above the floor, its 4 lip plates resting on it. Returns (plate joints, drivable labels)."""
    deck = bed + PLATE_T
    run = math.sqrt(RAMP_INCLINE**2 - deck**2)
    hinge_x = rear_x - tr.REAR_FRAME - RAMP_BUMPER
    top = hinge_x - RAMP_LEVEL + tr.REAR_FRAME + RAMP_BUMPER
    _static_box(b, "ramp_level", (top, y - RAMP_W / 2, deck - 0.05), (hinge_x, y + RAMP_W / 2, deck))
    theta = math.atan2(deck, run)
    length = math.hypot(run, deck)
    centre = np.array([top - run / 2, y, deck / 2]) - 0.025 * np.array([math.sin(theta), 0.0, math.cos(theta)])
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(*centre), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -theta)),
                    hx=length / 2, hy=RAMP_W / 2, hz=0.025, cfg=STATIC, label="ramp_incline")
    joints, labels = _ramp_plates(b, hinge_x, deck, bed, y)
    return joints, ["ramp_incline", "ramp_level", *labels]


def _ramp_plates(b: newton.ModelBuilder, hinge_x: float, deck: float, bed: float, y: float):
    width = (RAMP_W - (PLATES - 1) * PLATE_GAP) / PLATES
    mass = STEEL * PLATE_L * width * PLATE_T
    q0 = math.asin((deck - bed - PLATE_T - 0.002) / PLATE_L)            # tip 2 mm over the floor; + turns it down
    inertia = _plate_inertia(mass, PLATE_L, width, PLATE_T)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=tr.MU_STRUCTURE)
    joints, labels = [], []
    for k in range(PLATES):
        yk = y - RAMP_W / 2 + width / 2 + k * (width + PLATE_GAP)
        pin = wp.transform(wp.vec3(hinge_x, yk, deck), wp.quat_identity())
        body = b.add_link(xform=wp.transform_multiply(pin, wp.transform(wp.vec3(0.0, 0.0, 0.0),
                                                                        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0),
                                                                                                q0))),
                          mass=mass, com=wp.vec3(PLATE_L / 2, 0.0, -PLATE_T / 2), inertia=wp.mat33(*inertia.ravel()),
                          lock_inertia=True, label=f"ramp_lip_{k}")
        b.add_shape_box(body, xform=body_xform(PLATE_L / 2, 0.0, -PLATE_T / 2), hx=PLATE_L / 2, hy=width / 2,
                        hz=PLATE_T / 2, cfg=cfg, label=f"ramp_lip_{k}")
        for side in (-1.0, 1.0):                        # the toe's corners: MuJoCo Warp lost a thin plate's box-box
            b.add_shape_sphere(body, xform=body_xform(PLATE_L - PLATE_T / 2, side * (width / 2 - PLATE_T / 2),
                                                      -PLATE_T / 2), radius=PLATE_T / 2, cfg=cfg)   # toe contact
        j = b.add_joint_revolute(-1, body, parent_xform=pin, axis=(0.0, 1.0, 0.0), limit_lower=-math.pi / 2,
                                 limit_upper=math.asin(PLATE_TRAVEL / PLATE_L), limit_ke=STOP_KE, limit_kd=STOP_KD,
                                 friction=PLATE_FRICTION, label=f"ramp_lip_{k}")
        b.joint_q[b.joint_q_start[j]] = q0
        b.add_articulation([j], label=f"ramp_lip_{k}")
        joints.append(j)
        labels.append(f"ramp_lip_{k}")
    return joints, labels


def ramp_rest(rear_x: float) -> tr.Loads:
    """The yard trailer's extra static load: each plate pinned at the ramp and resting at its tip, half its weight
    (F/dock.py:342-347)."""
    width = (RAMP_W - (PLATES - 1) * PLATE_GAP) / PLATES
    weight = PLATES * STEEL * PLATE_L * width * PLATE_T * G
    return ((0.5 * weight, PLATE_L - tr.REAR_FRAME - RAMP_BUMPER),)


def add_rack(b: newton.ModelBuilder, front_x: float, y0: float, bays: int = 1) -> list[str]:
    """A single-row rack: its front beams' faces at ``front_x``, bays along +y from ``y0``, depth toward +x. Returns
    the beam labels (front and back, per level and bay)."""
    pitch = RACK_BEAM + RACK_UPRIGHT
    labels = []
    for k in range(bays + 1):
        yk = y0 + k * pitch
        for x in (front_x, front_x + RACK_DEPTH - RACK_UPRIGHT):
            label = f"rack_upright_{k}_{int(x > front_x)}"
            _static_box(b, label, (x, yk - RACK_UPRIGHT, 0.0), (x + RACK_UPRIGHT, yk, RACK_H))
    for k in range(bays):
        y_lo = y0 + k * pitch
        for level in RACK_LEVELS:
            for side, x in (("front", front_x), ("back", front_x + RACK_DEPTH - RACK_BEAM_W)):
                label = f"rack_beam_{k}_{level:.2f}_{side}"
                _static_box(b, label, (x, y_lo, level - RACK_BEAM_H), (x + RACK_BEAM_W, y_lo + RACK_BEAM, level))
                labels.append(label)
    return labels


__all__ = ["BUMPER", "CAPACITY", "DECK_L", "DOCK_H", "DRIVABLE", "RESTING", "LIP_L", "RACK_LEVELS", "add_dock",
           "add_leveler", "add_rack", "add_yard_ramp", "deployed", "dock_rest", "dock_spot", "lip_tip_load",
           "ramp_rest"]
