# SPDX-License-Identifier: Apache-2.0
"""The real roller on the surface-flow stores: Gym-Anything's 9 in x 3/8 in cover, its store, the tray's well, and a
kinematic script that loads it, holds it over the tray, rolls the strip's lanes and puts it back.

Until a robot holds it, the cover is a kinematic body (:func:`place`), put at each sub-period's pose with the twist of
rolling without slip on the surface it touches: the tray's floor or the wall. The film reads the cover's pose and
twist; the store lays and takes paint by its nip split, fills below the tray's paint level and drips what it holds
past its held share when lifted (``alloy.plugins.surface_flow.stores``).

Sources (Gym-Anything's roller, ``PT/params.py:122-133``):
- the cover: 0.2286 m long, a 42.4 mm core and a 9.5 mm nap, OD 61.4 mm (Wooster cover listings);
- its capacity: pore fraction 0.9 (estimate) x the nap's volume x 0.154 saturation (BASF, PCI Magazine 2018: ~40 g in
  a 4 in cover's 222 mL nap): 58 mL;
- the store keeps 0.25 of it for good and 0.75 when lifted (design note §7; Gym-Anything gives these to its brush),
  picks up at 1.5/s; the nip splits 0.5 (Walker and Fetsko 1955, Coyle et al. 1986); the drip 0.5 1/s (Gym-Anything's
  estimate);
- the nip gets only what the pressed nap expresses: 2 x its strain of the free load (Gym-Anything's BETA, calibrated
  so a loaded cover at 10 N lays ~100 µm a first pass, the ProMar 200 data sheet's 4 mil). The store's ``express``
  share takes it, one value per pass from the pass's press (:func:`express`). The press, 1.5 mm (0.32), is about
  19 N on Gym-Anything's nap law, not its 10 N point (0.98 mm, 0.21). Gym-Anything's roller keeps no retained share
  and drips only above 1.15x its saturated load, so its wall sees twice the free load of this store; the deeper
  press restores its ~100 µm first pass. Its 10 N point passes the full wall too (dry 25.7-34.9 µm, ΔE 0.136);
- the tray: a Wooster R404, its well 0.279 x 0.12 m (an estimate of the well/ramp split), 1 L of paint.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from alloy.tasks import paint_process as pp
from alloy.tasks.paint_scene import TRAY_SIZE, WallLayout

R, HW = 0.5 * 0.0614, 0.5 * 0.2286          # m: the cover's radius and half length
NAP, POROSITY, SATURATION = 0.0095, 0.9, 0.154
NAP_AREA = 2.0 * math.pi * R * 2.0 * HW     # m²
CAPACITY = POROSITY * NAP * NAP_AREA * SATURATION       # m³, 58 mL
RETAINED, HELD, SPLIT, PICKUP, DRIP = 0.25, 0.75, 0.5, 1.5, 0.5
SUB = 0.0125                 # s: a sub-period; at 0.25 m/s the cover turns 3.1 mm a step, under a bin's 6.0 mm arc
SPEED = 0.25                 # m/s on the wall
PRESS = 1.5e-3               # m: the nap pressed into the wall or the tray's floor (strain 0.16)
EXPRESS_PER_STRAIN = 2.0     # Gym-Anything's BETA
WELL = (0.279, 0.12)         # m: the well's floor across the tray (x) and along it (y)
TRAY_PAINT = 1.0e-3          # m³ in the well


def express(press: float = PRESS) -> float:
    """The share of the cover's free load the nip gets at ``press`` [m] into the nap: 2 x the nap's strain."""
    return min(1.0, EXPRESS_PER_STRAIN * press / NAP)


def roller_store(paint_name: str, initial: float = 0.0, press: float = PRESS):  # noqa: ANN201 - stores.Store
    """The cover's store: Gym-Anything's capacity, pick-up and drip, the retained and held shares, the 0.5 split and
    the nip's expressed share at ``press``; its load carries the paint's solids at φ0."""
    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415

    return Store(capacity=CAPACITY, retained=RETAINED, held=HELD, split=SPLIT, pickup=PICKUP, drip=DRIP,
                 initial=initial, express=express(press), species=pp.source_species(paint_name))  # fmt: skip


def roller(paint_name: str, initial: float = 0.0, body: str = "cover"):  # noqa: ANN201 - surface_flow Applicator
    """The cover as an applicator on chart 0, its axis along the body's x."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415

    return Applicator(body=body, shape="roller", size=(R, HW), axis=(1.0, 0.0, 0.0), stiffness=1.0e4,
                      store=roller_store(paint_name, initial))  # fmt: skip


def tray_well(layout: WallLayout, volume: float = TRAY_PAINT):  # noqa: ANN201 - surface_flow.stores.Tray
    """The tray's well as the store's pool: its floor at the tray's bottom, at the tray's end toward the room (-y),
    the tray's long side along y."""
    from alloy.plugins.surface_flow.stores import Tray  # noqa: PLC0415

    return Tray(center=well_floor(layout), half=(0.5 * WELL[0], 0.5 * WELL[1]), volume=volume)


def well_floor(layout: WallLayout) -> tuple[float, float, float]:
    x, y, z = layout.tray_pos
    return (x, y - 0.5 * TRAY_SIZE[0] + 0.5 * WELL[1], z + 0.002)


@dataclass(frozen=True)
class Move:
    """A straight move of the cover's axis centre to ``to`` [m, world] at ``speed`` [m/s], rolling on ``roll``
    ("wall", "tray") or turning freely in the air (None, it does not turn), then ``hold`` s still. A ``transfer`` is a
    long move in the air between the tray and the wall: the kinematic cover goes straight, a robot may take any path
    in the same time."""

    to: tuple[float, float, float]
    speed: float = SPEED
    roll: str | None = None
    hold: float = 0.0
    transfer: bool = False


CONTACT = {"wall": (0.0, 1.0, 0.0), "tray": (0.0, 0.0, -1.0)}     # from the axis to the contact line, world


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray             # the axis centre [m]
    vel: np.ndarray             # [m/s]
    angle: float                # about the cover's axis (world x) [rad]
    spin: float                 # [rad/s]
    roll: str | None            # what it rolls on this sub-period
    transfer: bool = False      # part of a transfer move (a robot may take another path)


def poses(start: tuple[float, float, float], moves: list[Move], dt: float = SUB) -> Iterator[Pose]:
    """The cover's pose at the start of each sub-period along ``moves``, and its twist over the sub-period. Rolling
    without slip on a surface with contact direction n: v + ω x (R n) = 0, so ω_x = -v·(x x n)/R."""
    pos, angle = np.asarray(start, np.float64), 0.0
    for m in moves:
        to = np.asarray(m.to, np.float64)
        n = max(1, int(math.ceil(float(np.linalg.norm(to - pos)) / (m.speed * dt) - 1e-9)))
        vel = (to - pos) / (n * dt)
        spin = 0.0 if m.roll is None else -float(vel @ np.cross((1.0, 0.0, 0.0), CONTACT[m.roll])) / R
        for k in range(n):
            yield Pose(pos + vel * (k * dt), vel, angle, spin, m.roll, m.transfer)
            angle += spin * dt
        pos = to
        for _ in range(int(round(m.hold / dt))):
            yield Pose(pos, np.zeros(3), angle, 0.0, None)


def place(sim, body: int, pose: Pose) -> None:  # noqa: ANN001 - alloy Simulation
    """Put the kinematic cover at ``pose`` (axis along world x, turned by its angle) with its twist (COM velocity
    first, then the angular velocity, world)."""
    q = sim.state.body_q.numpy().copy()
    q[body] = (*pose.pos, math.sin(0.5 * pose.angle), 0.0, 0.0, math.cos(0.5 * pose.angle))
    sim.state.body_q.assign(q)
    qd = sim.state.body_qd.numpy().copy()
    qd[body] = (*pose.vel, pose.spin, 0.0, 0.0)
    sim.state.body_qd.assign(qd)


def axis_on_wall(layout: WallLayout, x: float, y: float) -> tuple[float, float, float]:
    """The axis centre that presses the cover ``PRESS`` into the wall at chart point (x, y)."""
    return (x, layout.face_y - R + PRESS, layout.bottom + y)


def load_moves(layout: WallLayout, turns: float = 1.25, above: float = 0.30, hold: float = 26.0) -> list[Move]:
    """Into the well and forward along its floor, lifting back between strokes, until the cover has turned ``turns``
    in the paint (every bin dips); then up to ``above`` m over the tray and still for ``hold`` s, while the load past
    the held share drips back (one drop's worth left in its last 10 s)."""
    x, y, z = well_floor(layout)
    floor = z + R - PRESS
    back, fore = y - 0.5 * WELL[1] + R, y + 0.5 * WELL[1] - R
    strokes = int(math.ceil(turns * 2.0 * math.pi * R / (fore - back) - 1e-9))
    out = [Move((x, back, z + above), speed=0.5, transfer=True)]
    for _ in range(strokes):
        out += [Move((x, back, floor + 0.01), speed=0.2), Move((x, back, floor), speed=0.1),
                Move((x, fore, floor), speed=0.1, roll="tray"), Move((x, fore, floor + 0.01), speed=0.1)]
    return out + [Move((x, back, z + above), speed=0.2, hold=hold)]


def lane_moves(layout: WallLayout, x: float, passes: int = 3, margin: float | None = None) -> list[Move]:
    """One lane at ``x`` (chart frame): off the wall to its bottom, onto the wall, ``passes`` strokes up and down
    between the strip's lines (``margin`` inside them; by default half a cell past them, into the cut-in bands, so
    the nip crosses the strip's edge cells), and off the wall at the end."""
    s = layout.strip
    margin = -0.5 * layout.cell if margin is None else margin
    lo, hi = axis_on_wall(layout, x, s.y0 + margin), axis_on_wall(layout, x, s.y1 - margin)
    off = (lo[0], lo[1] - 0.05, lo[2])
    out = [Move(off, speed=0.5, transfer=True), Move(lo, speed=0.1)]
    for k in range(passes):
        out.append(Move(hi if k % 2 == 0 else lo, roll="wall"))
    end = out[-1].to
    return out + [Move((end[0], end[1] - 0.05, end[2]), speed=0.1)]


def lanes(layout: WallLayout) -> list[float]:
    """Lane centres across the strip: as few as cover it, the last one ending at its right line."""
    s = layout.strip
    n = int(math.ceil((s.x1 - s.x0) / (2.0 * HW) - 1e-9))
    return [float(v) for v in np.linspace(s.x0 + HW, s.x1 - HW, n)] if n > 1 else [0.5 * (s.x0 + s.x1)]


def script(layout: WallLayout, passes: int = 3) -> tuple[tuple[float, float, float], list[Move]]:
    """The whole job from over the tray: per lane, load, then the lane; then back into the well."""
    x, y, z = well_floor(layout)
    start = (x, y - 0.5 * WELL[1] + R, z + 0.30)
    moves: list[Move] = []
    for lx in lanes(layout):
        moves += load_moves(layout) + lane_moves(layout, lx, passes)
    return start, moves + [Move(start, speed=0.5, transfer=True), Move((start[0], start[1], z + R + 0.02), speed=0.2)]


def rolled(pose: Pose, layout: WallLayout, centres: np.ndarray) -> np.ndarray:
    """The wall cells under the cover's contact this sub-period (the cover on the wall): its width by the cells it
    crosses."""
    if pose.roll != "wall":
        return np.zeros(centres.shape[:2], dtype=bool)
    y = pose.pos[2] - layout.bottom
    reach = 0.5 * layout.cell + abs(float(pose.vel[2])) * SUB
    return (np.abs(centres[..., 0] - pose.pos[0]) <= HW) & (np.abs(centres[..., 1] - y) <= reach)


def roller_scene(layout: WallLayout, init, start: tuple[float, float, float]):  # noqa: ANN001, ANN201 - TestScene
    """The wall (its chart) and the kinematic cover at ``start``, no robot: XPBD, one world, one sub-period a period."""
    import warp as wp  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415
    from alloy.tasks.paint_scene import wall_box  # noqa: PLC0415

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        label, pos, quat, half = wall_box(layout)
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(*pos), wp.quat(*quat)), hx=half[0], hy=half[1], hz=half[2],
                        label=label)  # fmt: skip
        cover = b.add_body(xform=body_xform(*start), is_kinematic=True, label="cover")
        s = math.sqrt(0.5)                                              # the cylinder's z onto the cover's x
        b.add_shape_cylinder(cover, radius=R, half_height=HW, xform=body_xform(0.0, 0.0, 0.0, (0.0, s, 0.0, s)),
                             cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False))  # fmt: skip
        dummy = b.add_body(xform=body_xform(2.0, -2.0, 1.0), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=init, worlds=1, periods=1, control_dt=SUB, dt=SUB, solver="xpbd", ground=False)


def tray_frame(layout: WallLayout, pos) -> tuple[np.ndarray, tuple[float, float]]:  # noqa: ANN001
    """``pos`` in the tray's frame (its bottom's centre) and the tray's half extents (x across, y along it)."""
    x, y, z = layout.tray_pos
    return np.asarray(pos, np.float64) - (x, y, z), (0.5 * TRAY_SIZE[1], 0.5 * TRAY_SIZE[0])


class Kinematic:
    """The roller's script run kinematically: the cover is put at each pose (``run_roller``'s driver)."""

    cover = "cover"

    def __init__(self, layout: WallLayout, passes: int = 3) -> None:
        self.layout = layout
        self.start, moves = script(layout, passes)
        self.steps = list(poses(self.start, moves))
        self.body = -1

    def scene(self, init):  # noqa: ANN001, ANN201 - a harness TestScene
        return roller_scene(self.layout, init, self.start)

    def bind(self, sim) -> None:  # noqa: ANN001
        self.body = list(sim.model.body_label).index(self.cover)

    def apply(self, sim, step: Pose) -> None:  # noqa: ANN001
        place(sim, self.body, step)

    @staticmethod
    def pose(step: Pose) -> Pose:
        return step

    def cover_pos(self, sim) -> np.ndarray:  # noqa: ANN001
        return sim.state.body_q.numpy()[self.body][:3].astype(np.float64)

    @staticmethod
    def report() -> dict:
        return {}


__all__ = ["CAPACITY", "HW", "Kinematic", "Move", "NAP_AREA", "Pose", "R", "SUB", "axis_on_wall", "express",
           "lane_moves", "lanes", "load_moves", "place", "poses", "rolled", "roller", "roller_scene", "roller_store",
           "script", "tray_frame", "tray_well", "well_floor"]
