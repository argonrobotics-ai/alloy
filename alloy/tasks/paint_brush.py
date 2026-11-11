# SPDX-License-Identifier: Apache-2.0
"""The real round brush on the surface-flow stores: Gym-Anything's round #8 as a pointed disc tip on a metering store,
the palette's red well, and a kinematic script that loads it and paints the pencilled S, swelling from 2 mm to 6 mm
and back by pressing.

The tip is a level 4.5 mm disc whose underside rises ``TIP_SLOPE`` x the distance from its centre (the plug-in's
pointed round tip): pressed ``d`` into the paper it touches a disc of radius ``d/TIP_SLOPE``, so at a slope of 2 its
band is as wide as its press, at any heading (Gym-Anything's #8: 2 mm lines at a 1-2 mm press, 6 mm at 6 mm, 9 mm at
9 mm). The store meters the film by Gym-Anything's law ``h_m = min(L_c·sqrt(C·eta·U/w), 150 µm)`` over the whole
touching patch while it slides: ``L_c`` the patch's area over its width, ``w`` the pad's force over that width.
On this model's cone both grow together, so ``h_m = sqrt(6π·s·C·eta·U/k)/(2s)`` (:func:`metered_film`) whatever the
press: a property of the cone, not of Gym-Anything's tufts. ``eta`` is the acrylic's viscosity at the stroke's shear,
as Gym-Anything meters (0.34 Pa s, :func:`metering_viscosity`). The press sets the width (:func:`press_for`); the
speed sets the film. On 1 mm cells the patch is whole cells, so along a press ramp the film saw-tooths about that.

The metering constant is mapped, not copied. Gym-Anything calibrated its round #8's ``C`` (0.60) with its own
``L_c`` and ``w``: per tuft ``b·(1 + 2 sin a)`` with ``b`` 0.97 mm, and the brush's force, each tuft's capped by its
bending, over the touching tufts' widths (brush.py:72, 194, 381-385). Those follow its tufts' bend and contact and
have no closed form on a cone; on its own law at a full press they give 52-117 µm. So ``C`` here is the one that
lays the film Gym-Anything calibrated its ``C`` to, a loaded round's 40-60 µm wet (params.py:169-170): its middle,
50 µm. ``C = C_GA·(50 µm / h_m(C_GA))²`` = 1.33.

The paint lies under the whole patch, so it reaches ``width/2`` past the centre where a stroke ends: the script's
strokes stop short of the line's ends by that (:func:`end_inset`).

Until a robot holds it, the brush is a kinematic body (:func:`place`) put at each period's pose. Callers build poses
with :func:`stroke_pose` (a point, the stroke's direction, the width and the store's load).

Sources (Gym-Anything's round #8 and brush paint, ``PT/params.py:153-189``):
- the hair: 5 mm wide at the ferrule, 22.5 mm long, splaying to 9 mm at a 9 mm press (e-David, Guelzow 2024);
- the tip's stiffness: 20 N/m a tuft (Gym-Anything's kc, an estimate), its 21 tufts in the 5 mm circle (tools.py:
  153-155) over π/4 x 25 mm² (brush.py:71): 2.14e7 Pa/m;
- the metering constant: Gym-Anything's round #8's 0.60 (params.py:155), mapped to this tip (above); the film capped
  at 150 µm;
- the store: 0.25 never released, 0.75 held when lifted, pick-up 2/s, drip 0.5/s, capacity 0.276 mL (Gym-Anything's
  round brush, brush.py:536-553).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from alloy.tasks import paint_metrics as pm
from alloy.tasks import paint_process as pp

TIP_R = 4.5e-3                  # m: the tip disc's radius (the #8 splays to 9 mm)
TIP_SLOPE = 2.0                 # the pointed underside's rise per m from the centre: the band as wide as the press
KC, TUFTS = 20.0, 21            # N/m a tuft, and tufts in the 5 mm circle (Gym-Anything's round #8)
STIFFNESS = KC * TUFTS / (0.25 * math.pi * 5e-3 * 5e-3)   # Pa/m: 2.14e7
METER_GA = 0.60                 # Gym-Anything's round #8 metering constant (params.py:155)
FILM_GA = 50e-6                 # m wet: the middle of the 40-60 µm Gym-Anything calibrated it to (params.py:169-170)
METER_MAX = 150e-6              # m: the metered film's cap (Gym-Anything's h_max)
CAPACITY = 0.276e-6             # m³ (Gym-Anything's round brush)
RETAINED, HELD, PICKUP, DRIP = 0.25, 0.75, 2.0, 0.5
BAND_MAX = 2.0 * TIP_R          # m: the widest band the tip lays across a stroke (fully splayed)
SPEED = 0.05                    # m/s along the stroke
DT = 0.02                       # s: a control period (1 mm a step at SPEED, a cell)
WELL = (0.02, 0.02)             # m: the palette well's floor, half extents (Gym-Anything's: round, r 20 mm)
WELL_PAINT = 9.0e-6             # m³ (Gym-Anything's palette well: 5.6 mm deep here)


def metering_viscosity(speed: float = SPEED) -> float:
    """Pa·s: the fluid acrylic's viscosity where Gym-Anything meters a brush's film, its Cross law at the shear
    ``speed / 50 µm`` with no structure (brush.py:387, paint.py:98-101; ACRYLIC, params.py:53-55): eta_inf 0.30,
    eta0 1.0, gamma_c 1/s, n 0.4. 0.34 Pa·s at the stroke's 0.05 m/s; the paint card's 0.70 is its low-shear value.
    For this task's acrylic only."""
    eta_inf, eta0, gamma_c, n = 0.30, 1.0, 1.0, 0.4
    return eta_inf + (eta0 - eta_inf) / (1.0 + (speed / 50e-6 / gamma_c) ** n)


def metered_film(meter: float, speed: float = SPEED) -> float:
    """m wet: Gym-Anything's metering law on this tip's cone at any press, ``sqrt(6π·s·C·eta·U/k)/(2s)``."""
    s = TIP_SLOPE
    return math.sqrt(6.0 * math.pi * s * meter * metering_viscosity(speed) * speed / STIFFNESS) / (2.0 * s)


METER = METER_GA * (FILM_GA / metered_film(METER_GA)) ** 2       # 1.33: lays FILM_GA at SPEED on this tip


def brush_store(paint_name: str, initial: float = 0.0):  # noqa: ANN201 - surface_flow.stores.Store
    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415

    return Store(capacity=CAPACITY, retained=RETAINED, held=HELD, pickup=PICKUP, drip=DRIP, initial=initial,
                 meter=METER, meter_max=METER_MAX, meter_viscosity=metering_viscosity(),
                 species=pp.source_species(paint_name))  # fmt: skip


def brush(paint_name: str, initial: float = 0.0, body: str = "brush"):  # noqa: ANN201 - Applicator
    """The tip as a pointed disc applicator on chart 0: the underside's point at the body's origin, normal along -z.
    The pad is implicit (the hair's stiffness on a light tip is stable at any exchange step); the metering reads its
    spring load."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415

    return Applicator(body=body, shape="disc", size=(TIP_R, TIP_R), stiffness=STIFFNESS, tip_slope=TIP_SLOPE,
                      implicit=True, store=brush_store(paint_name, initial))  # fmt: skip


def palette_well(center: tuple[float, float, float]):  # noqa: ANN201 - surface_flow.stores.Tray
    from alloy.plugins.surface_flow.stores import Tray  # noqa: PLC0415

    return Tray(center=center, half=WELL, volume=WELL_PAINT)


def band_width(press: float, load: float = 0.0) -> float:  # noqa: ARG001 - the load no longer matters
    """The touching band's width across the stroke [m] at ``press`` (the tip's point into the paper)."""
    return min(2.0 * max(press, 0.0) / TIP_SLOPE, BAND_MAX)


def press_for(width: float, load: float = 0.0) -> float:  # noqa: ARG001 - the load no longer matters
    """How far [m] to press the tip's point into the paper for a band ``width`` wide."""
    return 0.5 * width * TIP_SLOPE


def end_inset(width: float) -> float:
    """How far [m] short of a line's end the tip's centre stops for a band ``width`` wide there: the paint lies under
    the whole touching disc, ``width/2`` past its centre."""
    return 0.5 * min(width, BAND_MAX)


@dataclass(frozen=True)
class BrushPose:
    pos: np.ndarray             # the tip's point [m, world]
    quat: tuple[float, float, float, float]
    vel: np.ndarray
    yaw_rate: float = 0.0       # rad/s (a round tip does not turn: 0)


LEVEL = (0.0, 0.0, 0.0, 1.0)    # (x, y, z, w): the tip level, pointing down


def tip_pose(point: np.ndarray, tangent: np.ndarray, width: float, press: float, z: float) -> BrushPose:  # noqa: ARG001
    """The pose that puts the tip's point on ``point`` (paper frame, on the paper at height ``z``) ``press`` into the
    paper, moving along ``tangent``. (``width`` is the band's; the point is its centre whatever the width.)"""
    t = np.asarray(tangent, np.float64) / np.linalg.norm(tangent)
    return BrushPose(np.array([point[0], point[1], z - press]), LEVEL, SPEED * np.array([t[0], t[1], 0.0]))


def _line(a, b, speed: float) -> list[BrushPose]:  # noqa: ANN001
    """Poses from ``a`` to ``b`` (the tip's point) at ``speed``, one a period."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / (speed * DT))))
    vel = (b - a) / (n * DT)
    return [BrushPose(a + (b - a) * k / n, LEVEL, vel) for k in range(1, n + 1)]


def _hold(p, seconds: float) -> list[BrushPose]:  # noqa: ANN001
    return [BrushPose(np.asarray(p, np.float64), LEVEL, np.zeros(3))] * int(round(seconds / DT))


def load_poses(well: tuple[float, float, float], dip: float = 2.5, hold: float = 6.0) -> list[BrushPose]:
    """Down into the well (the tip 1 mm over its floor, under the paint) for ``dip`` s: the store fills at its
    pick-up rate; up 30 mm and still for ``hold`` s, while the load past the held share drips back."""
    above, low = (well[0], well[1], well[2] + 0.05), (well[0], well[1], well[2] + 0.001)
    up = (well[0], well[1], well[2] + 0.03)
    return [*_line(above, low, 0.05), *_hold(low, dip), *_line(low, up, 0.05), *_hold(up, hold)]


def approach_poses(well: tuple[float, float, float], board_z: float,
                   start: tuple[np.ndarray, np.ndarray, float] | None = None) -> list[BrushPose]:
    """From over the well to over ``start`` (point, tangent, width: by default the S's first), then down until the
    tip's point meets the paper one period's travel short of it along the stroke. The store meters only while the tip
    slides, so the stroke's first period, sliding onto ``start``, lays its patch there (landed on it, the paint began
    a period late)."""
    point, tangent, width = start if start is not None else next(stroke_points())
    t = np.asarray(tangent, np.float64) / np.linalg.norm(tangent)
    pose = tip_pose(np.asarray(point) - SPEED * DT * t, t, width, 0.0, board_z)
    over = (pose.pos[0], pose.pos[1], pose.pos[2] + 0.02)
    return [*_line((well[0], well[1], well[2] + 0.03), over, 0.1), *_line(over, pose.pos, 0.02)]


def lift_poses(board_z: float, end: tuple[np.ndarray, np.ndarray, float] | None = None) -> list[BrushPose]:
    """Straight up 30 mm from ``end`` (point, tangent, width: by default the S's last)."""
    if end is None:
        *_, end = stroke_points()
    point, tangent, width = end
    pose = tip_pose(np.asarray(point), np.asarray(tangent), width, 0.0, board_z)
    return _line(pose.pos, pose.pos + (0.0, 0.0, 0.03), 0.05)


def brush_scene(init, board_z: float, paper: tuple[float, float],  # noqa: ANN001, ANN201 - a harness TestScene
                start: tuple[float, float, float]):
    """The paper (its chart's box) and the kinematic brush at ``start``, no robot: XPBD, one world, period ``DT``."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, board_z - 0.005), hx=0.5 * paper[0], hy=0.5 * paper[1],
                        hz=0.005, label="paper")  # fmt: skip
        tool = b.add_body(xform=body_xform(*start), is_kinematic=True, label="brush")
        b.add_shape_sphere(tool, radius=TIP_R, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False))
        dummy = b.add_body(xform=body_xform(2.0, -2.0, 1.0), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=init, worlds=1, periods=1, control_dt=DT, dt=DT, solver="xpbd", ground=False)


def stroke_points(n: int = 2000) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
    """Points along the template's centre line about ``SPEED·DT`` apart (a whole number of periods), with the tangent
    and the template's width, from :func:`end_inset` past its start to as far short of its end."""
    pts, half = pm.s_curve(n)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    first, last = end_inset(2.0 * float(half[0])), arc[-1] - end_inset(2.0 * float(half[-1]))
    for s in np.linspace(first, last, int(math.ceil((last - first) / (SPEED * DT))) + 1):
        k = min(int(np.searchsorted(arc, s)), n - 1)
        a = max(k - 1, 0)
        yield pts[k], pts[min(k + 1, n - 1)] - pts[a], 2.0 * float(half[k])


def place(sim, body: int, pose: BrushPose) -> None:  # noqa: ANN001 - alloy Simulation
    q = sim.state.body_q.numpy().copy()
    q[body] = (*pose.pos, *pose.quat)
    sim.state.body_q.assign(q)
    qd = sim.state.body_qd.numpy().copy()
    qd[body] = (*pose.vel, 0.0, 0.0, pose.yaw_rate)
    sim.state.body_qd.assign(qd)


def stroke_pose(point, tangent, width: float, load: float, z: float,  # noqa: ANN001
                yaw_rate: float = 0.0) -> BrushPose:  # noqa: ARG001 - a round tip does not turn
    """The tip's pose painting a band ``width`` wide centred on ``point`` (x, y on the paper at height ``z``), moving
    along ``tangent``, with ``load`` m³ in the store (the metered film no longer depends on it; the argument stays):
    the press for that width. The stable entry point for strokes; ``yaw_rate`` is kept and ignored (the level round
    tip does not turn)."""
    return tip_pose(np.asarray(point, np.float64), np.asarray(tangent, np.float64), width, press_for(width, load), z)


__all__ = ["BAND_MAX", "CAPACITY", "DT", "FILM_GA", "HELD", "KC", "LEVEL", "METER", "METER_GA", "METER_MAX",
           "RETAINED", "SPEED", "STIFFNESS", "TIP_R", "TIP_SLOPE", "TUFTS", "WELL", "WELL_PAINT", "BrushPose",
           "approach_poses", "band_width", "brush", "brush_scene", "brush_store", "end_inset", "lift_poses",
           "load_poses", "metered_film", "metering_viscosity", "palette_well", "place", "press_for", "stroke_points",
           "stroke_pose", "tip_pose"]
