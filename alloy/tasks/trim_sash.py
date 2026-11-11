# SPDX-License-Identifier: Apache-2.0
"""The angled sash brush on the surface-flow stores, for cut-in-trim: its metering store, its pressed footprint on the
wall, and a kinematic script (loading from the cut pail, strokes along a line).

The brush cuts in a band as wide as itself: its chisel edge across the stroke, the pressed tips a ``PATCH`` long
contact along it. The footprint is a rectangle ``WIDTH`` across the stroke and ``PATCH`` along it. While it slides,
the store meters every cell under it toward Gym-Anything's brushed film (WP15's ``Store(meter=C)``,
``PT/brush.py:381-391``)::

    h_m = min(L_c · sqrt(C · η · U / w), h_max)

laying from its free load where a cell is thinner and taking back up where it is thicker. ``L_c`` is the contact's
length along the slide (``PATCH``), ``U`` the slide speed, ``w`` the line load (the pad's spring load over the
contact's width) and ``η`` the paint's viscosity at the brush's shear ``U / 50 µm`` (``PT/brush.py:387``).

Values (Gym-Anything's sash and brush paint, ``PT/params.py:147-189``, unless marked):
- the Purdy XL Glide 2.5 in angled sash: 63.5 x 15.9 mm, 74.6 mm out, 0.10 kg; 24 x 6 tufts of tip stiffness
  100 N/m each: the pad's stiffness is their density over the head's section (``TUFT_DENSITY``, 1.43e7 Pa/m), an
  implicit pad, spread over the 20 mm contact (1.26x the head's section: 3.44 N at 0.19 mm is 1.26x the 144 tufts at
  that press; the film does not change, as the press is fitted);
- the metering constant C 0.075 and the cap h_max 150 µm;
- ``η``: Gym-Anything's latex Cross law (``PT/params.py:36``) at the brush's shear, 0.18 Pa·s at 0.1 m/s
  (:func:`meter_viscosity`), a fixed value on the store: the film's own viscosity is the low-shear Bingham value
  that sets sag (7.25 Pa·s); a Cross law on the film material is WP15's v2;
- the press: calibrated (``PRESS``, 0.19 mm, a 3.4 N load) so the metered film is ``FIRST_LAY`` 100 µm wet, the middle
  of Gym-Anything's E7 brushed film (80-115 µm, the Rhoplex SG-10M brush-out; ``PT/brush.py:38``,
  ``PT/params.py:166-168``), on which Gym-Anything calibrated C itself. 3.4 N is not Gym-Anything's press: with its
  own contact length (one tuft's strip, ``PT/brush.py:72, 381``) and line load (``:385``), the same C and η meter
  100 µm at 0.26-0.66 N. So C, ``L_c`` (``PATCH``) and the press are one fitted number here; only the law's
  ``sqrt(η·U)`` speed and viscosity dependence is Gym-Anything's;
- the store: 0.25 of the wetted pores never released, 0.75 held after withdrawal, pick-up 2/s, drip 0.5/s;
- capacity: the pores of the bristles dipped a third of their length (painters' guidance, an ESTIMATE) at 0.5 pore
  fraction (ESTIMATE: 6,642 filaments of 0.20-0.31 mm fill 0.2-0.5 of the head's section): 12.6 mL;
- the contact ``PATCH``: 20 mm along the stroke, the 15.9 mm head's tips pressed (ESTIMATE).

A load lays its free share at ``FIRST_LAY`` over ``WIDTH``: about 1 m of band. The script reloads after
``RUN_SHARE`` of it (:func:`run_length`), before the store runs short and the coat thins. Not modelled: the stipple a
lifting brush leaves (``PT/brush.py:464-470``) and the bristle mechanics behind ``L_c``; the band's edge is the
footprint's straight edge, so ``protected`` tests where the script puts the brush's edge.

Until a robot holds it, the brush is a kinematic body put at each period's pose (:func:`place`; SOFT-3's
``paint_brush.place`` sets a yaw rate as well; ``paint_spray_supply.place_can`` keeps a can upright).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WIDTH, THICK, LENGTH = 0.0635, 0.0159, 0.0746          # m (Gym-Anything's sash)
DIP = LENGTH / 3.0                                     # m dipped (ESTIMATE)
PORES = 0.5                                            # ESTIMATE
CAPACITY = WIDTH * THICK * DIP * PORES                 # m³, 12.6 mL
PATCH = 0.02                                           # m along the stroke (ESTIMATE)
RETAINED, HELD, SPLIT, PICKUP, DRIP = 0.25, 0.75, 0.5, 2.0, 0.5
FOOTPRINT = WIDTH * PATCH
TUFT_DENSITY = 100.0 * 24 * 6 / (WIDTH * THICK)        # Pa/m: Gym-Anything's tip stiffness per tuft over the head
METER_C, METER_MAX = 0.075, 150e-6                     # Gym-Anything's sash metering constant and film cap
SHEAR_GAP = 50e-6                                      # m: the brush's shear is U over this (PT/brush.py:383)
CROSS = (0.1796, 2.258, 0.03089, 0.676)                # latex eta_inf, eta0 (sheared), gamma_c, n (PT/params.py:36)
FIRST_LAY = 100e-6                                     # m wet: the metered film the press is set for (E7)
RUN_SHARE = 0.8                                        # of a load's run before the script reloads
DT = 0.02                                              # s, a control period
SPEED = 0.1                                            # m/s along a stroke (2 mm a period: one cell)
TRAVEL = 0.4                                           # m/s between strokes (kinematic, off the wall)
STANDOFF = 0.03                                        # m off the wall before and after a stroke
DIP_S, HOLD_S = 2.5, 10.0                              # s in the paint; held over the pail while it drips back
BOOKS_REL = 1e-12                                      # ledger drift relative to the paint picked up
MASS = 0.10                                            # kg (Gym-Anything's sash)


def head_half() -> tuple[float, float, float]:
    """Half extents of the brush's head as the body's mass shape: along the chisel (across the stroke), through the
    head (along it), out [m]."""
    return 0.5 * WIDTH, 0.5 * THICK, 0.5 * LENGTH


def head_density() -> float:
    h = head_half()
    return MASS / (8.0 * h[0] * h[1] * h[2])


def meter_viscosity(speed: float | None = None) -> float:
    """Gym-Anything's latex Cross viscosity [Pa·s] at the brush's shear, ``speed / 50 µm`` (default: ``SPEED``)."""
    eta_inf, eta0, gamma_c, n = CROSS
    rate = (SPEED if speed is None else speed) / SHEAR_GAP
    return eta_inf + (eta0 - eta_inf) / (1.0 + (rate / gamma_c) ** n)


def metered_film(press: float, speed: float | None = None) -> float:
    """The film [m] the law meters at ``press`` [m into the wall] and ``speed``: the pad's spring load
    ``TUFT_DENSITY · press · FOOTPRINT`` over ``WIDTH`` is the line load, ``PATCH`` the contact's length."""
    u = SPEED if speed is None else speed
    line = TUFT_DENSITY * press * FOOTPRINT / WIDTH
    return min(PATCH * math.sqrt(METER_C * meter_viscosity(u) * u / line), METER_MAX)


def press_for(film: float = FIRST_LAY, speed: float | None = None) -> float:
    """The press [m] at which the law meters ``film`` [m] (below the cap)."""
    u = SPEED if speed is None else speed
    line = METER_C * meter_viscosity(u) * u * (PATCH / film) ** 2
    return line * WIDTH / (TUFT_DENSITY * FOOTPRINT)


PRESS = press_for()                                    # m into the wall: 0.19 mm, a 3.4 N load


def run_length() -> float:
    """How far [m] the script lets a loaded brush run: ``RUN_SHARE`` of its free load laid ``FIRST_LAY`` thick over
    ``WIDTH``."""
    return RUN_SHARE * (HELD - RETAINED) * CAPACITY / (WIDTH * FIRST_LAY)


def sash_store(paint_name: str, initial: float = 0.0):  # noqa: ANN201 - surface_flow.stores.Store
    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415
    from alloy.tasks import paint_process as pp  # noqa: PLC0415

    return Store(capacity=CAPACITY, retained=RETAINED, held=HELD, split=SPLIT, pickup=PICKUP, drip=DRIP,
                 initial=initial, meter=METER_C, meter_max=METER_MAX, meter_viscosity=meter_viscosity(),
                 species=pp.source_species(paint_name))  # fmt: skip


def sash(paint_name: str, chart: int, body: str, initial: float = 0.0):  # noqa: ANN201 - surface_flow Applicator
    """The pressed head as a rect on ``chart``: the underside's centre at the body's origin, its normal along the body's
    -z, ``WIDTH`` along the body's x and ``PATCH`` along its y (a stroke along the body's y); an implicit pad at the
    tufts' stiffness density; ``initial``: the store's loaded share at the start."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415

    return Applicator(body=body, shape="rect", size=(0.5 * WIDTH, 0.5 * PATCH), axis=(1.0, 0.0, 0.0), chart=chart,
                      stiffness=TUFT_DENSITY, implicit=True, store=sash_store(paint_name, initial))  # fmt: skip


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray             # the underside's centre [m, world]
    quat: tuple[float, float, float, float]
    vel: np.ndarray             # m/s


def wall_quat(turn: float) -> tuple[float, float, float, float]:
    """(x, y, z, w): the body's -z into the wall (+y), its y up, turned ``turn`` [rad] about the wall's normal (0: a
    stroke up or down the wall, π/2: along it)."""
    s = math.sqrt(0.5)
    c, t = math.cos(0.5 * turn), math.sin(0.5 * turn)
    return (s * c, -s * t, s * t, s * c)          # R_x(90°) · R_z(turn)


def on_wall(x: float, z: float, gap: float, face_y: float) -> np.ndarray:
    """The underside's centre ``gap`` [m] off the wall face (< 0: pressed into it) at wall point (x, z)."""
    return np.array([x, face_y - gap, z])


def line(a, b, speed: float, quat: tuple[float, float, float, float]) -> list[Pose]:  # noqa: ANN001
    """Poses from ``a`` to ``b`` at ``speed``, one a period (``a`` excluded)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / (speed * DT) - 1e-9)))
    vel = (b - a) / (n * DT)
    return [Pose(a + (b - a) * k / n, quat, vel) for k in range(1, n + 1)]


def hold(p, seconds: float, quat: tuple[float, float, float, float]) -> list[Pose]:  # noqa: ANN001
    return [Pose(np.asarray(p, np.float64), quat, np.zeros(3))] * int(round(seconds / DT))


def load(at, pail_floor, level: float) -> list[Pose]:  # noqa: ANN001
    """From ``at`` over the pail, down until the underside is 10 mm under the paint (``level`` [m] over the pail's
    floor) for ``DIP_S``; up 0.2 m and still for ``HOLD_S`` while the load past the held share drips back."""
    q = wall_quat(0.0)
    x, y, z = pail_floor
    over, low = (x, y, z + level + 0.2), (x, y, z + max(level - 0.01, 0.002))
    return [*line(at, over, TRAVEL, q), *line(over, low, TRAVEL, q), *hold(low, DIP_S, q),
            *line(low, over, TRAVEL, q), *hold(over, HOLD_S, q)]  # fmt: skip


def stroke(at, start: tuple[float, float], end: tuple[float, float], face_y: float, press: float = PRESS) -> list[Pose]:
    """From ``at`` to ``STANDOFF`` off the wall at ``start`` (x, z), straight in until pressed ``press`` into it, along
    the wall to ``end`` at ``SPEED``, and straight out. Moving in and out along the wall's normal it does not slide,
    and a metering store meters only while sliding, so it lays only along the stroke (under its whole contact)."""
    q = wall_quat(0.0 if abs(end[0] - start[0]) < abs(end[1] - start[1]) else 0.5 * math.pi)
    off_a, on_a = on_wall(*start, STANDOFF, face_y), on_wall(*start, -press, face_y)
    on_b, off_b = on_wall(*end, -press, face_y), on_wall(*end, STANDOFF, face_y)
    return [*line(at, off_a, TRAVEL, q), *line(off_a, on_a, TRAVEL, q), *line(on_a, on_b, SPEED, q),
            *line(on_b, off_b, TRAVEL, q)]  # fmt: skip


def brush_books(read: dict, tray0: float, world: int = 0, rel: float = BOOKS_REL):  # noqa: ANN201 - paint_metrics.Check
    """The film's ledger (with what the tools took and laid), the tools' stores' (load + dripped - picked - taken from
    the film) and the trays' (paint + picked = ``tray0``, their start volume [m³]) close, relative to the paint
    picked up. Any store tool: the sash, SOFT-3's round brush."""
    from alloy.tasks.paint_metrics import Check  # noqa: PLC0415

    def one(key: str) -> float:
        return float(np.asarray(read[key]).reshape(-1)[world])

    picked = float(np.asarray(read["tool_picked"])[world].sum())
    drift = max(abs(one("ledger")), abs(one("store_ledger")), abs(one("tray_ledger") - tray0)) / max(picked, 1e-15)
    return Check("books", drift, rel, drift <= rel)


def place(sim, body: int, pose: Pose) -> None:  # noqa: ANN001 - alloy Simulation
    """Put the kinematic brush at ``pose`` for the next period."""
    q = sim.state.body_q.numpy().copy()
    q[body] = (*pose.pos, *pose.quat)
    sim.state.body_q.assign(q)
    qd = sim.state.body_qd.numpy().copy()
    qd[body] = (*pose.vel, 0.0, 0.0, 0.0)
    sim.state.body_qd.assign(qd)


__all__ = ["CAPACITY", "CROSS", "DT", "FIRST_LAY", "MASS", "METER_C", "METER_MAX", "PATCH", "PRESS", "Pose", "SPEED",
           "TUFT_DENSITY", "WIDTH", "brush_books", "head_density", "head_half", "hold", "line", "load",
           "meter_viscosity", "metered_film", "on_wall", "place", "press_for", "run_length", "sash", "sash_store",
           "stroke", "wall_quat"]
