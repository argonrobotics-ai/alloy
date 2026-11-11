# SPDX-License-Identifier: Apache-2.0
"""Test scenes and laws of the deposition plug-in (engine item: contact-driven deposition, XB4-7 whiteboard gap).

- ``line_density`` (rate against force and speed): a marker stroke at speed v under force F leaves ``flow·F/v`` of
  ink per metre (summed across the stroke, along the steady part), within 3 %, at two speeds and two forces.
- ``stroke_width`` (width against the contact patch): the inked width equals the domed nib's contact diameter
  ``2a``, with ``a = sqrt(2·R·δ0)`` and ``F = k·π·R·δ0²``, within two cells, at both forces.
- ``ink_ledger``: a marker draws while a wiper follows it. Per world, map + reservoirs + sinks stays constant to
  float64 rounding (1e-12); what the wiper removed was laid by the marker.
- ``wipe_lifts_ink``: one wiper pass over uniformly inked board leaves ``exp(-2R/L)`` on the cells under its centre
  line (5 %); the sink holds exactly what left the board.
- ``no_ink_when_lifted``: a marker held 2 mm above the board lays nothing and pushes nothing.

Each tool is a block driven by a ``Carriage`` (speed along +x, y and attitude held, pressed down): a marker on a domed
nib is an inverted pendulum and needs the hold. Each block's rigid shape sits 5 mm above its tip apex, so the solver
never touches the board: only the plug-in's tips do.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.deposition.carriage import Carriage
from alloy.plugins.deposition.tools import tool_specs
from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform

G = 9.81
MARKER = {"tools": ("marker",), "kinds": ("marker",), "colours": (0,), "tip_radius": (0.003,), "dome_radius": (0.003,),
          "stiffness": (1.0e9,), "damping": (1.0e-3,), "friction": (0.0,), "flow": (1.0e-7,), "reservoir": (3.0e-3,),
          "wipe_length": (0.0,), "cells": (480, 160), "saturation": 0.0}
WIPER = {"tools": ("wiper",), "kinds": ("wiper",), "colours": (0,), "tip_radius": (0.015,), "dome_radius": (0.0,),
         "stiffness": (3.0e6,), "damping": (2.0e-3,), "friction": (0.0,), "flow": (0.0,), "reservoir": (0.0,),
         "wipe_length": (0.03,), "cells": (240, 80), "saturation": 0.0}
SPEEDS, PRESSES = (0.05, 0.1, 0.05, 0.1), (1.0, 1.0, 4.0, 4.0)     # per world, line laws [m/s], [N]
SMALL_BOARD = (0.06, 0.02, 0.005)                                  # half extents [m]
BIG_BOARD = (0.1, 0.05, 0.005)
DIRT = 1.0e-9                                                      # kg per cell pre-inked for the wipe law


@dataclass(frozen=True)
class Rig:
    """One tool block: label, half extents, mass, start x, the carriage's speed and press (per world), start y."""

    label: str
    half: tuple[float, float, float]
    mass: float
    x0: float
    speed: tuple[float, ...]
    press: tuple[float, ...]
    lift: float = 0.0
    y0: float = 0.0


def merged(c: Any, overrides: dict[str, Any]) -> Any:
    return dataclasses.replace(c, **overrides)


def bench(c: Any, rigs: list[Rig], board: tuple[float, float, float], *, worlds: int, periods: int,
          dirt: float = 0.0, gravity: tuple[float, float, float] = (0.0, 0.0, -G)) -> TestScene:
    apex = {s.label: s.center[2] for s in tool_specs(c)}

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, board[2]), hx=board[0], hy=board[1], hz=board[2],
                        label=c.board)
        for r in rigs:
            body = b.add_body(xform=body_xform(r.x0, r.y0, 2.0 * board[2] - apex[r.label] + r.lift), label=r.label)
            density = r.mass / (8.0 * r.half[0] * r.half[1] * r.half[2])
            b.add_shape_box(body, hx=r.half[0], hy=r.half[1], hz=r.half[2],
                            cfg=newton.ModelBuilder.ShapeConfig(density=density))

    def companions() -> list[Carriage]:
        return [Carriage(body=r.label, speed=r.speed[0], press=r.press[0], label=f"carriage_{r.label}") for r in rigs]

    def init(sim: Any) -> None:
        n = sim.model.world_count
        for r in rigs:
            carriage = sim.plugins[f"carriage_{r.label}"]
            carriage.set_input("speed", np.resize(np.asarray(r.speed, np.float32), n))
            carriage.set_input("press", np.resize(np.asarray(r.press, np.float32), n))
        if dirt > 0.0:
            for p in sim.plugins.plugins.values():
                if hasattr(p, "set_ink"):
                    p.set_ink(np.full(p.ink.shape[1:], dirt, np.float32))

    return TestScene(build=build, worlds=worlds, periods=periods, ground=False, gravity=gravity,
                     companions=companions, init=init)


def marker_rig(**kw: Any) -> Rig:
    return Rig("marker", (0.01, 0.01, 0.01), 0.1, kw.pop("x0", -0.04), kw.pop("speed", SPEEDS),
               kw.pop("press", PRESSES), **kw)


def marker_and_wiper_scene(c: Any) -> TestScene:
    """A marker draws along the board's centre line at 0.1 m/s while a wiper follows it 50 mm behind."""
    rigs = [Rig("marker", (0.01, 0.01, 0.01), 0.1, -0.03, (0.1,), (1.0,)),
            Rig("wiper", (0.03, 0.03, 0.01), 0.3, -0.08, (0.1,), (3.0,))]
    return bench(c, rigs, BIG_BOARD, worlds=3, periods=25)


def deposition_laws(c: Any) -> list[Law]:
    marker, wiper = merged(c, MARKER), merged(c, WIPER)
    line = bench(marker, [marker_rig()], SMALL_BOARD, worlds=4, periods=30)
    wipe = bench(wiper, [Rig("wiper", (0.02, 0.02, 0.01), 0.2, -0.04, (0.1,), (2.0,))], SMALL_BOARD, worlds=2,
                 periods=30, dirt=DIRT)
    hover = bench(marker, [marker_rig(speed=(0.1,), press=(0.0,), lift=0.002)], SMALL_BOARD, worlds=2, periods=10,
                  gravity=(0.0, 0.0, 0.0))
    return [
        Law("line_density", check_density, line, "ink per metre of stroke = flow·F/v at two speeds and forces",
            config=MARKER),
        Law("stroke_width", check_width, line, "inked width = the domed nib's contact diameter 2a(F)", config=MARKER),
        Law("ink_ledger", check_ledger, None, "map + reservoirs + sinks constant; the wiper holds what it lifted"),
        Law("wipe_lifts_ink", check_wipe, wipe, "one pass leaves exp(-2R/L); the sink holds exactly what left",
            config=WIPER),
        Law("no_ink_when_lifted", check_lifted, hover, "a marker 2 mm above the board lays nothing", config=MARKER),
    ]


# --- checks ------------------------------------------------------------------------------------------------------

def _series(run: Any, name: str) -> np.ndarray:
    return np.asarray(run.trace.plugin(run.plugin.label, name), np.float64)


def _steady(run: Any, w: int, label: str, skip: int = 5) -> tuple[float, float, float, float]:
    """(mean load [N], mean speed [m/s], first and last x of the tip [m]) over the steady periods of world ``w``."""
    p = run.plugin
    t = p.tool_index(label)
    b = run.bodies(label)[w]
    load = _series(run, "load")[skip:, w, t].mean()
    speed = float(run.trace.body_qd[skip:, b, 0].mean())
    x = run.trace.body_q[:, b, 0].astype(np.float64)
    return float(load), speed, float(x[skip]), float(x[-1])


def _columns(run: Any, x0: float, x1: float, margin: float) -> np.ndarray:
    """Map columns whose centres lie in [x0 + margin, x1 - margin] (the nib fully passed them)."""
    p = run.plugin
    nx = p.ink.shape[1]
    xc = (np.arange(nx) + 0.5) * float(p.grid.cell[0]) - float(p.grid.half[0])
    return np.nonzero((xc >= x0 + margin) & (xc <= x1 - margin))[0]


def _patch_radius(spec: Any, load: float) -> float:
    """Contact radius a of a Winkler dome under load F: F = k·π·R·δ0², a = sqrt(2·R·δ0)."""
    delta = math.sqrt(load / (spec.stiffness * math.pi * spec.dome_radius))
    return math.sqrt(2.0 * spec.dome_radius * delta)


def check_density(run: Any) -> LawResult:
    p = run.plugin
    spec = p.specs[0]
    ink = _series(run, "ink")[-1]
    dx = float(p.grid.cell[0])
    worst, rows = 0.0, []
    for w in range(ink.shape[0]):
        load, speed, x0, x1 = _steady(run, w, spec.label)
        cols = _columns(run, x0, x1, 2.0 * spec.radius)
        if cols.size < 3 or load <= 0.0:
            return LawResult(False, math.nan, 0.03, f"world {w}: no steady stroke", "the marker must touch and move")
        density = float(ink[w, cols, :, spec.colour].sum(axis=1).mean()) / dx
        want = spec.flow * load / speed
        worst = max(worst, abs(density / want - 1.0))
        rows.append(f"{density * 1e6:.4g} vs {want * 1e6:.4g} µg/mm at {load:.3g} N, {speed:.3g} m/s")
    return LawResult.within(worst, 0.03, "; ".join(rows), "lay flow·p·A·dt per cell: the stroke then carries flow·F/v")


def check_width(run: Any) -> LawResult:
    p = run.plugin
    spec = p.specs[0]
    ink = _series(run, "ink")[-1]
    dy = float(p.grid.cell[1])
    worst, rows = 0.0, []
    for w in range(ink.shape[0]):
        load, _, x0, x1 = _steady(run, w, spec.label)
        cols = _columns(run, x0, x1, 2.0 * spec.radius)
        width = float((ink[w, cols, :, spec.colour] > 0.0).sum(axis=1).mean()) * dy
        want = 2.0 * _patch_radius(spec, load)
        worst = max(worst, abs(width - want) / dy)
        rows.append(f"{width * 1e3:.3f} vs 2a {want * 1e3:.3f} mm at {load:.3g} N")
    return LawResult.within(worst, 2.0, f"width error in cells (cell {dy * 1e3:.2f} mm): " + "; ".join(rows),
                            "lower the nib face by its sag r²/2R: the contact (and the ink) is the disc a(F)")


def check_ledger(run: Any) -> LawResult:
    ledger = _series(run, "ledger")
    drift = float(np.abs(ledger - ledger[0]).max() / np.abs(ledger[0]).max())
    p = run.plugin
    laid = _series(run, "laid")[-1, 0, p.tool_index("marker")]
    lifted = _series(run, "sink")[-1, 0, p.tool_index("wiper")]
    if laid <= 0.0 or lifted <= 0.0:
        return LawResult(False, math.nan, 1e-12, f"laid {laid:.3g} kg, lifted {lifted:.3g} kg",
                         "the marker must draw and the wiper must pass over the ink")
    return LawResult.within(drift, 1.0e-12, f"relative ledger change (laid {laid * 1e6:.4g} mg, wiped "
                            f"{lifted * 1e6:.4g} mg)", "move exactly float64(new) - float64(old) of every cell")


def check_wipe(run: Any) -> LawResult:
    p = run.plugin
    spec = p.specs[0]
    ink = _series(run, "ink")[-1, 0, :, :, 0]
    ny = ink.shape[1]
    _, _, x0, x1 = _steady(run, 0, spec.label)
    cols = _columns(run, x0, x1, spec.radius + float(p.grid.cell[0]))     # the whole pad passed these columns
    if cols.size == 0:
        return LawResult(False, math.nan, 0.05, "the wiper passed no column fully", "the wiper must slide")
    centre = ink[cols][:, ny // 2 - 1: ny // 2 + 1] / DIRT
    want = math.exp(-2.0 * spec.radius / spec.wipe_length)
    shape = float(np.abs(centre / want - 1.0).max())
    total = _series(run, "ink_total")
    exact = float(np.abs((total[0] - total[-1]) - _series(run, "sink")[-1].sum(axis=1)).max() / total[0].max())
    detail = (f"centre line keeps {float(centre.mean()):.4f} vs exp(-2R/L) {want:.4f} ({shape:.2%}); board loss - "
              f"sink {exact:.1e} relative")
    if exact > 1.0e-12:
        return LawResult(False, exact, 1.0e-12, detail, "move exactly float64(old) - float64(new) into the sink")
    return LawResult.within(shape, 0.05, detail, "lift 1 - exp(-cover·ds/L) of every channel, ds the slip over the cell")


def check_lifted(run: Any) -> LawResult:
    moved = float(np.abs(_series(run, "laid")).max() + np.abs(_series(run, "load")).max())
    return LawResult.within(moved, 0.0, "ink laid + tip load of a marker 2 mm above the board",
                            "press and lay only where the nib face lies below the board surface")


__all__ = ["bench", "deposition_laws", "marker_and_wiper_scene", "merged"]
