# SPDX-License-Identifier: Apache-2.0
"""task_whiteboard_draw_erase: draw a blank table in blue on a whiteboard, then wipe the board clean.

Gym-Anything's whiteboard tasks (draw-table, erase-board; XB4-7 REPORT §2) need ink laid and lifted by contact. The
scene is a 400 x 240 mm board (static, the ``deposition`` plug-in's ``whiteboard`` face, 1.25 mm cells, black and
blue channels) with two tools, each held by a ``PenHead``:
- a 0.3 kg marker head with a 3 mm domed felt nib draws the table at 0.1 m/s with 1.5 N on top of its weight: outer
  frame 240 x 120 mm, a header line, two column lines; then it parks off the board;
- a 0.6 kg eraser head with a 25 mm felt pad then wipes the drawing area in 30 mm rows at 0.15 m/s with 6 N.

Metrics (world 0):
- after drawing: how much of the table's lines is inked (coverage), how much of the ink lies on them (precision),
  the mean line width against the nib's contact diameter, and the ink per metre;
- after erasing: the ink left on the board, the eraser's share of what was laid, and the ledger.

Run: ``python -m alloy.tasks.whiteboard_draw_erase [--worlds W] [--device cuda:0] [--npy maps.npz]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.deposition import Deposition
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.tasks.whiteboard_head import PenHead, strokes_to_path

log = logging.getLogger("alloy.tasks.whiteboard")

BOARD_HALF = (0.2, 0.12, 0.01)            # m
TOP = 2.0 * BOARD_HALF[2]
APEX = 0.025                              # m: tip apex below each head's body origin
HEADS = {"marker": ((0.015, 0.015, 0.02), 0.3), "eraser": ((0.03, 0.03, 0.02), 0.6)}   # half extents, kg
HOVER = TOP + APEX + 0.005                # body height while a pen is up (tip 5 mm above the board)
CONTROL_DT = 0.02


def table_strokes() -> list[list[tuple[float, float]]]:
    """The blank table: outer frame, header line, two column lines (draw-table)."""
    frame = [(-0.12, -0.06), (0.12, -0.06), (0.12, 0.06), (-0.12, 0.06), (-0.12, -0.06)]
    return [frame, [(-0.12, 0.03), (0.12, 0.03)], [(-0.04, -0.06), (-0.04, 0.06)], [(0.04, -0.06), (0.04, 0.06)]]


def raster(x0: float, x1: float, y0: float, y1: float, spacing: float) -> list[tuple[float, float]]:
    """One serpentine polyline over the rectangle, rows along x."""
    rows = max(int(math.ceil((y1 - y0) / spacing)) + 1, 1)
    pts = []
    for k, y in enumerate(np.linspace(y0, y1, rows)):
        a, b = (x0, x1) if k % 2 == 0 else (x1, x0)
        pts += [(a, float(y)), (b, float(y))]
    return pts


@dataclass(frozen=True)
class Whiteboard:
    """Task parameters; ``deposition`` overrides the plug-in's config, the rest set the heads and paths."""

    deposition: dict[str, Any] = field(default_factory=lambda: {
        "tools": ("marker", "eraser"), "kinds": ("marker", "wiper"), "channels": ("black", "blue"), "colours": (1,),
        "tip_center": (0.0, 0.0, -APEX), "board": "whiteboard", "cells": (320, 192)})
    strokes: tuple[Any, ...] = field(default_factory=lambda: tuple(tuple(s) for s in table_strokes()))
    erase_area: tuple[float, float, float, float] = (-0.13, 0.13, -0.075, 0.075)   # x0, x1, y0, y1 of pad centres
    erase_rows: float = 0.03                       # m between eraser rows
    marker_speed: float = 0.1
    eraser_speed: float = 0.15
    marker_press: float = 1.5
    eraser_press: float = 6.0
    park: dict[str, tuple[float, float]] = field(default_factory=lambda: {"marker": (0.26, 0.16),
                                                                          "eraser": (-0.26, -0.16)})

    def plugins(self) -> tuple[Deposition, PenHead, PenHead]:
        drawn = PenHead(head="marker", path=strokes_to_path(self.strokes, self.park["marker"], self.park["marker"]),
                        speed=self.marker_speed,
                        press=self.marker_press, hover_z=HOVER, label="marker_head")
        x0, x1, y0, y1 = self.erase_area
        sweep = raster(x0, x1, y0, y1, self.erase_rows)
        wipe = PenHead(head="eraser", path=strokes_to_path([sweep], self.park["eraser"], self.park["eraser"]),
                       speed=self.eraser_speed,
                       press=self.eraser_press, hover_z=HOVER, start=drawn.duration + 0.5, label="eraser_head")
        return Deposition(**self.deposition), drawn, wipe


def build_scene(task: Whiteboard):
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, BOARD_HALF[2]), hx=BOARD_HALF[0], hy=BOARD_HALF[1],
                        hz=BOARD_HALF[2], label=task.deposition["board"])
        for name, (half, mass) in HEADS.items():
            body = b.add_body(xform=body_xform(*task.park[name], HOVER), label=name)
            density = mass / (8.0 * half[0] * half[1] * half[2])
            b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                            cfg=newton.ModelBuilder.ShapeConfig(density=density))
    return build


@dataclass
class BoardResult:
    """World-0 metrics (masses in mg, lengths in mm) and the cost per exchange step."""

    seconds: float
    coverage: float              # fraction of the table's centre lines that is inked
    precision: float             # fraction of inked cells within the nib's reach of a table line
    width_mm: float              # mean inked width
    width_expected_mm: float     # 2a of the nib under the marker head's weight + press
    ink_per_m_ug_mm: float       # laid ink per mm of drawn line
    laid_mg: float
    left_after_erase: float      # board ink after erasing / laid
    erased: float                # eraser sink / laid
    ledger_rel: float
    worlds: int
    ms_per_step: float
    ms_per_step_without: float


def line_mask(dep: Deposition, strokes: Any, reach: float) -> np.ndarray:
    """Map cells whose centres lie within ``reach`` of a stroke segment."""
    nx, ny = dep.ink.shape[1:3]
    cx, cy = float(dep.grid.cell[0]), float(dep.grid.cell[1])
    x = (np.arange(nx) + 0.5) * cx - float(dep.grid.half[0])
    y = (np.arange(ny) + 0.5) * cy - float(dep.grid.half[1])
    px, py = np.meshgrid(x, y, indexing="ij")
    dist = np.full((nx, ny), np.inf)
    for stroke in strokes:
        for (ax, ay), (bx, by) in zip(stroke[:-1], stroke[1:]):
            ex, ey = bx - ax, by - ay
            u = np.clip(((px - ax) * ex + (py - ay) * ey) / max(ex * ex + ey * ey, 1e-18), 0.0, 1.0)
            dist = np.minimum(dist, np.hypot(px - ax - u * ex, py - ay - u * ey))
    return dist <= reach


def draw_metrics(task: Whiteboard, dep: Deposition, read: dict[str, np.ndarray], load: float) -> dict[str, float]:
    spec = dep.specs[0]
    ink = read["ink"][0, :, :, spec.colour]
    inked = ink > 0.0
    a = math.sqrt(2.0 * spec.dome_radius * math.sqrt(load / (spec.stiffness * math.pi * spec.dome_radius)))
    cell = max(float(dep.grid.cell[0]), float(dep.grid.cell[1]))
    centre = line_mask(dep, task.strokes, 0.5 * cell)
    length = sum(float(np.linalg.norm(np.diff(np.asarray(s), axis=0), axis=1).sum()) for s in task.strokes)
    laid = float(read["laid"][0, 0])
    return {"coverage": float(inked[centre].mean()),
            "precision": float(line_mask(dep, task.strokes, a + cell)[inked].mean()) if inked.any() else 0.0,
            "width_mm": 1e3 * float(inked.sum()) * dep.cell_area / length, "width_expected_mm": 2e3 * a,
            "ink_per_m_ug_mm": laid / length * 1e6, "laid_mg": laid * 1e6}


def run(task: Whiteboard | None = None, *, worlds: int = 1, device: str = "cpu",
        timing: bool = True) -> tuple[BoardResult, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Draw, read, erase, read. Returns world 0's metrics and the plug-in's reads after drawing and after erasing."""
    task = task or Whiteboard()
    dep, marker, eraser = task.plugins()
    scene = TestScene(build=build_scene(task), worlds=worlds, ground=False, control_dt=CONTROL_DT, dt=1.25e-3)
    sim = build_simulation(scene, [marker, eraser, dep], device)
    draw_periods = int(math.ceil(eraser.config.start / CONTROL_DT))
    loads = []
    for _ in range(draw_periods):
        sim.step_period()
        loads.append(float(dep.load.numpy()[0, 0]))
    drawn = dep.read()
    for _ in range(int(math.ceil((eraser.duration - eraser.config.start) / CONTROL_DT)) + 10):
        sim.step_period()
    wiped = dep.read()
    pressing = [x for x in loads if x > 0.0]
    load = float(np.median(pressing)) if pressing else 0.0
    laid = float(wiped["laid"][0].sum())
    nan = float("nan")
    cost = (_cost(task, scene, device, True), _cost(task, scene, device, False)) if timing else (nan, nan)
    res = BoardResult(seconds=float(sim.clock.t), **draw_metrics(task, dep, drawn, load),
                      left_after_erase=float(wiped["ink_total"][0]) / max(laid, 1e-30),
                      erased=float(wiped["sink"][0].sum()) / max(laid, 1e-30),
                      ledger_rel=float(abs(wiped["ledger"][0] - drawn["ledger"][0]) / drawn["ledger"][0]),
                      worlds=worlds, ms_per_step=cost[0], ms_per_step_without=cost[1])
    log.info("task_whiteboard_draw_erase: %s", res)
    return res, drawn, wiped


def _cost(task: Whiteboard, scene: TestScene, device: str, with_ink: bool, periods: int = 25) -> float:
    """ms per exchange step while drawing, with or without the deposition plug-in (best of two, warm)."""
    dep, marker, eraser = task.plugins()
    sim = build_simulation(scene, [marker, eraser, dep] if with_ink else [marker, eraser], device)
    for _ in range(int(1.0 / CONTROL_DT)):                       # reach the board first (1 s), warms kernels too
        sim.step_period()
    best = float("inf")
    for _ in range(2):
        wp.synchronize_device(device)
        steps, t0 = sim.steps, time.perf_counter()
        for _ in range(periods):
            sim.step_period()
        wp.synchronize_device(device)
        best = min(best, 1e3 * (time.perf_counter() - t0) / max(sim.steps - steps, 1))
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npy", default=None, help="save world 0's maps after drawing and erasing (.npz)")
    args = ap.parse_args(argv)
    res, drawn, wiped = run(worlds=args.worlds, device=args.device)
    print(json.dumps(asdict(res), indent=1))
    if args.npy:
        np.savez(args.npy, drawn=drawn["ink"][0], wiped=wiped["ink"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
