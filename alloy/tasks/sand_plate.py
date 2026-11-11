# SPDX-License-Identifier: Apache-2.0
"""task_sand_plate: a rotary sanding head rasters an aluminium plate (engine item 10, material removal v1).

The scene: a 300 x 200 x 10 mm aluminium plate (static, the workpiece ``plate``) and a 1.5 kg sanding head (body
``sander``) with a 50 mm foam pad 5 mm below its housing. ``SandingHead`` (``sand_head.py``) rasters the head in 5
rows 30 mm apart at 0.1 m/s, presses with 20 N on top of its weight and spins the pad at 60 rad/s; the ``removal``
plug-in turns pad pressure and slip into removed depth (Archard / Preston), with 80 % of the dust captured.

The task reads the depth grid over the target area (the raster's interior): mean removed depth, uniformity
(coefficient of variation), coverage (cells with at least half the mean depth), plus the mass ledger.

Run: ``python -m alloy.tasks.sand_plate [--worlds W] [--device cuda:0] [--seconds S] [--npy depth.npy]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.removal import Removal
from alloy.tasks.sand_head import SandingHead

log = logging.getLogger("alloy.tasks.sand_plate")

PLATE_HALF = (0.15, 0.1, 0.005)       # m
HEAD_HALF = 0.03                      # m: cubic housing half extent
HEAD_MASS = 1.5                       # kg
PAD_DROP = 0.005                      # m: pad face below the housing
G = 9.81


@dataclass(frozen=True)
class SandPlate:
    """Task parameters: the removal plug-in's and the head's configs (any field of either can be overridden)."""

    removal: dict[str, Any] = field(default_factory=lambda: {
        "tool": "sander", "workpiece": "plate", "pad_center": (0.0, 0.0, -HEAD_HALF - PAD_DROP),
        "pad_radius": 0.025, "cells": (96, 64), "capture": 0.8})
    head: dict[str, Any] = field(default_factory=dict)
    margin: float = 0.005             # m: the target area (``target_mask``) is shrunk by this

    def plugins(self) -> tuple[Removal, SandingHead]:
        return Removal(**self.removal), SandingHead(**self.head)


def build_scene(task: SandPlate):
    removal, head = task.plugins()
    rc, hc = removal.config, head.config
    load = HEAD_MASS * G + hc.downforce
    sink = load / (rc.pad_stiffness * np.pi * rc.pad_radius**2)
    z = 2.0 * PLATE_HALF[2] - rc.pad_center[2] - sink

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, PLATE_HALF[2]), hx=PLATE_HALF[0], hy=PLATE_HALF[1],
                        hz=PLATE_HALF[2], label=rc.workpiece)
        sander = b.add_body(xform=body_xform(hc.start[0], hc.start[1], z), label=rc.tool)
        density = HEAD_MASS / (2.0 * HEAD_HALF) ** 3
        b.add_shape_box(sander, hx=HEAD_HALF, hy=HEAD_HALF, hz=HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=density))
    return build


@dataclass
class SandResult:
    """World-0 metrics (lengths in µm, masses in mg) and cost per exchange step."""

    seconds: float
    mean_depth_um: float
    cv: float                         # std / mean removed depth over the target area
    coverage: float                   # fraction of target cells with >= 0.5 x the mean depth
    max_depth_um: float
    removed_mg: float
    captured_mg: float
    loose_mg: float
    ledger_rel: float                 # |removed - captured - loose| / removed (no dirt at the start)
    track_mm: float                   # head's worst distance from the path (pad centre, x-y) after the first row starts
    worlds: int
    ms_per_step: float
    ms_per_step_without: float


def target_mask(task: SandPlate, removal: Removal) -> np.ndarray:
    """Target cells: the pad-centre path's rows, each owning half a row spacing either side (half a pad radius for a
    single row), shrunk by ``margin``."""
    hc = SandingHead(**task.head).config
    band = 0.5 * (hc.spacing if hc.rows > 1 else removal.config.pad_radius)
    nx, ny = removal.depth.shape[1:]
    cx, cy = (float(removal.grid.cell[k]) for k in range(2))
    x = (np.arange(nx) + 0.5) * cx - float(removal.grid.half[0])
    y = (np.arange(ny) + 0.5) * cy - float(removal.grid.half[1])
    x0, x1 = hc.start[0] + task.margin, hc.start[0] + hc.length - task.margin
    y0 = hc.start[1] - band + task.margin
    y1 = hc.start[1] + (hc.rows - 1) * hc.spacing + band - task.margin
    return ((x >= x0) & (x <= x1))[:, None] & ((y >= y0) & (y <= y1))[None, :]


def metrics(task: SandPlate, removal: Removal, read: dict[str, np.ndarray]) -> dict[str, float]:
    depth = read["depth"][0]
    inside = depth[target_mask(task, removal)]
    mean = float(inside.mean())
    removed, captured, loose = (float(read[k][0]) for k in ("removed", "sink", "loose_total"))
    return {"mean_depth_um": mean * 1e6, "cv": float(inside.std() / mean) if mean > 0 else float("inf"),
            "coverage": float((inside >= 0.5 * mean).mean()) if mean > 0 else 0.0,
            "max_depth_um": float(depth.max()) * 1e6, "removed_mg": removed * 1e6, "captured_mg": captured * 1e6,
            "loose_mg": loose * 1e6, "ledger_rel": abs(removed - captured - loose) / max(removed, 1e-30)}


def run(task: SandPlate | None = None, *, worlds: int = 1, device: str = "cpu", seconds: float | None = None,
        timing: bool = True) -> tuple[SandResult, dict[str, np.ndarray]]:
    """Run the task; returns world 0's metrics and the removal plug-in's read (all worlds)."""
    task = task or SandPlate()
    removal, head = task.plugins()
    scene = TestScene(build=build_scene(task), worlds=worlds, ground=False, control_dt=0.02, dt=1.25e-3)
    sim = build_simulation(scene, [head, removal], device)
    seconds = head.duration if seconds is None else seconds
    periods = max(int(round(seconds / scene.control_dt)), 1)
    b = int(head.head.numpy()[0])
    worst = 0.0
    for k in range(periods):
        sim.step_period()
        worst = max(worst, _track_error(head, sim, b, (k + 1) * scene.control_dt))
    read = removal.read()
    nan = float("nan")
    cost = (_cost(task, scene, device, True), _cost(task, scene, device, False)) if timing else (nan, nan)
    res = SandResult(seconds=periods * scene.control_dt, **metrics(task, removal, read), track_mm=worst * 1e3,
                     worlds=worlds, ms_per_step=cost[0], ms_per_step_without=cost[1])
    log.info("task_sand_plate: %s", res)
    return res, read


def _track_error(head: SandingHead, sim: Any, b: int, t: float) -> float:
    c = head.config
    if t * c.speed < 0.02:                                   # the head starts at rest: skip the first 20 mm
        return 0.0
    target = _host_raster(c, min(t * c.speed, c.rows * c.length + (c.rows - 1) * c.spacing))
    q = sim.state.body_q.numpy()[b]
    return float(np.hypot(q[0] - target[0], q[1] - target[1]))


def _host_raster(c: Any, s: float) -> tuple[float, float]:
    period = c.length + c.spacing
    k = min(int(s // period), c.rows - 1)
    rest = s - k * period
    y = c.start[1] + k * c.spacing
    if rest <= c.length:
        return (c.start[0] + (rest if k % 2 == 0 else c.length - rest), y)
    return (c.start[0] + (c.length if k % 2 == 0 else 0.0), y + rest - c.length)


def _cost(task: SandPlate, scene: TestScene, device: str, with_removal: bool, periods: int = 25) -> float:
    """ms per exchange step of the scene with the head controller, with or without the removal plug-in (best of
    two timed runs after a warm-up period; no host reads inside the timed loop)."""
    removal, head = task.plugins()
    sim = build_simulation(scene, [head, removal] if with_removal else [head], device)
    sim.step_period()                                        # warm-up (kernel loads)
    best = float("inf")
    for _ in range(2):
        _sync(device)
        steps, t0 = sim.steps, time.perf_counter()
        for _ in range(periods):
            sim.step_period()
        _sync(device)
        best = min(best, 1e3 * (time.perf_counter() - t0) / max(sim.steps - steps, 1))
    return best


def _sync(device: str) -> None:
    import warp as wp  # noqa: PLC0415

    wp.synchronize_device(device)


def save_depth(path: str, read: dict[str, np.ndarray]) -> None:
    """World 0's removed depth grid [m] as a ``.npy`` file, x along the first axis (plot it with any tool)."""
    np.save(path, read["depth"][0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seconds", type=float, default=None, help="default: the whole raster")
    ap.add_argument("--npy", default=None, help="save world 0's depth grid [m] here")
    args = ap.parse_args(argv)
    task = SandPlate()
    res, read = run(task, worlds=args.worlds, device=args.device, seconds=args.seconds)
    print(json.dumps(asdict(res), indent=1))
    if args.npy:
        save_depth(args.npy, read)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
