# SPDX-License-Identifier: Apache-2.0
"""task_deburr_seam: a spinning abrasive wheel follows a seam and grinds its burr off (engine item 10, deburring).

The scene: a 200 x 100 x 10 mm aluminium plate (static, the workpiece ``plate``) with a burr along a 40 mm seam on
its top face: a triangular ridge 0.15 mm tall and 1 mm wide at its base, as a shear or saw cut leaves. A 1.5 kg
deburring head (body ``deburrer``, 80 x 80 x 30 mm) carries a 10 mm abrasive wheel face (a stiff, rubber-bonded abrasive: 1e11 Pa/m)
5 mm below its housing. ``SandingHead`` (``alloy/tasks/sand_head.py``) drives it once along the seam at 15 mm/s with
15 N on top of its weight and spins the wheel at 200 rad/s; the ``removal`` plug-in, with geometry feedback, lets
the burr carry the wheel, so the burr wears first and the base around it only once the burr is down.

The task reads the surface along the seam's interior (away from its ends): the burr height above the base before and
after (max and mean over the seam), the base over-removal next to the burr (mean and max removed depth of the base
cells under the wheel's path), the mass ledger, and the cost per exchange step with and without the plug-in.

Run: ``python -m alloy.tasks.deburr_seam [--worlds W] [--device cuda:0] [--seconds S]``.
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

log = logging.getLogger("alloy.tasks.deburr_seam")

PLATE_HALF = (0.1, 0.05, 0.005)       # m
HEAD_HALF = (0.04, 0.04, 0.015)       # m: a flat housing, so the wheel sits close to its COM (tilt stability)
HEAD_MASS = 1.5                       # kg
WHEEL_DROP = 0.005                    # m: wheel face below the housing
G = 9.81
STEP = 3.125e-4                       # s: the stiff wheel needs omega·dt < 1 (Removal.check_dt)
CONTROL_DT = 0.02                     # s


@dataclass(frozen=True)
class DeburrSeam:
    """Task parameters: the burr, the removal plug-in's and the head's configs (any field can be overridden)."""

    burr_height: float = 1.5e-4       # m
    burr_width: float = 1.0e-3        # m at the base
    seam: float = 0.04                # m: burr length along x, centred
    removal: dict[str, Any] = field(default_factory=lambda: {
        "tool": "deburrer", "workpiece": "plate", "pad_center": (0.0, 0.0, -HEAD_HALF[2] - WHEEL_DROP),
        "pad_radius": 0.005, "pad_stiffness": 1.0e11, "pad_damping": 2.0e-4, "friction": 0.5,
        "wear_coefficient": 0.1, "size": (0.07, 0.012), "surface_offset": PLATE_HALF[2], "cells": (140, 48),
        "capture": 0.5})
    head: dict[str, Any] = field(default_factory=lambda: {
        "head": "deburrer", "start": (-0.03, 0.0), "length": 0.06, "spacing": 0.0, "rows": 1, "speed": 0.015,
        "downforce": 15.0, "spin": 200.0})
    margin: float = 0.005             # m: seam cells this far from the burr's ends are scored

    def plugins(self) -> tuple[Removal, SandingHead]:
        return Removal(**self.removal), SandingHead(**self.head)


def burr_profile(task: DeburrSeam, removal: Removal) -> np.ndarray:
    """Heights [m] of the burr ridge over the grid (a triangle across y, the seam's length along x)."""
    x, y = _cells(removal)
    across = np.clip(1.0 - np.abs(y) / (0.5 * task.burr_width), 0.0, None)
    along = np.abs(x) <= 0.5 * task.seam
    return task.burr_height * along[:, None] * across[None, :]


def build_scene(task: DeburrSeam):
    removal, head = task.plugins()
    rc, hc = removal.config, head.config
    load = HEAD_MASS * G + hc.downforce
    sink = load / (rc.pad_stiffness * np.pi * rc.pad_radius**2)
    z = 2.0 * PLATE_HALF[2] - rc.pad_center[2] - sink           # the wheel starts on the base, before the seam

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, PLATE_HALF[2]), hx=PLATE_HALF[0], hy=PLATE_HALF[1],
                        hz=PLATE_HALF[2], label=rc.workpiece)
        tool = b.add_body(xform=body_xform(hc.start[0], hc.start[1], z), label=rc.tool)
        b.add_shape_box(tool, hx=HEAD_HALF[0], hy=HEAD_HALF[1], hz=HEAD_HALF[2],
                        cfg=newton.ModelBuilder.ShapeConfig(density=float(HEAD_MASS / (8.0 * np.prod(HEAD_HALF)))))
    return build


def with_burr(task: DeburrSeam):
    def init(sim: Any) -> None:
        removal = sim.plugins.plugins.get("removal")          # absent in the cost baseline without the plug-in
        if removal is not None:
            removal.set_profile(burr_profile(task, removal))
    return init


@dataclass
class DeburrResult:
    """World-0 metrics (heights in µm, masses in mg) and cost per exchange step."""

    seconds: float
    burr_before_um: float             # max burr height above the base along the scored seam
    burr_after_max_um: float
    burr_after_mean_um: float
    base_removed_mean_um: float       # removed depth of the base cells beside the burr, under the wheel's path
    base_removed_max_um: float
    removed_mg: float
    captured_mg: float
    loose_mg: float
    ledger_rel: float
    worlds: int
    ms_per_step: float
    ms_per_step_without: float


def metrics(task: DeburrSeam, removal: Removal, read: dict[str, np.ndarray]) -> dict[str, float]:
    x, y = _cells(removal)
    burr = burr_profile(task, removal)
    surface = burr - read["depth"][0]                      # height above the nominal face
    rows = np.abs(x) <= 0.5 * task.seam - task.margin
    on = (np.abs(y) < 0.5 * task.burr_width)[None, :] & rows[:, None]
    base = ((np.abs(y) >= 0.5 * task.burr_width) & (np.abs(y) <= removal.config.pad_radius * 0.8))[None, :] & rows[:, None]
    base_level = float(surface[base].mean())               # the ground base, which may itself be worn
    after = surface[rows][:, np.abs(y) < 0.5 * task.burr_width].max(axis=1) - base_level
    removed, captured, loose = (float(read[k][0]) for k in ("removed", "sink", "loose_total"))
    return {"burr_before_um": float(burr[on].max()) * 1e6, "burr_after_max_um": float(after.max()) * 1e6,
            "burr_after_mean_um": float(after.mean()) * 1e6,
            "base_removed_mean_um": float(read["depth"][0][base].mean()) * 1e6,
            "base_removed_max_um": float(read["depth"][0][base].max()) * 1e6, "removed_mg": removed * 1e6,
            "captured_mg": captured * 1e6, "loose_mg": loose * 1e6,
            "ledger_rel": abs(removed - captured - loose) / max(removed, 1e-30)}


def run(task: DeburrSeam | None = None, *, worlds: int = 1, device: str = "cpu", seconds: float | None = None,
        timing: bool = True) -> tuple[DeburrResult, dict[str, np.ndarray]]:
    """Run the task; returns world 0's metrics and the removal plug-in's read (all worlds)."""
    task = task or DeburrSeam()
    removal, head = task.plugins()
    scene = TestScene(build=build_scene(task), worlds=worlds, ground=False, control_dt=CONTROL_DT, dt=STEP,
                      init=with_burr(task))
    sim = build_simulation(scene, [head, removal], device)          # runs the scene's init (the burr)
    seconds = head.duration if seconds is None else seconds
    for _ in range(max(int(round(seconds / CONTROL_DT)), 1)):
        sim.step_period()
    read = removal.read()
    nan = float("nan")
    cost = (_cost(task, scene, device, True), _cost(task, scene, device, False)) if timing else (nan, nan)
    res = DeburrResult(seconds=seconds, **metrics(task, removal, read), worlds=worlds, ms_per_step=cost[0],
                       ms_per_step_without=cost[1])
    log.info("task_deburr_seam: %s", res)
    return res, read


def _cells(removal: Removal) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = removal.depth.shape[1:]
    cx, cy = (float(removal.grid.cell[k]) for k in range(2))
    x = (np.arange(nx) + 0.5) * cx - float(removal.grid.half[0])
    y = (np.arange(ny) + 0.5) * cy - float(removal.grid.half[1])
    return x, y


def _cost(task: DeburrSeam, scene: TestScene, device: str, with_removal: bool, periods: int = 25) -> float:
    """ms per exchange step with the head controller, with or without the removal plug-in (best of two)."""
    import warp as wp  # noqa: PLC0415

    removal, head = task.plugins()
    sim = build_simulation(scene, [head, removal] if with_removal else [head], device)
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
    ap.add_argument("--seconds", type=float, default=None, help="default: the whole pass")
    args = ap.parse_args(argv)
    res, _ = run(DeburrSeam(), worlds=args.worlds, device=args.device, seconds=args.seconds)
    print(json.dumps(asdict(res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
