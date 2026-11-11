# SPDX-License-Identifier: Apache-2.0
"""task_drill_plate: a feed-controlled spindle drills through holes in an aluminium plate (engine item 10, drilling).

The scene: a 100 x 100 x 10 mm 6061 plate (static, the workpiece ``plate``) and a 3 kg spindle (body ``spindle``) with
a 6 mm, 118° HSS drill 40 mm below its housing. ``FeedHead`` (``alloy.plugins.drilling.head``) holds the hole position,
spins the drill at 3000 rpm and feeds it down; the ``drill`` plug-in turns the feed per revolution into Shaw-Oxford
thrust and torque, advances the hole and books the chips (80 % extracted). Each world drills at its own feed per
revolution (``feeds``), so one batch gives thrust and torque against feed.

Per world the task reports the steady thrust and torque (and the Shaw-Oxford law at the realized feed), the time the
drill enters and breaks through, the removed mass against density·π·r²·t, the ledger, the hole position error, and
the cost per exchange step with and without the plug-in.

Run: ``python -m alloy.tasks.drill_plate [--worlds W] [--device cuda:0] [--feeds 0.05,0.1,0.2] [--seconds S]``.
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

from alloy._vendor import newton

from alloy.plugins.drilling import Drill, FeedHead
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation

log = logging.getLogger("alloy.tasks.drill_plate")

PLATE_HALF = (0.05, 0.05, 0.005)      # m
HEAD_HALF = 0.03                      # m: cubic housing half extent
HEAD_MASS = 3.0                       # kg
BIT_LEN = 0.04                        # m: drill tip below the housing
GAP = 5.0e-4                          # m: tip above the face at the start
HOLE = (0.02, -0.01)                  # m: hole position on the plate
SPIN = 100.0 * math.pi                # rad/s (3000 rpm)
STEP = 6.25e-4                        # s: exchange step (control_dt / 32)
CONTROL_DT = 0.02                     # s


@dataclass(frozen=True)
class DrillPlate:
    """Task parameters: the drill's and the head's configs (any field can be overridden) and the feeds per world."""

    drill: dict[str, Any] = field(default_factory=lambda: {
        "tool": "spindle", "workpiece": "plate", "tip": (0.0, 0.0, -HEAD_HALF - BIT_LEN), "diameter": 0.006,
        "capture": 0.8})
    head: dict[str, Any] = field(default_factory=lambda: {"head": "spindle", "spin": SPIN})
    feeds: tuple[float, ...] = (0.08e-3, 0.12e-3, 0.16e-3, 0.2e-3)   # m/rev, one per world (the last repeats)

    def plugins(self) -> tuple[Drill, FeedHead]:
        return Drill(**self.drill), FeedHead(**self.head)

    def feed_of(self, world: int) -> float:
        return self.feeds[min(world, len(self.feeds) - 1)]


def build_scene(task: DrillPlate):
    drill, _ = task.plugins()
    dc = drill.config

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, PLATE_HALF[2]), hx=PLATE_HALF[0], hy=PLATE_HALF[1],
                        hz=PLATE_HALF[2], label=dc.workpiece)
        z = 2.0 * PLATE_HALF[2] + GAP - float(dc.tip[2])
        spindle = b.add_body(xform=body_xform(HOLE[0], HOLE[1], z), label=dc.tool)
        b.add_shape_box(spindle, hx=HEAD_HALF, hy=HEAD_HALF, hz=HEAD_HALF,
                        cfg=newton.ModelBuilder.ShapeConfig(density=HEAD_MASS / (2.0 * HEAD_HALF) ** 3))
    return build


def start(task: DrillPlate):
    """init: each world's feed rate (its feed per rev at the head's spin) and the drill already spinning."""
    def init(sim: Any) -> None:
        W = sim.model.world_count
        spin = float(task.plugins()[1].config.spin)
        sim.plugins["feed_head"].set_input("feed_rate", [task.feed_of(w) * spin / (2.0 * math.pi) for w in range(W)])
        qd = sim.state.joint_qd.numpy().reshape(-1, 6)
        qd[:] = 0.0
        qd[:, 5] = spin
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    return init


@dataclass
class HoleResult:
    """One world's hole (forces in N and N·m, times in s, masses in mg, lengths in mm)."""

    feed_mm_rev: float                # realized feed per revolution, steady
    thrust_n: float
    torque_nm: float
    thrust_law_n: float               # Shaw-Oxford at the realized feed
    torque_law_nm: float
    t_entry: float                    # first period with the tip in the plate
    t_through: float                  # first period with the whole drill point out of the plate's back
    removed_mg: float
    exact_mg: float                   # density·π·r²·t of the through hole
    captured_mg: float
    ledger_rel: float                 # |removed - captured - loose| / removed
    position_mm: float                # hole entry point vs the commanded hole position


@dataclass
class DrillResult:
    seconds: float
    worlds: int
    holes: list[HoleResult]
    ms_per_step: float
    ms_per_step_without: float


def run(task: DrillPlate | None = None, *, worlds: int = 4, device: str = "cpu", seconds: float = 3.6,
        timing: bool = True) -> tuple[DrillResult, dict[str, np.ndarray]]:
    """Run the task; returns the per-world results and the drill's read (all worlds)."""
    task = task or DrillPlate()
    drill, head = task.plugins()
    scene = TestScene(build=build_scene(task), worlds=worlds, ground=False, control_dt=CONTROL_DT, dt=STEP,
                      init=start(task))
    sim = build_simulation(scene, [head, drill], device)            # runs the scene's init (feeds, spin)
    periods = max(int(round(seconds / CONTROL_DT)), 1)
    trace: dict[str, list[np.ndarray]] = {k: [] for k in ("depth", "thrust", "torque", "feed")}
    for _ in range(periods):
        sim.step_period()
        r = drill.read()
        for k in trace:
            trace[k].append(r[k])
    read = drill.read()
    series = {k: np.asarray(v, np.float64) for k, v in trace.items()}
    holes = [hole_result(drill, read, series, w) for w in range(worlds)]
    nan = float("nan")
    cost = (_cost(task, scene, device, True), _cost(task, scene, device, False)) if timing else (nan, nan)
    res = DrillResult(seconds=periods * CONTROL_DT, worlds=worlds, holes=holes, ms_per_step=cost[0],
                      ms_per_step_without=cost[1])
    log.info("task_drill_plate: %s", res)
    return res, read


def hole_result(drill: Drill, read: dict[str, np.ndarray], series: dict[str, np.ndarray], w: int) -> HoleResult:
    c, t = drill.config, float(drill.work.thickness)
    times = (np.arange(series["depth"].shape[0]) + 1) * CONTROL_DT
    point = 0.5 * c.diameter / math.tan(math.radians(0.5 * c.point_angle))
    steady = _steady_window(series["depth"][:, w], point, t)
    feed = float(series["feed"][steady, w].mean()) if steady.any() else 0.0
    removed, sink, loose = (float(read[k][w]) for k in ("removed", "sink", "loose"))
    exact = c.density * math.pi * (0.5 * float(np.float32(c.diameter))) ** 2 * t
    entry = read["entry"][w].astype(np.float64)
    return HoleResult(
        feed_mm_rev=feed * 1e3, thrust_n=_mean(series["thrust"][steady, w]), torque_nm=_mean(series["torque"][steady, w]),
        thrust_law_n=float(drill.thrust_law(feed)), torque_law_nm=float(drill.torque_law(feed)),
        t_entry=_first(times, series["depth"][:, w] > 0.0), t_through=_first(times, series["depth"][:, w] >= t + point),
        removed_mg=removed * 1e6, exact_mg=exact * 1e6, captured_mg=sink * 1e6,
        ledger_rel=abs(removed - sink - loose) / max(removed, 1e-30),
        position_mm=float(np.hypot(entry[0] - HOLE[0], entry[1] - HOLE[1])) * 1e3)


def _steady_window(depth: np.ndarray, point: float, t: float, margin: float = 1.0e-3) -> np.ndarray:
    """Periods with the whole drill point in the plate and the tip short of the back face by ``margin``: the second
    half of them (past the entry transient)."""
    idx = np.flatnonzero((depth >= point + 0.5 * margin) & (depth <= t - margin))
    mask = np.zeros(depth.shape, bool)
    if idx.size:
        mask[idx[len(idx) // 2]: idx[-1] + 1] = True
    return mask


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


def _first(times: np.ndarray, hit: np.ndarray) -> float:
    idx = np.flatnonzero(hit)
    return float(times[idx[0]]) if idx.size else float("nan")


def _cost(task: DrillPlate, scene: TestScene, device: str, with_drill: bool, periods: int = 25) -> float:
    """ms per exchange step with the head controller, with or without the drill (best of two after a warm-up)."""
    drill, head = task.plugins()
    sim = build_simulation(scene, [head, drill] if with_drill else [head], device)
    sim.step_period()
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feeds", default="0.08,0.12,0.16,0.2", help="mm/rev per world (the last repeats)")
    ap.add_argument("--seconds", type=float, default=3.6)
    args = ap.parse_args(argv)
    task = DrillPlate(feeds=tuple(float(f) * 1e-3 for f in args.feeds.split(",")))
    res, _ = run(task, worlds=args.worlds, device=args.device, seconds=args.seconds)
    print(json.dumps(asdict(res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
