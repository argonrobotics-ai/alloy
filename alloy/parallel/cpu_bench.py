# SPDX-License-Identifier: Apache-2.0
"""CPU mode A throughput rows: world-steps per second of one scene per material at 1..N worker processes.

``python -m alloy.parallel.cpu_bench --materials rigid,cloth,liquid,paste --worlds 16 --processes 1,4,8,16
--out rows.jsonl``. Each scene has per-world starts (or per-world targets), so worlds differ. Per material and
process count: build (timed), one warm-up period (kernel loads), then ``--seconds`` of stepping timed (wall clock);
one JSON row each, appended as it finishes. ``parity``: the end state equals the one-process run's, bit for bit
(``max_diff`` otherwise). The materials:

- rigid: the shipped Franka Panda scene (arm, table, free box, cylinder and mug), joints 1, 2 and 4 ramped per world;
- cloth: a 20 x 20 sheet dropped on a stand (VBD), per-world start x;
- liquid: water at rest in an open box (SPH), per-world start;
- paste: mayonnaise in an open box (MPM paste), per-world start.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("alloy.parallel")

SECONDS = {"rigid": 2.0, "cloth": 0.1, "liquid": 0.1, "paste": 0.05}
RAMPS = {"panda/joint1": 0.6, "panda/joint2": 0.4, "panda/joint4": -0.5}


def _spread(worlds: int, base: tuple[float, float, float], dx: float) -> np.ndarray:
    xs = np.linspace(-dx, dx, worlds) if worlds > 1 else np.zeros(1)
    return np.array([[base[0] + x, base[1], base[2]] for x in xs])


def rigid(worlds: int) -> Any:
    import alloy  # noqa: PLC0415
    from alloy.cards.library import default_root  # noqa: PLC0415

    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")
    return scene


def cloth(worlds: int) -> Any:
    import alloy  # noqa: PLC0415

    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("stand", size=(0.2, 0.2, 0.3), pos=(0, 0, 0.15), static=True)
    scene.add_cloth("sheet", size=(0.5, 0.5), pos=_spread(worlds, (0.0, 0.0, 0.45), 0.03), resolution=20, mass=0.1)
    return scene


def _tub(worlds: int, material: str) -> Any:
    import alloy  # noqa: PLC0415

    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_open_box("tub", inner=(0.08, 0.08, 0.06), pos=(0, 0, 0))
    scene.add_liquid("fill", size=(0.05, 0.05, 0.03), pos=_spread(worlds, (0.0, 0.0, 0.025), 0.01),
                     material=material)
    return scene


SCENES: dict[str, Callable[[int], Any]] = {"rigid": rigid, "cloth": cloth,
                                            "liquid": lambda w: _tub(w, "water"),
                                            "paste": lambda w: _tub(w, "mayonnaise")}


def _targets(material: str, worlds: int) -> dict[str, Any] | None:
    if material != "rigid":
        return None
    k = np.linspace(0.5, 1.0, worlds)
    return {j: (np.zeros(worlds), v * k) for j, v in RAMPS.items()}


def measure(material: str, worlds: int, processes: int, seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """One row and the end state."""
    import alloy  # noqa: PLC0415

    scene = SCENES[material](worlds)
    t0 = time.perf_counter()
    sim = alloy.build(scene, device="cpu", seed=0, processes=processes if processes > 1 else None)
    build_s = time.perf_counter() - t0
    try:
        sim.step(1)                                                   # warm-up: kernel loads, first contacts
        s0, t0 = sim.physics_steps, time.perf_counter()
        traj = sim.run(seconds, record=[], targets=_targets(material, worlds))
        end = sim.state()
        wall = time.perf_counter() - t0
        steps = sim.physics_steps - s0
        row = {"material": material, "worlds": worlds, "processes": processes, "seconds": seconds,
               "build_s": round(build_s, 2), "wall_s": round(wall, 3), "physics_steps": steps,
               "world_steps_per_s": round(worlds * steps / wall, 1), "sim_s_per_wall_s": round(seconds / wall, 5),
               "worker_wall_s": traj.meta.get("worker_wall_s"), "ok": bool(traj.meta["health"]["ok"]),
               "routes": sorted({r["route"] for r in sim.routes().values()})}
    finally:
        if hasattr(sim, "close"):
            sim.close()
    return row, end


def parity(end: dict[str, Any], ref: dict[str, Any]) -> tuple[bool, float]:
    keys = [k for k in ref if k != "t"]
    same = all(np.array_equal(end[k], ref[k]) for k in keys)
    diff = max((float(np.nanmax(np.abs(np.asarray(end[k], float) - np.asarray(ref[k], float)))) for k in keys
                if np.size(ref[k])), default=0.0)
    return same, diff


def host() -> dict[str, Any]:
    from alloy.parallel.cpu_worlds import cores  # noqa: PLC0415

    model = ""
    try:
        model = next(ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                     if ln.startswith("model name"))
    except (OSError, StopIteration):
        model = platform.processor()
    return {"host": platform.node(), "cpus": cores(), "cpu_model": model,
            "cpu_route": os.environ.get("ALLOY_CPU_ROUTE", "mjc (default)")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--materials", default="rigid,cloth,liquid,paste")
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--processes", default="1,4,8,16")
    ap.add_argument("--seconds", type=float, default=None, help="simulated seconds (default per material)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    info = host()
    for material in a.materials.split(","):
        seconds = a.seconds or SECONDS[material]
        ref = None
        for p in (int(x) for x in a.processes.split(",")):
            row, end = measure(material, a.worlds, p, seconds)
            ref = end if ref is None else ref
            row["parity"], row["max_diff"] = parity(end, ref)
            row.update(info)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            log.info("%s %d procs: %.0f world-steps/s, parity %s", material, p, row["world_steps_per_s"],
                     row["parity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
