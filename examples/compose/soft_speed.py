# SPDX-License-Identifier: Apache-2.0
"""Build time and step rate of saved composed scenes, at one world and at larger world counts.

For each scene card: load it (``alloy.compose.load_scene``), build it at each world count (``alloy.api.build``),
step one control period to compile and warm up, then time ``--seconds`` of simulated time, cut to whole control
periods of at most ``--max-steps`` physics steps (some scenes step far more often than others: sand in a free bin
takes about 770 physics steps per 20 ms period, water in a mug 16). One JSON line per scene and world count: build
seconds, seconds timed, world-steps per second (physics steps x worlds / wall seconds), the solver entries.

Run:

    python examples/compose/soft_speed.py --scenes soft_suite/s00/scenes/scene.yaml --worlds 1          # CPU is fine
    python examples/compose/soft_speed.py --scenes soft_suite/*/scenes/scene.yaml --worlds 1 64 256 1024 \\
        --device cuda:0 --out speed.jsonl                                                              # one L4

Each scene is measured in a process of its own: a CUDA error (out of memory in a step graph, for one) can leave
its process without a GPU, and every later row there would fail at once. A world count that fails is reported as
an error row, the scene's larger counts are skipped, and the next scene starts in a fresh process. The exit code
is 1 when any row is an error row, so a table with a hole in it does not read as a pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import alloy.api
import alloy.compose

log = logging.getLogger("soft_speed")


def measure(path: str, worlds: int, seconds: float, device: str | None, max_steps: int = 500) -> dict:
    """One row: build ``path`` at ``worlds`` and time ``seconds`` of it, cut to whole periods of at most ``max_steps``
    physics steps (one period at least). Example: ``measure("s00/scenes/scene.yaml", 1, 0.25, "cpu")["wall_s"]``."""
    row: dict = {"scene": path, "worlds": worlds}
    t0 = time.perf_counter()
    try:
        sim = alloy.api.build(alloy.compose.load_scene(path), worlds=worlds, device=device)
        row["build_s"] = round(time.perf_counter() - t0, 2)
        steps0 = sim.physics_steps
        sim.step(1)                                                  # compile and warm up
        per_period = max(sim.physics_steps - steps0, 1)
        periods = max(1, min(round(seconds / sim.dt), max_steps // per_period))
        steps0, t1 = sim.physics_steps, time.perf_counter()
        sim.run(periods * sim.dt)
        wall = time.perf_counter() - t1
        row.update(seconds=round(periods * sim.dt, 6), wall_s=round(wall, 3), physics_steps=sim.physics_steps - steps0,
                   world_steps_per_s=round((sim.physics_steps - steps0) * worlds / max(wall, 1e-9), 1),
                   routes=sorted({r.get("route", "") for r in sim.routes().values()}), device=sim.device)
        sim.close()
    except Exception as exc:  # noqa: BLE001 - an example reports the failure as a row and goes on
        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return row


def one_scene(path: str, args: argparse.Namespace) -> int:
    """The rows of one scene, smallest world count first, printed and appended to ``args.out``; stops at the first
    error row. Returns the number of error rows (0 or 1). Example: ``one_scene("s00/scenes/scene.yaml", args)``."""
    for w in sorted(args.worlds):
        row = measure(path, w, args.seconds, args.device, args.max_steps)
        print(json.dumps(row), flush=True)
        if args.out:
            with Path(args.out).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if "error" in row:
            log.error("%s: %d worlds failed, larger world counts skipped", path, w)
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--worlds", type=int, nargs="+", default=[1])
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=500, help="physics steps to time at most (whole periods)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="append the rows to this JSONL file")
    ap.add_argument("--in-process", action="store_true", help="measure the scenes in this process (one scene: default)")
    args = ap.parse_args()
    if args.in_process or len(args.scenes) == 1:
        return min(1, sum(one_scene(path, args) for path in args.scenes))
    failed = 0
    for path in args.scenes:                                          # a fresh process, and a fresh GPU context, each
        cmd = [sys.executable, __file__, "--scenes", path, "--worlds", *map(str, args.worlds), "--seconds",
               str(args.seconds), "--max-steps", str(args.max_steps)]
        cmd += (["--device", args.device] if args.device else []) + (["--out", args.out] if args.out else [])
        code = subprocess.run(cmd, check=False).returncode
        if code not in (0, 1):                                        # the process died without writing its row
            row = {"scene": path, "error": f"the scene's process ended with exit code {code}"}
            print(json.dumps(row), flush=True)
            if args.out:
                with Path(args.out).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        failed += code != 0
    if failed:
        log.error("%d of %d scenes have an error row", failed, len(args.scenes))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
