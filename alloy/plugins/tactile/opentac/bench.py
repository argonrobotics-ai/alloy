# SPDX-License-Identifier: Apache-2.0
"""Cost of the OpenTac board reading, batched: the example gripper (``example.py``) in N worlds, timed per control
period with no touch sensor, with ``Tactile`` alone (newton channels), and with the board reading on.

Each variant runs in its own process (a second simulation in one process can keep GPU memory, see the Alloy notes),
and prints one JSON line. The board variant also checks, in every world, that the board's taxel loads add up to
the engine's pad force (the force-sum law, batched)::

    python -m alloy.plugins.tactile.opentac.bench --worlds 1024 --device cuda:0          # all variants
    python -m alloy.plugins.tactile.opentac.bench --variant opentac --worlds 1024 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.tactile.opentac import example as E

VARIANTS = ("none", "tactile", "opentac", "opentac_plain")


def _plugins(variant: str, g: E.Gripper) -> list:
    if variant == "none":
        return []
    if variant == "tactile":
        return [_newton_only(g)]
    return [E.tactile(g, opentac_bumps=(variant == "opentac"))]


def _newton_only(g: E.Gripper) -> Any:
    """``Tactile`` on the same taxel points as the board, without the board reading."""
    from alloy.plugins.tactile import Tactile  # noqa: PLC0415
    from alloy.plugins.tactile.opentac import resolve  # noqa: PLC0415

    origins, rotations = E.pad_poses(g)
    pts = tuple((float(x), float(y)) for x, y in resolve(E.BOARD).xy)
    return Tactile(bodies=("left_finger", "right_finger"), origins=origins, rotations=rotations, points=pts)


def _count_measures(plugins: list) -> list[int]:
    calls = [0]
    for p in plugins:
        inner = p.measure

        def counted(*a: Any, _f: Any = inner, **k: Any) -> None:
            calls[0] += 1
            _f(*a, **k)
        p.measure = counted
    return calls


def _drive(sim: Any, cols: dict, g: E.Gripper, phase: str, n: int) -> None:
    for k in range(n):
        E.set_targets(sim, cols, *E.targets(g, phase, (k + 1) / n))
        sim.step_period()


def _force_sum(plugins: list) -> float:
    """Largest relative gap, over worlds and pads, between the board's loads and the engine's pad normal force."""
    r = plugins[0]
    load = r.board_load.numpy().sum(-1)                 # [W, P]
    pad = r.pad_force.numpy()[..., 0]
    return float(np.max(np.abs(load - pad) / np.maximum(np.abs(pad), 1e-9)))


def _board_only(plugin: Any, sim: Any, n: int = 200) -> float:
    """Microseconds per call of the board reading's three kernels alone, on the last step's contacts."""
    plugin._board.measure(plugin, sim.state)
    wp.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        plugin._board.measure(plugin, sim.state)
    wp.synchronize()
    return (time.perf_counter() - t) / n * 1e6


def run_variant(variant: str, worlds: int, device: str, periods: int) -> dict:
    g = E.Gripper()
    plugins = _plugins(variant, g)
    t0 = time.perf_counter()
    sim = E.simulation(worlds, device, g, plugins)
    cols = E._columns(sim)
    calls = _count_measures(plugins)
    _drive(sim, cols, g, "close", 15)                    # warm-up: compiles, contact settles
    wp.synchronize()
    build_s = time.perf_counter() - t0
    calls[0] = 0
    t1 = time.perf_counter()
    _drive(sim, cols, g, "lift", periods)
    wp.synchronize()
    dt = (time.perf_counter() - t1) / periods
    out = {"variant": variant, "worlds": worlds, "device": device, "periods": periods, "s_per_period": dt,
           "measures_per_period": calls[0] / periods if plugins else 0.0, "build_and_warmup_s": build_s}
    if variant.startswith("opentac"):
        out["board_kernels_us_per_measure"] = _board_only(plugins[0], sim)
        out["force_sum_max_rel"] = _force_sum(plugins)
        out["counts_max"] = float(plugins[0].counts.numpy().max())
        out["worlds_identical"] = bool(np.all(plugins[0].counts.numpy() == plugins[0].counts.numpy()[:1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--periods", type=int, default=25)
    a = ap.parse_args()
    if a.variant:
        print(json.dumps(run_variant(a.variant, a.worlds, a.device, a.periods)), flush=True)
        return
    rows = []
    for v in VARIANTS:
        cmd = [sys.executable, "-m", "alloy.plugins.tactile.opentac.bench", "--variant", v, "--worlds", str(a.worlds),
               "--device", a.device, "--periods", str(a.periods)]
        line = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
        rows.append(json.loads(line))
        print(line, flush=True)
    by = {r["variant"]: r for r in rows}
    m = max(by["opentac"]["measures_per_period"], 1.0)
    for v in ("opentac", "opentac_plain"):
        extra = by[v]["s_per_period"] - by["tactile"]["s_per_period"]
        print(json.dumps({"summary": v, "board_reading_ms_per_period": extra * 1e3,
                          "board_reading_us_per_measure": extra / m * 1e6,
                          "tactile_ms_per_period": (by["tactile"]["s_per_period"] - by["none"]["s_per_period"]) * 1e3,
                          "physics_ms_per_period": by["none"]["s_per_period"] * 1e3}), flush=True)


if __name__ == "__main__":
    main()
