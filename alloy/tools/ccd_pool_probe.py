# SPDX-License-Identifier: Apache-2.0
"""T12 with MuJoCo Warp's CCD scratch pool sized per world, or left at the default (SCR-WP8-7 M1 gate).

    python -m alloy.tools.ccd_pool_probe --worlds 64 --periods 25 --slots 32 --out rows.jsonl [--graph auto|off]

One arm per process, so the memory high-water belongs to that arm (``--slots 0``: MuJoCo Warp's default, naconmax
slots). The scene is ``task_t12`` as ``alloy.integration.t12_graph`` builds it: the card at the bully's per-world
capacities (nconmax 256, njmax 1024), the recorded start state and command, WP5's ordered route, graphs on. The arm
adds the entry kwarg ``alloy:ccd_slots`` (``alloy.precision.mjw_ccd``) and holds the start command.

One JSON row per arm is appended to ``--out``:
- ``naccdmax``;
- the worlds whose ``Data.overflow`` had the CCD bit after any period (gate: none);
- worlds INVALID in health;
- the mempool high-water and the device memory in use at the end, which includes the CUDA graph pool;
- the median period time over periods 3 on (after warm-up and capture).

Coordinator gate (2026-11-06 13:32Z): 64 worlds, 0 overflow, the same speed as the default. The 16,384-world row goes
to the reconciliation pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.tools.ccd_pool_probe")


def build(worlds: int, device: str, slots: int, graph: str = "auto") -> Any:
    """``t12_graph.make_sim``'s simulation, with ``slots`` CCD slots per world on its MuJoCo entry (0: default)."""
    from alloy.bully.tasks import _scene_info, _set_initial
    from alloy.integration import t12_graph as T
    from alloy.integration.wiring import build_simulation
    from alloy.precision.mjw_ccd import with_ccd_slots

    spec = T._spec()
    info = _scene_info(spec)
    built = T._assemble(worlds, device, spec.control_dt)
    if slots:
        built = with_ccd_slots(built, slots)
    _set_initial(built.model, info, worlds)
    with wp.ScopedDevice(device):
        sim = build_simulation(built, "standard", graph=graph, mjw_route=True)
    sim.control.mujoco.ctrl.assign(np.tile(info.ctrl0, worlds).astype(np.float32))
    return sim


def _mujoco(sim: Any) -> Any:
    return next(s for s in sim.parts.solvers.values() if type(s).__name__ == "SolverMuJoCo")


def _used_mib(device: str) -> float:
    d = wp.get_device(device)
    return (d.total_memory - d.free_memory) / 2**20 if d.is_cuda else 0.0


def run(worlds: int, periods: int, slots: int, device: str, graph: str) -> dict:
    from alloy._vendor.mujoco_warp import OverflowType

    wp.init()
    t0 = time.perf_counter()
    sim = build(worlds, device, slots, graph)
    build_s = time.perf_counter() - t0
    data = _mujoco(sim).mjw_data
    ccd_worlds = np.zeros(worlds, bool)
    times = []
    for _ in range(periods):
        wp.synchronize_device(device)
        t = time.perf_counter()
        sim.step_period()
        wp.synchronize_device(device)
        times.append(time.perf_counter() - t)
        ccd_worlds |= (data.overflow.numpy() & int(OverflowType.CCD)) != 0
    health = getattr(sim, "health", None)
    invalid = list(health.poll().invalid_worlds) if health is not None else None
    cuda = wp.get_device(device).is_cuda
    return {"worlds": worlds, "periods": periods, "slots": slots, "graph": graph, "device": device,
            "naccdmax": int(data.naccdmax), "naconmax": int(data.naconmax),
            "ccd_overflow_worlds": np.flatnonzero(ccd_worlds).tolist(), "invalid_worlds": invalid,
            "mempool_high_mib": round(wp.get_mempool_used_mem_high(device) / 2**20, 1) if cuda else None,
            "device_used_mib": round(_used_mib(device), 1),
            "period_ms_median": round(float(np.median(times[3:] if len(times) > 3 else times)) * 1e3, 3),
            "build_s": round(build_s, 1), "use_graph": bool(getattr(sim, "use_graph", False))}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="python -m alloy.tools.ccd_pool_probe")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--periods", type=int, default=25)
    p.add_argument("--slots", type=int, default=32, help="CCD slots per world (0: MuJoCo Warp's default)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--graph", default="auto", choices=("auto", "off"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    row = run(a.worlds, a.periods, a.slots, a.device, a.graph)
    with open(Path(a.out), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    return 0 if not row["ccd_overflow_worlds"] else 1


if __name__ == "__main__":
    sys.exit(main())
