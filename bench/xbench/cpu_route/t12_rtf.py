# SPDX-License-Identifier: Apache-2.0
"""T12 RTF row of the CPU route (SCR-WP8-6 gate 3): Alloy's SB-7 target ``task_t12`` on the MuJoCo C core, 1 world.

    OMP_NUM_THREADS=1 PYTHONPATH=. python bench/xbench/cpu_route/t12_rtf.py --backend auto --out rtf.jsonl

The target is built as WP7's SB-7 runs it (scene card, routed solver, dt 0.625 ms, 32 steps per 20 ms tick) with
``ALLOY_CPU_ROUTE=mjc``. It plays B0 seed 100000 row 0 (``b0.applied_traces``) one control row per tick: one
``DirectSim.step_period`` per tick (the CPU route fast path, one route call per world and tick) or, with
``--per-step``, the per-step loop (one solver call per physics step, forced by a no-op step observer, the path
per-step hooks need). After a
warm-up tick, ``--repeats`` rounds of ``--ticks`` ticks each start from the reset state; the row reports the median
and min ms per physics step, RTF (simulated / wall time), the host load and the route's counters. ``--backend
numpy`` forces the numpy plug-ins (the fallback when WP2's extension is missing). Run several processes for the
spread; the host is shared.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parent / "ours")]

log = logging.getLogger("xbench.cpu_route")


def build(backend: str):
    """The task_t12 DirectSim on the CPU route (``backend`` "numpy" forces the numpy plug-ins)."""
    os.environ["ALLOY_CPU_ROUTE"] = "mjc"
    from alloy.bully.targets import get_target

    sim = get_target("task_t12").make_sim(1, "cpu")
    if backend == "numpy":
        s = sim.solver
        s.route = s.route.replace(backend="numpy")
        s._make_outputs()
    return sim


def play(sim, trace: np.ndarray, ticks: int) -> float:
    """Wall seconds for ``ticks`` ticks from the reset state."""
    sim.reset()
    ctrl = sim.control.mujoco.ctrl
    t0 = time.perf_counter()
    for t in range(ticks):
        ctrl.assign(trace[min(t, trace.shape[0] - 1)].astype(np.float32))
        sim.step_period()
    return time.perf_counter() - t0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("auto", "numpy"), default="auto")
    p.add_argument("--per-step", action="store_true", help="DirectSim's per-step loop instead of the fast path")
    p.add_argument("--ticks", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import b0
    from packs_io import load_pack

    pack = load_pack()
    trace = b0.applied_traces(pack["action_space"], b0.SEED_BATCH, b0.FIRST_SEED)[0]
    sim = build(a.backend)
    if a.per_step:
        sim.add_step_observer(lambda *args: None)                         # forces DirectSim's per-step loop
    play(sim, trace, 1)                                                   # warm-up (kernels, first push)
    load0 = os.getloadavg()[0]
    walls = [play(sim, trace, a.ticks) for _ in range(a.repeats)]
    steps = a.ticks * sim.substeps
    med = float(np.median(walls))
    row = {"scene": "task_t12", "route": "mjc_core", "backend": sim.solver.backend, "per_step": a.per_step,
           "dt": float(sim.dt), "substeps": sim.substeps, "ticks": a.ticks, "repeats": a.repeats,
           "ms_per_step": 1e3 * med / steps, "ms_per_step_min": 1e3 * min(walls) / steps,
           "rtf": steps * float(sim.dt) / med, "rtf_best": steps * float(sim.dt) / min(walls),
           "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
           "load1": [load0, os.getloadavg()[0]], "pid": os.getpid(), "report": sim.solver.route_report()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print(f"{row['backend']:5s} per_step={a.per_step} {row['ms_per_step']:.4f} ms/step "
          f"(min {row['ms_per_step_min']:.4f})  RTF {row['rtf']:.2f} (best {row['rtf_best']:.2f})  "
          f"load {row['load1'][0]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
