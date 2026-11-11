# SPDX-License-Identifier: Apache-2.0
"""T12 SB-7 S1 on the CPU route (SCR-WP8-6 gate 5): B0 episodes on the MuJoCo C core, frames every physics step, scored
offline by Alloy's SB-7 checker as the rival rows are (``bench/xbench/ours/sb7_watch.py``).

    ALLOY_CPU_ROUTE=mjc OMP_NUM_THREADS=1 PYTHONPATH=. python bench/xbench/cpu_route/sb7_cpu.py OUT_DIR \\
        --seed 100000 --rows 0-127 --batch 32
    PYTHONPATH=. python bench/xbench/ours/sb7_watch.py watch OUT_DIR --device cpu   # touch OUT_DIR/STOP first
    PYTHONPATH=. python bench/xbench/ours/sb7_watch.py summary OUT_DIR

The target is WP7's ``task_t12`` (scene card, routed solver, dt 0.625 ms, 32 steps per 20 ms tick) built with the
CPU route. A batch is rows ``row0 .. row0 + B - 1`` of ``b0.applied_traces(space, 2048, seed)`` (the episodes every
engine plays), one world each; each tick applies the trace row through ``bind_actions`` (as ``EpisodeRunner``) and
runs one route call per world (``DirectSim.step_period``'s CPU route fast path) with the trajectory buffer on, so
every physics step's body_q / body_qd is recorded (the numpy equivalent of the ABI's ``traj_*`` rows when the C
extension is missing). Frame 0 is the reset state. Each batch is ``batch_<seed>r<row0>_<B>.npz`` in sb7_watch's
contract plus a ``.done`` marker. A world the route froze non-finite keeps NaN frames (S1 ``nonfinite`` for the
scorer) and is listed in ``meta.nonfinite_worlds``. Batches that exist are skipped, so a killed run resumes.
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


def build(worlds: int, backend: str):
    os.environ["ALLOY_CPU_ROUTE"] = "mjc"
    from alloy.bully.targets import get_target

    target = get_target("task_t12")
    sim = target.make_sim(worlds, "cpu")
    s = sim.solver
    if backend == "numpy":
        s.route = s.route.replace(backend="numpy")
    s.set_trajectory(sim.substeps)                       # rebuilds the outputs with the backend asked for
    return target, sim


def play(target, sim, traces: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Frames [T, W, B, 7] / [T, W, B, 6] (float32) of the batch, and the wall time."""
    from alloy.bully.actions import bind_actions

    s, w = sim.solver, traces.shape[0]
    sim.reset()
    bound = bind_actions(sim, sim.model, target.action_kind, tuple(target.action_index), traces)
    nb = sim.model.body_count // w
    T = 1 + bound.ticks * sim.substeps
    q, qd = np.empty((T, w, nb, 7), np.float32), np.empty((T, w, nb, 6), np.float32)
    q[0] = sim.state.body_q.numpy().reshape(w, nb, 7)
    qd[0] = sim.state.body_qd.numpy().reshape(w, nb, 6)
    t0 = time.perf_counter()
    for tick in range(bound.ticks):
        bound.apply(tick)
        frozen = s._frozen.copy()                        # skipped this tick: its frames stay non-finite
        sim.step_period()
        rows = slice(1 + tick * sim.substeps, 1 + (tick + 1) * sim.substeps)
        for k, out in enumerate(s.outs):
            q[rows, k] = np.nan if frozen[k] else out.traj_body_q[: sim.substeps]
            qd[rows, k] = np.nan if frozen[k] else out.traj_body_qd[: sim.substeps]
    return q, qd, time.perf_counter() - t0


def write_batch(path: Path, sim, seed: int, row0: int, q: np.ndarray, qd: np.ndarray, wall: float) -> None:
    from alloy_t12 import body_names, versions

    rep = sim.solver.route_report()
    bad = [int(i) for i in np.flatnonzero(sim.solver._frozen)]   # non-finite: their NaN frames are S1 nonfinite
    meta = {"engine": "alloy", "engine_version": versions()["alloy"], "versions": versions(), "test": "task_t12_b0",
            "alloy_target": "task_t12", "pack": "t12", "dt": float(sim.dt), "substeps": int(sim.substeps),
            "record_every": 1, "seed": seed, "seeds": [seed], "row0": row0, "worlds": int(q.shape[1]),
            "row": "cpu_route", "nonfinite_worlds": bad, "wall_s": wall,
            "settings": {"route": "get_target('task_t12'), ALLOY_CPU_ROUTE=mjc (MuJoCo C core)",
                         "backend": rep["backend"], "solver": rep["settings"], "noslip": rep["noslip"],
                         "line": rep["line"]},
            "route_counters": rep["counters"],
            "notes": "SCR-WP8-6 CPU route; frames every physics step from the route's trajectory rows"}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), body_names=np.array(body_names(sim.model)),
                        t=np.arange(q.shape[0]) * float(sim.dt), body_q=q, body_qd=qd)
    tmp.rename(path)
    path.with_suffix(".done").touch()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("alloy", "newton", "warp"):
        logging.getLogger(name).setLevel(logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--seed", type=int, default=100_000)
    p.add_argument("--rows", default="0-511", help="first-last row of the seed's 2,048, inclusive")
    p.add_argument("--batch", type=int, default=32, help="worlds per batch npz")
    p.add_argument("--backend", choices=("auto", "numpy"), default="auto")
    a = p.parse_args()
    import b0
    from packs_io import load_pack

    lo, hi = (int(x) for x in a.rows.split("-"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    traces = b0.applied_traces(load_pack()["action_space"], b0.SEED_BATCH, a.seed)
    target, sim = build(a.batch, a.backend)
    for row0 in range(lo, hi + 1, a.batch):
        path = a.out_dir / f"batch_{a.seed}r{row0}_{a.batch}.npz"
        if path.with_suffix(".done").exists():
            continue
        q, qd, wall = play(target, sim, traces[row0:row0 + a.batch])
        write_batch(path, sim, a.seed, row0, q, qd, wall)
        steps = (q.shape[0] - 1) * a.batch
        log.info("%s: %.0f s, %.3f ms per world-step, frozen %s", path.name, wall, 1e3 * wall / steps,
                 np.flatnonzero(sim.solver._frozen).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
