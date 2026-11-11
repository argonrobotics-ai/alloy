# SPDX-License-Identifier: Apache-2.0
"""Alloy beside, SB-7: Alloy's own episodes scored by Alloy online AND by the rivals' scorer path (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/alloy_sb7.py OUT_DIR --seeds 100000-100007 [--device cuda:0] [--chunk 1024]

Per seed, one ``EpisodeRunner`` batch of 2,048 episodes on the pack's Alloy target (``--pack``, T12 by default: WP7's
SB-7 run, MuJoCo-native contacts, the route in force: ``ALLOY_MJW_ROUTE``, recorded in the JSON), with a
``score_s1.PipelineMonitors`` step observer in the same process: Newton-pipeline contacts on the pre-step state, the
exact path ``sb7_watch.py`` runs on a rival's exported frames (WP7: one scorer for all engines; its pipeline collides
``target_model``'s twin of the live model, mesh geoms as convex hulls). Each batch writes
``batch_<seed>_2048.s1.json`` in ``sb7_watch``'s schema (so ``sb7_watch.py summary OUT_DIR`` pools it), with the
online verdict beside (``online_*``). An S1 world whose tripped rows are all robot-contact-excusable is replayed
alone with frames recorded (``alloy_t12``) and re-scored with pre- or post-step robot touch, as for the rivals.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

import b0  # noqa: E402
from alloy_t12 import Recorder, versions  # noqa: E402
from packs_io import T12, load_pack, runner as make_runner, set_route, target_model, target_of  # noqa: E402
from sb7_watch import EXCUSED_ROWS, _post_touch, _s1_rows  # noqa: E402
from score_s1 import PipelineMonitors, verdict  # noqa: E402

log = logging.getLogger("xbench.ours")


def _trip_frames(pack: dict, seed: int, k: int, device: str):
    """Episode (seed, k) alone, frames every step (the ordered route makes it equal to its batch run)."""
    trace = b0.applied_traces(pack["action_space"], b0.SEED_BATCH, seed)[k:k + 1]
    runner = make_runner(target_of(pack), 1, device)
    rec = Recorder(1)
    runner.sim.add_step_observer(rec)
    rec.start(runner.sim.state, 1)
    res = runner.run(trace)
    t = np.arange(len(rec.q)) * float(runner.sim.dt)
    return np.stack(rec.q), np.stack(rec.qd), t, int(res.severity[0])


def run_seed(pack: dict, seed: int, runner, pm: PipelineMonitors, device: str, replays: dict, row0: int = 0) -> dict:
    """Episodes ``row0 .. row0 + worlds - 1`` of ``seed``'s batch (rows of one seed can run in chunks)."""
    traces = b0.applied_traces(pack["action_space"], b0.SEED_BATCH, seed)[row0:row0 + runner.worlds]
    if runner.episodes:   # what EpisodeRunner.run(reset=True) does, so both monitor sets start from the same state
        runner.sim.reset()
    pm.reset(runner.sim.state)
    t0 = time.perf_counter()
    online = runner.run(traces, reset=False)
    wall = time.perf_counter() - t0
    grow = pm.needed()
    if grow is not None:   # the scorer's pooled contact rows overflowed: run the seed again with a larger buffer
        log.warning("seed %d: scorer contact buffers overflowed (%s); again at %s per world", seed,
                    pm.contact_stats(), grow)
        pm.resize(*grow)
        return run_seed(pack, seed, runner, pm, device, replays, row0)
    extra = pm.report()
    res = verdict(extra, pm.limits)
    res.pop("monitors")
    res["episodes"] = [[seed, row0 + int(i)] for i in res["s1_worlds"]]
    res["checks"] = {}
    for i in res["s1_worlds"]:
        rows = _s1_rows(extra, pm.limits, i)
        chk: dict = {"s1_rows": rows, "confirmed": True}
        if rows and set(rows) <= EXCUSED_ROWS:
            q, qd, t, _ = _trip_frames(pack, seed, row0 + i, device)
            chk.update(_post_touch(q, qd, t, replays, device, target_of(pack)))
            chk["confirmed"] = chk["union_touch_s1"]
        res["checks"][str(i)] = chk
    from alloy.bully.severity import world_cause
    from alloy.precision.types import HealthReport

    on = np.nonzero(online.severity == 1)[0]
    res.update({"worlds": int(traces.shape[0]), "seeds": [seed], "row0": row0, "row": "alloy", "wall_s": wall,
                "scorer_contacts": pm.contact_stats(),
                "online_s1_worlds": on.tolist(), "online_s1_causes": {str(int(i)): world_cause(
                    HealthReport.ok(len(online.severity)), online.monitors, pm.limits, int(i))[0] for i in on}})
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--seeds", default="100000-100007", help="first-last, inclusive")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--worlds", type=int, default=b0.SEED_BATCH, help="rows 0..worlds-1 of each seed (tests)")
    p.add_argument("--chunk", type=int, help="worlds per run (default --worlds): a seed's rows run in chunks of this "
                   "many (GPU memory); the episodes are the same")
    p.add_argument("--pack", type=Path, default=T12, help="scene pack (T12 or a P3-2b embodiment)")
    a = p.parse_args()
    pack = load_pack(a.pack)
    target = target_of(pack)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    chunk = a.chunk or a.worlds
    if a.worlds % chunk:
        p.error(f"--chunk {chunk} must divide --worlds {a.worlds}")
    hulls = target_model(target, chunk, a.device)   # the scorer's twin (mesh geoms as hulls), before the live sim
    runner = make_runner(target, chunk, a.device)
    pm = PipelineMonitors(runner.model, "pre", target, collision_model=hulls)
    runner.sim.add_step_observer(pm.observer)
    route = {"ALLOY_MJW_ROUTE": set_route(None), "alloy_target": target, "versions": versions(),
             "dt": float(runner.sim.dt)}
    replays: dict = {}
    for seed, row0 in ((s, r) for s in range(lo, hi + 1) for r in range(0, a.worlds, chunk)):
        tag = f"{seed}_{a.worlds}" if chunk == a.worlds else f"{seed}_r{row0}_{chunk}"
        out = a.out_dir / f"batch_{tag}.s1.json"
        if out.exists():
            continue
        res = run_seed(pack, seed, runner, pm, a.device, replays, row0)
        res["route"] = route
        out.write_text(json.dumps(res, indent=1))
        log.info("seed %d rows %d+: scorer S1 %s, online S1 %s (%.0f s; scorer contacts %s)", seed, row0,
                 res["s1_worlds"], res["online_s1_worlds"], res["wall_s"], res["scorer_contacts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
