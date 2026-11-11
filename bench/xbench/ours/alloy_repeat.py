# SPDX-License-Identifier: Apache-2.0
"""Alloy beside, SB-11: T12 repeat identity on Alloy's SB-7 target, digests per world (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/alloy_repeat.py --worlds 1024 --device cuda:0 --det on --out p1.json

One process = two runs of the same B0 batch (seed 100000, rows 0..W-1; R1: the second after ``sim.reset()``) and one
run of world 0 alone (R4 batch invariance). Each run ends with per-world digests of the final state: ``ours``
(``digest.world_digests`` on the raw ``body_q, body_qd, joint_q, joint_qd`` arrays, world-major, as for an engine)
and Alloy's own ``alloy.precision.digest.world_digests``. R2 = this script in 3 fresh processes, compared by
``score_repeat.py``. ``--det on`` is WP5's ordered MuJoCo Warp route (``ALLOY_MJW_ROUTE`` on, the SB-7 default and
Alloy's RUN_TO_RUN path); ``off`` is ``ALLOY_MJW_ROUTE=off`` (MuJoCo Warp unordered, determinism NONE).
``--pack`` picks the scene (T12 or a P3-2b embodiment; its ``alloy_target``).
The steady step time of each setting is reported (the switch's cost).
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
import digest  # noqa: E402
from alloy_t12 import versions  # noqa: E402
from packs_io import T12, load_pack, runner as make_runner, set_route, target_of  # noqa: E402

log = logging.getLogger("xbench.ours")


def _raw(model, state) -> dict[str, np.ndarray]:
    w = model.world_count
    return {n: getattr(state, n).numpy().reshape(w, -1, *getattr(state, n).numpy().shape[1:]).copy()
            for n in ("body_q", "body_qd", "joint_q", "joint_qd")}


def _digests(runner) -> dict[str, list[str]]:
    import warp as wp

    from alloy.precision.digest import world_digests

    st = runner.sim.state
    wp.synchronize()
    alloy = world_digests(runner.model, st)
    return {"ours": digest.world_digests(_raw(runner.model, st)),
            "alloy": [f"{int(a):016x}{int(b):016x}" for a, b in alloy]}


def _runner(target: str, worlds: int, device: str, det: bool):
    set_route(det)   # ALLOY_MJW_ROUTE: on = WP5's ordered route (the SB-7 default), off = standard
    return make_runner(target, worlds, device)


def _timed_run(runner, traces: np.ndarray) -> dict:
    import warp as wp

    t0 = time.perf_counter()
    runner.run(traces)
    wp.synchronize()
    wall = time.perf_counter() - t0
    steps = traces.shape[1] * runner.sim.substeps
    return {"wall_s": wall, "step_ms": 1e3 * wall / steps, **_digests(runner)}


def run(worlds: int, device: str, det: bool, pack_dir: Path = T12) -> dict:
    pack = load_pack(pack_dir)
    target = target_of(pack)
    traces = b0.applied_traces(pack["action_space"], b0.SEED_BATCH, b0.FIRST_SEED)[:worlds]
    big = _runner(target, worlds, device, det)
    first = _timed_run(big, traces)
    second = _timed_run(big, traces)   # EpisodeRunner.run resets the sim before a second episode
    alone = _timed_run(_runner(target, 1, device, det), traces[:1])
    return {"engine": "alloy", "versions": versions(), "scene": Path(pack_dir).name, "alloy_target": target,
            "seed": b0.FIRST_SEED, "worlds": worlds,
            "det": "ordered route (WP5)" if det else "standard (unordered)", "dt": float(big.sim.dt),
            "runs": {"a": first, "b_rerun": second, "n1": alone}}


def _law_run(law_id: str, cond_index: int, worlds: int, device: str, det: bool) -> dict:
    """One ``step_twin`` pass of ``worlds`` copies of one condition; digests of the final state and step time."""
    import warp as wp

    from alloy.bench.runner import RunContext, _World, step_twin
    from alloy.precision.digest import world_digests
    from alloy_laws import _route_on
    from law_pack import law_twin

    spec, twin = law_twin(law_id)
    cond = twin.conditions(False)[cond_index]
    refused = _route_on(twin, law_id) if det else []
    t0 = time.perf_counter()
    run = step_twin(spec, twin, [_World(cond, {}) for _ in range(worlds)], RunContext(spec, device, 0, "mujoco", False))
    wp.synchronize()
    wall = time.perf_counter() - t0
    alloy = world_digests(run.model, run.state)
    return {"wall_s": wall, "step_ms": 1e3 * wall / run.monitors["steps"], "scene_id": cond.id, "refused": refused,
            "ours": digest.world_digests(_raw(run.model, run.state)),
            "alloy": [f"{int(a):016x}{int(b):016x}" for a, b in alloy]}


def run_law(law_id: str, cond_index: int, worlds: int, device: str, det: bool) -> dict:
    a = _law_run(law_id, cond_index, worlds, device, det)
    b = _law_run(law_id, cond_index, worlds, device, det)
    n1 = _law_run(law_id, cond_index, 1, device, det)
    from alloy_laws import _route_state

    return {"engine": "alloy", "versions": versions(), "scene": f"{law_id}/{a['scene_id']}", "worlds": worlds,
            "det": "ordered route (WP5)" if det else "standard (unordered)",
            "route": _route_state(det, a["refused"] + b["refused"] + n1["refused"]),   # in force, not asked for
            "runs": {"a": a, "b_rerun": b, "n1": n1},
            "note": "R1 here is a rebuild in the same process (as D1.rerun does), not a state reset"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--det", choices=("on", "off"), default="on")
    p.add_argument("--pack", type=Path, default=T12, help="scene pack (T12 or a P3-2b embodiment)")
    p.add_argument("--law", help="law scene instead of a scene pack, e.g. A5.slide")
    p.add_argument("--cond", type=int, default=0, help="condition index of --law")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    det = a.det == "on"
    res = run_law(a.law, a.cond, a.worlds, a.device, det) if a.law else run(a.worlds, a.device, det, a.pack)
    a.out.write_text(json.dumps(res, indent=1))
    r = res["runs"]
    same = np.mean([x == y for x, y in zip(r["a"]["ours"], r["b_rerun"]["ours"], strict=True)])
    log.info("%s det=%s: R1 %.4f, R4 world 0 %s, step %.3f ms", res["scene"], a.det, same,
             r["a"]["ours"][0] == r["n1"]["ours"][0], r["a"]["step_ms"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
