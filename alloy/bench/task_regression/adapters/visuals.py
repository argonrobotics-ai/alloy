# SPDX-License-Identifier: Apache-2.0
"""SOFT-3 visuals: the arm packs' visual layer adds display geometry only. Each target's Alloy CPU route (bully target,
1 world) runs 2 s of a fixed sinusoid control with the layer (ALLOY_ROBOT_VISUALS=1) and without it (=0), each setting
in its own process; every period's state (joint_q, joint_qd, body_q, body_qd) must hash the same. The control sequence
and targets are the source run's ([internal note], row "SOFT-3 visuals" of
task-regression-sources.md); the MuJoCo C half of that run is the gate test tests/tasks/test_emb_visuals.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

#: the source run's 11 targets: emb e1-e4, their route / stiff variants and T12
TARGETS: tuple[str, ...] = (
    "emb_e1_fr3", "emb_e1_fr3_route", "emb_e2_panda", "emb_e2_panda_route", "emb_e3_ur10e_2f85",
    "emb_e3_ur10e_2f85_route", "emb_e3_ur10e_2f85_stiff", "emb_e4_fr3_bimanual", "emb_e4_fr3_bimanual_route",
    "emb_e4_fr3_bimanual_stiff", "task_t12")
PERIODS = 100                          # 2 s at the targets' 20 ms period
SWITCH = "ALLOY_ROBOT_VISUALS"         # alloy.scene.robot_visuals.ENV_SWITCH ("0": loads skip the layer)


def layer_bitwise(seeds: Sequence[int], device: str) -> dict[str, float]:
    """Run every target with and without the layer (two processes side by side) and compare the per-period hashes."""
    import tempfile

    from alloy.bench.task_regression.adapters import keep_json

    with tempfile.TemporaryDirectory(prefix="taskreg-visuals-") as tmp:
        outs = {v: Path(tmp) / f"visuals_{v}.json" for v in ("1", "0")}
        procs = [subprocess.Popen([sys.executable, "-m", __name__, "--out", str(out), *TARGETS],
                                  env={**os.environ, SWITCH: v}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  text=True) for v, out in outs.items()]
        errs = [p.communicate()[1] for p in procs]
        for p, err in zip(procs, errs, strict=True):
            if p.returncode:
                raise RuntimeError(f"visuals rollout exited {p.returncode}: {err[-2000:]}")
        on, off = (json.loads(outs[v].read_text()) for v in ("1", "0"))
    rows = [_compare(name, on[name], off[name]) for name in TARGETS]
    keep_json("soft3_visuals_rows.json", rows)
    return {"targets": float(len(rows)), "differing": float(sum(not r["identical"] for r in rows)),
            "attached": float(sum(r["shapes_on"] > r["shapes_off"] for r in rows))}


def _compare(name: str, on: dict, off: dict) -> dict:
    same = [a == b for a, b in zip(on["hashes"], off["hashes"], strict=True)]
    first = next((i for i, s in enumerate(same) if not s), None)
    return {"target": name, "identical": all(same), "first_differing_period": first,
            "shapes_on": on["shapes"], "shapes_off": off["shapes"], "wall_s": [on["wall_s"], off["wall_s"]]}


def ctrl_sequence(low, high, initial, max_delta, periods: int = PERIODS):
    """The source run's control: sinusoids inside the legal box, rate-limited like the bully's actions."""
    import numpy as np

    t = np.arange(periods)[:, None] * 0.02
    k = np.arange(low.size)[None, :]
    mid, half = 0.5 * (low + high), 0.5 * (high - low)
    want = mid + 0.6 * half * np.sin(2 * np.pi * (0.4 + 0.07 * k) * t + 0.9 * k)
    out, cur = [], initial.astype(np.float64).copy()
    for row in want:
        cur = cur + np.clip(row - cur, -max_delta, max_delta)
        out.append(np.clip(cur, low, high))
    return np.asarray(out)


def rollout_hashes(name: str) -> dict:
    """One target on the CPU route, 1 world: the shape count and each period's state sha256."""
    import time

    import numpy as np

    import alloy.bully.tasks  # noqa: F401  (registers task_t12)
    import alloy.tasks.emb_targets  # noqa: F401  (registers emb_*)
    from alloy.bully.targets import get_target

    t0 = time.perf_counter()
    target = get_target(name)
    a = target.actions
    sim = target.make_sim(1, "cpu")
    hashes = []
    for row in ctrl_sequence(a.low, a.high, a.initial, a.max_delta_per_tick):
        sim.control.mujoco.ctrl.assign(row.astype(np.float32))
        sim.step_period()
        h = hashlib.sha256()
        for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
            h.update(getattr(sim.state, field).numpy().tobytes())
        hashes.append(h.hexdigest())
    return {"shapes": int(sim.model.shape_count), "hashes": hashes, "wall_s": round(time.perf_counter() - t0, 1)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="per-period state hashes of bully targets on the CPU route")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("targets", nargs="+")
    args = ap.parse_args(argv)
    args.out.write_text(json.dumps({name: rollout_hashes(name) for name in args.targets}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
