# SPDX-License-Identifier: Apache-2.0
"""Offline CPU MuJoCo reference for R03 (never in the simulation loop): every condition, scored together.

    python -m bench.scenarios._rigid_ref r03 [--out bench/results] [--horizon 4.0]

Newton's native CPU MuJoCo backend (SolverMuJoCo(use_mujoco_cpu=True)) holds one world, and R03's law needs all 74
conditions in one evaluate() (onset brackets, 16 azimuths). So each condition runs as its own one-world harness
run, with the harness's own stepping and sample cadence, and InclineFriction.evaluate() scores the stacked samples.
Rows go to <out>/rigid/R03_incline_friction.ref.jsonl with solver "ref:mujoco_cpu".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from bench import harness as H
from bench.scenarios import rigid as R

REF_CONFIGS = ("ref_mujoco_cpu_elliptic", "ref_mujoco_cpu_pyramidal")


class _Collect(R.InclineFriction):
    """R03 with evaluate() replaced by a stash of the per-world samples."""

    stash: dict = {}

    def evaluate(self, b: H.Built, rec: dict) -> None:
        x = b.extra
        _Collect.stash = {"ts": list(x["ts"]), "pos": list(x["pos"]), "vel": list(x["vel"]),
                          "p_kick": x["p_kick"].numpy().copy(), "cs": list(x["cs"]), "mu": x["mu"].copy(),
                          "theta": x["theta"].copy(), "phi": x["phi"].copy(), "g": x["g"]}


def _run_condition(scn: _Collect, cfg: H.Config, cond: dict, opts: H.RunOptions) -> tuple[dict, dict]:
    full = R.R03_COND
    R.R03_COND = (cond,)  # build() tiles conditions as w % len(R03_COND): world 0 runs this one
    try:
        rec = H._base_record(scn, cfg, 1, opts)
        H._execute(scn, cfg, 1, opts, rec)
    finally:
        R.R03_COND = full
    return rec, dict(_Collect.stash)


def _merge(parts: list[dict]) -> SimpleNamespace:
    """One fake Built whose world w is condition w, for InclineFriction.evaluate()."""
    ts = parts[0]["ts"]
    for p in parts:
        if p["ts"] != ts:
            raise RuntimeError("sample times differ between condition runs")
    extra = {"ts": ts,
             "pos": [np.concatenate([p["pos"][i] for p in parts]) for i in range(len(ts))],
             "vel": [np.concatenate([p["vel"][i] for p in parts]) for i in range(len(ts))],
             "p_kick": wp.array(np.concatenate([p["p_kick"] for p in parts]), dtype=wp.vec3, device="cpu"),
             "cs": [p["cs"][0] for p in parts], "mu": np.concatenate([p["mu"] for p in parts]),
             "theta": np.concatenate([p["theta"] for p in parts]), "phi": np.concatenate([p["phi"] for p in parts]),
             "g": parts[0]["g"]}
    return SimpleNamespace(extra=extra, nworld=len(parts))


def r03_reference(out_dir: Path, horizon: float | None) -> list[dict]:
    scn = _Collect()
    opts = H.RunOptions(device="cpu", horizon_s=horizon, budget_s=1e9, repeat_max_nworld=0, out_dir=out_dir,
                        command="python -m bench.scenarios._rigid_ref r03")
    rows = []
    for cfg in (c for c in scn.configs if c.name in REF_CONFIGS):
        t0 = time.perf_counter()
        recs, parts = zip(*(_run_condition(scn, cfg, c, opts) for c in R.R03_COND))
        rec = H._base_record(scn, cfg, len(parts), opts)
        rec.update(solver="ref:mujoco_cpu", device_name="cpu", steps=recs[0]["steps"],
                   complete=all(r["complete"] for r in recs), stable=all(r["stable"] for r in recs),
                   ms_per_step=float(np.mean([r["ms_per_step"] for r in recs])),
                   single_world_realtime_factor=float(np.mean([r["single_world_realtime_factor"] for r in recs])),
                   notes=f"{len(parts)} one-world CPU MuJoCo runs (one per condition) scored together; "
                         f"wall {time.perf_counter() - t0:.0f} s")
        pens = [r["penetration_p99_mm"] for r in recs if r["penetration_p99_mm"] is not None]
        rec["penetration_p99_mm"] = float(max(pens)) if pens else None
        if rec["stable"]:
            R.InclineFriction.evaluate(scn, _merge(list(parts)), rec)
        rows.append(rec)
        path = out_dir / "rigid" / "R03_incline_friction.ref.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=H._json_default) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.scenarios._rigid_ref")
    ap.add_argument("what", choices=["r03"])
    ap.add_argument("--out", type=Path, default=H.REPO / "bench" / "results")
    ap.add_argument("--horizon", type=float, default=None)
    a = ap.parse_args()
    wp.config.quiet = True
    if hasattr(wp, "LOG_WARNING"):  # warp >= 1.18 prints module loads at LOG_INFO (as bench/run.py)
        wp.config.log_level = wp.LOG_WARNING
    wp.init()
    for r in r03_reference(a.out, a.horizon):
        m = r["metrics"]
        print(json.dumps({"config": r["config"], "ms_per_step": r["ms_per_step"], "stable": r["stable"],
                          **{k: m.get(k) for k in ("onset_err_deg", "accel_err_pct", "stop_err_pct",
                                                   "creep_mm_per_10s", "isotropy_pct", "law_pass")}}))


if __name__ == "__main__":
    main()
