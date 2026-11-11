# SPDX-License-Identifier: Apache-2.0
"""Offline CPU MuJoCo reference for the G01 rigid-sphere collapse (never in the simulation loop).

    python -m bench.scenarios._granular_ref [--out bench/results] [--horizon 3.0] [--config spheres_d10_mujoco,...]

CPU MuJoCo 3.16 (float64 MjData) through Newton's SolverMuJoCo(use_mujoco_cpu=True, update_data_interval=0), the
same scene, solver settings, dt and measurement as the GPU sphere configs. Newton's CPU MuJoCo backend holds one
world, so each collapse aspect runs as its own one-world harness run (with a bitwise repeat), and a combined row
gathers the three. Sphere jitter: world 0's draws for every aspect (the GPU worlds 1 and 2 draw later in the same
stream), so the start states match the GPU's only for a = 0.5. Rows go to
<out>/granular/G01_sand_collapse_repose.ref.jsonl with solver "ref:mujoco_cpu".
"""
from __future__ import annotations

import argparse
import copy
import math
from dataclasses import replace

from bench import harness as H
from bench.scenarios import granular as G


class _OneAspect(G.SandCollapse):
    """G01 with a single collapse condition (one world per run)."""

    def __init__(self, index: int) -> None:
        self.index = index

    def conditions(self, config: H.Config) -> list:
        return [super().conditions(config)[self.index]]


def ref_config(cfg: H.Config) -> H.Config:
    return replace(cfg, name=f"ref_mujoco_cpu_{cfg.name}", solver="mujoco", cuda_graph=False, max_nworld=1,
                   params={**cfg.params, "use_mujoco_cpu": True, "update_data_interval": 0})


def _combine(recs: list[dict], name: str) -> dict:
    """One row over the three aspects: worst runout error, per-condition values merged, timing summed."""
    out = copy.deepcopy(recs[0])
    per = {}
    for r in recs:
        per.update(r["metrics"].get("per_condition", {}))
    errs = [v for r in recs for d in r["metrics"].get("per_condition", {}).values()
            for v in d.get("runout_err_pct", []) if v is not None and math.isfinite(v)]
    worst = max(errs, key=abs) if errs else None
    pens = [r["penetration_p99_mm"] for r in recs if r["penetration_p99_mm"] is not None]
    out.update(config=name, nworld=1, accuracy_error=worst, stable=all(r["stable"] for r in recs),
               complete=all(r["complete"] for r in recs), repeat_identical=all(r["repeat_identical"] for r in recs),
               penetration_p99_mm=max(pens) if pens else None,
               ms_per_step=sum(r["ms_per_step"] or 0.0 for r in recs) / len(recs),
               notes="combined row: max over the three one-world aspect runs (penetration: max of the p99s; "
                     "ms_per_step: mean)")
    checks = dict(runout=bool(errs) and all(abs(e) <= 10.0 for e in errs))
    out["metrics"] = dict(recs[0]["metrics"], per_condition=per, runout_err_pct=worst, conditions_covered=True,
                          conditions=[c for r in recs for c in r["metrics"].get("conditions", [])],
                          law_checks=checks, law_pass=all(checks.values()))
    out["solver"] = "ref:mujoco_cpu"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(H.REPO / "bench" / "results"))
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--config", default="spheres_d10_mujoco,spheres_d10_roll0.5mm_mujoco")
    a = ap.parse_args()
    H.setup_logging()
    import warp as wp

    wp.config.quiet = True
    if hasattr(wp, "LOG_WARNING"):  # warp >= 1.18 prints module loads at LOG_INFO
        wp.config.log_level = wp.LOG_WARNING
    wp.init()
    base = G.SandCollapse()
    opts = H.RunOptions(device="cpu", horizon_s=a.horizon, budget_s=1e9, repeat_max_nworld=1,
                        out_dir=H.Path(a.out), command="python -m bench.scenarios._granular_ref " +
                        " ".join(f"--{k} {v}" for k, v in vars(a).items() if v is not None))
    for name in a.config.split(","):
        cfg = ref_config(next(c for c in base.configs if c.name == name))
        recs = []
        for k in range(len(G.COLLAPSE_L0)):
            scn = _OneAspect(k)
            rec = H.run_one(scn, cfg, 1, opts)
            a_k = list(G.COLLAPSE_L0)[k]
            rec.update(solver="ref:mujoco_cpu", config=f"{cfg.name}_a{a_k:g}")
            recs.append(rec)
            H.log.info("%s a=%g %s ms/step=%.3g acc=%s pen=%s repeat=%s", cfg.name, a_k, rec["status"],
                       rec["ms_per_step"] or math.nan, rec["accuracy_error"], rec["penetration_p99_mm"],
                       rec["repeat_identical"])
        for rec in recs + ([_combine(recs, cfg.name)] if all(r["status"] == "ok" for r in recs) else []):
            _write_ref(rec, H.Path(a.out))
    return 0


def _write_ref(rec: dict, out: H.Path) -> None:
    path = out / "granular" / f"{rec['scenario']}.ref.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(H.json.dumps(rec, default=H._json_default) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
