# SPDX-License-Identifier: Apache-2.0
"""Alloy bench CLI.

    python -m bench.run --family rigid --device cuda:0 --ladder 1,16,256
    python -m bench.run --family _example --device cpu --ladder 1,16          # harness smoke test (Mac CPU)
    python -m bench.run --family all --list                                    # what would run
    python -m bench.run --summary bench/results/_example                       # table of recorded rows

A family is the module bench/scenarios/<family>.py exposing SCENARIOS (a list of Scenario instances).
Records append to <out>/<family>/<scenario>.jsonl (default out: bench/results).
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import pkgutil
import shlex
import sys
from pathlib import Path

from bench.harness import REPO, RunOptions, Scenario, run_scenario, setup_logging

log = logging.getLogger("alloy.bench")


def list_families() -> list[str]:
    import bench.scenarios as pkg

    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__) if not m.name.startswith("_"))


def load_family(key: str) -> list[Scenario]:
    mod = importlib.import_module(f"bench.scenarios.{key}")
    scenarios = list(getattr(mod, "SCENARIOS", []))
    for s in scenarios:
        s.family = s.family or key
    return scenarios


def _csv(text: str | None, cast=str) -> list | None:
    return [cast(x) for x in text.split(",") if x.strip()] if text else None


def summarize(paths: list[Path]) -> None:
    """Print one line per record: scenario, config, nworld, status, speed and accuracy."""
    files = [f for p in paths for f in (sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])]
    cols = ("scenario", "config", "device", "nworld", "status", "ms_per_step", "world_steps_per_s",
            "single_world_realtime_factor", "gpu_mem_mb_per_world", "stable", "energy_drift_pct",
            "penetration_p99_mm", "accuracy_error", "accuracy_unit", "repeat_identical")
    print("\t".join(cols))
    for f in files:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            print("\t".join(_cell(r.get(c)) for c in cols))


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "-" if v is None else str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", help="family key (bench/scenarios/<key>.py) or 'all'")
    ap.add_argument("--scenario", help="comma list of scenario ids to keep")
    ap.add_argument("--config", help="comma list of config names to keep")
    ap.add_argument("--device", default="cpu", help="warp device, e.g. cpu or cuda:0")
    ap.add_argument("--ladder", help="comma list of world counts (default: each scenario's own ladder)")
    ap.add_argument("--horizon", type=float, help="override every scenario's horizon [s]")
    ap.add_argument("--budget-s", type=float, default=120.0, help="wall cap on each timed loop [s]")
    ap.add_argument("--repeat-max-nworld", type=int, default=16,
                    help="rerun and compare bitwise when nworld <= this (0 disables)")
    ap.add_argument("--out", type=Path, default=REPO / "bench" / "results", help="results root")
    ap.add_argument("--list", action="store_true", help="list scenarios and configs, run nothing")
    ap.add_argument("--summary", nargs="+", type=Path, help="print recorded rows from files or directories")
    ap.add_argument("--log", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log)
    if a.summary:
        summarize(a.summary)
        return 0
    if not a.family:
        ap.error("--family is required (or --summary)")

    families = list_families() if a.family == "all" else [a.family]
    keep_ids = set(_csv(a.scenario) or [])
    scenarios = [s for fam in families for s in load_family(fam) if not keep_ids or s.id in keep_ids]
    if not scenarios:
        log.error("no scenarios matched (family=%s scenario=%s)", a.family, a.scenario)
        return 2
    if a.list:
        for s in scenarios:
            print(f"{s.family}/{s.id}: {s.title}  ladder={list(s.ladder)}  configs={[c.name for c in s.configs]}")
        return 0

    import warp as wp

    wp.config.quiet = True
    if hasattr(wp, "LOG_WARNING"):  # warp >= 1.18 prints module loads at LOG_INFO
        wp.config.log_level = wp.LOG_WARNING
    wp.init()
    opts = RunOptions(device=a.device, ladder=tuple(_csv(a.ladder, int)) if a.ladder else None,
                      horizon_s=a.horizon, budget_s=a.budget_s, repeat_max_nworld=a.repeat_max_nworld,
                      out_dir=a.out, command=" ".join(["python -m bench.run", *map(shlex.quote, argv or sys.argv[1:])]))
    keep_cfg = set(_csv(a.config) or [])
    n_err = 0
    for s in scenarios:
        n_err += sum(r["status"] == "error" for r in run_scenario(s, opts, keep_cfg or None))
    log.info("done: %d scenario(s), %d error record(s)", len(scenarios), n_err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
