# SPDX-License-Identifier: Apache-2.0
"""Build the Newton baseline summary from the bench result rows.

Reads the GPU rows of every family (NVIDIA L4, 2026-11-01 runs), keeps the newest row per
(config, nworld, horizon), summarises each config and writes ``bench/baseline_scoreboard.json`` plus the
per-scenario markdown tables used in ``bench/BASELINE.md``.

    python -m bench.tools.baseline                      # writes bench/baseline_scoreboard.json
    python -m bench.tools.baseline --tables out.md      # also writes the markdown tables

Pure stdlib; no Newton or Warp import, so it runs anywhere.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BENCH, "results")

# Which files and hosts are "the baseline" per family (earlier smoke/builder VMs are excluded).
SOURCES: dict[str, dict[str, Any]] = {
    "rigid": {"glob": "rigid/l4/R0*.jsonl", "exclude": (".ref.",), "hosts": None},
    "precision": {"glob": "precision/P0*.gpu_l4.jsonl", "exclude": (), "hosts": None},
    "deform": {"glob": "deform/D0*.gpu.jsonl", "exclude": (), "hosts": None},
    "granular": {"glob": "granular/G0*.jsonl", "exclude": (".ref.",), "hosts": {"alloy-bench-granular"}},
    "coupled": {"glob": "coupled/C0*.jsonl", "exclude": (), "hosts": {"alloy-bench-coupled"}},
    "scale": {"glob": "scale/S0*.jsonl", "exclude": (), "hosts": {"alloy-bench-scale"}},
}
FAMILY_ORDER = ("rigid", "precision", "deform", "granular", "coupled", "scale")


@dataclass
class ConfigSummary:
    """Baseline numbers for one (scenario, config) on the L4."""

    config: str
    solver: str
    best_world_steps_per_s: float | None = None
    best_at_nworld: int | None = None
    best_sim_s_per_gpu_s: float | None = None
    best_is_unstable: bool = False
    single_world_rtf: float | None = None
    gpu_mem_mb_per_world: float | None = None
    mem_at_nworld: int | None = None
    max_ok_nworld: int | None = None
    stable: bool | None = None
    unstable_at: list[int] = field(default_factory=list)
    accuracy_error: float | None = None
    accuracy_unit: str | None = None
    accuracy_nworld: int | None = None
    accuracy_complete: bool | None = None
    law_pass: bool | None = None
    accuracy_at_largest: float | None = None
    largest_complete_nworld: int | None = None
    penetration_p99_mm: float | None = None
    repeat_identical: bool | None = None
    repeat_detail: str = ""
    contact_overflow_rows: int = 0
    first_error: str | None = None
    first_error_nworld: int | None = None
    law_metrics: dict[str, Any] = field(default_factory=dict)


def _num(x: Any) -> float | None:
    """Return x as a finite float, else None."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def load_family(family: str) -> list[dict]:
    """Load the GPU rows of one family, newest row per (scenario, config, nworld, horizon)."""
    src = SOURCES[family]
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(RES, src["glob"]))):
        if any(x in os.path.basename(path) for x in src["exclude"]):
            continue
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not str(r.get("device", "")).startswith("cuda"):
                    continue
                host = (r.get("versions") or {}).get("host")
                if src["hosts"] and host not in src["hosts"]:
                    continue
                if r.get("status") == "skipped":
                    continue
                rows.append(r)
    latest: dict[tuple, dict] = {}
    for r in rows:  # files are append-only, so later lines supersede earlier ones
        latest[(r["scenario"], r["config"], r["nworld"], r.get("horizon_s"))] = r
    return list(latest.values())


def _covered(r: dict, max_cc: float | None) -> bool:
    cc = (r.get("metrics") or {}).get("conditions_covered")
    if cc is None or cc is True:
        return True
    if cc is False:
        return False
    v = _num(cc)
    return v is not None and max_cc is not None and v >= max_cc


def _law_row(ok: list[dict]) -> dict | None:
    """Smallest-nworld complete row at the full horizon that covers every condition."""
    if not ok:
        return None
    hmax = max(r.get("horizon_s") or 0 for r in ok)
    full = [r for r in ok if (r.get("horizon_s") or 0) == hmax]
    ccs = [_num((r.get("metrics") or {}).get("conditions_covered")) for r in full]
    max_cc = max((c for c in ccs if c is not None), default=None)
    good = [r for r in full if r.get("complete") and _covered(r, max_cc)]
    pool = good or [r for r in full if _covered(r, max_cc)] or full
    return min(pool, key=lambda r: r["nworld"])


def _scalar_metrics(m: dict) -> dict:
    """Keep the scalar scenario metrics (drop per-condition tables and long traces)."""
    out = {}
    for k, v in (m or {}).items():
        if isinstance(v, bool) or (isinstance(v, (int, float)) and _num(v) is not None):
            out[k] = v
        elif isinstance(v, str) and len(v) <= 80:
            out[k] = v
    return out


def _repeat(ok: list[dict]) -> tuple[bool | None, str]:
    vals = sorted((r["nworld"], r["repeat_identical"]) for r in ok if r.get("repeat_identical") is not None)
    if not vals:
        return None, "not run"
    flags = {v for _, v in vals}
    detail = ", ".join(f"{n}:{'y' if v else 'n'}" for n, v in vals)
    if flags == {True}:
        return True, detail
    if flags == {False}:
        return False, detail
    return False, detail  # any differing rerun counts as not repeatable


def summarise_config(rows: list[dict]) -> ConfigSummary:
    """Summarise every rung of one config."""
    rows = sorted(rows, key=lambda r: r["nworld"])
    s = ConfigSummary(config=rows[0]["config"], solver=rows[0].get("solver", ""))
    ok = [r for r in rows if r.get("status") == "ok"]
    errs = [r for r in rows if r.get("status") == "error"]
    if errs:
        e = errs[0]
        s.first_error = (e.get("error") or "").splitlines()[0][:160] if e.get("error") else "error"
        s.first_error_nworld = e["nworld"]
    if not ok:
        return s
    stable = [r for r in ok if r.get("stable")]
    s.unstable_at = sorted({r["nworld"] for r in ok if not r.get("stable")})
    s.stable = not s.unstable_at
    s.max_ok_nworld = max(r["nworld"] for r in ok)
    speed_pool = [r for r in (stable or ok) if _num(r.get("world_steps_per_s"))]
    if speed_pool:
        b = max(speed_pool, key=lambda r: r["world_steps_per_s"])
        s.best_world_steps_per_s, s.best_at_nworld = b["world_steps_per_s"], b["nworld"]
        s.best_sim_s_per_gpu_s = _num(b.get("sim_seconds_per_gpu_second"))
        s.best_is_unstable = not stable
    one = [r for r in ok if r["nworld"] == 1 and _num(r.get("single_world_realtime_factor"))]
    if one:
        s.single_world_rtf = max(one, key=lambda r: r.get("horizon_s") or 0)["single_world_realtime_factor"]
    mem = [r for r in ok if _num(r.get("gpu_mem_mb_per_world"))]
    if mem:
        m = max(mem, key=lambda r: r["nworld"])
        s.gpu_mem_mb_per_world, s.mem_at_nworld = m["gpu_mem_mb_per_world"], m["nworld"]
    law = _law_row(ok)
    if law:
        s.accuracy_error = _num(law.get("accuracy_error"))
        s.accuracy_unit = law.get("accuracy_unit")
        s.accuracy_nworld = law["nworld"]
        s.accuracy_complete = bool(law.get("complete"))
        lp = (law.get("metrics") or {}).get("law_pass")
        s.law_pass = lp if isinstance(lp, bool) else None
        s.law_metrics = _scalar_metrics(law.get("metrics") or {})
        hmax = law.get("horizon_s")
        big = [r for r in ok if r.get("complete") and r.get("horizon_s") == hmax and _num(r.get("accuracy_error"))]
        if big:
            g = max(big, key=lambda r: r["nworld"])
            s.accuracy_at_largest, s.largest_complete_nworld = g["accuracy_error"], g["nworld"]
    pens = [_num(r.get("penetration_p99_mm")) for r in (stable or ok)]
    pens = [p for p in pens if p is not None]
    s.penetration_p99_mm = max(pens) if pens else None
    s.repeat_identical, s.repeat_detail = _repeat(ok)
    s.contact_overflow_rows = sum(
        1
        for r in ok
        if (_num((r.get("metrics") or {}).get("contact_overflow")) or 0) > 0
        or (_num((r.get("metrics") or {}).get("reduction_overflow_passes")) or 0) > 0
    )
    return s


def summarise_family(family: str) -> dict[str, dict[str, ConfigSummary]]:
    """{scenario: {config: ConfigSummary}} for one family."""
    by: dict[str, dict[str, list[dict]]] = {}
    for r in load_family(family):
        by.setdefault(r["scenario"], {}).setdefault(r["config"], []).append(r)
    return {sc: {c: summarise_config(rs) for c, rs in sorted(cfgs.items())} for sc, cfgs in sorted(by.items())}


# S04 deliberately runs at a quarter of the needed contact capacity, so its speed is not a baseline.
NO_SPEED_BASELINE = {"S04_memory_capacity"}


def _best(cfgs: dict[str, ConfigSummary], key: str) -> dict | None:
    pool = [c for c in cfgs.values() if getattr(c, key) is not None and not c.best_is_unstable]
    if not pool:
        return None
    b = max(pool, key=lambda c: getattr(c, key))
    nworld = 1 if key == "single_world_rtf" else b.best_at_nworld
    return {"config": b.config, "value": getattr(b, key), "nworld": nworld}


def scenario_entry(scenario: str, family: str, cfgs: dict[str, ConfigSummary]) -> dict:
    """Scenario-level baseline block for baseline_scoreboard.json."""
    judged = [c for c in cfgs.values() if c.law_pass is not None]
    speed = scenario not in NO_SPEED_BASELINE
    return {
        "family": family,
        "best_throughput_world_steps_per_s": _best(cfgs, "best_world_steps_per_s") if speed else None,
        "best_sim_s_per_gpu_s": _best(cfgs, "best_sim_s_per_gpu_s") if speed else None,
        "best_single_world_rtf": _best(cfgs, "single_world_rtf") if speed else None,
        "law_pass_configs": sorted(c.config for c in judged if c.law_pass),
        "law_judged_configs": len(judged),
        "configs": {k: v.__dict__ for k, v in cfgs.items()},
    }


def _share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


def spec_metrics(all_s: dict[str, dict[str, dict[str, ConfigSummary]]]) -> dict:
    """Baselines for the SPEC §5 headline metrics that the stock-Newton rows can measure."""
    flat = [(fam, sc, c) for fam, scs in all_s.items() for sc, cfgs in scs.items() for c in cfgs.values()]
    physics = [c.law_pass for fam, _, c in flat if fam != "scale" and c.law_pass is not None]
    rep = [c.repeat_identical for _, _, c in flat if c.repeat_identical is not None]
    prec = all_s.get("precision", {})
    p01 = prec.get("P01_peg_hole_0p2", {})
    p05 = prec.get("P05_thin_wall_tunnel", {})
    scen = {sc: (fam, cfgs) for fam, scs in all_s.items() for sc, cfgs in scs.items() if sc not in NO_SPEED_BASELINE}
    return {
        "SB-3_precision_penetration_p99_mm": {
            "scenario": "P01_peg_hole_0p2", "target": "<= 0.05 mm (0.2 mm clearance); bench law 0.02 mm",
            "per_config": {k: v.penetration_p99_mm for k, v in p01.items()},
        },
        "SB-4_pass_through": {
            "scenario": "P05_thin_wall_tunnel", "target": 0,
            "per_config_tunnel_rate": {k: v.law_metrics.get("tunnel_rate") for k, v in p05.items()},
        },
        "SB-5_best_sim_s_per_gpu_s": {sc: _best(cfgs, "best_sim_s_per_gpu_s") for sc, (_, cfgs) in scen.items()},
        "SB-6_best_single_world_rtf": {sc: _best(cfgs, "single_world_rtf") for sc, (_, cfgs) in scen.items()},
        "SB-11_repeat_identity_share": {
            "value": _share(rep), "configs_judged": len(rep), "target": 1.0,
            "note": "share of GPU configs whose bitwise reruns (n <= 16-256) all matched",
        },
        "SB-12_law_pass_share": {
            "value": _share(physics), "configs_judged": len(physics), "target": 1.0,
            "note": "physics families only (scale laws are determinism/graph checks); law row as in row_rules",
        },
        "CT-12_rows_with_contact_overflow": {
            "value": sum(c.contact_overflow_rows for fam, sc, c in flat if sc != "S04_memory_capacity"),
            "note": "excludes S04, whose quarter-capacity configs overflow on purpose",
        },
    }


# ---------------------------------------------------------------- markdown tables


def _g(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "-"
    if x == 0:
        return "0"
    if abs(x) >= 1e5:
        return f"{x:.2e}".replace("e+0", "e").replace("e+", "e")
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:.{nd}g}"


def _row(c: ConfigSummary) -> str:
    speed = f"{_g(c.best_world_steps_per_s)} ({c.best_at_nworld})" if c.best_world_steps_per_s else "-"
    if c.best_is_unstable:
        speed += " †"
    mem = f"{_g(c.gpu_mem_mb_per_world)} ({c.mem_at_nworld})" if c.gpu_mem_mb_per_world else "-"
    stable = "yes" if c.stable else ("-" if c.stable is None else "no @" + ",".join(map(str, c.unstable_at)))
    acc = "-"
    if c.accuracy_error is not None:
        acc = f"{_g(c.accuracy_error)} {c.accuracy_unit or ''} (n={c.accuracy_nworld})"
        if c.largest_complete_nworld and c.largest_complete_nworld != c.accuracy_nworld:
            acc += f"; {_g(c.accuracy_at_largest)} @{c.largest_complete_nworld}"
        if not c.accuracy_complete:
            acc += " partial"
    law = {True: "pass", False: "FAIL", None: "-"}[c.law_pass]
    rep = {True: "yes", False: "no", None: "-"}[c.repeat_identical]
    end = f"{c.first_error_nworld}: {c.first_error[:70]}" if c.first_error else f"ok to {c.max_ok_nworld}"
    cells = [c.config, speed, _g(c.single_world_rtf), mem, stable, acc, law, _g(c.penetration_p99_mm), rep, end]
    return "| " + " | ".join(x.replace("|", "/") for x in cells) + " |"


HEADER = (
    "| config | best world-steps/s (n) | 1-world RTF | GPU MB/world (n) | stable | accuracy vs law (law rung; worst rung) "
    "| law | pen p99 mm | repeat | ladder end |\n|---|---|---|---|---|---|---|---|---|---|"
)


def tables_md(all_s: dict) -> str:
    out = []
    for fam in FAMILY_ORDER:
        for sc, cfgs in all_s.get(fam, {}).items():
            out += [f"#### {sc}", "", HEADER] + [_row(c) for c in cfgs.values()] + [""]
    return "\n".join(out)


def build() -> dict[str, dict[str, dict[str, ConfigSummary]]]:
    return {fam: summarise_family(fam) for fam in FAMILY_ORDER}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(BENCH, "baseline_scoreboard.json"))
    ap.add_argument("--tables", default=None, help="also write the markdown tables here")
    args = ap.parse_args()
    all_s = build()
    doc = {
        "schema": "alloy.bench.baseline/1",
        "generated": _dt.date.today().isoformat(),
        "command": "python -m bench.tools.baseline",
        "baseline_of": "stock Newton ac7f7830 (1.8.0.dev0), warp-lang 1.18.0, mujoco/mujoco-warp 3.16.0",
        "device": "NVIDIA L4 (g2-standard-8 spot), driver 580.178.04, CUDA graphs on",
        "row_rules": {
            "rows": "GPU rows only; newest row per (config, nworld, horizon); skipped rows dropped",
            "best_throughput": "max world_steps_per_s over ok+stable rows (ok rows if none stable, flagged)",
            "accuracy": "law row = smallest nworld, full horizon, complete, all conditions covered",
            "penetration_p99_mm": "max over ok+stable rows",
            "repeat_identical": "false if any bitwise rerun differed",
        },
        "spec_metrics": spec_metrics(all_s),
        "scenarios": {sc: scenario_entry(sc, fam, cfgs) for fam in FAMILY_ORDER for sc, cfgs in all_s[fam].items()},
    }
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False, default=str)
        fh.write("\n")
    if args.tables:
        with open(args.tables, "w") as fh:
            fh.write(tables_md(all_s))
    print(f"wrote {args.out}" + (f" and {args.tables}" if args.tables else ""))


if __name__ == "__main__":
    main()
