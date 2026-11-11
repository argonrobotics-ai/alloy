# SPDX-License-Identifier: Apache-2.0
"""Rows and tables of the mixed-batch scorecard rows from the raw point files of ``mixed.py``.

    python3 bench/scorecard/mixed/build.py RAW_DIR OUT_DIR

Reads every ``RAW_DIR/**/mix*.json`` (one per point), writes ``OUT_DIR/rows.jsonl`` (one line per speed point, per
failed point and per task of a quality run) and ``OUT_DIR/tables.md`` (best batch, sweep, memory, quality).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

LABELS = {"alloy_multi": "Alloy as shipped, one sim per type", "alloy_one": "Alloy as shipped, all robots in one scene",
          "alloy5ms_multi": "Alloy 5 ms probe (unsupported), one sim per type",
          "mjw_sub4": "MuJoCo Warp (Newton alone, default buffers: overflows), 4 x 1.25 ms", "mjw_sub2": "MuJoCo Warp (Newton alone, default buffers: overflows), 2 x 2.5 ms",
          "mjw_sub1": "MuJoCo Warp (Newton alone, default buffers: overflows), 1 x 5 ms",
          "mjws_sub4": "MuJoCo Warp (Newton alone, measured buffers), 4 x 1.25 ms",
          "mjws_sub2": "MuJoCo Warp (Newton alone, measured buffers), 2 x 2.5 ms",
          "mjws_sub1": "MuJoCo Warp (Newton alone, measured buffers), 1 x 5 ms"}
TAG = re.compile(r"^(mix\d+)_(.+)_(speed|quality)_(\d+)$")
BASE = "mjws_sub2"         # the step count of Isaac Lab's MuJoCo Warp presets
FROZEN_MAX = 0.05          # a speed point with more INVALID (frozen) worlds than this in a group is a failed point


def up95(k: int, n: int) -> float:
    """Clopper-Pearson one-sided 95 % upper bound on a rate with k of n."""
    if k >= n:
        return 1.0
    from math import comb  # noqa: PLC0415

    lo, hi = k / n, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        cdf = sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
        lo, hi = (p, hi) if cdf > 0.05 else (lo, p)
    return hi


def load(raw: Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(raw.rglob("mix*.json")):
        m = TAG.match(f.stem)
        if not m:
            continue
        d = json.loads(f.read_text())
        base = {"protocol": "hold-mixed", "mix": m[1], "label": m[2], "mode": m[3], "num_envs": int(m[4]),
                "file": str(f.relative_to(raw)), "gpu": d.get("gpu"), "task_types": d.get("task_types"),
                "tasks": d.get("tasks"), "engine": d.get("engine"), "form": d.get("form"),
                "solver_steps_per_env_step": d.get("solver_steps_per_env_step")}
        if "error" in d:
            rows.append(base | {"metric_name": "failed", "error": d["error"][:300],
                                "memory_mb": (d.get("memory_end") or {}).get("nvidia_smi_used_mb")})
        elif m[3] == "speed":
            sp, ends = d["speed"], d.get("group_end") or []
            frozen = max(((e.get("health") or {}).get("invalid", 0) / max((e.get("health") or {}).get("n", 1), 1)
                          for e in ends), default=0.0)
            full = [i for i, e in enumerate(ends) if (e.get("buffers") or {}).get("full_at_end")]
            if frozen > FROZEN_MAX:          # mostly frozen worlds: not a physics speed (README, Fairness notes)
                rows.append(base | {"metric_name": "failed", "error": f"frozen worlds: {frozen:.0%} INVALID",
                                    "value_not_a_speed": sp["env_steps_per_s"],
                                    "memory_mb": sp["memory"].get("nvidia_smi_used_mb")})
                continue
            rows.append(base | {"metric_name": "env_steps_per_s", "value": sp["env_steps_per_s"],
                                "frozen_share_max": frozen, "buffers_full_groups": full,
                                "block_walls_s": sp["block_walls_s"], "memory_mb": sp["memory"].get("nvidia_smi_used_mb"),
                                "warp_pool_high_mb": sp["memory"].get("warp_pool_high_mb"),
                                "group_wall_s": sp["group_wall_s"],
                                "invalid_worlds": sum((e.get("health") or {}).get("invalid", 0) for e in ends)})
        else:
            for i, q in enumerate(d["quality"]):
                s, ends = q["score"], d.get("group_end") or []
                h = (ends[i] if len(ends) > i else ends[0] if ends else {}).get("health") or {}
                s1 = max(s["s1"], h.get("invalid") or 0)   # a frozen (INVALID) world is a glitch the outside rule misses
                rows.append(base | {"metric_name": "quality", "task": q["task"], "worlds": q["worlds"], "s1": s1,
                                    "s1_outside_rule": s["s1"], "s1_up95": up95(s1, s["episodes"]), "causes": s["causes"], "held": s["held"],
                                    "kind": s["kind"], "drift_mm": s["drift_mm"], "alloy_invalid": h.get("invalid"),
                                    "alloy_flags": h.get("flags")})
    return rows


def num(v: float | None) -> str:
    """A drift value: none when every world went non-finite."""
    return "-" if v is None else f"{v:.2g}"


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:,.0f}"


def tables(rows: list[dict[str, Any]]) -> str:
    speed = defaultdict(dict)                    # (mix, label) -> envs -> row
    fails = defaultdict(dict)
    for r in rows:
        if r["metric_name"] == "env_steps_per_s":
            speed[r["mix"], r["label"]][r["num_envs"]] = r
        elif r["metric_name"] == "failed" and r["mode"] == "speed":
            fails[r["mix"], r["label"]][r["num_envs"]] = r
    out = []
    for mix in sorted({r["mix"] for r in rows}, key=lambda s: int(s[3:])):        # a mix with quality rows only too
        labels = [lb for lb in LABELS if (mix, lb) in speed]
        best = {lb: max(speed[mix, lb].values(), key=lambda r: r["value"]) for lb in labels}
        out += [f"### {mix}: best measured batch", "",
                f"| engine | env-steps/s | total envs | GPU memory (MiB) | vs {LABELS[BASE]} | first batch that failed |",
                "|---|---|---|---|---|---|"]
        for lb in labels:
            b, fail = best[lb], min(fails.get((mix, lb), {}), default=None)
            ratio = f"{b['value'] / best[BASE]['value']:.2f}x" if BASE in best else "-"
            lost = sum(r.get("invalid_worlds") or 0 for r in [b])
            out.append(f"| {LABELS[lb]} | {fmt(b['value'])} | {fmt(b['num_envs'])} | {fmt(b['memory_mb'])} | {ratio} | "
                       f"{fmt(fail) if fail else '-'} |")
        sizes = sorted({n for lb in labels for n in speed[mix, lb]} | {n for lb in labels for n in fails.get((mix, lb), {})})
        for title, key in (("env-steps/s by total envs", "value"), ("GPU memory (MiB) by total envs", "memory_mb")):
            out += ["", f"### {mix}: {title}", "", "| engine | " + " | ".join(fmt(n) for n in sizes) + " |",
                    "|---|" + "---|" * len(sizes)]
            for lb in labels:
                cells = [fmt(speed[mix, lb][n][key]) if n in speed[mix, lb] else
                         ("failed" if n in fails.get((mix, lb), {}) else "-") for n in sizes]
                out.append(f"| {LABELS[lb]} | " + " | ".join(cells) + " |")
        q = [r for r in rows if r["metric_name"] == "quality" and r["mix"] == mix]
        if q:
            out += ["", f"### {mix}: quality (4 s of the same loop)", "",
                    "| task | engine | worlds | S1 k/N (95 % upper bound) | causes | held | drift p95 / max (mm) | Alloy INVALID |",
                    "|---|---|---|---|---|---|---|---|"]
            for r in sorted(q, key=lambda r: (r["task"], list(LABELS).index(r["label"]))):
                d = r["drift_mm"]
                out.append(f"| {r['task']} | {LABELS[r['label']]} | {fmt(r['worlds'])} | {r['s1']}/{r['worlds']} "
                           f"(<= {100 * r['s1_up95']:.2f} %) | {r['causes'] or '-'} | {r['held']}/{r['worlds']} ({r['kind']}) | "
                           f"{num(d['p95'])} / {num(d['max'])} | {'-' if r['alloy_invalid'] is None else r['alloy_invalid']} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = load(raw)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    (out / "tables.md").write_text(tables(rows))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
