# SPDX-License-Identifier: Apache-2.0
"""SB-11 shares from repeat JSONs (any engine; numpy only).

    python bench/xbench/ours/score_repeat.py p1.json p2.json p3.json [--key ours] [--out shares.json]

Each file is one fresh process of one (engine, scene, determinism setting): ``runs.a`` (N worlds), ``runs.b_rerun``
(the same batch again in the same process after a reset) and ``runs.n1`` (world 0 alone), each with per-world
digests under ``ours`` (``digest.world_digests`` of the engine's raw state) and, for Alloy, ``alloy``
(``alloy.precision.digest``). Shares:

- R1 same-process rerun: worlds with a == b_rerun, pooled over the processes;
- R2 fresh processes: worlds whose ``a`` digest is the same in every process;
- R4 batch invariance: world 0 of ``a`` == world 0 of ``n1``, per process;
- SB-11 share: equal digests over all R1 and R2 comparisons (world-comparisons equal / compared), R4 beside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def shares(procs: list[dict], key: str = "ours") -> dict:
    a = [p["runs"]["a"][key] for p in procs]
    r1 = [x == y for p in procs for x, y in zip(p["runs"]["a"][key], p["runs"]["b_rerun"][key], strict=True)]
    r2 = [len(set(ws)) == 1 for ws in zip(*a, strict=True)] if len(procs) > 1 else []
    r4 = [p["runs"]["a"][key][0] == p["runs"]["n1"][key][0] for p in procs if "n1" in p["runs"]]
    eq = sum(r1) + sum(r2)
    n = len(r1) + len(r2)
    step = [p["runs"]["a"].get("step_ms") for p in procs]
    return {"engine": procs[0].get("engine"), "scene": procs[0].get("scene"), "det": procs[0].get("det"),
            "worlds": procs[0].get("worlds"), "processes": len(procs), "key": key,
            "r1_share": sum(r1) / len(r1) if r1 else None, "r2_share": sum(r2) / len(r2) if r2 else None,
            "r4_equal": f"{sum(r4)}/{len(r4)}", "sb11_share": eq / n if n else None,
            "step_ms": [s for s in step if s is not None]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--key", default="ours")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    res = shares([json.loads(f.read_text()) for f in a.files], a.key)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
