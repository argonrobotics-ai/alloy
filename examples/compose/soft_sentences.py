# SPDX-License-Identifier: Apache-2.0
"""The soft sentence suite: each sentence -> a checked scene plus N variations, one JSON line per sentence.

Covers fills (liquid and grains), a pour from a held vessel, a pile, cloth flat / draped / folded in half / folded over a bin /
hanging over the table edge, cables loose and plugged in, two liquids, a robot arm scene and a cluttered kitchen.

Run:

    python examples/compose/soft_sentences.py                              # 4 sentences, load checks (minutes)
    python examples/compose/soft_sentences.py --all                        # every sentence, load checks
    python examples/compose/soft_sentences.py --all --check full --variations 8 --device cuda:0 --out soft_suite
    python examples/compose/soft_sentences.py --only 0 3 --check full      # a few, with the full check

It searches the objects shipped with Alloy (``Catalog(imports=False)``), so every machine builds the same scenes;
``--all-libraries`` also searches imported and fetched libraries. ``--all --check full`` takes over 25 minutes on
one L4 GPU (each full check runs its scene for a few seconds), and each variation adds as much again.

Each line: sentence, ok, scenes built and passed, routes, missing, estimated, wall seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import alloy.compose
from alloy.compose.catalog import Catalog

SENTENCES = [
    "a mug with 200 ml of water",
    "pour 150 ml of water from a jar into a mug",
    "pour 150 ml of water from a jar tilted at 30 deg/s into a bin",
    "a pile of 200 g of sand on a table",
    "a bin with 300 g of sand",
    "a jar of rice",
    "a towel lying flat on a table",
    "a towel draped over a mug",
    "a towel folded over the edge of a bin",
    "a towel folded in half",
    "a towel hanging over the edge of the table",
    "a 1 m usb cable on a table",
    "a usb cable plugged into a jar",
    "a mug with 150 ml of olive oil and a jar of honey",
    "a robot arm next to a bin of rice and a mug",           # rice beside a towel cannot share a scene yet
    "a cluttered kitchen table with a mug of water, a towel, a spoon and a jar",
]
QUICK = (0, 3, 7, 12)            # by default: a fill, a pile, a drape and a cable plugged in


def run(sentence: str, args: argparse.Namespace, out: Path | None, catalog: Catalog) -> dict:
    t0 = time.perf_counter()
    try:
        r = alloy.compose.compose(sentence, variations=args.variations, check=args.check, device=args.device,
                                  worlds=args.worlds, catalog=catalog)
    except alloy.compose.ComposeError as exc:
        return {"sentence": sentence, "ok": False, "error": exc.to_dict(), "wall_s": round(time.perf_counter() - t0, 1)}
    if out is not None:
        r.save(out / f"s{SENTENCES.index(sentence):02d}")
    soft = [row for s in r.scenes for row in s.soft.get("items", [])]
    return {"sentence": sentence, "ok": r.ok, "scenes": len(r.scenes), "passed": sum(s.ok for s in r.scenes),
            "asked": args.variations + 1, "routes": sorted({row["route"] for row in soft if row.get("route")}),
            "missing": r.scenes[0].missing, "estimated": r.scenes[0].soft.get("estimated", []),
            "checks": [s.report.get("checks") if s.report else None for s in r.scenes][:1],
            "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variations", type=int, default=0)
    ap.add_argument("--check", default="load", choices=("full", "load", "none"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--worlds", type=int, default=None, help="worlds in each full check (default: alloy.check's)")
    ap.add_argument("--only", type=int, nargs="*", help="sentence indices to run (default: 4 of them)")
    ap.add_argument("--all", action="store_true", help="run every sentence")
    ap.add_argument("--all-libraries", action="store_true", help="search imported and fetched libraries too")
    ap.add_argument("--out", default=None, help="folder to save each sentence's scenes")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    catalog = Catalog() if args.all_libraries else Catalog(imports=False)
    pick = range(len(SENTENCES)) if args.all else (args.only or QUICK)
    rows = []
    for i, s in enumerate(SENTENCES):
        if i in pick:
            rows.append(run(s, args, out, catalog))
            print(json.dumps(rows[-1]), flush=True)                  # one row per sentence as it finishes
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
