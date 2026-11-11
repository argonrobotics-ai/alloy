# SPDX-License-Identifier: Apache-2.0
"""Top-level files of the library dataset: ``index.jsonl``, ``ATTRIBUTION.md``, ``LICENSES/`` and the dataset card.

Maintainer tool, run after every library is staged (``alloy.assets.hub.build``). Rows come from the per-library JSON
lines files the builder wrote; licence texts from a folder of ``<SPDX id>.txt`` files.

Example::

    python -m alloy.assets.hub.card --rows rows --licenses licenses --out ds     # prints counts per library
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from alloy.assets.hub.layout import ATTRIBUTION, INDEX, LICENSES
from alloy.assets.hub.provenance import mass_is_estimate
from alloy.assets.hub.readme import readme

# Extra licence files: Objaverse 1.0 (the compilation) is ODC-By; Gym-Anything's MIT notice names its copyright holder.
EXTRA = {"objaverse": ("ODC-By-1.0",)}
MIT_HOLDERS = {"gymanything": "2026 cmu-l3"}


def load_rows(rows_dir: Path) -> list[dict[str, Any]]:
    """Every row of ``rows_dir/*.jsonl`` (the last row per name wins), sorted by library and name."""
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(rows_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["name"]] = r
    return sorted(rows.values(), key=lambda r: (r["library"], r["name"]))


#: licences that ask for no credit; every other object must have a credit line
NO_CREDIT = {"CC0-1.0"}


def uncredited(rows: list[dict[str, Any]]) -> list[str]:
    """Names of the objects whose licence asks for credit and that have none (CC BY 4.0 requires it).
    Example: ``uncredited([{"name": "a", "licence": "CC-BY-4.0", "attribution": ""}]) == ["a"]``."""
    return [r["name"] for r in rows if r["licence"] not in NO_CREDIT and not str(r.get("attribution") or "").strip()]


def attribution_md(rows: list[dict[str, Any]]) -> str:
    """Every object's credit line, one table per library."""
    out = ["# Attribution", "",
           "Every object keeps its own licence (`licence` in index.jsonl, full texts in LICENSES/). Objects under CC BY "
           "4.0 must be credited as below when you redistribute them or show them.", ""]
    for lib in sorted({r["library"] for r in rows}):
        out += [f"## {lib}", "", "| Object | Credit | Licence |", "|---|---|---|"]
        out += [f"| {r['name']} | {str(r.get('attribution', '')).replace('|', '/')} | {r['licence']} |"
                for r in rows if r["library"] == lib]
        out.append("")
    return "\n".join(out)


def write_licenses(rows: list[dict[str, Any]], texts: Path, out: Path) -> list[str]:
    """Copy the text of every licence the rows use (plus each library's extras) into ``LICENSES/``."""
    ids = {r["licence"] for r in rows} | {x for lib in {r["library"] for r in rows} for x in EXTRA.get(lib, ())}
    (out / LICENSES).mkdir(parents=True, exist_ok=True)
    written = []
    for lid in sorted(ids):
        src = Path(texts) / f"{lid}.txt"
        if not src.is_file():
            raise FileNotFoundError(f"no licence text {src}: add <SPDX id>.txt for {lid} to {texts}")
        text = src.read_text(encoding="utf-8")
        (out / LICENSES / f"{lid}.txt").write_text(text, encoding="utf-8")
        written.append(lid)
        if lid == "MIT":
            for lib, holder in MIT_HOLDERS.items():
                notice = text.replace("<year> <copyright holders>", holder)
                (out / LICENSES / f"MIT-{lib}.txt").write_text(notice, encoding="utf-8")
                written.append(f"MIT-{lib}")
    return written


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per library: objects, bytes, licences, showcase_ok, estimates (``estimate``: the mass or any material
    value; ``mass_estimate``: the mass), flagged cards."""
    libs: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = libs.setdefault(r["library"], {"objects": 0, "bytes": 0, "files": 0, "showcase_ok": 0, "estimate": 0,
                                           "mass_estimate": 0, "flagged": 0, "licences": Counter()})
        s["objects"] += 1
        s["bytes"] += r["size"]
        s["files"] += len(r["files"])
        s["showcase_ok"] += bool(r.get("showcase_ok"))
        s["estimate"] += bool(r.get("estimate"))
        s["mass_estimate"] += mass_is_estimate(str(r.get("mass_basis", "")))
        s["flagged"] += bool(r.get("flags"))
        s["licences"][r["licence"]] += 1
    for s in libs.values():
        s["licences"] = dict(s["licences"])
    return libs


def write_top(rows_dir: Path, texts: Path, out: Path) -> dict[str, Any]:
    """Write index.jsonl, ATTRIBUTION.md, LICENSES/ and README.md into ``out``; return the per-library summary.
    Raises ``ValueError`` when an object that needs a credit line has none (``uncredited``)."""
    rows = load_rows(rows_dir)
    bare = uncredited(rows)
    if bare:
        raise ValueError(f"{len(bare)} objects have a licence that asks for credit and no credit line "
                         f"({', '.join(bare[:5])}, ...): stage their library again so its rows carry `attribution`")
    out.mkdir(parents=True, exist_ok=True)
    (out / INDEX).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (out / ATTRIBUTION).write_text(attribution_md(rows), encoding="utf-8")
    lic = write_licenses(rows, texts, out)
    libs = summary(rows)
    (out / "README.md").write_text(readme(libs, lic), encoding="utf-8")
    return {"objects": len(rows), "bytes": sum(s["bytes"] for s in libs.values()), "libraries": libs, "licenses": lic}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.assets.hub.card", description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, type=Path, help="folder of <library>.jsonl row files")
    ap.add_argument("--licenses", required=True, type=Path, help="folder of <SPDX id>.txt licence texts")
    ap.add_argument("--out", required=True, type=Path, help="dataset folder")
    a = ap.parse_args(argv)
    json.dump(write_top(a.rows, a.licenses, a.out), sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
