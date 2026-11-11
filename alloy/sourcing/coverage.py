# SPDX-License-Identifier: Apache-2.0
"""Coverage: the share of imported object values that are sourced, estimated or missing.

Counted per (object, field) over the queue scan, using the effective tag (record if any, else card):

- ``sourced``: ``DOC`` (a fetched page or PDF with an exact quote);
- ``weak``: ``SNIPPET`` or ``UNVERIFIED`` (cited, not re-read);
- ``estimated``: ``ESTIMATE`` (category or material prior, with a range and a reason); ``estimated_documented``
  counts those whose range rests on fetched or quoted pages (DOC/SNIPPET evidence), the rest are priors;
- ``missing``: no value (card source ``default``) and nothing recorded.

Example:
    >>> from alloy.sourcing.queue import Scan
    >>> summarize({"none": Scan()})["total"]["values"]
    0
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from alloy.sourcing.queue import Scan

log = logging.getLogger("alloy.sourcing")

GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "sourced": ("DOC",), "weak": ("SNIPPET", "UNVERIFIED"), "estimated": ("ESTIMATE",), "missing": ("MISSING",)}


def _shares(counts: Counter[str], documented: int = 0) -> dict[str, Any]:
    n = sum(counts.values())
    out: dict[str, Any] = {"values": n, "by_tag": dict(sorted(counts.items()))}
    for group, tags in GROUPS.items():
        k = sum(counts.get(t, 0) for t in tags)
        out[group] = k
        out[f"{group}_share"] = round(k / n, 4) if n else 0.0
    out["estimated_documented"] = documented
    return out


def summarize(scans: dict[str, Scan]) -> dict[str, Any]:
    """Coverage per library, per field and in total (counts and shares)."""
    total: Counter[str] = Counter()
    total_doc = 0
    libs: dict[str, Any] = {}
    for lib, scan in sorted(scans.items()):
        per_lib: Counter[str] = Counter()
        per_field: dict[str, Counter[str]] = {}
        doc: Counter[str] = Counter()
        objects = set()
        for it in scan.items:
            per_lib[it.tag] += 1
            per_field.setdefault(it.field, Counter())[it.tag] += 1
            if it.tag == "ESTIMATE" and it.documented:
                doc[it.field] += 1
            objects.add(it.object)
        total.update(per_lib)
        total_doc += sum(doc.values())
        libs[lib] = {**_shares(per_lib, sum(doc.values())), "objects": len(objects),
                     "skipped_files": len(scan.skipped),
                     "fields": {f: _shares(c, doc[f]) for f, c in sorted(per_field.items())}}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    return {"schema": "alloy.sourcing.coverage/1", "generated": stamp, "total": _shares(total, total_doc),
            "libraries": libs}


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def to_markdown(cov: dict[str, Any]) -> str:
    """A short Markdown report: one row per library, then per-field rows."""
    head = ("| scope | objects | values | sourced (DOC) | weak (SNIPPET, UNVERIFIED) | estimated | missing "
            "| estimates with documented range |")
    rule = "|---|---|---|---|---|---|---|---|"
    lines = [f"# Import value coverage ({cov['generated']})", "", head, rule]

    def row(name: str, s: dict[str, Any], objects: str = "") -> str:
        cells = [f"{s[g]} ({_pct(s[g + '_share'])})" for g in GROUPS] + [str(s["estimated_documented"])]
        return f"| {name} | {objects} | {s['values']} | " + " | ".join(cells) + " |"

    for lib, s in cov["libraries"].items():
        lines.append(row(lib, s, str(s["objects"])))
    lines.append(row("**total**", cov["total"]))
    for lib, s in cov["libraries"].items():
        lines += ["", f"## {lib} by field", "", head, rule]
        lines += [row(f, fs) for f, fs in s["fields"].items()]
    lines += ["", "Tags: DOC = page fetched and quote read; SNIPPET = search summary only; UNVERIFIED = cited, "
              "not re-read; estimated = a prior with a range and the reason (category or material)."]
    return "\n".join(lines) + "\n"


def write_report(cov: dict[str, Any], sources: Path) -> tuple[Path, Path]:
    """Write ``coverage.json`` and ``COVERAGE.md`` under ``sources``; returns both paths."""
    sources.mkdir(parents=True, exist_ok=True)
    js, md = sources / "coverage.json", sources / "COVERAGE.md"
    js.write_text(json.dumps(cov, indent=1) + "\n", encoding="utf-8")
    md.write_text(to_markdown(cov), encoding="utf-8")
    log.info("coverage: wrote %s and %s", js, md)
    return js, md
