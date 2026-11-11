# SPDX-License-Identifier: Apache-2.0
"""Read and write scorecard rows as JSON lines, keeping unpublishable numbers out of public files.

Numbers measured on PhysX / Isaac Sim / Isaac Lab may not be published (NVIDIA licence): such a record must be
`internal=True`, and internal records go to `rows.internal.jsonl`, whose first line is the restricted-numbers header.
"""

from __future__ import annotations

import dataclasses
import json
import os
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from alloy.bench.scorecard.types import HEADLINE_SCHEMA, ROW_SCHEMA, VERDICTS, Headline, Row

PUBLIC_NAME = "rows.jsonl"
INTERNAL_NAME = "rows.internal.jsonl"
HEADLINE_NAME = "headline.json"
# A lab that keeps such files can set its own first line (ALLOY_SCORECARD_RESTRICTED_HEADER).
INTERNAL_HEADER = os.environ.get("ALLOY_SCORECARD_RESTRICTED_HEADER") or (
    "Restricted numbers: this file holds figures that may not be published (NVIDIA licence).")
_RESTRICTED = re.compile(r"physx|isaac|omniverse", re.IGNORECASE)
_FIELDS = {f.name for f in dataclasses.fields(Row)}


def problems(row: Row) -> list[str]:
    """What is wrong with a record (empty when it may be written)."""
    out = []
    where = f"row {row.row} {row.engine}/{row.task}/{row.metric}"
    if not (row.row and row.engine and row.task and row.metric):
        out.append(f"{where}: row, engine, task and metric are all required")
    if row.verdict not in VERDICTS:
        out.append(f"{where}: unknown verdict {row.verdict!r}")
    if row.value is not None and not math.isfinite(row.value):
        out.append(f"{where}: value must be finite or null (write the failure in verdict and caveat)")
    if (row.k is None) != (row.n is None):
        out.append(f"{where}: k and n come together")
    if row.lo is not None and row.hi is not None and row.lo > row.hi:
        out.append(f"{where}: lo > hi")
    if (row.lo is not None or row.hi is not None) and not row.bound_kind:
        out.append(f"{where}: a bound needs bound_kind (wilson95, tolerance, min_max)")
    measured = (row.value is not None or row.k is not None) and not row.documented
    if measured and _RESTRICTED.search(row.engine) and not row.internal:
        out.append(f"{where}: a measured {row.engine} number must be internal=True")
    return out


def to_dict(row: Row) -> dict[str, Any]:
    return {"schema": ROW_SCHEMA, **dataclasses.asdict(row)}


def from_dict(d: dict[str, Any]) -> Row:
    if d.get("schema") != ROW_SCHEMA:
        raise ValueError(f"not a {ROW_SCHEMA} record: schema={d.get('schema')!r}")
    return Row(**{k: v for k, v in d.items() if k in _FIELDS})


def _write_lines(path: Path, rows: Iterable[Row], header: str | None) -> None:
    lines = [json.dumps({"_header": header})] if header else []
    lines += [json.dumps(to_dict(r), sort_keys=True) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def write_rows(rows: Sequence[Row], out: Path) -> list[Path]:
    """Write `rows.jsonl` (public) and, when there are any, `rows.internal.jsonl`. Raises on a bad record."""
    bad = [p for r in rows for p in problems(r)]
    if bad:
        raise ValueError("scorecard rows refused:\n  " + "\n  ".join(bad))
    out.mkdir(parents=True, exist_ok=True)
    public = [r for r in rows if not r.internal]
    internal = [r for r in rows if r.internal]
    _write_lines(out / PUBLIC_NAME, public, None)
    paths = [out / PUBLIC_NAME]
    if internal:
        _write_lines(out / INTERNAL_NAME, internal, INTERNAL_HEADER)
        paths.append(out / INTERNAL_NAME)
    return paths


def read_rows(path: Path) -> list[Row]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "_header" not in d:
            rows.append(from_dict(d))
    return rows


def write_headline(h: Headline, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / HEADLINE_NAME
    path.write_text(json.dumps({"schema": HEADLINE_SCHEMA, **dataclasses.asdict(h)}, indent=2, sort_keys=True) + "\n")
    return path


def read_headline(path: Path) -> Headline:
    d = json.loads(Path(path).read_text())
    if d.get("schema") != HEADLINE_SCHEMA:
        raise ValueError(f"{path}: not a {HEADLINE_SCHEMA} file")
    names = {f.name for f in dataclasses.fields(Headline)}
    return Headline(**{k: v for k, v in d.items() if k in names})


__all__ = ["HEADLINE_NAME", "INTERNAL_HEADER", "INTERNAL_NAME", "PUBLIC_NAME", "from_dict", "problems",
           "read_headline", "read_rows", "to_dict", "write_headline", "write_rows"]
