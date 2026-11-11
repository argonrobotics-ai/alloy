# SPDX-License-Identifier: Apache-2.0
"""First full scoreboard: every package's result rows -> report/1 files the WP6 scoreboard reads (SPEC §5, §7 WP12).

Packages record their metrics as rows in ``docs/notes/wp*-results.md`` (brief rule: metric id, value,
pass/fail/not_run, command, commit) and WP12's tests also write JSON lines under ``bench/results/wp12``. The
scoreboard (``python -m alloy.bench scoreboard``) reads only ``alloy.bench.report/1`` and ``alloy.bench.run/1``
files, so ``collect`` turns each row whose first cell starts with a §5 id (``alloy.bench.metric_catalogue.BY_ID``)
into a ``MetricVerdict`` named ``<id>:<package>:<line>`` (the scoreboard feeds ``<id>``), one ``LawResult`` per
package. The first number in the value cell becomes the verdict's error/value; the row text, command and commit
stay in the report (``monitors["rows"]``), and not_run rows carry their reason in ``monitors["reasons"]``.

    python -m alloy.integration.results [--notes docs/notes] [--out bench/results/collected]
    python -m alloy.bench scoreboard bench/results --md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from alloy.bench.metric_catalogue import BY_ID
from alloy.bench.report import write_report
from alloy.bench.types import LawResult, MetricVerdict, Quantity

log = logging.getLogger("alloy.integration")

__all__ = ["ResultRow", "collect", "jsonl_rows", "markdown_rows", "to_law_results"]

ID_RE = re.compile(r"^\**\s*([A-Z]{2,4}-\d{1,2})\b")
NUM_RE = re.compile(r"(?<![A-Za-z_\d.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
VERDICT_COLS = ("verdict", "status", "result")
SOLVER = "results"


@dataclass(frozen=True)
class ResultRow:
    """One metric row from a package's results (``source`` = file:line)."""

    package: str
    metric_id: str
    label: str
    value_text: str
    verdict: str
    command: str
    commit: str
    source: str

    @property
    def value(self) -> float:
        """First number of the value cell (NaN when none or not_run)."""
        if self.verdict == "not_run":
            return math.nan
        m = NUM_RE.search(self.value_text.replace(",", ""))
        return float(m.group()) if m else math.nan


def _verdict(cell: str) -> str | None:
    """pass / fail / not_run / inconclusive from a verdict cell; None for info rows (not a verdict)."""
    c = cell.strip().strip("*").lower()
    for word, verdict in (("not_run", "not_run"), ("not run", "not_run"), ("not-run", "not_run"),
                          ("fail", "fail"), ("pass", "pass"), ("inconclusive", "inconclusive")):
        if c.startswith(word):
            return verdict
    if c.startswith(("info", "deferred", "n/a", "-")):
        return None
    return "inconclusive"


def _cells(line: str) -> list[str]:
    """Table cells; ``\\|`` inside a cell is an escaped pipe, not a separator."""
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", body)]


def _columns(header: Sequence[str]) -> dict[str, int] | None:
    low = [h.lower() for h in header]
    pick = lambda keys: next((i for i, h in enumerate(low) if any(k in h for k in keys)), None)  # noqa: E731
    cols = {"value": pick(("value",)), "verdict": pick(VERDICT_COLS), "command": pick(("command",)),
            "commit": pick(("commit",))}
    return None if cols["verdict"] is None else {k: v for k, v in cols.items() if v is not None}


def markdown_rows(path: Path, package: str) -> list[ResultRow]:
    """§5 rows of every table in one results file."""
    out: list[ResultRow] = []
    cols: dict[str, int] | None = None
    for no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.lstrip().startswith("|"):
            cols = None
            continue
        cells = _cells(line)
        if cols is None:
            cols = _columns(cells)
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        row = _row(cells, cols, package, f"{path}:{no}")
        if row is not None:
            out.append(row)
    return out


def _row(cells: list[str], cols: dict[str, int], package: str, source: str) -> ResultRow | None:
    m = ID_RE.match(cells[0])
    if m is None or m.group(1) not in BY_ID or cols["verdict"] >= len(cells):
        return None
    verdict = _verdict(cells[cols["verdict"]])
    if verdict is None:
        return None
    get = lambda key: cells[cols[key]] if key in cols and cols[key] < len(cells) else ""  # noqa: E731
    return ResultRow(package, m.group(1), cells[0], get("value"), verdict, get("command").strip("`"),
                     get("commit").strip("`"), source)


def jsonl_rows(path: Path, package: str) -> list[ResultRow]:
    """Rows written by ``alloy.testing.integration.record`` (``metric``, ``value``, ``status``, ...)."""
    out = []
    for no, line in enumerate(path.read_text().splitlines(), 1):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        mid = str(d.get("metric", "")).split(":", 1)[0]
        if mid not in BY_ID or d.get("status") not in ("pass", "fail", "not_run", "inconclusive"):
            continue
        text = "" if d.get("value") is None else repr(d["value"])
        extra = {k: v for k, v in d.items() if k not in ("metric", "value", "status", "command", "commit")}
        out.append(ResultRow(package, mid, json.dumps(extra, default=str)[:300], text, d["status"],
                             d.get("command", ""), d.get("commit", ""), f"{path}:{no}"))
    return out


def _metric_verdict(row: ResultRow, index: int) -> MetricVerdict:
    value = row.value
    q = Quantity(row.metric_id, "", value, math.nan, 1 if math.isfinite(value) else 0, math.nan)
    name = f"{row.metric_id}:{row.package}:{index}"
    return MetricVerdict(name, "", None, q, value, math.nan, None, "mean", row.verdict)  # type: ignore[arg-type]


def _reason(row: ResultRow) -> str:
    """Why a row is not_run: the label, value and command cells that say something, plus the source line."""
    parts = [c for c in (row.label, row.value_text, row.command) if c.strip(" -—`")]
    return "; ".join(parts) + f" ({row.source})"


def to_law_results(rows: Iterable[ResultRow], date: str) -> list[LawResult]:
    """One LawResult per package; verdict names ``<id>:<package>:<i>`` feed the §5 id."""
    by_pkg: dict[str, list[ResultRow]] = defaultdict(list)
    for r in rows:
        by_pkg[r.package].append(r)
    out = []
    for pkg, items in sorted(by_pkg.items()):
        verdicts = tuple(_metric_verdict(r, i) for i, r in enumerate(items))
        reasons = {v.name: _reason(r) for v, r in zip(verdicts, items) if r.verdict == "not_run"}
        rows_meta = {v.name: {"label": r.label, "value": r.value_text, "command": r.command, "commit": r.commit,
                              "source": r.source} for v, r in zip(verdicts, items)}
        commits = sorted({r.commit for r in items if r.commit})
        out.append(LawResult(f"{pkg}-results", SOLVER, "mixed", verdicts, {"reasons": reasons, "rows": rows_meta},
                             {"alloy": commits[-1] if commits else "", "date": date, "suite": "results"}, 0.0,
                             False, command="python -m alloy.integration.results"))
    return out


def collect(notes: Path, extra_jsonl: Sequence[Path], out: Path, date: str) -> list[Path]:
    """Write one report/1 file per package under ``out``; returns the paths."""
    rows: list[ResultRow] = []
    for path in sorted(notes.glob("wp*-results.md")):
        rows += markdown_rows(path, path.name.split("-", 1)[0].upper())
    for path in extra_jsonl:
        rows += jsonl_rows(path, "WP12")
    log.info("collected %d §5 rows from %s", len(rows), notes)
    return [write_report(r, out) for r in to_law_results(rows, date)]


def main(argv: Sequence[str] | None = None) -> int:
    import time  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="python -m alloy.integration.results", description=__doc__.split("\n\n")[0])
    ap.add_argument("--notes", type=Path, default=Path("docs/notes"))
    ap.add_argument("--jsonl", type=Path, nargs="*", default=[Path("bench/results/wp12/results.jsonl")])
    ap.add_argument("--out", type=Path, default=Path("bench/results/collected"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    paths = collect(args.notes, [p for p in args.jsonl if p.exists()], args.out, time.strftime("%Y-%m-%d"))
    for p in paths:
        log.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
