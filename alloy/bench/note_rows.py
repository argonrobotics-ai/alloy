# SPDX-License-Identifier: Apache-2.0
"""Producer rows the results collector does not take, read straight from a results note (SB headline feed).

`alloy.integration.results` collects rows whose first cell starts with a §5 id and whose verdict cell holds a
verdict. Some producers report a headline number in rows with neither, e.g. WP12's T09 rows: "WP12-T09-RT default
route (the SB-5 verdict)" and the fast-profile row of the table beside it, which has no verdict column. A headline
feed names such a row with a `NoteSource`; `read` finds it in `NOTES/<package>-results.md`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("alloy.bench")
NOTES = Path(__file__).resolve().parents[2] / "docs" / "notes"
_SEP = re.compile(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$")
_VERDICTS = ("not_run", "fail", "pass", "inconclusive")


@dataclass(frozen=True)
class NoteSource:
    """One kind of row in one results note."""

    package: str                     # "wp12" -> docs/notes/wp12-results.md
    row: str                         # regex on the row's label (first cell, ", <label_column cell>" when set)
    column: str = "value"            # header (prefix, case-insensitive) of the cell that holds the number
    verdict_column: str = "verdict"  # "" for a table without verdicts (informational rows)
    label_column: str = ""           # a column the label adds (e.g. WP5's "mode": the same metric in several modes)


@dataclass(frozen=True)
class NoteRow:
    package: str
    label: str
    text: str
    verdict: str | None              # None: no verdict cell, or no verdict word in it
    command: str
    commit: str
    source: str                      # docs/notes/<file>.md:<line>


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_rows(lines: Sequence[str]) -> Iterator[tuple[list[str], int, list[str]]]:
    """(lower-cased header, 1-based line number, cells) of every table body row."""
    header: list[str] | None = None
    for no, line in enumerate(lines, 1):
        if not line.lstrip().startswith("|"):
            header = None
        elif _SEP.match(line):
            continue
        elif header is None:
            header = [c.lower() for c in _cells(line)]
        else:
            yield header, no, _cells(line)


def _cell(header: Sequence[str], cells: Sequence[str], name: str) -> str:
    idx = next((i for i, h in enumerate(header) if name and h.startswith(name)), None)
    return cells[idx] if idx is not None and idx < len(cells) else ""


def _verdict(cell: str) -> str | None:
    c = cell.strip(" *").lower()
    return next((v for v in _VERDICTS if c.startswith(v)), None)


def read(src: NoteSource, notes: Path | None = None) -> list[NoteRow]:
    """Every row of `src`'s note whose first cell matches `src.row` (none, with a warning, if the note is missing)."""
    path = (notes or NOTES) / f"{src.package}-results.md"
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        log.warning("note rows: cannot read %s (%s)", path, exc)
        return []
    out = []
    for header, no, cells in _table_rows(lines):
        extra = _cell(header, cells, src.label_column)
        label = f"{cells[0]}, {extra}" if extra else cells[0]
        if not re.search(src.row, label):
            continue
        verdict = _verdict(_cell(header, cells, src.verdict_column)) if src.verdict_column else None
        commit = _cell(header, cells, "commit") or _cell(header, cells, "source")
        out.append(NoteRow(src.package.upper(), label, _cell(header, cells, src.column), verdict,
                           _cell(header, cells, "command").strip("`"), commit.strip("`"),
                           f"docs/notes/{path.name}:{no}"))
    return out


def verdict_cell(source: str, notes: Path | None = None) -> str:
    """The whole verdict cell of the producer row at `docs/notes/<file>.md:<line>` ("" when it cannot be read). The
    results collector keeps the verdict word only; the cell's prose says more, e.g. "fail; superseded by <commit>,
    re-run pending"."""
    name, _, no = source.rpartition(":")
    try:
        lines = ((notes or NOTES) / Path(name).name).read_text().splitlines()
    except OSError:
        return ""
    return next((_cell(header, cells, "verdict") for header, n, cells in _table_rows(lines) if str(n) == no), "")


__all__ = ["NOTES", "NoteRow", "NoteSource", "read", "verdict_cell"]
