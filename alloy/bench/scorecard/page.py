# SPDX-License-Identifier: Apache-2.0
"""Build the one-page scorecard from the rows' output folders (`<root>/row<id>/headline.json`).

A row with no headline file is printed as "not measured": the page never shows a number it cannot point to.
Headlines marked internal are left out unless `internal=True` (the internal page carries the restricted-numbers header).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alloy.bench.scorecard.registry import ROWS
from alloy.bench.scorecard.rows_io import HEADLINE_NAME, INTERNAL_HEADER, read_headline
from alloy.bench.scorecard.types import Headline

COLUMNS = ("#", "Row", "Number", "Bound", "Command", "Date", "Commit", "Caveat")


def row_dir(root: Path, row: str) -> Path:
    return Path(root) / f"row{row}"


def _missing(row: str) -> Headline:
    spec = ROWS[row]
    return Headline(row=row, title=spec.title, number="not measured", caveat=f"keeper: {spec.lead}",
                    state="not_measured")


def load_headlines(root: Path, *, internal: bool = False) -> list[Headline]:
    """One headline per registered row, in scorecard order."""
    out = []
    for row in ROWS:
        path = row_dir(root, row) / HEADLINE_NAME
        head = read_headline(path) if path.is_file() else _missing(row)
        out.append(_missing(row) if head.internal and not internal else head)
    return out


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ") or "-"


def _line(h: Headline) -> str:
    number = h.number if h.state != "draft" else f"{h.number} (draft)"
    command = f"`{h.command}`" if h.command else ""
    cells = (h.row, h.title, number, h.bound, command, h.date[:10], h.commit, h.caveat)
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def render(headlines: Sequence[Headline], *, internal: bool = False) -> str:
    lines = [INTERNAL_HEADER, ""] if internal else []
    lines += ["# Alloy benchmark scorecard", "",
              "Each row is one command; a row where Alloy loses is shown as a loss. Drafts are marked.", "",
              "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += [_line(h) for h in headlines]
    return "\n".join(lines) + "\n"


def write_page(root: Path, out: Path, *, internal: bool = False) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(load_headlines(root, internal=internal), internal=internal))
    return out


__all__ = ["COLUMNS", "load_headlines", "render", "row_dir", "write_page"]
