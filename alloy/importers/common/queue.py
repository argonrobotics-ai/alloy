# SPDX-License-Identifier: Apache-2.0
"""The hand-off queue for the sourcing tool (``alloy.sourcing``): one JSON line per card value that is still an ESTIMATE.

Row fields: ``card`` (object card name), ``field`` (dotted path in the card that holds the value, e.g.
``material.density`` or ``material.surface.mu_static``), ``value`` and ``unc`` (SI), ``note`` (how it was estimated),
``category`` (what kind of object), ``query`` (a search string that should find a published value), plus ``extra``
(library-specific hints such as a GTIN, brand or product description). One row per (card, field); writing a row
again replaces it, and ``drop`` removes rows once a sourced value lands.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class QueueRow:
    """One value that needs a source.

    Example::

        QueueRow("gso_x", "material.density", 300.0, (150.0, 600.0), "ESTIMATE: packaged goods", "Consumer Goods",
                 "Brisk iced tea 12 x 355 ml net weight", {"gtin": "012000..."})
    """

    card: str
    field: str
    value: float
    unc: tuple[float, float] | None
    note: str
    category: str
    query: str
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        """``(card, field)``, the row's identity in the queue.

        Example::

            row.key   # ("gso_x", "material.density")
        """
        return (self.card, self.field)


def read(path: Path) -> dict[tuple[str, str], dict]:
    """Rows of a queue file keyed by (card, field); empty if the file does not exist.

    Example::

        rows = read(Path("imports/ycb_gso/queue.jsonl"))
    """
    p = Path(path)
    if not p.is_file():
        return {}
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["card"], r["field"]): r for r in rows}


def write(path: Path, rows: dict[tuple[str, str], dict]) -> None:
    """Write rows sorted by (card, field), one JSON object per line.

    Example::

        write(Path("queue.jsonl"), rows)
    """
    lines = [json.dumps(rows[k], sort_keys=True) for k in sorted(rows)]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def add(path: Path, new: list[QueueRow]) -> int:
    """Insert or replace rows; returns the queue length.

    Example::

        add(Path("queue.jsonl"), [row])
    """
    rows = read(path)
    for r in new:
        d = asdict(r)
        d["unc"] = list(r.unc) if r.unc is not None else None
        rows[r.key] = d
    write(path, rows)
    return len(rows)


def drop(path: Path, keys: list[tuple[str, str]]) -> int:
    """Remove rows (values that are now sourced); returns the queue length.

    Example::

        drop(Path("queue.jsonl"), [("gso_x", "material.density")])
    """
    rows = read(path)
    for k in keys:
        rows.pop(tuple(k), None)
    write(path, rows)
    return len(rows)
