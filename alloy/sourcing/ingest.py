# SPDX-License-Identifier: Apache-2.0
"""Merge per-object findings (JSON lines) into the per-object records.

Each line: ``{"card", "field", "value", "lo", "hi", "tag", "evidence": [...], "reason", "derivation", "conditions"}``
(SI values; the same keys as a record value plus ``card``). A finding replaces the record's value for that field
only if its tag is stronger (``DOC`` > ``SNIPPET`` > ``UNVERIFIED`` > ``ESTIMATE``), or if both are ESTIMATE and
the record's is a category prior (written by ``fill``, reason starting "category ") while the finding is about this
object; a weaker or equal one is skipped, so re-running an ingest is harmless. Every finding is validated like any record value; a bad line is
reported with its line number and the rest still land.

Example:
    >>> import json, tempfile
    >>> from pathlib import Path
    >>> d = Path(tempfile.mkdtemp())
    >>> line = {"card": "gso_x", "field": "mass", "value": 0.05, "lo": 0.05, "hi": 0.05, "tag": "DOC",
    ...         "evidence": [{"url": "https://example.org/x", "quote": "Weight 50 g", "tag": "DOC"}]}
    >>> _ = (d / "f.jsonl").write_text(json.dumps(line) + "\\n")
    >>> ingest_file(d / "f.jsonl", "demo", d / "sources").to_json()["applied"]
    1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alloy.sourcing.fields import SourcingError
from alloy.sourcing.record import is_category_prior, rank, value_from_dict
from alloy.sourcing.store import ObjectRecord, load_records, write_record

log = logging.getLogger("alloy.sourcing")


@dataclass
class IngestReport:
    """Findings applied, skipped as not stronger, and rejected (with the reason)."""

    applied: int = 0
    weaker: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Plain dict for the CLI."""
        return {"applied": self.applied, "weaker_or_equal": self.weaker, "rejected": self.rejected}


def _stronger(new: Any, old: Any) -> bool:
    """True if ``new`` should replace ``old``: a stronger tag, or an object-level estimate over a category prior."""
    if rank(new.tag) != rank(old.tag):
        return rank(new.tag) < rank(old.tag)
    return is_category_prior(old) and not is_category_prior(new)


def _parse(line: str, n: int, path: Path) -> tuple[str, Any]:
    raw = json.loads(line)
    if not isinstance(raw, dict) or not raw.get("card") or not raw.get("field"):
        raise SourcingError(f"{path}:{n}: each line needs 'card' and 'field'")
    card, fname = str(raw.pop("card")), str(raw.pop("field"))
    return card, value_from_dict(raw, fname, f"{path.name}:{n}:{card}.{fname}")


def ingest_file(path: Path, library: str, sources: Path | None = None, dry_run: bool = False) -> IngestReport:
    """Apply every valid finding in ``path`` to ``<sources>/<library>/<card>.yaml`` (records created as needed)."""
    report = IngestReport()
    records = load_records(library, sources)
    changed: dict[str, ObjectRecord] = {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            card, value = _parse(line, n, Path(path))
        except (SourcingError, json.JSONDecodeError, ValueError) as e:
            report.rejected.append({"line": n, "error": str(e)[:300]})
            continue
        rec = changed.get(card) or records.get(card) or ObjectRecord(library=library, object=card)
        old = rec.values.get(value.field)
        if old is not None and not _stronger(value, old):
            report.weaker += 1
            continue
        changed[card] = ObjectRecord(library=library, object=card, values={**rec.values, value.field: value},
                                     category=rec.category, material=rec.material, updated=today)
        report.applied += 1
    for rec in changed.values() if not dry_run else ():
        write_record(rec, sources)
    log.info("ingest %s: %d applied, %d not stronger, %d rejected", path, report.applied, report.weaker,
             len(report.rejected))
    return report
