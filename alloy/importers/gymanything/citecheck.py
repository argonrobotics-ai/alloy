# SPDX-License-Identifier: Apache-2.0
"""Does each cited number appear at the line it cites? A grounding check against a Gym-Anything checkout.

A value counts as found when the cited line or the 12 lines after it (a dict entry, a multi-line call) or the
spec's quote contain the number in SI or a common scaled unit (mm, g, cm, um, half/double for extents). Computed
values (basis ``derived``) cite the function that computes them and are reported, not failed.

Example::

    >>> rows = cite_rows(load_spec("specs/wine_pouring/glass_bordeaux.json"), Path("gym-anything-robotics"))  # doctest: +SKIP
    >>> rows[0]["found"]                                                                                       # doctest: +SKIP
    True
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from alloy.importers.gymanything.provenance import ENV_DIR
from alloy.importers.gymanything.spec import Cited, GarSpec

NUM_RE = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE]-?\d+)?")
SCALES = (1.0, 1e3, 1e6, 1e2, 1e-3, 5e2, 2e3, 2.0, 0.5)
REL_TOL, LINES_AFTER = 0.005, 12


def resolve(repo: Path, sim: str, cite: str) -> tuple[Path | None, int | None]:
    """(file, 1-based line) of a ``path:line`` cite inside the checkout; (None, None) if it does not resolve."""
    path, _, line = cite.strip().rpartition(":")
    if not line.isdigit():
        return None, None
    for base in (repo / path, repo / ENV_DIR / path, repo / ENV_DIR / sim / path):
        if base.is_file():
            return base, int(line)
    return None, int(line)


def contains_number(value: float, text: str) -> bool:
    """True if ``text`` holds ``value`` at any of ``SCALES`` within 0.5 %."""
    for tok in NUM_RE.findall(text):
        try:
            n = float(tok)
        except ValueError:
            continue
        if any(abs(n - value * k) <= 1e-9 + REL_TOL * abs(value * k) for k in SCALES):
            return True
    return False


def _cited_values(spec: GarSpec) -> list[tuple[str, Cited]]:
    out = [(n, getattr(spec, n)) for n in ("mass", "friction", "restitution", "thickness")]
    out += [(f"material.{n}", getattr(spec.material, n)) for n in ("density", "youngs", "poisson")]
    return [(n, c) for n, c in out if c is not None]


def cite_rows(spec: GarSpec, repo: Path) -> list[dict[str, Any]]:
    """One row per cited value: field, basis, cite, resolved (file exists), found (number at the line)."""
    rows = []
    for name, c in _cited_values(spec):
        path, line = resolve(repo, spec.sim, c.cite)
        found = False
        if path is not None and line is not None:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            window = "\n".join(lines[max(0, line - 1):line + LINES_AFTER])
            found = contains_number(c.value, window + "\n" + c.quote)
        rows.append({"field": name, "basis": c.basis, "cite": c.cite, "resolved": path is not None, "found": found})
    return rows


def summarize(rows_by_card: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Counts over every card: found, derived-not-literal, not found (non-derived), unresolved cites."""
    flat = [(card, r) for card, rows in rows_by_card.items() for r in rows]
    missing = [f"{card}:{r['field']} ({r['cite']})" for card, r in flat
               if r["resolved"] and not r["found"] and r["basis"] != "derived"]
    return {"values": len(flat), "found": sum(r["found"] for _, r in flat),
            "derived_not_literal": sum(1 for _, r in flat if not r["found"] and r["basis"] == "derived"),
            "not_found": missing, "unresolved": [f"{card}:{r['field']} ({r['cite']})" for card, r in flat
                                                  if not r["resolved"]]}
