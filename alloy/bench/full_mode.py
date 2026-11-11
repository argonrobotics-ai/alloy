# SPDX-License-Identifier: Apache-2.0
"""Laws SB-12 counts at their full-mode verdict (coordinator ruling 2026-11-05 05:48Z, A13 in SB-12).

When a law's quick mode lies outside the law's own stated validity range, the quick verdict cannot decide it, and
turning it into not_run would drop the law from SB-12's denominator. So SB-12 counts the producer's full-mode row
instead, the stronger evidence, and the board says which mode fed the law. A law whose quick mode is inside its range
keeps its quick verdict, a fail included. A full-mode row that cannot be read counts as inconclusive (never pass),
so the law stays in the denominator either way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from alloy.bench import gpu_repeat as GR
from alloy.bench import note_rows as NR

log = logging.getLogger("alloy.bench")

RULING = "coordinator 2026-11-05 05:48Z"


@dataclass(frozen=True)
class FullMode:
    """A law whose quick mode is outside its stated range, and the producer row holding its full-mode verdict."""

    outside: str                # why quick is outside the law's stated range (from the law's own text)
    row: NR.NoteSource          # the full-mode row in the producer's results note
    device: str = "cuda"        # where the full-mode row ran (the producer's GPU run)


@dataclass(frozen=True)
class Counted:
    """The verdict SB-12 counts for one law result, and the board's note on which mode fed it."""

    verdict: str
    note: str
    source: str                 # docs/notes/<file>.md:<line> of the full-mode row; "" when it was not found


_SEAL_TWINS = "alloy/solvers/sph/seal_twins.py"

FULL_MODE: dict[str, FullMode] = {
    "A13.piston_displacement": FullMode(
        outside=(f"the law fits the outflow slope over the push after 0.1 s ({_SEAL_TWINS}); quick pushes 0.04 s "
                 "(about 2 layers) and fits its last 0.03 s, which cannot resolve the 1 % bound (WP9)"),
        row=NR.NoteSource("wp9", r"^A13\.piston_displacement full\b", verdict_column="status")),
    "A13.sealed_column": FullMode(
        outside=("the law's drift bound is over 1 s ('column drift < 0.1 particle spacing over 1 s', "
                 "alloy/solvers/sph/laws.py) at C18's 110-layer column; quick runs a 20-layer column for 0.05 s"),
        row=NR.NoteSource("wp9", r"^A13\.sealed_column full\b", verdict_column="status")),
}


def counted(law: str, quick_verdict: str, notes: Path | None = None) -> Counted | None:
    """The full-mode verdict SB-12 counts for `law` (None for a law that keeps its quick verdict).

    The newest matching row decides: the last one in note order, since producers append.
    """
    fm = FULL_MODE.get(law)
    if fm is None:
        return None
    rows = [r for r in NR.read(fm.row, notes) if r.verdict]
    sample = GR.sample_label(fm.device)
    head = (f"{law} counted at its full-mode verdict on {fm.device}{f' ({sample})' if sample else ''} ({RULING}; "
            f"quick {quick_verdict} not counted: {fm.outside})")
    if not rows:
        log.warning("full mode: no %s row with a verdict for %s; counted inconclusive", fm.row.package, law)
        return Counted("inconclusive", f"{head}: no full-mode row found in docs/notes/{fm.row.package}-results.md, "
                                       "counted inconclusive", "")
    row = rows[-1]
    return Counted(row.verdict or "inconclusive", f"{head}: {row.verdict} ({row.source}, {row.commit})", row.source)


__all__ = ["FULL_MODE", "RULING", "Counted", "FullMode", "counted"]
