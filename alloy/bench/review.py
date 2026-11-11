# SPDX-License-Identifier: Apache-2.0
"""Producer rows under review (coordinator rulings): listed under the board, and any board row they feed is marked.

A row is under review when a ruling found the code that made it suspect but no rerun has replaced it yet. The mark
names the ruling and the reason; the verdict stays as reported (the producer reruns, then the entry goes), unless
the ruling withdrew it too (`withhold`): then every row it feeds reads inconclusive, value kept, until the re-feed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from alloy.bench.types import ScoreRow

log = logging.getLogger("alloy.bench")


@dataclass(frozen=True)
class UnderReview:
    """Producer rows a ruling put under review: where they are, what they measure, and why."""

    source: str                 # results file, repo-relative (docs/notes/...) or under [internal note]
    lines: tuple[int, ...]      # 1-based lines of the rows in it
    rows: str                   # what the rows are
    owner: str                  # the producer that reruns them
    why: str                    # the ruling and the reason
    withhold: bool = False      # the ruling withdrew the verdict: fed rows read inconclusive ("under review, not pass")


WITHHELD = "inconclusive"       # the verdict a withheld row shows (value kept)

_ROUTE_BUG = ("coordinator ruling 2026-11-03 ~23:15Z item 3 ([internal note]): made with WP5's ordered MuJoCo "
              "route on (T09 covered since 8d513582) in a scene with a precision pair appended into MuJoCo; the "
              "route's contact sort moves the appended precision contacts while the binding reads the old slots "
              "(WP5 cause, fix on main aa2a613a; rerun with the route on to clear): bad worlds from about 0.62 x N (WP12 ROUTE AUDIT, REVISED, "
              "[internal note])")

# Cleared: SB-4's liquid half (WP9, 14:33Z) by WP9's census, e9eb9035. SB-8 / CPL-4's single-height rows (WP4, withheld
# 2026-11-04 20:46Z, SCR-WP11-3) by the re-feed on the amended protocol: both amendments on main (CPL-4 protocol
# 20141cb6, D20 welded-body E_scale aa418263), 8-height 60 s rows in docs/notes/wp6-results.md; WP4's rows feed neither.
UNDER_REVIEW: tuple[UnderReview, ...] = (
    UnderReview("[internal note]", (15,), "T09 fast-profile route V1c", "WP2",
                f"{_ROUTE_BUG}; void until rerun with the route off or fixed"),
    UnderReview("[internal note]", (24,), "T09 fast-profile route V1h", "WP2",
                f"{_ROUTE_BUG}; void until rerun with the route off or fixed"),
    UnderReview("[internal note]", (13, 14), "T09 fast-profile route F1c, F0c", "WP2",
                f"{_ROUTE_BUG}; void until rerun with the route off or fixed"),
    UnderReview("[internal note]", (17, 18), "T09 fast-profile route F1s, F0s", "WP2",
                f"{_ROUTE_BUG}; void until rerun with the route off or fixed"),
    UnderReview("[internal note]", (22, 23), "T09 fast-profile route F1h, F0h", "WP2",
                f"{_ROUTE_BUG}; void until rerun with the route off or fixed"),
)


def _pointers(entries: Sequence[UnderReview]) -> dict[str, UnderReview]:
    """`<source>:<line>` -> its entry, for matching board `source_rows`."""
    return {f"{e.source}:{n}": e for e in entries for n in e.lines}


def _note(hits: Sequence[str], ptr: dict[str, UnderReview]) -> str:
    """One clause per entry hit, `<source>:<lines> (<why>)`, so a row fed by several lines of one entry reads once."""
    by: dict[UnderReview, list[str]] = {}
    for s in hits:
        by.setdefault(ptr[s], []).append(s.rsplit(":", 1)[1])
    return "; ".join(f"{e.source}:{','.join(sorted(ns, key=int))} ({e.why})" for e, ns in by.items())


def _marked(r: ScoreRow, hits: Sequence[str], ptr: dict[str, UnderReview]) -> ScoreRow:
    """`r` with the "under review: ..." prefix; verdict withheld (inconclusive) when a hit entry's ruling withdrew it."""
    tail = f"; {r.reason}" if r.reason else ""
    if not any(ptr[s].withhold for s in hits):
        return replace(r, reason=f"under review: {_note(hits, ptr)}{tail}")
    head = f"under review, verdict withheld (reported {r.verdict}): {_note(hits, ptr)}"
    return replace(r, verdict=WITHHELD, reason=head + tail)


def mark_under_review(rows: Sequence[ScoreRow], entries: Sequence[UnderReview] = UNDER_REVIEW) -> list[ScoreRow]:
    """Prefix "under review: ..." to the reason of every row fed by a row under review; verdicts unchanged unless an
    entry withholds them (then inconclusive, value kept)."""
    ptr = _pointers(entries)
    out = []
    for r in rows:
        hits = [s for s in r.source_rows if s in ptr]
        if not hits:
            out.append(r)
            continue
        log.info("scoreboard: %s fed by rows under review: %s", r.metric_id, ", ".join(hits))
        out.append(_marked(r, hits, ptr))
    return out


def review_json(entries: Sequence[UnderReview] = UNDER_REVIEW) -> list[dict[str, object]]:
    return [{"source": e.source, "lines": list(e.lines), "rows": e.rows, "owner": e.owner, "why": e.why,
             "withhold": e.withhold} for e in entries]


def review_md(entries: Sequence[UnderReview] = UNDER_REVIEW) -> list[str]:
    """Board section listing every row under review (rows feeding no board row are listed too)."""
    if not entries:
        return []
    out = ["", "## Rows under review", ""]
    for e in entries:
        where = f"{e.source}:{','.join(map(str, e.lines))}"
        held = f" Verdict withheld: rows it feeds read {WITHHELD} until the re-feed." if e.withhold else ""
        out.append(f"- **{e.owner}** {where}: {e.rows}. Under review: {e.why}.{held}")
    return out


__all__ = ["UNDER_REVIEW", "UnderReview", "WITHHELD", "mark_under_review", "review_json", "review_md"]
