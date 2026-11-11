# SPDX-License-Identifier: Apache-2.0
"""Rows under review (coordinator ruling 2026-11-03 ~23:15Z item 3): listed under the board, fed rows marked."""

from __future__ import annotations

from dataclasses import replace

from alloy.bench import review as RV
from alloy.bench.scoreboard import render_markdown
from alloy.bench.types import ScoreRow


def _row(metric_id: str, sources: tuple[str, ...], reason: str = "") -> ScoreRow:
    return ScoreRow(metric_id=metric_id, value=1.64, unit="x", target="<= 1.3x", verdict="fail",
                    source_rows=sources, command="", commit="", date="", status="active", reason=reason)


def test_a_row_fed_by_a_row_under_review_is_marked_and_keeps_its_verdict():
    rows = [_row("SB-11:P3-1 cost, T09", ("docs/notes/wp5-results.md:415",), "informational"),
            _row("SB-11:P3-1 cost, T12", ("docs/notes/wp5-results.md:416",), "informational")]
    entry = RV.UnderReview("docs/notes/wp5-results.md", (415,), "a row", "WP5", "coordinator ruling X")
    marked, clean = RV.mark_under_review(rows, (entry,))
    assert marked.reason.startswith("under review: docs/notes/wp5-results.md:415 (coordinator ruling")
    assert marked.reason.endswith("; informational") and marked.verdict == "fail" and marked.value == 1.64
    assert clean == rows[1]


def test_every_named_row_is_listed_under_the_board():
    md = "\n".join(RV.review_md())
    assert "## Rows under review" in md
    for e in RV.UNDER_REVIEW:
        assert f"{e.source}:{','.join(map(str, e.lines))}" in md
    assert "## Rows under review" in render_markdown([])
    assert {e.owner for e in RV.UNDER_REVIEW} == {"WP2"}   # WP2's T09 fast-route rows (WP4's CPL-4: re-fed, cleared)
    assert not any("dc5" in e.source or "wp5" in e.source for e in RV.UNDER_REVIEW)  # withdrawn (no append)
    assert all(d["why"] and d["lines"] for d in RV.review_json())


def test_a_row_fed_by_several_lines_of_one_entry_names_it_once_with_its_verdict_kept():
    """SB-4's liquid half was under review 14:33Z to WP9's census (e9eb9035): a row fed by two lines reads once."""
    entry = RV.UnderReview("docs/notes/wp9-results.md", (45, 47, 48, 54), "LIQ-8 leak rows", "WP9", "wall-seam leak")
    sb4 = _row("SB-4", ("docs/notes/wp3-results.md:35", "docs/notes/wp9-results.md:54",
                        "docs/notes/wp9-results.md:45"), "verdict on the Alloy default route")
    sb4 = replace(sb4, value=0.0, verdict="pass")
    (marked,) = RV.mark_under_review([sb4], (entry,))
    assert marked.reason == ("under review: docs/notes/wp9-results.md:45,54 (wall-seam leak); "
                             "verdict on the Alloy default route")
    assert (marked.value, marked.verdict) == (0.0, "pass")
    rigid = _row("SB-4:routed / mujoco (WP3)", ("docs/notes/wp3-results.md:35",))
    assert RV.mark_under_review([rigid], (entry,)) == [rigid]
    assert RV.mark_under_review([sb4]) == [sb4]   # the census cleared it: no live entry names WP9's rows


def test_a_withholding_entry_reads_inconclusive_with_its_value_kept():
    """Coordinator 2026-11-04 20:46Z (SCR-WP11-3): SB-8's single-height CPL-4 pass was a phase lottery, so the board
    showed SB-8 under review, not pass (withhold), until the re-feed on the amended protocol cleared the entry."""
    entry = RV.UnderReview("docs/notes/wp4-results.md", (31, 32), "CPL-4 rows", "WP4", "phase lottery", withhold=True)
    sb8 = replace(_row("SB-8", ("docs/notes/wp4-results.md:31", "docs/notes/wp4-results.md:32"), "coupling injection"),
                  value=0.4, verdict="pass")
    (marked,) = RV.mark_under_review([sb8], (entry,))
    assert (marked.verdict, marked.value) == (RV.WITHHELD, 0.4)
    assert marked.reason == ("under review, verdict withheld (reported pass): docs/notes/wp4-results.md:31,32 "
                             "(phase lottery); coupling injection")
    assert "Verdict withheld" in "\n".join(RV.review_md((entry,)))
    assert RV.mark_under_review([sb8]) == [sb8]       # re-fed: no live entry names WP4's rows
    assert not any(d["withhold"] for d in RV.review_json())
