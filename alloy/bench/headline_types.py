# SPDX-License-Identifier: Apache-2.0
"""Types of the SB headline feed (`alloy.bench.headline`): which producer rows feed a headline, and the rows matched."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from alloy.bench import note_rows as NR

NOT_MEASURED = ("not_run",)


@dataclass(frozen=True)
class Feed:
    """Producer rows that feed one headline."""

    metric: str                     # producer §5 id, e.g. "CT-1"
    value: str                      # regex; group 1 is the number, first match in the value cell
    unit: str
    tag: str = ""                   # sub-row variant; "" = the row label without its id
    label: str = ""                 # regex the row label must match (case-insensitive)
    exclude: str = ""               # regex the row label must not match
    decides: bool = True            # False: reported beside the verdict only
    pass_value: float | None = None  # value of a pass row whose text states it in words (e.g. "equal" = share 1)
    default: str | None = None      # regex on the label of rows on Alloy's default route; None: every row is
    alt: bool = False               # a configuration Alloy does not run by default (shown in "all configurations")
    quantity: str = ""              # rows of feeds with the same quantity (default: the metric) share one value
    info: str = ""                  # informational line (never decides, may re-read a row another feed uses): why
    fail_above: float | None = None  # info line verdict: fail when its value exceeds this (else pass)
    fail_below: float | None = None  # info line verdict: fail when its value is under this (else pass)
    note: NR.NoteSource | None = None  # read the rows from a results note instead of the collected reports
    marker: str = ""                # regex the label or value cell must match (e.g. an opt-in route profile)
    not_marker: str = ""            # regex neither the label nor the value cell may match
    caveat: str = ""                # printed beside every number this feed shows
    scale: float = 1.0              # number read x scale (e.g. 0.01: a "100.0 %" cell as a share)
    every: Literal["", "max", "min"] = ""  # the max / min over every match in the value cell (a row listing several
                                    # scenes or repeats), not the first match
    newest_last: bool = False       # with `newest`: the producer lists a class oldest first (WP3), not newest first
    newest: str = ""                # regex on the label naming a class: its first row in note order decides (the
                                    # producer lists a class newest first), later rows go beside as superseded
                                    # (an info feed drops them: it shows the class's newest number only)
    aside: Callable[[str, str], str] | None = None  # text shown beside a row's number, built from its label and
                                    # value cell (SB-7's S2 label); a superseded row drops it
    required: str = ""              # a class that must be measured: while no measured deciding row comes from this
                                    # feed, the headline reads fail and names it (SB-7's route variants, 21:04Z)


@dataclass(frozen=True)
class Headline:
    id: str
    feeds: tuple[Feed, ...]
    better: Literal["lower", "higher"]
    note: str = ""
    default_route: bool = False     # verdict on the default-route rows, "all configurations" sub-row beside it
    beside: str = "all configurations"
    beside_feed: str = ""           # tag of the one alt feed shown beside (else every row: "all configurations")


@dataclass(frozen=True)
class ProducerRow:
    """One matched producer row."""

    feed: Feed
    package: str
    label: str
    text: str
    value: float | None
    verdict: str
    source: str
    commit: str
    command: str
    date: str
    reason: str

    @property
    def tag(self) -> str:
        if self.feed.tag:
            return self.feed.tag
        head = re.sub(rf"^\**\s*{re.escape(self.feed.metric)}\b\s*(/\s*SB-\d+)?", "", self.label).strip(" *")
        if head.startswith("(") and head.endswith(")"):  # "SB-7 (xpbd_box_stack class)"; keep "T12 (N = 1)" whole
            head = head[1:-1]
        return head.strip() or self.feed.metric

    @property
    def measured(self) -> bool:
        return self.verdict not in NOT_MEASURED


IN_NOTE = re.compile(r"\bsuperseded\b", re.IGNORECASE)      # in a row's label or its note's verdict cell
NO_CURRENT = "superseded in its note, no current row"


def demote_superseded(rows: list[ProducerRow]) -> list[ProducerRow]:
    """A deciding row that its own results note marks superseded (the word in its label, or in its verdict cell:
    "fail; superseded by <commit>, re-run pending") stops deciding. It is shown beside, tagged, and the headline says
    it has no current row for it (audit AB-inv-reports-12: SB-7's 0.687 was the bound of such a row)."""
    out = []
    for p in rows:
        marked = IN_NOTE.search(p.label) or IN_NOTE.search(NR.verdict_cell(p.source))
        if marked and p.feed.decides and not p.feed.info:
            p = dataclasses.replace(p, feed=dataclasses.replace(p.feed, decides=False, aside=None,
                                                                tag=f"{p.tag} ({NO_CURRENT})"))
        out.append(p)
    return out


@dataclass(frozen=True)
class Summary:
    """Value, verdict and reason over one population of producer rows."""

    value: float | None
    unit: str
    verdict: str
    reason: str
    key: ProducerRow | None


__all__ = ["Feed", "Headline", "NOT_MEASURED", "NO_CURRENT", "ProducerRow", "Summary", "demote_superseded"]
