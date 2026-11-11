# SPDX-License-Identifier: Apache-2.0
"""Summary table: pass k/n with a 95 % Wilson interval, median wall minutes and tokens, per engine and per group.

    python3 -m bench.newenv summarize ROWS.jsonl [--public] [--out summary.md]

Columns: share of runs at L2 or better (95 % Wilson) over all requests and over the requests the engine's docs say it
supports (engines.json `declines` removes the others), and L3 for robot tasks. Runs lost to the harness or API (an
agent run with no transcript, or an API error) and runs whose checker is missing are listed but not counted in n.

Example:
    >>> cell([{"success": True}, {"success": False}])
    '1/2 (9-91 %)'
    >>> cell([])
    '-'
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from bench.newenv.score import Row, drop_restricted, restricted_notices
from bench.newenv.stats import median, wilson


VERDICTS = ("L0", "L1", "L2", "L3", "declined", "error")
LEVEL_HEAD = ["L2+ (all requests)", "L2+ (supported requests)", "L3 (robot tasks)", "verdicts"]


def cell(rows: list[Any], key: str = "success") -> str:
    """``k/n (lo-hi %)`` with a 95 % Wilson interval; "-" for no rows."""
    if not rows:
        return "-"
    k, n = sum(bool(r.get(key)) for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({lo * 100:.0f}-{hi * 100:.0f} %)"


def _fmt(x: float | None, nd: int = 0) -> str:
    return "-" if x is None else f"{x:,.{nd}f}"


def verdict_counts(rows: list[Any]) -> str:
    counts = Counter(r.get("status", "?") for r in rows)
    return ", ".join(f"{v} {counts[v]}" for v in VERDICTS if counts[v]) or "-"


def _level_cells(rows: list[Any]) -> list[str]:
    """L2+ over all requests, L2+ over supported requests, L3 (robot tasks only), verdict counts."""
    robot = [r for r in rows if r.get("group") == "robot"]
    return [cell(rows), cell([r for r in rows if r.get("supported", True)]), cell(robot, "solved"),
            verdict_counts(rows)]


def _stat_cells(rows: list[Any]) -> list[str]:
    wall = median([r["agent_wall_s"] / 60 if r.get("agent_wall_s") is not None else None for r in rows])
    return [*_level_cells(rows), _fmt(wall, 1), _fmt(median([r.get("tokens_total") for r in rows])),
            _fmt(median([r.get("output_tokens") for r in rows]))]


def _table(head: list[str], body: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body]


TIMEOUT_RC = (124, 137)


def is_infra(row: Row) -> bool:
    """Lost to the harness or the API, not to the agent: an API error, or an agent that ran (claude_rc written) but
    left no transcript. A run scored without any agent (claude_rc -1, e.g. a reference env) is not lost.

    >>> is_infra({"claude_rc": 0, "turns": None}), is_infra({"claude_rc": -1, "turns": None})
    (True, False)
    """
    if row.get("api_error_status"):
        return True
    rc = row.get("claude_rc", -1)
    return rc >= 0 and rc not in TIMEOUT_RC and not row.get("timed_out") and row.get("turns") is None


def split_infra(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """(counted, lost to the harness or API)."""
    return [r for r in rows if not is_infra(r)], [r for r in rows if is_infra(r)]


def split_unscored(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """(scored, rows whose checker is missing or failed to import: no evidence either way)."""
    return [r for r in rows if r.get("status") != "error"], [r for r in rows if r.get("status") == "error"]


def engine_table(rows: list[Row]) -> list[str]:
    by = defaultdict(list)
    for r in rows:
        by[r["engine"]].append(r)
    return _table(["engine", *LEVEL_HEAD, "median wall min", "median tokens", "median output tokens"],
                  [[e, *_stat_cells(rs)] for e, rs in sorted(by.items())])


def group_table(rows: list[Row]) -> list[str]:
    by = defaultdict(list)
    for r in rows:
        by[(r["engine"], r["group"])].append(r)
    return _table(["engine", "group", *LEVEL_HEAD, "median wall min", "median tokens", "median output tokens"],
                  [[e, g, *_stat_cells(rs)] for (e, g), rs in sorted(by.items())])


def render(rows: list[Row], public: bool = False) -> str:
    """The whole summary text; starts with the notice of every restricted arm in it (--public leaves those arms out)."""
    rows = drop_restricted(rows) if public else rows
    counted, lost = split_infra(rows)
    counted, unscored = split_unscored(counted)
    notices = restricted_notices(rows)
    lines = [*notices, ""] if notices else []
    lines += ["## By engine", "", *engine_table(counted), "", "## By engine and group", "", *group_table(counted)]
    if lost:
        lines += ["", f"Lost to the harness or API (not counted): {', '.join(r['id'] for r in lost)}"]
    if unscored:
        lines += ["", "Not scored (checker missing or broken): " + ", ".join(f"{r['id']} ({r.get('error')})" for r in unscored)]
    return "\n".join(lines) + "\n"


__all__ = ["cell", "engine_table", "group_table", "is_infra", "render", "split_infra"]
