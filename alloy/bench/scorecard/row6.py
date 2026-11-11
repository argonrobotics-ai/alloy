# SPDX-License-Identifier: Apache-2.0
"""Row 6, coverage: the capability matrix and the 200 Gym-Anything task verdicts, counted from their data files.

Nothing is measured here: the row reads `capabilities.jsonl` (one cell per capability and engine, each with its
evidence) and `ga_tasks.jsonl` (one verdict per task, with the evidence tier behind it) and counts them. Every
Gym-Anything verdict is Alloy's re-implementation of GA's criteria on Alloy's port (scripted privileged actor),
not run through GA's verifier. The headline is three counts and never a "tasks pass" figure: tasks run and scored
by Gym-Anything's own verifier (none until such a run exists), tasks with a port judged by our own checks, and
tasks whose physics classes we read as supported (a reading, not a run).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alloy.bench.scorecard.types import Headline, Row, RunOptions

ROW = "6"
TITLE = "Coverage"
DEFAULT_DATA = Path("bench/scorecard/row6")
FULL_PORT_TIER = "A_port_judges_every_GA_gate"
GA_PHRASING = ("Alloy's re-implementation of GA's criteria on Alloy's port (scripted privileged actor); "
               "not run through GA's verifier")
PORT_PHRASING = ("a port is a scripted Alloy version of the task with our own checks: Alloy's re-implementation of "
                 "the task's success gates, with Alloy's scripted tools on a fixed start, not the benchmark's robot; "
                 "not run through GA's verifier")
CLASS_PHRASING = ("the class count is a reading of physics coverage from commits and tests, not a run of the task: "
                  "none of these tasks has been run and scored as Gym-Anything scores it")
_STATUS = {"yes": "yes", "partial": "partial", "no": "no"}


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"row 6 needs {path}: pass --data <folder with capabilities.jsonl and ga_tasks.jsonl>")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def capability_rows(cells: Sequence[dict[str, Any]], source: str) -> list[Row]:
    """One record per (capability, engine); a status the data does not state is `unknown`, never a guess."""
    return [Row(row=ROW, engine=str(c["engine"]), task=f"capability/{c['capability']}", metric="status",
                verdict=_STATUS.get(str(c.get("status")), "unknown"), caveat=str(c.get("caveat") or ""),
                source=source, extra={"evidence": list(c.get("evidence") or [])})
            for c in cells]


def ga_rows(tasks: Sequence[dict[str, Any]], source: str) -> list[Row]:
    """One record per Gym-Anything task: the verdict by physics class, with the evidence tier and gate counts."""
    return [Row(row=ROW, engine="alloy", task=f"gym-anything/{t['task_id']}", metric="verdict_class",
                verdict=_STATUS.get(str(t.get("verdict_class")), "unknown"), caveat=str(t.get("phrasing") or GA_PHRASING),
                source=source, extra={"evidence_tier": t.get("evidence_tier"), "has_port": bool(t.get("alloy_env")),
                                      "split": t.get("split"), "gates_total": t.get("gates_total"),
                                      "gates_at_ga_rule": t.get("gates_at_ga_rule")})
            for t in tasks]


def _rate(metric: str, k: int, n: int, caveat: str, source: str) -> Row:
    """A count over all of Gym-Anything's tasks: a census, not a sample, so it carries no interval."""
    return Row(row=ROW, engine="alloy", task="gym-anything", metric=metric, value=k / n, unit="fraction", k=k, n=n,
               caveat=caveat, source=source)


def summary_rows(cells: Sequence[dict[str, Any]], tasks: Sequence[dict[str, Any]], source: str) -> list[Row]:
    n = len(tasks)
    scored = sum(bool(t.get("scored_by_ga_verifier")) for t in tasks)
    ports = [t for t in tasks if t.get("alloy_env")]
    full_ports = [t for t in ports if t.get("evidence_tier") == FULL_PORT_TIER]
    full = len(full_ports)
    other = sorted(int(t.get("gates_judged_other_measure") or 0) for t in full_ports) or [0]
    by_class = sum(t.get("verdict_class") == "yes" for t in tasks)
    rows = [_rate("tasks_scored_by_ga_verifier", scored, n, "run and scored by Gym-Anything's own verifier", source),
            _rate("tasks_with_port_own_checks", len(ports), n, PORT_PHRASING, source),
            _rate("ports_judging_all_gates_own_checks", full, n,
                  f"{PORT_PHRASING}; each of these judges {other[0]} to {other[-1]} gates by another measure than "
                  "Gym-Anything's", source),
            _rate("tasks_physics_class_supported", by_class, n, CLASS_PHRASING, source)]
    for engine in sorted({str(c["engine"]) for c in cells}):
        counts = Counter(_STATUS.get(str(c.get("status")), "unknown") for c in cells if c["engine"] == engine)
        total = sum(counts.values())
        rows += [Row(row=ROW, engine=engine, task="capability", metric=f"cells_{status}", k=counts[status], n=total,
                     value=counts[status] / total, unit="fraction", source=source, documented=True,
                     caveat="Alloy's yes needs a passing test; a rival's yes means the feature is documented")
                 for status in ("yes", "partial", "no", "unknown")]
    return rows


def headline(summary: Sequence[Row], command: str) -> Headline:
    """Three counts, in this order, and no "N tasks pass" figure: nothing here was scored by Gym-Anything."""
    by = {(r.engine, r.task, r.metric): r for r in summary}
    scored = by["alloy", "gym-anything", "tasks_scored_by_ga_verifier"]
    ports = by["alloy", "gym-anything", "tasks_with_port_own_checks"]
    full = by["alloy", "gym-anything", "ports_judging_all_gates_own_checks"]
    cls = by["alloy", "gym-anything", "tasks_physics_class_supported"]
    caps = {s: by["alloy", "capability", f"cells_{s}"] for s in ("yes", "partial", "no")}
    number = (f"Gym-Anything's {scored.n} tasks: run and scored by Gym-Anything's own verifier {scored.k}; "
              f"ports with our own checks {ports.k} (k of n gates each; {full.k} judge all n); "
              f"physics class supported {cls.k} of {cls.n} (by reading, not run). "
              f"Capabilities: {caps['yes'].k} yes, {caps['partial'].k} partial, {caps['no'].k} no of {caps['yes'].n}")
    return Headline(row=ROW, title=TITLE, number=number, command=command, state="measured",
                    bound=f"counts over all {cls.n} tasks, not a sample: no interval",
                    caveat=f"{PORT_PHRASING}. {CLASS_PHRASING}. Rival capability cells are read from their docs, "
                           "not tested")


def run(opts: RunOptions) -> tuple[list[Row], Headline]:
    data = opts.data or DEFAULT_DATA
    cells = _records(data / "capabilities.jsonl")
    tasks = _records(data / "ga_tasks.jsonl")
    source = str(data)
    summary = summary_rows(cells, tasks, source)
    return summary + capability_rows(cells, source) + ga_rows(tasks, source), headline(summary, opts.command)


__all__ = ["CLASS_PHRASING", "DEFAULT_DATA", "GA_PHRASING", "PORT_PHRASING", "ROW", "TITLE", "capability_rows",
           "ga_rows", "headline", "run", "summary_rows"]
