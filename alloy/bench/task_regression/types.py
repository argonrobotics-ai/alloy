# SPDX-License-Identifier: Apache-2.0
"""Types of the task-regression suite: a task's checks, its last recorded result, and the verdict of a run.

Coordinator ruling 2026-11-06 05:55Z (scr/XB4-rulings.md, "engine focus"): every finished task keeps its last recorded
result (metric, value, build) and a small fixed seed set; a run passes a task when every metric stays within the
tolerance stated for it, and a tolerance is never loosened to make a run pass. The "cpu" part (CPU-sized: minutes)
runs in the post-merge full run (WP1's hook) and gates last-green; the "nightly" part (GPU runs, and CPU runs too long
for a post-merge run) runs in the reconciliation pass (coordinator ruling 2026-11-06, end of scr/XB4-rulings.md;
WP5's nightly GPU run is paused).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Rule = Literal["min", "max", "abs", "rel", "band"]
Part = Literal["cpu", "nightly"]
Verdict = Literal["pass", "fail", "error", "unbaselined"]


@dataclass(frozen=True)
class MetricCheck:
    """One metric of a task and the tolerance stated for it.

    rule: "min" passes value >= last - tol (a count or rate that must not drop); "max" passes value <= last + tol (a
    depth or error that must not grow); "abs" passes |value - last| <= tol; "rel" passes |value - last| <= tol * |last|;
    "band" passes lo <= value <= hi (a band from the task's own spec, independent of the last value).
    """

    name: str
    rule: Rule
    tol: float = 0.0
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    why: str = ""                   # where the tolerance comes from (a spec, a measured spread); never "to pass"

    def bounds(self, last: float | None) -> tuple[float, float]:
        """(low, high) the value must lie in, given the last recorded value."""
        if self.rule == "band":
            return (-math.inf if self.lo is None else self.lo, math.inf if self.hi is None else self.hi)
        if last is None:
            raise ValueError(f"{self.name}: rule {self.rule!r} needs a last recorded value")
        width = self.tol * abs(last) if self.rule == "rel" else self.tol
        low = -math.inf if self.rule == "max" else last - width
        high = math.inf if self.rule == "min" else last + width
        return low, high

    def holds(self, value: float, last: float | None) -> bool:
        low, high = self.bounds(last)
        return math.isfinite(value) and low <= value <= high


@dataclass(frozen=True)
class TaskCheck:
    """A finished task the engine must keep doing: how to re-run its small check and what it must still give."""

    id: str                         # e.g. "DC1.cat6_s4"
    title: str
    owner: str                      # the WP a regression goes to (as ASK)
    part: Part                      # cpu: post-merge full run, gates last-green; nightly: GPU or long CPU runs
    runner: str                     # "module:function"; called as fn(seeds, device) -> dict[str, float]
    seeds: tuple[int, ...]
    metrics: tuple[MetricCheck, ...]
    est_s: float                    # expected wall seconds on its device (the runner's timeout is a multiple of it)
    device: str                     # "cpu" or "cuda:0": where the check runs (and was baselined)
    command: str                    # the equivalent shell command, for a human re-run
    source: str                     # where the task and its pass criterion are documented
    note: str = ""
    route: bool = False             # its tool asks for WP5's ordered MuJoCo route, so its GPU rows repeat bit for bit


@dataclass(frozen=True)
class LastResult:
    """A task's last recorded result on a named build (from last_results.json)."""

    values: dict[str, float]
    build: str
    date: str
    source: str                     # the run that recorded it (report path or note row)


@dataclass(frozen=True)
class MetricOutcome:
    name: str
    value: float | None
    last: float | None
    low: float
    high: float
    ok: bool


@dataclass
class TaskOutcome:
    """The verdict of one task in one run."""

    task: str
    owner: str
    part: Part
    verdict: Verdict
    metrics: list[MetricOutcome] = field(default_factory=list)
    wall_s: float = 0.0
    detail: str = ""

    @property
    def gates(self) -> bool:
        """A CPU task that is baselined decides last-green; an unbaselined one is reported only."""
        return self.part == "cpu" and self.verdict != "unbaselined"


def judge(task: TaskCheck, last: LastResult | None, measured: dict[str, float]) -> TaskOutcome:
    """Compare a run's metrics with the task's last recorded result under each metric's stated tolerance."""
    out = TaskOutcome(task.id, task.owner, task.part, "pass")
    for m in task.metrics:
        prev = None if last is None else last.values.get(m.name)
        if m.rule != "band" and prev is None:
            out.verdict = "unbaselined" if out.verdict == "pass" else out.verdict
            out.detail = f"no last recorded {m.name}: run `rebaseline` on current main first"
            continue
        low, high = m.bounds(prev)
        value = measured.get(m.name)
        ok = value is not None and m.holds(float(value), prev)
        out.metrics.append(MetricOutcome(m.name, None if value is None else float(value), prev, low, high, ok))
        if not ok and out.verdict != "fail":            # a band is the task's spec: it fails even before a baseline
            out.verdict = "fail"
    return out


__all__ = ["LastResult", "MetricCheck", "MetricOutcome", "Part", "Rule", "TaskCheck", "TaskOutcome", "Verdict",
           "judge"]
