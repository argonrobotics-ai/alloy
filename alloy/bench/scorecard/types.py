# SPDX-License-Identifier: Apache-2.0
"""Record types of the scorecard: one `Row` per measured number, one `Headline` per scorecard row."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ROW_SCHEMA = "alloy.scorecard.row/1"
HEADLINE_SCHEMA = "alloy.scorecard.headline/1"
Verdict = Literal["pass", "fail", "inconclusive", "not_run", "error", "yes", "partial", "no", "unknown", ""]
VERDICTS: tuple[str, ...] = ("pass", "fail", "inconclusive", "not_run", "error", "yes", "partial", "no", "unknown", "")


@dataclass(frozen=True)
class Row:
    """One number of one scorecard row. `lo` / `hi` bound `value` (a 95 % interval, or a tolerance: say which in
    `bound_kind`); `k` of `n` for rates. `internal=True` keeps the record out of every public file."""

    row: str                                  # "1", "1b", "2", ... "8"
    engine: str                               # "alloy", "mujoco_warp", "newton", "isaac_lab_physx", ...
    task: str                                 # task, law, capability or request id
    metric: str                               # "steps_per_s", "glitch_rate", "error", "status", ...
    value: float | None = None
    unit: str = ""
    lo: float | None = None
    hi: float | None = None
    bound_kind: str = ""                      # "wilson95", "tolerance", "min_max", ""
    k: int | None = None
    n: int | None = None
    verdict: Verdict = ""
    caveat: str = ""
    internal: bool = False
    documented: bool = False                  # a count of facts read from public docs, not a benchmark measurement
    device: str = ""
    command: str = ""
    commit: str = ""
    date: str = ""
    source: str = ""                          # file the number was read from, or the harness that measured it
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Headline:
    """What the one-page scorecard prints for a row: the number, its bound, the command, date, commit, one caveat."""

    row: str
    title: str
    number: str                               # e.g. "41 of 47 laws pass" or "not measured"
    bound: str = ""
    command: str = ""
    date: str = ""
    commit: str = ""
    caveat: str = ""
    state: Literal["measured", "draft", "not_measured"] = "draft"
    internal: bool = False


@dataclass(frozen=True)
class RunOptions:
    out: Path
    device: str = "cpu"
    quick: bool = False
    seed: int = 0
    command: str = ""
    data: Path | None = None                  # verdict / result files a row reads instead of measuring
    jobs: int = 1                             # processes a row may use (row 3: one law per process)


Runner = Callable[[RunOptions], tuple[Sequence[Row], Headline]]


@dataclass(frozen=True)
class RowSpec:
    """A scorecard row: its title, who keeps it, the machine it is measured on and the function that runs it
    (None until its lead wires it: the command then says so and exits 2)."""

    id: str
    title: str
    lead: str
    machine: str = ""                         # cloud recipe key (docs/benchmarks/reproduce.md)
    runner: Runner | None = None
    note: str = ""


__all__ = ["HEADLINE_SCHEMA", "ROW_SCHEMA", "VERDICTS", "Headline", "Row", "RowSpec", "RunOptions", "Runner",
           "Verdict"]
