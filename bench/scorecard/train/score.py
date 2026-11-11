# SPDX-License-Identifier: Apache-2.0
"""Row 4 scorer: wall-clock and env-steps to a fixed threshold, the same code for every framework.

A run "reaches" a threshold at the first point where the metric meets it and keeps meeting it for ``hold``
consecutive points (so one lucky reading does not count). A run that never does is censored: it is reported as
not reached, never dropped, and a cell's median is "not reached" unless more than half its seeds reached.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from bench.scorecard.train.schema import Run, read_runs


@dataclass(frozen=True)
class Threshold:
    metric: str
    value: float
    higher_is_better: bool
    hold: int = 3

    def met(self, x: float) -> bool:
        return x >= self.value if self.higher_is_better else x <= self.value


@dataclass(frozen=True)
class Reach:
    reached: bool
    wall_s: float | None
    env_steps: int | None
    iteration: int | None


@dataclass(frozen=True)
class Cell:
    """One (framework, task, variant) cell over its seeds."""

    framework: str
    task: str
    variant: str
    internal_only: bool
    seeds: int
    seeds_reached: int
    wall_s_median: float | None
    wall_s_min: float | None
    wall_s_max: float | None
    env_steps_median: float | None
    steps_per_s_median: float | None  # training throughput over the whole run (env steps / wall s)


def time_to_threshold(run: Run, th: Threshold) -> Reach:
    """First point from which the metric holds the threshold for ``th.hold`` consecutive points."""
    streak_start = None
    streak = 0
    for i, p in enumerate(run.points):
        x = p.metrics.get(th.metric)
        if x is not None and x == x and th.met(x):  # x == x: NaN never counts
            if streak == 0:
                streak_start = i
            streak += 1
            if streak >= th.hold:
                first = run.points[streak_start]
                return Reach(True, first.wall_s, first.env_steps, first.iteration)
        else:
            streak = 0
    return Reach(False, None, None, None)


def _throughput(run: Run) -> float | None:
    if not run.points or run.points[-1].wall_s <= 0:
        return None
    last = run.points[-1]
    return last.env_steps / last.wall_s


def score_cell(runs: list[Run], th: Threshold) -> Cell:
    """Median over seeds; "not reached" (None) unless more than half the seeds reached the threshold."""
    h = runs[0].header
    reaches = [time_to_threshold(r, th) for r in runs]
    hit = [r for r in reaches if r.reached]
    rates = [x for x in (_throughput(r) for r in runs) if x is not None]
    majority = len(hit) * 2 > len(runs)
    # Censored seeds count as +inf, so the median is honest when a minority fails.
    walls = sorted([r.wall_s for r in hit] + [float("inf")] * (len(runs) - len(hit)))
    steps = sorted([float(r.env_steps) for r in hit] + [float("inf")] * (len(runs) - len(hit)))
    return Cell(
        framework=h.framework,
        task=h.task,
        variant=h.variant,
        internal_only=any(r.header.internal_only for r in runs),
        seeds=len(runs),
        seeds_reached=len(hit),
        wall_s_median=statistics.median(walls) if majority else None,
        wall_s_min=min(r.wall_s for r in hit) if hit else None,
        wall_s_max=max(r.wall_s for r in hit) if len(hit) == len(runs) else None,
        env_steps_median=statistics.median(steps) if majority else None,
        steps_per_s_median=statistics.median(rates) if rates else None,
    )


def score_files(paths: list[Path], thresholds: dict[str, Threshold]) -> list[Cell]:
    groups: dict[tuple[str, str, str], list[Run]] = defaultdict(list)
    for path in paths:
        for run in read_runs(path):
            groups[(run.header.framework, run.header.task, run.header.variant)].append(run)
    cells = []
    for (_, task, _), runs in sorted(groups.items()):
        if task not in thresholds:
            raise KeyError(f"no threshold for task {task!r}; known: {sorted(thresholds)}")
        cells.append(score_cell(runs, thresholds[task]))
    return cells


def load_thresholds(path: Path) -> dict[str, Threshold]:
    """``{"go2_flat": {"metric": ..., "value": ..., "higher_is_better": ..., "hold": 3}, ...}``"""
    return {task: Threshold(**spec) for task, spec in json.loads(path.read_text()).items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score time-to-train runs (JSON lines) against the protocol thresholds.")
    ap.add_argument("runs", nargs="+", type=Path, help="run files (JSON lines) or directories of *.jsonl")
    ap.add_argument("--thresholds", type=Path, default=Path(__file__).with_name("thresholds.json"))
    ap.add_argument("--out", type=Path, help="write the cells here as JSON lines (default: stdout)")
    ap.add_argument("--public", action="store_true", help="leave out cells whose runs set internal_only")
    args = ap.parse_args(argv)
    paths = [p for r in args.runs for p in (sorted(r.glob("**/*.jsonl")) if r.is_dir() else [r])]
    cells = score_files(paths, load_thresholds(args.thresholds))
    if args.public:
        cells = [c for c in cells if not c.internal_only]
    text = "".join(json.dumps(asdict(c), sort_keys=True) + "\n" for c in cells)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
