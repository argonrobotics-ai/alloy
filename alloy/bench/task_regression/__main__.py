# SPDX-License-Identifier: Apache-2.0
"""Task-regression suite CLI (coordinator 2026-11-06 05:55Z).

    python -m alloy.bench.task_regression list [--part cpu|nightly]
    python -m alloy.bench.task_regression run --part cpu|nightly [--only ID ...] [--jobs N] [--device D] [--out DIR]
    (nightly has the aliases gpu and reconcile)
    python -m alloy.bench.task_regression rebaseline --task ID --reason TEXT

`run` exits 0 when the part is green (every task within its stated tolerance), 1 when a task regressed or its run
crashed or timed out (a code regression: bisectable), 2 on an environment error found before any task runs (the
installed Newton is not the pin, an unknown task id). WP1's post-merge hook runs the CPU part and advances last-green
only on 0 (with the full CPU suite green). The nightly part (GPU tasks, and CPU tasks too long for a post-merge run)
runs in the coordinated reconciliation pass (coordinator ruling 2026-11-06, end of scr/XB4-rulings.md: the
task-regression GPU part is deferred there and the nightly GPU run is paused). Each task runs on its own device (the
one it was baselined on). The report (report.json / .md in --out, default
bench/results/task_regression/<time>-<build>) lists each regression with its owner WP; report.json's "rows" are one
per (task, metric). `rebaseline` re-takes a task's last result on the current build after a ruled
change; it records the reason and never touches a tolerance.
"""

from __future__ import annotations

import argparse
import logging
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from alloy.bench.task_regression import store
from alloy.bench.task_regression.registry import NO_ADAPTERS, TASKS, available, get
from alloy.bench.task_regression.run import head_commit, measure, run_part

log = logging.getLogger("alloy.bench.task_regression")
DEFAULT_OUT = Path("bench/results/task_regression")
PARTS = ("cpu", "nightly")


def _part(name: str) -> str:
    return "nightly" if name in ("gpu", "reconcile") else name


def _list(part: str | None) -> int:
    if not available():
        print(NO_ADAPTERS)
        return 0
    for t in available():
        if part is None or t.part == part:
            last = store.last_result(t.id)
            when = f"last {last.values} on {last.build}" if last else "unbaselined"
            print(f"{t.id:<28} {t.part:<7} {t.device:<6} ~{t.est_s:>5.0f} s  owner {t.owner:<5} {when}")
    return 0


def _preflight() -> str:
    """'' when the environment can run the suite, else why not (exit 2: not green, not a regression)."""
    from alloy.core.newton_compat import NEWTON_PIN, installed_newton_commit

    got = installed_newton_commit()
    return "" if got == NEWTON_PIN else f"installed Newton {got} is not the pin {NEWTON_PIN}"


def _run(args: argparse.Namespace) -> int:
    why = _preflight()
    unknown = sorted(set(args.only or ()) - {t.id for t in TASKS})
    if why or unknown:
        log.error("task regression: environment error, nothing run: %s", why or f"unknown task(s) {unknown}")
        return 2
    if not available([t for t in TASKS if t.part == args.part and (not args.only or t.id in args.only)]):
        print(f"task regression {args.part}: {NO_ADAPTERS}")
        return 0
    out = args.out or DEFAULT_OUT / f"{time.strftime('%Y%m%d-%H%M%S')}-{head_commit()}"
    report = run_part(args.part, out_dir=out, only=args.only or (), jobs=args.jobs, device=args.device)
    print(f"task regression {args.part}: green={report['green']} {report['counts']} -> {out}")
    for r in report["regressions"]:
        print(f"  REGRESSION {r['task']} (owner {r['owner']}): {r['verdict']} {r['detail']} {r['metrics']}")
    return 0 if report["green"] else 1


def _rebaseline(args: argparse.Namespace) -> int:
    """Re-take a task's last result; the run's result file and log stay in DEFAULT_OUT/rebaseline-<time>-<task>."""
    task = get(args.task)
    work = DEFAULT_OUT / f"rebaseline-{time.strftime('%Y%m%d-%H%M%S')}-{task.id}"
    metrics, wall, error = measure(task, work)
    if metrics is None:
        trace = work / f"{task.id}.json"
        detail = json.loads(trace.read_text()).get("trace", "") if trace.exists() else ""
        log.error("rebaseline %s: the run failed (%s); nothing recorded (%s)\n%s", task.id, error, work, detail)
        return 2
    build = head_commit()
    store.record(task.id, metrics, build=build, source=f"rebaseline on {build} ({wall:.0f} s, {task.device})",
                 reason=args.reason)
    print(f"{task.id}: recorded {metrics} on {build}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.bench.task_regression", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--part", choices=PARTS, type=_part)
    run = sub.add_parser("run")
    run.add_argument("--part", choices=PARTS, type=_part, required=True,
                     help="cpu, or nightly (aliases gpu, reconcile)")
    run.add_argument("--device", default=None, help="nightly tasks run here instead of their own device (cuda:1)")
    run.add_argument("--only", "--task", dest="only", action="append", help="this task only (repeatable)")
    run.add_argument("--jobs", type=int, default=1, help="tasks run at once, each in its own process")
    run.add_argument("--out", type=Path, default=None, help="report folder (default: a stamped one)")
    rb = sub.add_parser("rebaseline")
    rb.add_argument("--task", required=True)
    rb.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.cmd == "list":
        return _list(args.part)
    return _run(args) if args.cmd == "run" else _rebaseline(args)


if __name__ == "__main__":
    sys.exit(main())
