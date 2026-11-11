# SPDX-License-Identifier: Apache-2.0
"""Run the task-regression suite (one part) and write its report.

Each task runs in its own subprocess (`alloy.bench.task_regression.one`) with a timeout of a few times its expected
time, so one crash or hang cannot take the suite down and every task's error is reported, never read as a pass. The
report lists each regression with its owner WP, so the post-merge runner can route it as ASK (WP1's hook), and says
whether the part is green: every baselined task passed (a CPU-green report is one of last-green's two conditions).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alloy.bench import gpu_repeat as GR
from alloy.bench.task_regression import store
from alloy.bench.task_regression.adapters import WORK_ENV
from alloy.bench.task_regression.registry import TASKS, available
from alloy.bench.task_regression.types import Part, TaskCheck, TaskOutcome, judge

log = logging.getLogger("alloy.bench.task_regression")

SCHEMA = "alloy.bench.task_regression.report/1"
TIMEOUT_FACTOR = 4.0
MIN_TIMEOUT_S = 300.0


def head_commit() -> str:
    """The build under test: ALLOY_COMMIT when set (a synced tree on a VM has no .git), else git's HEAD."""
    if os.environ.get("ALLOY_COMMIT"):
        return os.environ["ALLOY_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def measure(task: TaskCheck, work: Path) -> tuple[dict[str, float] | None, float, str]:
    """(metrics or None, wall seconds, error text) of one task, run in a subprocess on the task's device."""
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{task.id}.json"
    cmd = [sys.executable, "-m", "alloy.bench.task_regression.one", "--task", task.id, "--device", task.device,
           "--json", str(out)]
    timeout = max(MIN_TIMEOUT_S, TIMEOUT_FACTOR * task.est_s)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, WORK_ENV: str(work / task.id)})
    except subprocess.TimeoutExpired:
        return None, time.monotonic() - t0, f"timed out after {timeout:.0f} s"
    wall = time.monotonic() - t0
    (work / f"{task.id}.log").write_text(proc.stdout[-20000:] + proc.stderr[-20000:])
    if not out.exists():
        return None, wall, f"exit {proc.returncode}, no result file (log {task.id}.log)"
    payload = json.loads(out.read_text())
    if "error" in payload:
        return None, wall, payload["error"]
    return payload["metrics"], wall, ""


def run_one(task: TaskCheck, out_dir: Path) -> TaskOutcome:
    """Measure one task and judge it against its last recorded result."""
    log.info("task regression: %s (%s, ~%.0f s)", task.id, task.device, task.est_s)
    metrics, wall, error = measure(task, out_dir)
    if metrics is None:
        outcome = TaskOutcome(task.id, task.owner, task.part, "error", wall_s=wall, detail=error)
    else:
        outcome = judge(task, store.last_result(task.id), metrics)
        outcome.wall_s = wall
    log.info("task regression: %s %s (%.0f s) %s", task.id, outcome.verdict, wall, outcome.detail)
    return outcome


def run_part(part: Part, *, out_dir: Path, only: Sequence[str] = (), jobs: int = 1,
             device: str | None = None) -> dict[str, Any]:
    """Run every task of `part` (or the named ones), each in its own process, judge each against its last result,
    and write out_dir/report.json, rows.jsonl and report.md. `device` moves the GPU tasks to another GPU; CPU tasks
    stay on the CPU. Scheduling: see `run_tasks`."""
    unknown = set(only) - {t.id for t in TASKS}
    if unknown:
        raise KeyError(f"unknown task-regression task(s): {sorted(unknown)}")
    picked = [t for t in TASKS if t.part == part and (not only or t.id in only)]
    for t in sorted(set(picked) - set(available(picked)), key=lambda t: t.id):
        log.warning("task regression: %s skipped, its adapter %s is not installed", t.id, t.runner)
    tasks = [t if device is None or not _on_gpu(t) else dataclasses.replace(t, device=device)
             for t in available(picked)]
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes = run_tasks(tasks, out_dir, jobs)
    report = build_report(part, outcomes, {t.id: t.device for t in tasks})
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, allow_nan=True) + "\n")
    (out_dir / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in report["rows"]))
    (out_dir / "report.md").write_text(render(report))
    return report


def run_tasks(tasks: Sequence[TaskCheck], out_dir: Path, jobs: int = 1) -> list[TaskOutcome]:
    """The tasks on each GPU run one at a time (one lane per device: concurrent episode runs on one L4 slow each
    other past their timeouts), while the CPU tasks run `jobs` at a time beside them, so the nightly part's CPU
    tasks cost no extra wall time. Outcomes come back in the order of `tasks`."""
    lanes: dict[str, list[TaskCheck]] = {}
    for t in tasks:
        if _on_gpu(t):
            lanes.setdefault(t.device, []).append(t)
    cpu = [t for t in tasks if not _on_gpu(t)]
    done: dict[str, TaskOutcome] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(lanes))) as gpu_pool, \
            ThreadPoolExecutor(max_workers=max(1, jobs)) as cpu_pool:
        cpu_runs = {t.id: cpu_pool.submit(run_one, t, out_dir) for t in cpu}
        lane_runs = [(lane, gpu_pool.submit(lambda ts: [run_one(t, out_dir) for t in ts], lane))
                     for lane in lanes.values()]
        for lane, fut in lane_runs:
            done.update({t.id: o for t, o in zip(lane, fut.result())})
        done.update({tid: fut.result() for tid, fut in cpu_runs.items()})
    return [done[t.id] for t in tasks]


def _on_gpu(task: TaskCheck) -> bool:
    return task.device.startswith("cuda")


def build_report(part: Part, outcomes: Sequence[TaskOutcome], devices: dict[str, str] | None = None
                 ) -> dict[str, Any]:
    counts = {v: sum(o.verdict == v for o in outcomes) for v in ("pass", "fail", "error", "unbaselined")}
    judged = counts["pass"] + counts["fail"] + counts["error"]
    regressions = [{"task": o.task, "owner": o.owner, "verdict": o.verdict, "detail": o.detail,
                    "metrics": [asdict(m) for m in o.metrics if not m.ok]}
                   for o in outcomes if o.verdict in ("fail", "error")]
    return {"schema": SCHEMA, "part": part, "build": head_commit(), "rows": flat_rows(outcomes, devices or {}),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "green": counts["fail"] == 0 and counts["error"] == 0, "judged": judged,
            "pass_rate": counts["pass"] / judged if judged else None, "counts": counts,
            "regressions": regressions, "tasks": [asdict(o) for o in outcomes]}


def flat_rows(outcomes: Sequence[TaskOutcome], devices: dict[str, str]) -> list[dict[str, Any]]:
    """One row per (task, metric) for the post-merge hook (WP1) and the nightly (WP5): task, owner_wp, metric, value,
    last_value, tolerance (the bounds the value had to lie in), pass, seeds, build, device; a task that did not run
    gives one row with metric None."""
    build = head_commit()
    seeds = {t.id: list(t.seeds) for t in TASKS}
    routed = {t.id for t in TASKS if t.route}
    rows: list[dict[str, Any]] = []
    for o in outcomes:
        device = devices.get(o.task, "")
        base = {"task": o.task, "owner_wp": o.owner, "verdict": o.verdict, "seeds": seeds.get(o.task, []),
                "build": build, "device": device, "repeats": GR.repeats(device, route_requested=o.task in routed)}
        if not o.metrics:
            rows.append({**base, "metric": None, "value": None, "last_value": None, "tolerance": None,
                         "pass": o.verdict == "pass", "detail": o.detail})
        for m in o.metrics:
            rows.append({**base, "metric": m.name, "value": m.value, "last_value": m.last,
                         "tolerance": [_finite(m.low), _finite(m.high)], "pass": m.ok})
    return rows


def _finite(x: float) -> float | None:
    """An open end of a tolerance is None in the report (strict JSON has no infinity)."""
    return None if math.isinf(x) else x


def render(report: dict[str, Any]) -> str:
    """A short markdown summary of a report."""
    lines = [f"# Task regression, {report['part']} part ({report['build']}, {report['date']})", "",
             f"green: {report['green']}; counts {report['counts']}; pass rate {report['pass_rate']}", "",
             "| task | owner | verdict | metrics (value vs [low, high]) | wall s |", "|---|---|---|---|---|"]
    for t in report["tasks"]:
        ms = "; ".join(f"{m['name']} {m['value']} vs [{m['low']:.4g}, {m['high']:.4g}]" for m in t["metrics"])
        lines.append(f"| {t['task']} | {t['owner']} | {t['verdict']} | {ms or t['detail']} | {t['wall_s']:.0f} |")
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "build_report", "flat_rows", "head_commit", "measure", "render", "run_one", "run_part"]
