# SPDX-License-Identifier: Apache-2.0
"""Row 3, physics against exact answers: the analytic laws, each compared with its closed-form reference.

Runs the analytic suite (`python -m alloy.bench run --suite analytic`) and writes one record per law, solver and
metric: the error against the exact answer, the law's tolerance as the bound, and the verdict. A law that did not
run is written as `not_run` with its reason and never counts as passed.

The headline counts Alloy's default route only (the solver Alloy picks when a scene overrides nothing,
`alloy.bench.default_route`), with the measured verdicts as they are: no ruling turns a fail into a pass here.
Results on the stock Newton solvers the suite also runs are written as engine `newton/<solver>`, for comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
from pathlib import Path

from alloy.bench import default_route as DR
from alloy.bench.scorecard.types import Headline, Row, RunOptions
from alloy.bench.types import LawResult, MetricVerdict

ROW = "3"
SELFTEST = "A16.selftest"                 # the comparator's own check: listed, not a physics law
TITLE = "Physics against exact answers"
COUNTED = ("pass", "fail", "inconclusive", "error")       # verdicts of laws that ran (not_run is listed apart)


def _finite(x: float | None) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _caveat(result: LawResult, v: MetricVerdict) -> str:
    notes = []
    if result.fit_dt_only:
        notes.append("run at the law's own step, not the production step")
    if str(result.provenance.get("quick", "False")) == "True":
        notes.append("quick mode (16 worlds or fewer, short duration)")
    if v.verdict == "not_run":
        notes.append(str(result.monitors.get("not_run_reason", "not run")))
    return "; ".join(notes)


def law_rows(result: LawResult) -> list[Row]:
    """One record per metric of one law result."""
    prov = result.provenance
    default = result.solver in ("any", "none") or DR.on_default_route(result.law, result.solver)
    engine = f"alloy/{result.solver}" if default else f"newton/{result.solver}"
    return [Row(row=ROW, engine=engine, task=result.law, metric=v.name, value=_finite(v.error),
                unit=v.unit, lo=0.0 if _finite(v.bound) is not None else None, hi=_finite(v.bound),
                bound_kind="tolerance" if _finite(v.bound) is not None else "", verdict=v.verdict,
                caveat=_caveat(result, v), device=result.device, command=result.command,
                commit=str(prov.get("alloy", "")), date=str(prov.get("date", "")), source="alloy.bench analytic suite",
                extra={"device_name": str(prov.get("device_name", "")), "wall_s": round(result.wall_s, 3)})
            for v in result.verdicts]


def _law_verdicts(rows: Sequence[Row]) -> dict[tuple[str, str], str]:
    """Worst verdict per (law, engine): a law passes only when every one of its metrics passes."""
    rank = {"pass": 0, "inconclusive": 1, "fail": 2, "not_run": 3, "error": 4}
    worst: dict[tuple[str, str], str] = {}
    for r in rows:
        key = (r.task, r.engine)
        if key not in worst or rank[r.verdict] > rank[worst[key]]:
            worst[key] = r.verdict
    return worst


def headline(rows: Sequence[Row], command: str) -> Headline:
    verdicts = _law_verdicts([r for r in rows if r.engine.startswith("alloy/") and r.task != SELFTEST])
    ran = [v for v in verdicts.values() if v in COUNTED]
    passed = sum(v == "pass" for v in ran)
    skipped = sum(v == "not_run" for v in verdicts.values())
    if not ran:
        return Headline(row=ROW, title=TITLE, number="no law ran", command=command, state="not_measured")
    quick = any("quick mode" in r.caveat for r in rows)
    caveat = f"{skipped} laws did not run and are not counted" if skipped else ""
    if quick:
        caveat = "; ".join(filter(None, [caveat, "quick mode: not the launch number"]))
    first = rows[0]
    return Headline(row=ROW, title=TITLE, number=f"{passed} of {len(ran)} laws within tolerance on Alloy's default route",
                    bound="a fixed list of laws, not a sample: no interval; one run per law", command=command, date=first.date,
                    commit=first.commit, caveat=caveat, state="draft" if quick else "measured")


def _args(opts: RunOptions, laws: Sequence[str]) -> argparse.Namespace:
    return argparse.Namespace(suite="analytic", law=list(laws), solver=[], device=opts.device, quick=opts.quick,
                              seed=opts.seed, rigs=Path("bench/rigs"))


def _run_one_law(law: str, opts: RunOptions) -> list[Row]:
    """One law in its own process; its records are kept in `<out>/partial/` so a stopped run resumes."""
    from alloy.bench.cli import run_suite  # noqa: PLC0415 - heavy import, in the worker only

    partial = opts.out / "partial" / f"{law}.jsonl"
    if partial.exists():
        return [Row(**json.loads(line)) for line in partial.read_text().splitlines() if line.strip()]
    rows = [row for result in run_suite(_args(opts, [law]), opts.command) for row in law_rows(result)]
    partial.parent.mkdir(parents=True, exist_ok=True)
    tmp = partial.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in rows))
    tmp.replace(partial)
    return rows


def _run_parallel(opts: RunOptions) -> list[Row]:
    """Each law in its own process, `opts.jobs` at a time. The records equal the serial run's: same laws, same
    settings; the comparator's self-test, which every process repeats, is kept once."""
    from alloy.bench import law as L  # noqa: PLC0415

    L.discover()
    laws = [spec.id for spec, _ in L.registered_laws("analytic") if spec.id != SELFTEST]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=opts.jobs, mp_context=ctx) as pool:
        per_law = list(pool.map(_run_one_law, laws, [opts] * len(laws)))
    rows: list[Row] = []
    seen: set[tuple[str, str, str]] = set()
    for row in [r for chunk in per_law for r in chunk] + _missing_rows(opts):
        key = (row.task, row.engine, row.metric)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def _missing_rows(opts: RunOptions) -> list[Row]:
    """Laws the suite lists but has no code for: written as not_run, as the serial run writes them."""
    from alloy.bench import law as L  # noqa: PLC0415

    return [row for entry, reason in L.missing_laws("analytic")
            for row in law_rows(dataclasses.replace(L.not_run_result(entry.prefix, "any", opts.device, reason),
                                                    command=opts.command))]


def run(opts: RunOptions) -> tuple[list[Row], Headline]:
    if opts.jobs > 1:
        rows = _run_parallel(opts)
    else:
        from alloy.bench.cli import run_suite  # noqa: PLC0415 - imports warp and every law: only when the row runs

        rows = [row for result in run_suite(_args(opts, []), opts.command) for row in law_rows(result)]
    return rows, headline(rows, opts.command)


__all__ = ["ROW", "TITLE", "headline", "law_rows", "run"]
