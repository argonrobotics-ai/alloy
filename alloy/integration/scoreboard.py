# SPDX-License-Identifier: Apache-2.0
"""The full scoreboard in one command (SPEC §5, §7 WP12): `bench/SCOREBOARD.json` and `bench/SCOREBOARD.md`.

    python -m alloy.integration.scoreboard [--run-analytic] [--roots DIR ...]

1. optionally runs the analytic suite (``python -m alloy.bench run --suite analytic --device cpu --quick``) into
   ``bench/results/analytic`` (feeds SB-12 and the ``law:`` rows);
2. turns every package's ``docs/notes/wp*-results.md`` §5 rows and WP12's JSON lines into report/1 files
   (``alloy.integration.results.collect``) under ``bench/results/collected``;
3. runs WP6's scoreboard over those roots plus ``--roots`` (``alloy.bench.scoreboard``), which values each §5
   metric from the results that feed it and gives every not_run row its reason.

Equivalent by hand: ``python -m alloy.integration.results`` then ``python -m alloy.bench scoreboard bench/results``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from alloy.bench.scoreboard import pass_rate, render_markdown, scoreboard, write_scoreboard
from alloy.integration.results import collect

log = logging.getLogger("alloy.integration")

__all__ = ["build", "main", "summary"]


def run_analytic(out: Path, quick: bool = True) -> int:
    """The analytic suite on CPU into ``out`` (report/1 files); ``quick`` is the CI subset (the full suite with the
    SPH laws takes > 20 min on a 4-thread CPU). A failing law still writes its report, so a non-zero exit is logged,
    not raised."""
    cmd = [sys.executable, "-m", "alloy.bench", "run", "--suite", "analytic", "--device", "cpu", "--out", str(out)]
    cmd += ["--quick"] if quick else []
    log.info("running %s", " ".join(cmd))
    code = subprocess.run(cmd, check=False).returncode
    if code:
        log.warning("analytic suite exited %d (reports written for the laws that ran)", code)
    return code


def build(notes: Path, jsonl: Sequence[Path], roots: Sequence[Path], out: Path, collected: Path) -> list:
    """Collect (``collected`` is regenerated), score, write ``out`` and ``out`` with suffix .md; returns the rows."""
    if collected.is_dir():
        shutil.rmtree(collected)
    collect(notes, [p for p in jsonl if p.exists()], collected, time.strftime("%Y-%m-%d"))
    rows = scoreboard([collected, *[r for r in roots if r.exists()]])
    write_scoreboard(rows, out)
    out.with_suffix(".md").write_text(render_markdown(rows) + "\n")
    return rows


def summary(rows: Sequence) -> str:
    """One line: active catalogue verdict counts and the pass rate (not_run never passes)."""
    active = [r for r in rows if r.status == "active" and ":" not in r.metric_id]
    counts = Counter(r.verdict for r in active)
    no_reason = [r.metric_id for r in active if r.verdict == "not_run" and not getattr(r, "reason", "")]
    return (f"{len(active)} active metrics: {dict(sorted(counts.items()))}; pass rate {pass_rate(rows):.3f}; "
            f"not_run without a reason: {no_reason or 'none'}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.integration.scoreboard", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-analytic", action="store_true", help="run the quick analytic suite first (CPU)")
    ap.add_argument("--full-analytic", action="store_true", help="with --run-analytic: the full suite (> 20 min)")
    ap.add_argument("--notes", type=Path, default=Path("docs/notes"))
    ap.add_argument("--jsonl", type=Path, nargs="*", default=[Path("bench/results/wp12/results.jsonl")])
    ap.add_argument("--roots", type=Path, nargs="*", default=[Path("bench/results/analytic")])
    ap.add_argument("--collected", type=Path, default=Path("bench/results/collected"))
    ap.add_argument("--out", type=Path, default=Path("bench/SCOREBOARD.json"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    if args.run_analytic:
        run_analytic(Path("bench/results/analytic"), quick=not args.full_analytic)
    rows = build(args.notes, args.jsonl, args.roots, args.out, args.collected)
    log.info("wrote %s and %s: %s", args.out, args.out.with_suffix(".md"), summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
