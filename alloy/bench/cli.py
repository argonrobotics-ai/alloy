# SPDX-License-Identifier: Apache-2.0
"""`python -m alloy.bench run|scoreboard` (SPEC §3.10 CLI).

run: discover laws (missing producers become `not_run`), run A16 first (a failure blocks every other law), run each
law on each applicable solver, write one report/1 file per result, then compare with the baseline (D23): exit 1 on
any regression. Compare mode never writes the committed baseline (the merge gate needs a clean tree); keys the
baseline lacks are listed as NEW with their verdict, and one that does not pass is a regression (exit 1: a law cannot
fail unseen for want of a key), and so is a key of the suite, device and mode that a whole-suite run (no `--law`,
no `--solver`) produced no result for. `--add-new-keys` appends new keys first, which records a known failure;
`--update-baseline REASON` rewrites every key of the run (recorded in the baseline history).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from alloy.bench import baseline as BL
from alloy.bench import law as L
from alloy.bench import rigs
from alloy.bench import scoreboard as SB
from alloy.bench.report import write_report
from alloy.bench.runner import run_law
from alloy.bench.solvers import BASELINE_SOLVERS
from alloy.bench.types import LawResult, LawSpec

log = logging.getLogger("alloy.bench")
SELFTEST = "A16.selftest"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.bench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a suite of laws")
    r.add_argument("--suite", choices=("analytic", "determinism", "rig"), default="analytic")
    r.add_argument("--law", action="append", default=[], help="law id or prefix (repeatable)")
    r.add_argument("--solver", action="append", default=[], help="solver name (repeatable); default: the law's")
    r.add_argument("--device", default="cpu")
    r.add_argument("--quick", action="store_true", help="<= 16 worlds and quick.duration_s per law")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", type=Path, default=Path("bench/results"))
    r.add_argument("--rigs", type=Path, default=Path("bench/rigs"))
    r.add_argument("--baseline", type=Path, default=None)
    r.add_argument("--update-baseline", metavar="REASON", default=None)
    r.add_argument("--add-new-keys", action="store_true", help="append keys the baseline lacks (never overwrites)")
    b = sub.add_parser("baseline", help="add report/1 results under ROOTS to the baseline (e.g. GPU runs)")
    b.add_argument("roots", nargs="+", type=Path)
    b.add_argument("--baseline", type=Path, default=BL.DEFAULT_PATH)
    b.add_argument("--update", metavar="REASON", default=None, help="also overwrite existing keys")
    s = sub.add_parser("scoreboard", help="build bench/SCOREBOARD.json from report/1 and run/1 files")
    s.add_argument("roots", nargs="*", type=Path, default=[Path("bench/results")])
    s.add_argument("--out", type=Path, default=Path("bench/SCOREBOARD.json"))
    s.add_argument("--md", action="store_true", help="also print a Markdown table")
    return p


def _selected(spec: LawSpec, wanted: Sequence[str]) -> bool:
    return not wanted or any(spec.id == w or L.law_prefix(spec.id) == w for w in wanted)


def _solvers_for(spec: LawSpec, wanted: Sequence[str]) -> list[str]:
    own = list(spec.solvers) or list(BASELINE_SOLVERS)
    return [s for s in own if s in wanted] if wanted else own


def _datasets(spec: LawSpec, root: Path) -> list:
    if spec.suite != "rig":
        return [None]
    return [rigs.load_rig(p) for p in rigs.datasets_for(spec.id, root)] or [None]


def _run_one(spec: LawSpec, factory, solver: str, args: argparse.Namespace, command: str) -> list[LawResult]:
    out = []
    for data in _datasets(spec, args.rigs):
        if spec.suite == "rig" and data is None:
            out.append(L.not_run_result(spec.id, solver, args.device, f"no dataset under {args.rigs}/{spec.id}"))
            continue
        res = run_law(spec, factory(), data, device=args.device, seed=args.seed, solver=solver, quick=args.quick)
        out.append(dataclasses.replace(res, command=command))
    return out


def _blocked(spec: LawSpec, solver: str, device: str, command: str) -> LawResult:
    res = L.not_run_result(spec.id, solver, device, "blocked: A16 comparator self-test failed")
    v = dataclasses.replace(res.verdicts[0], verdict="error")
    return dataclasses.replace(res, verdicts=(v,), command=command)


def run_suite(args: argparse.Namespace, command: str) -> list[LawResult]:
    """Run every selected law; returns all results (including not_run and blocked ones)."""
    L.discover()
    laws = [(s, f) for s, f in L.registered_laws(args.suite) if _selected(s, args.law)]
    gate_ok = True
    if args.suite == "analytic" or any(L.law_prefix(s.id) == "A16" for s, _ in laws):
        spec, factory = next((s, f) for s, f in L.registered_laws() if s.id == SELFTEST)
        gate = run_law(spec, factory(), None, device=args.device, solver="none", quick=args.quick)
        gate_ok = gate.verdicts[0].verdict == "pass"
        results = [dataclasses.replace(gate, command=command)]
    else:
        results = []
    for spec, factory in laws:
        if spec.id == SELFTEST:
            continue
        for solver in _solvers_for(spec, args.solver):
            if not gate_ok:
                results.append(_blocked(spec, solver, args.device, command))
                continue
            log.info("run %s on %s/%s", spec.id, solver, args.device)
            results += _run_one(spec, factory, solver, args, command)
    for entry, reason in L.missing_laws(args.suite):
        if _selected_prefix(entry.prefix, args.law):
            results.append(dataclasses.replace(L.not_run_result(entry.prefix, "any", args.device, reason),
                                               command=command))
    return results


def _selected_prefix(prefix: str, wanted: Sequence[str]) -> bool:
    return not wanted or any(L.law_prefix(w) == prefix for w in wanted)


def _summary(results: Sequence[LawResult]) -> str:
    lines = [f"{'law':28s} {'solver':14s} {'verdict':12s} worst metric"]
    for r in results:
        worst = max(r.verdicts, key=lambda v: BL.RANK[v.verdict])
        detail = f"{worst.name} error={worst.error:.4g} bound={worst.bound:.4g}"
        lines.append(f"{r.law:28s} {r.solver:14s} {BL.law_verdict(r):12s} {detail} ({r.wall_s:.1f} s)")
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    command = "python -m alloy.bench " + " ".join(shlex.quote(a) for a in args.argv)
    results = run_suite(args, command)
    for r in results:
        write_report(r, args.out)
    print(_summary(results))  # CLI output (library code logs instead)
    if args.baseline is None:
        return 0
    if args.update_baseline:
        keys = BL.update_baseline(results, args.baseline, reason=args.update_baseline, overwrite=True)
        print(f"baseline {args.baseline}: wrote {len(keys)} keys ({args.update_baseline})")
        return 0
    known = BL.load_baseline(args.baseline)["entries"]
    new = {BL.key_of(r): BL.law_verdict(r) for r in results if BL.key_of(r) not in known}
    if new and args.add_new_keys:       # before the compare: an added key records its verdict, known failures too
        BL.update_baseline(results, args.baseline, reason="new keys from run")
    regressions = BL.compare_to_baseline(results, args.baseline)
    if not args.law and not args.solver:        # the whole suite: a keyed law with no result is a regression too
        suites = {e.prefix: e.suite for e in L.CATALOGUE}
        regressions += BL.missing_from_run(results, args.baseline, device=args.device,
                                           mode="quick" if args.quick else "full",
                                           in_scope=lambda law: suites.get(L.law_prefix(law)) == args.suite)
    for key, verdict in new.items():
        print(f"NEW {key}{' (added)' if args.add_new_keys else ''}: {verdict}")
    for line in regressions:
        print(f"REGRESSION {line}")
    return 1 if regressions else 0


def cmd_baseline(args: argparse.Namespace) -> int:
    results = [r for _, r in SB.load_reports(args.roots)]
    keys = BL.update_baseline(results, args.baseline, reason=args.update or "keys from reports",
                              overwrite=bool(args.update))
    print(f"baseline {args.baseline}: wrote {len(keys)} keys from {len(results)} reports")
    return 0


def cmd_scoreboard(args: argparse.Namespace) -> int:
    rows = SB.scoreboard(args.roots)
    path = SB.write_scoreboard(rows, args.out)
    print(f"{path}: {len(rows)} rows, active pass rate {SB.pass_rate(rows):.3f} (not_run never passes)")
    if args.md:
        print(SB.render_markdown(rows))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import warp as wp

    wp.config.quiet = True
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stderr)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    args.argv = argv
    return {"run": cmd_run, "baseline": cmd_baseline, "scoreboard": cmd_scoreboard}[args.cmd](args)


__all__ = ["main", "run_suite"]
