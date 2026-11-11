# SPDX-License-Identifier: Apache-2.0
"""Command line (see README.md).

Public: ``task`` writes a request's task directory (drive.py, request.json, script.py, the prompt block); ``check``
scores the out/ directory that ``python drive.py`` wrote there; ``score`` / ``summarize`` turn run directories into
rows and tables. Cloud harness (needs the cold-agent kit in bench/agents): plan | prompt | overlay | launch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from bench.newenv import ENGINES
from bench.newenv.requests import ConfigError, load_engines, load_requests, unfinished_engines

log = logging.getLogger("bench.newenv")


def _csv(text: str) -> list[str]:
    return [t for t in text.split(",") if t]


def _labels(text: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in text.split(",") if "=" in p)


def _add_plan_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--engines", type=_csv, default=list(ENGINES))
    ap.add_argument("--requests", type=_csv, default=None, help="request ids (default: all)")
    ap.add_argument("--reps", type=int, default=3)


def _request(rid: str):
    reqs = {r.id: r for r in load_requests()}
    if rid not in reqs:
        raise ConfigError(f"unknown request {rid!r}; known: {', '.join(sorted(reqs))}")
    return reqs[rid]


def cmd_task(a: argparse.Namespace) -> int:
    from bench.newenv.make_task import prompt_block, write_task_dir
    req = _request(a.request)
    a.dest.mkdir(parents=True, exist_ok=True)
    write_task_dir(req, a.dest)
    (a.dest / "REQUEST.md").write_text(prompt_block(req))
    print(a.dest)
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    from bench.newenv.score import run_checker
    v = run_checker(_request(a.request), a.out, load_engines().get(a.engine) if a.engine else None)
    print(json.dumps(v, indent=1, default=float))
    return 0 if v.get("level") is not None else 1


def cmd_plan(a: argparse.Namespace) -> int:
    from bench.newenv.plan import make_plan
    runs = make_plan(a.engines, a.requests, a.reps)
    print(json.dumps({"runs": [r.id for r in runs], "unfinished_engines": unfinished_engines(a.engines)}, indent=1))
    return 0


def cmd_prompt(a: argparse.Namespace) -> int:
    import tempfile

    from bench.newenv.harness import build_overlay
    from bench.newenv.prompt import build
    with tempfile.TemporaryDirectory() as tmp:
        sys.stdout.write(build(a.engine, a.request, build_overlay(Path(tmp) / "overlay")))
    return 0


def cmd_overlay(a: argparse.Namespace) -> int:
    from bench.newenv.harness import build_overlay
    print(build_overlay(a.dest))
    return 0


def missing_request_files(ids: set[str]) -> dict[str, list[str]]:
    """Request id -> script / checker files still missing (empty dict = every selected request is runnable)."""
    from bench.newenv.make_task import missing_files
    from bench.newenv.requests import load_requests
    gaps = {r.id: missing_files(r) for r in load_requests() if r.id in ids}
    return {k: v for k, v in gaps.items() if v}


def cmd_launch(a: argparse.Namespace) -> int:
    from bench.newenv.launch import LaunchOptions, run_round
    from bench.newenv.plan import make_plan
    bad = unfinished_engines(a.engines)
    if bad:
        log.error("engines with TODO fields, fix bench/newenv/engines.json first: %s", bad)
        return 2
    runs = [r for r in make_plan(a.engines, a.requests, a.reps) if r.id not in a.skip]
    unfinished = missing_request_files({r.task for r in runs})
    if unfinished:
        log.error("requests without their script / checker file (group agents have not delivered): %s", unfinished)
        return 2
    rid = time.strftime("%Y%m%dT%H%MZ", time.gmtime()) + "-newenv"
    local = a.work / "newenv" / rid
    local.mkdir(parents=True, exist_ok=True)
    (local / "plan.json").write_text(json.dumps({"runs": [r.id for r in runs]}, indent=1))
    if a.plan_only:
        print(local)
        return 0
    zones = {"zones": a.zones} if a.zones else {}
    opts = LaunchOptions(a.token_file, a.work, a.repo, vm_prefix=a.vm_prefix, labels=a.labels, n_vms=a.n_vms,
                         model=a.model, agent_bin=a.agent_bin, snapshot=a.snapshot, machine=a.machine,
                         keep_vm=a.keep_vm, **zones)
    run_round(runs, opts, local)
    return cmd_score(argparse.Namespace(runs=local / "runs", out=local / "rows.jsonl", public=False,
                                        summary=local / "summary.md"))


def cmd_score(a: argparse.Namespace) -> int:
    from bench.newenv.score import score_dir, write_rows
    from bench.newenv.summarize import render
    rows = score_dir(a.runs)
    write_rows(rows, a.out, public=a.public)
    text = render(rows, public=a.public)
    if getattr(a, "summary", None):
        a.summary.write_text(text)
    sys.stdout.write(text)
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    from bench.newenv.score import read_rows
    from bench.newenv.summarize import render
    text = render(read_rows(a.rows), public=a.public)
    if a.out:
        a.out.write_text(text)
    sys.stdout.write(text)
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python3 -m bench.newenv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("task", help="write a request's task directory")
    p.add_argument("request"); p.add_argument("dest", type=Path); p.set_defaults(fn=cmd_task)
    p = sub.add_parser("check", help="score the out/ dir drive.py wrote for a request")
    p.add_argument("request"); p.add_argument("out", type=Path)
    p.add_argument("--engine", default="", help="engine key, for the correct-decline rule")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("plan"); _add_plan_args(p); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("prompt"); p.add_argument("engine"); p.add_argument("request")
    p.set_defaults(fn=cmd_prompt)
    p = sub.add_parser("overlay"); p.add_argument("dest", type=Path); p.set_defaults(fn=cmd_overlay)
    p = sub.add_parser("launch"); _add_plan_args(p)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path(os.environ.get("NEWENV_WORK", "newenv-work")))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--vm-prefix", default=os.environ.get("NEWENV_VM_PREFIX", "alloy-newenv"))
    p.add_argument("--labels", type=_labels, default=_labels(os.environ.get("NEWENV_LABELS", "purpose=newenv")))
    p.add_argument("--n-vms", type=int, default=1); p.add_argument("--model", default="sonnet")
    p.add_argument("--agent-bin", type=Path); p.add_argument("--zones", type=_csv, default=None)
    p.add_argument("--skip", type=_csv, default=[], help="run ids already home (e.g. alloy-E01-r1)")
    p.add_argument("--snapshot", default=os.environ.get("NEWENV_SNAPSHOT", ""),
                   help="disk snapshot holding the pre-built arm images (new VMs boot from it)")
    p.add_argument("--machine", default="g2-standard-8")
    p.add_argument("--keep-vm", action="store_true"); p.add_argument("--plan-only", action="store_true")
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("score"); p.add_argument("runs", type=Path); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary", type=Path); p.add_argument("--public", action="store_true"); p.set_defaults(fn=cmd_score)
    p = sub.add_parser("summarize"); p.add_argument("rows", type=Path); p.add_argument("--out", type=Path)
    p.add_argument("--public", action="store_true"); p.set_defaults(fn=cmd_summarize)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    a = parser().parse_args(argv)
    try:
        return a.fn(a)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
