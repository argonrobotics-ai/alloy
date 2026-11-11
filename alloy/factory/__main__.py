# SPDX-License-Identifier: Apache-2.0
"""Factory CLI: ``python -m alloy.factory {evaluate,loop} ...`` (FAC-1, FAC-2).

``evaluate`` is what ``factory_loop`` runs in a subprocess on the scratch worktree; ``loop`` drives a whole
create-evaluate run with a shell agent command, or with ``--copy-template`` (the scripted fake runner).
Each command writes one JSON object to stdout and logs to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from alloy.bully.types import SearchConfig
from alloy.factory.evaluate import evaluate_plugin, load_manifest, verdict_to_json
from alloy.factory.loop import LoopConfig, factory_loop
from alloy.factory.runner import ShellAgentRunner


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = SearchConfig(worlds=args.worlds, generations=args.generations, horizon_s=args.horizon, knots=args.knots,
                       sigma0=args.sigma0, budget_gpu_h=args.budget_h, seed=args.seed)
    verdict = verdict_to_json(evaluate_plugin(load_manifest(Path(args.manifest)), args.device, cfg=cfg,
                                              replay_runs=args.replay_runs))
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True))
    sys.stdout.write(json.dumps(verdict, sort_keys=True) + "\n")
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    if args.copy_template:
        from alloy.testing.factory import CopyTemplateRunner

        runner = CopyTemplateRunner()
    else:
        runner = ShellAgentRunner(args.runner_cmd)
    attack = SearchConfig(worlds=args.worlds, generations=args.generations, horizon_s=args.horizon, knots=args.knots,
                          sigma0=args.sigma0, budget_gpu_h=args.budget_h, seed=args.seed)
    cfg = LoopConfig(device=args.device, attack=attack, replay_runs=args.replay_runs,
                     export_to=Path(args.export) if args.export else None)
    verdicts = factory_loop(Path(args.brief), runner, args.name, args.rounds, cfg=cfg)
    sys.stdout.write(json.dumps({"command": " ".join(sys.argv), "name": args.name,
                                 "rounds": [verdict_to_json(v) for v in verdicts]}, sort_keys=True) + "\n")
    return 0 if verdicts and verdicts[-1].ship else 1


def _attack_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cpu")
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--replay-runs", type=int, default=10)
    p.add_argument("--knots", type=int, default=8)
    p.add_argument("--sigma0", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-h", type=float, default=0.05)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m alloy.factory")
    p.add_argument("--log", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="evaluate one plug-in package from its manifest")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--out", default="")
    _attack_args(ev)
    lp = sub.add_parser("loop", help="create-evaluate rounds with an agent runner")
    lp.add_argument("--brief", required=True)
    lp.add_argument("--name", required=True)
    lp.add_argument("--rounds", type=int, default=3)
    lp.add_argument("--runner-cmd", default="")
    lp.add_argument("--copy-template", action="store_true", help="scripted fake runner (copies the template)")
    lp.add_argument("--export", default="", help="copy a shipped package here")
    _attack_args(lp)
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.cmd == "loop" and not (args.copy_template or args.runner_cmd):
        p.error("loop needs --runner-cmd or --copy-template")
    return cmd_evaluate(args) if args.cmd == "evaluate" else cmd_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
