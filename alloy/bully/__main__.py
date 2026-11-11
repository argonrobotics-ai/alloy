# SPDX-License-Identifier: Apache-2.0
"""Bully CLI: ``python -m alloy.bully {run,b0,replay,try,sb7,targets} ...`` (SPEC §5.2 BUL-1, BUL-2, SB-7).

Each command writes one JSON object to stdout (the record a results row cites) and logs progress to stderr.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import ctypes
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Any, TextIO

import numpy as np
import warp as wp

from alloy.bully.corpus import Corpus
from alloy.bully.rate import blowup_rate_from_severity
from alloy.bully.replay import EpisodeRunner
from alloy.bully.search import search_detailed
from alloy.bully.severity import MONITORS, classify, world_cause
from alloy.bully.stress import named_stress_phases, random_legal_traces, tile_phases
from alloy.bully.targets import get_target, target_names
from alloy.bully.types import BullyLimits, EpisodeResult, SearchConfig
from alloy.core.errors import BullyError
from alloy.precision.types import HealthFlags

log = logging.getLogger("alloy.bully")


_RECORD_STREAM: TextIO = sys.stdout  # the original stdout while main() runs; everything else goes to stderr


def _emit(obj: dict[str, Any]) -> None:
    _RECORD_STREAM.write(json.dumps(obj, default=_jsonable, sort_keys=True) + "\n")
    _RECORD_STREAM.flush()


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    return str(x)


def cmd_run(args: argparse.Namespace) -> int:
    target = get_target(args.target)
    cfg = SearchConfig(worlds=args.worlds, generations=args.generations, knots=args.knots, horizon_s=args.horizon,
                       sigma0=args.sigma0, budget_gpu_h=args.budget_h, seed=args.seed)
    out = search_detailed(target, cfg, BullyLimits(), args.device, seed_from_b0=not args.no_b0,
                          replay_runs=args.replay_runs)
    corpus = Corpus(Path(args.corpus))
    filed = []
    for f in out.findings:
        trace = out.traces[f.id]
        path = corpus.add(f, trace, dt=1.0 / target.actions.rate_hz, seed=args.seed, target=target.name)
        filed.append(path.name)
    _emit({"command": " ".join(sys.argv), "target": target.name, "device": args.device, "worlds": args.worlds,
           "generations_run": out.generations_run, "wall_s": out.wall_s, "best_badness": out.best_badness,
           "first_trip_generation": out.first_trip_generation, "replays": out.replays, "filed": filed,
           "findings": [dataclasses.asdict(f) for f in out.findings]})
    return 0


def cmd_b0(args: argparse.Namespace) -> int:
    target = get_target(args.target)
    named = named_stress_phases(target, args.horizon)
    runner = EpisodeRunner(target, len(named), args.device, BullyLimits())
    res = runner.run(tile_phases([t for _, t in named], len(named)))
    _emit({"target": target.name, "device": args.device, "phases": [n for n, _ in named],
           "severity": res.severity, "badness": res.badness, "monitors": res.monitors})
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    corpus = Corpus(Path(args.corpus))
    res = corpus.replay(args.finding, args.device, args.runs, workdir=Path(args.workdir))
    if args.close and res.trips == 0:
        corpus.set_status(args.finding, "closed")
    _emit({"replay": dataclasses.asdict(res), "status": corpus.record(args.finding)["status"]})
    return 0 if (res.counts or args.close) else 1


def cmd_try(args: argparse.Namespace) -> int:
    target = get_target(args.target)
    with np.load(args.trace) as z:
        actions = np.asarray(z["actions"])
    res = EpisodeRunner(target, 1, args.device, BullyLimits()).run(actions[None])
    _emit({"target": target.name, "severity": res.severity, "first_step": res.first_step,
           "badness": res.badness, "monitors": res.monitors})
    return 0


def _d12_severity(result: EpisodeResult, limits: BullyLimits) -> np.ndarray:
    """[W] severity under D12 before SCR-WP7-7: ``impossible_speed`` = any free-body speed without robot contact."""
    mon = dict(result.monitors)
    mon["impossible_speed"] = mon.get("impossible_speed_d12", mon.get("impossible_speed"))
    return classify(result.health, mon, limits)


def _s1_entries(result: EpisodeResult, d12: np.ndarray, seed: int, keep: int, limits: BullyLimits
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """S1 episodes as ``{seed, world, cause, value, s1_d12}``, and those S1 only under the old D12 row."""
    s1, d12_only = [], []
    for w in np.nonzero(result.severity[:keep] == 1)[0]:
        name, value, _, _ = world_cause(result.health, result.monitors, limits, int(w))
        s1.append({"seed": seed, "world": int(w), "cause": name, "value": float(value), "s1_d12": bool(d12[w] == 1)})
    speed = result.monitors.get("impossible_speed_d12", np.full(d12.shape, np.nan))
    for w in np.nonzero((d12[:keep] == 1) & (result.severity[:keep] != 1))[0]:
        d12_only.append({"seed": seed, "world": int(w), "impossible_speed_d12": float(speed[w])})
    return s1, d12_only


def _cause_or_unattributed(result: EpisodeResult, limits: BullyLimits, world: int) -> str:
    try:
        return world_cause(result.health, result.monitors, limits, world)[0]
    except BullyError:  # an S2 set by no listed row: counted, not fatal to a campaign piece
        return "unattributed"


def _s2_counts(result: EpisodeResult, keep: int, limits: BullyLimits) -> tuple[collections.Counter[str], int]:
    """S2 episodes by the row that set their severity, and how many of them tripped the max_pen row (a contact past
    ``limits.pen_limit``): the penetrations the thin-part rule keeps below S1 stay visible beside the rate."""
    s2 = np.nonzero(result.severity[:keep] == 2)[0]
    causes = collections.Counter(_cause_or_unattributed(result, limits, int(w)) for w in s2)
    pen = result.monitors.get("max_pen")
    over = 0 if pen is None else int(np.sum(np.asarray(pen)[s2] > MONITORS["max_pen"].threshold(limits)))
    return causes, over


def _fallback_entries(result: EpisodeResult, seed: int, keep: int) -> list[dict[str, Any]]:
    """Episodes a solver guard rescued (HealthFlags.FALLBACK_USED, e.g. the BUL-0016 pivot clamp) as ``{seed, world,
    severity}``: DEGRADED by health, so not S1 unless a bench monitor tripped as well."""
    flags = np.asarray(result.health.flags)[:keep].astype(np.int64)
    return [{"seed": seed, "world": int(w), "severity": int(result.severity[w])}
            for w in np.nonzero(flags & int(HealthFlags.FALLBACK_USED))[0]]


def cmd_sb7(args: argparse.Namespace) -> int:
    """SB-7 rate (§4.3 as amended by SCR-WP7-7), S1 causes, and every S1 episode as ``{seed, world, cause, value,
    s1_d12}`` (``fallback_worlds``: those a solver guard rescued; ``s2_causes`` and ``s2_max_pen_episodes``: S2
    episodes by setting row, and those past the max_pen line): its trace is ``random_legal_traces(target, worlds,
    horizon, seed)[world]``, so it can be replayed or filed without a rerun. ``rate_d12`` counts the same episodes
    under D12 before SCR-WP7-7 (free fall trips)."""
    from alloy.bench.sb7 import population_line  # noqa: PLC0415 - bench imports the bully's rows

    target = get_target(args.target)
    limits = BullyLimits()
    runner = EpisodeRunner(target, args.worlds, args.device, limits)
    severities, old, causes, s1_worlds, d12_only, fallback = [], [], collections.Counter(), [], [], []
    s2_causes, s2_max_pen = collections.Counter(), 0
    for batch in range(int(np.ceil(args.episodes / args.worlds))):
        seed = args.seed + batch
        result = runner.run(random_legal_traces(target, args.worlds, args.horizon, seed=seed))
        keep = min(args.worlds, args.episodes - batch * args.worlds)
        d12 = _d12_severity(result, limits)
        severities.append(result.severity[:keep])
        old.append(d12[:keep])
        s1, only = _s1_entries(result, d12, seed, keep, limits)
        causes.update(e["cause"] for e in s1)
        s1_worlds += s1
        d12_only += only
        fallback += _fallback_entries(result, seed, keep)
        c2, over = _s2_counts(result, keep, limits)
        s2_causes.update(c2)
        s2_max_pen += over
    sev, sev_d12 = np.concatenate(severities), np.concatenate(old)
    rate, rate_d12 = blowup_rate_from_severity(sev), blowup_rate_from_severity(sev_d12)
    _emit({"command": " ".join(sys.argv), "metric": "SB-7", "target": target.name, "device": args.device,
           "rate": dataclasses.asdict(rate), "passes": rate.passes, "s1_causes": dict(causes),
           "s1_worlds": s1_worlds, "s2_episodes": int(np.sum(sev == 2)), "s2_causes": dict(s2_causes),
           "s2_max_pen_episodes": s2_max_pen, "fallback_worlds": fallback,
           "rate_d12": dataclasses.asdict(rate_d12), "passes_d12": rate_d12.passes, "d12_only_worlds": d12_only,
           "population": population_line()})          # audit AB-laws-7: the count beside any published rate
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """``alloy.bully.audit`` on any scene card: the report as JSON on stdout and in ``--out``; exit 1 with an S1."""
    from alloy.bully.scene_audit import audit  # noqa: PLC0415
    from alloy.bully.scene_target import AuditOptions  # noqa: PLC0415

    opts = AuditOptions(control_dt=args.control_dt, channel=args.channel, push_g=args.push_g)
    report = audit(args.scene, episodes=args.episodes, seed=args.seed, device=args.device, horizon_s=args.horizon,
                   batch=args.batch, options=opts, attribute=args.attribute)
    if args.out:
        Path(args.out).write_text(report.to_json() + "\n")
    _emit(report.to_dict())
    return 0 if report.ok else 1


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.bully")
    p.add_argument("--log", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="B0 + B1 search; file replay-confirmed findings")
    run.add_argument("--target", required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--worlds", type=int, default=64)
    run.add_argument("--generations", type=int, default=20)
    run.add_argument("--knots", type=int, default=8)
    run.add_argument("--horizon", type=float, default=2.0)
    run.add_argument("--sigma0", type=float, default=0.3)
    run.add_argument("--budget-h", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--replay-runs", type=int, default=10)
    run.add_argument("--no-b0", action="store_true")
    run.add_argument("--corpus", default="bench/bully/corpus")
    b0 = sub.add_parser("b0", help="run the B0 stress phases, one per world")
    b0.add_argument("--target", required=True)
    b0.add_argument("--device", default="cpu")
    b0.add_argument("--horizon", type=float, default=2.0)
    rep = sub.add_parser("replay", help="replay a corpus finding")
    rep.add_argument("finding")
    rep.add_argument("--device", default="cpu")
    rep.add_argument("--runs", type=int, default=10)
    rep.add_argument("--close", action="store_true", help="mark closed if the replay no longer trips")
    rep.add_argument("--corpus", default="bench/bully/corpus")
    rep.add_argument("--workdir", default=".", help="command findings: the checkout the program runs from")
    tr = sub.add_parser("try", help="run one trace file (B2 submissions)")
    tr.add_argument("--target", required=True)
    tr.add_argument("--trace", required=True)
    tr.add_argument("--device", default="cpu")
    sb = sub.add_parser("sb7", help="SB-7 random-legal rollouts")
    sb.add_argument("--target", required=True)
    sb.add_argument("--device", default="cpu")
    sb.add_argument("--episodes", type=int, default=1024)
    sb.add_argument("--worlds", type=int, default=256)
    sb.add_argument("--horizon", type=float, default=2.0)
    sb.add_argument("--seed", type=int, default=0)
    sub.add_parser("targets", help="list registered targets")
    au = sub.add_parser("audit", help="random-legal episodes on any scene card, as findings (alloy.bully.audit)")
    au.add_argument("scene", help="scene YAML path, library scene name or YAML text")
    au.add_argument("--episodes", type=int, default=256)
    au.add_argument("--seed", type=int, default=0)
    au.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
    au.add_argument("--horizon", type=float, default=2.0)
    au.add_argument("--batch", type=int, default=None, help="worlds per batch (default 1024 GPU, 8 CPU)")
    au.add_argument("--channel", default="auto", choices=("auto", "ctrl", "push", "passive"))
    au.add_argument("--push-g", type=float, default=0.2)
    au.add_argument("--control-dt", type=float, default=0.02)
    au.add_argument("--attribute", type=int, default=3, help="findings per (class, cause) given a body")
    au.add_argument("--out", default=None, help="also write the JSON report here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    wp.config.log_level = wp.LOG_WARNING  # no module-load lines in the log
    handlers = {"run": cmd_run, "b0": cmd_b0, "replay": cmd_replay, "try": cmd_try, "sb7": cmd_sb7, "audit": cmd_audit}
    global _RECORD_STREAM
    with _stdout_to_stderr() as record:
        _RECORD_STREAM = record
        if args.cmd == "targets":
            _emit({"targets": target_names()})
            return 0
        return handlers[args.cmd](args)


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[TextIO]:
    """Point file descriptor 1 and ``sys.stdout`` at stderr while a command runs, so prints from Python and from
    kernels (MuJoCo Warp's ``nefc overflow`` warnings are kernel printf) reach the log; yields a stream on the original
    stdout for the JSON record. C stdio and Warp's device printf buffers are flushed before fd 1 is restored."""
    sys.stdout.flush()
    saved = os.dup(1)
    record = os.fdopen(os.dup(saved), "w")
    os.dup2(2, 1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield record
    finally:
        _flush_native()
        record.close()
        os.dup2(saved, 1)
        os.close(saved)


def _flush_native() -> None:
    try:
        wp.synchronize()
    except RuntimeError:  # Warp never initialised in this process
        pass
    ctypes.CDLL(None).fflush(None)


if __name__ == "__main__":
    raise SystemExit(main())
