# SPDX-License-Identifier: Apache-2.0
"""Score finished run dirs: checker verdict plus wall time, turns, tokens, cost and tool errors.

A run dir is the kit's (sandbox/run_one.sh): meta.json, transcript.jsonl, claude_rc, rerun_rc, modules/ and
task/out. Transcript numbers come from the cold-agent kit (bench/agents) when it is installed, else they
are left empty; the checker is
bench/newenv/checkers/<name>.py named by the request: ``check(rollout, request) -> CheckResult``, run through
checkers/common.py (``run_checker``). Verdicts (PROTOCOL.md section 4): L0-L3 from the checker's level, "declined"
when the agent wrote supported:false and the engine's docs say so (``declines`` in engines.json), else that decline
is L0; "error" when the checker module is missing or crashes before any level (the row says which).

    python3 -m bench.newenv score RUNS_DIR [--out rows.jsonl] [--public]

Example:
    >>> verdict_of({"level": 2, "declined": False}, False)
    'L2'
    >>> verdict_of({"level": None, "declined": True}, True)
    'declined'
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

from bench.newenv.checkers import common
from bench.newenv.requests import EngineSpec, Request, is_correct_decline, load_engines, load_requests

log = logging.getLogger("bench.newenv")
PASS_LEVEL = 2          # "checked": the row's headline level
SOLVED_LEVEL = 3        # robot tasks only


class Row(TypedDict, total=False):
    id: str
    engine: str
    task: str            # the request id (kit column name, so the kit's helpers work on rows)
    group: str
    rep: str
    status: str          # the verdict: L0 | L1 | L2 | L3 | declined | error
    level: int | None
    declined: bool
    correct_decline: bool
    supported: bool      # False where the engine's docs say it cannot do this request
    solved: bool         # L3 and the engine was imported (robot tasks)
    failed: list[str]
    metrics: dict[str, Any]
    error: str | None
    success: bool
    agent_wall_s: float | None
    rerun_wall_s: float | None
    turns: int | None
    tokens_total: int
    output_tokens: int | None
    cost_usd: float | None
    tool_errors: int
    timed_out: bool
    engine_imported: bool
    claude_rc: int
    api_error_status: Any


def load_checker(name: str) -> Any:
    """The checker module ``bench.newenv.checkers.<name>``; ModuleNotFoundError when it is not written yet."""
    return importlib.import_module(f"bench.newenv.checkers.{name}")


def verdict_of(v: dict[str, Any], correct_decline: bool) -> str:
    if v.get("error") or v.get("missing"):
        return "error"
    if v.get("declined"):
        return "declined" if correct_decline else "L0"
    return f"L{int(v['level'])}"


def _checker_result(req: Request, out_dir: Path) -> dict[str, Any]:
    """common.run_checker's dict; a missing or unimportable checker is an `error` result, never an exception."""
    name = f"bench.newenv.checkers.{req.checker}"
    try:
        module = load_checker(req.checker)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            meta = common.Rollout.load(out_dir).meta
            if meta.get("supported") is False:      # a decline needs no checker
                return {"declined": True, "level": None, "reason": str(meta.get("reason", ""))[:300]}
            return {"missing": True, "error": f"checker missing: {name} does not exist yet", "level": None}
        return {"error": f"checker {req.checker} failed to import: {exc}", "level": None}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"checker {req.checker} failed to import: {type(exc).__name__}: {exc}", "level": None}
    return common.run_checker(module, out_dir, req.entry)


def run_checker(req: Request, out_dir: Path, engine: EngineSpec | None = None) -> dict[str, Any]:
    """The verdict dict: status, level, declined, correct_decline, supported, failed, checks, metrics, error."""
    correct = engine is not None and is_correct_decline(engine, req)
    try:
        v = _checker_result(req, out_dir)
    except Exception as exc:  # noqa: BLE001 (a bad run dir must not stop scoring the other runs)
        log.error("checker %s on %s failed: %s", req.checker, out_dir, exc)
        v = {"error": f"checker {req.checker}: {type(exc).__name__}: {exc}", "level": None}
    error = v.get("error") or (f"checker raised: {v['checker_error']}" if v.get("checker_error") else None)
    return {"status": verdict_of(v, correct), "level": v.get("level"), "declined": bool(v.get("declined")),
            "correct_decline": bool(v.get("declined")) and correct, "supported": not correct,
            "failed": list(v.get("failed", [])), "checks": v.get("checks", []),
            "metrics": dict(v.get("metrics", {})), "error": error}


def _read(p: Path, default: str = "") -> str:
    try:
        return p.read_text()
    except OSError:
        return default


def engine_imported(run: Path, module: str) -> bool:
    """Did the harness re-run import the engine's top-level module? (audit lines `top a b c`)"""
    for f in (run / "modules").glob("*.txt"):
        for line in _read(f).splitlines():
            if line.startswith("top ") and module in line.split()[1:]:
                return True
    return False


def run_meta(run: Path) -> dict[str, Any]:
    meta = json.loads(_read(run / "meta.json", "{}") or "{}")
    if "engine" not in meta:
        engine, task, rep = run.name.rsplit("-", 2)
        meta.update(engine=engine, task=task, rep=rep.lstrip("r"))
    return meta


def transcript_numbers(path: Path) -> dict[str, Any]:
    """Turns, tokens, cost and tool errors from an agent transcript, read by the kit's own scorer.

    The kit is not part of the public package, so without it (or without a transcript) the numbers are empty.
    """
    try:
        from bench.agents.ci.compare import total_tokens
        from bench.agents.score_runs import transcript_stats
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    st = transcript_stats(path)
    return {**st, "tokens_total": total_tokens(st)}


def score_run(run: Path, requests: dict[str, Request], modules: dict[str, str],
              engines: dict[str, EngineSpec] | None = None) -> Row:
    meta = run_meta(run)
    req = requests[meta["task"]]
    v = run_checker(req, run / "task" / "out", (engines or {}).get(meta["engine"]))
    st = transcript_numbers(run / "transcript.jsonl")
    claude_rc = int(_read(run / "claude_rc", "-1").strip() or -1)
    imported = engine_imported(run, modules.get(meta["engine"], meta["engine"]))
    return Row(id=run.name, engine=meta["engine"], task=req.id, group=req.group, rep=str(meta.get("rep", "")),
               status=v["status"], level=v["level"], declined=v["declined"],
               correct_decline=v["correct_decline"], supported=v["supported"], failed=v["failed"],
               metrics=v["metrics"], error=v["error"],
               success=(v["level"] or 0) >= PASS_LEVEL and not v["declined"] and imported,
               solved=(v["level"] or 0) >= SOLVED_LEVEL and not v["declined"] and imported, agent_wall_s=meta.get("agent_wall_s"),
               rerun_wall_s=meta.get("rerun_wall_s"), turns=st.get("turns"), tokens_total=st.get("tokens_total"),
               output_tokens=st.get("output_tokens"), cost_usd=st.get("cost_usd"),
               tool_errors=st.get("tool_errors", 0), timed_out=claude_rc in (124, 137),
               engine_imported=imported, claude_rc=claude_rc, api_error_status=st.get("api_error_status"))


def score_dir(runs_dir: Path, requests_path: Path | None = None, modules: dict[str, str] | None = None) -> list[Row]:
    engines = load_engines()
    reqs = {r.id: r for r in load_requests(requests_path)}
    mods = modules or {k: e.module for k, e in engines.items()}
    rows = []
    for d in sorted(p for p in runs_dir.glob("*") if (p / "DONE").is_file()):
        try:
            rows.append(score_run(d, reqs, mods, engines))
        except KeyError as exc:
            log.error("%s: unknown request or engine %s; skipped", d.name, exc)
    return rows


def restricted_notices(rows: list[Row], engines: dict[str, EngineSpec] | None = None) -> list[str]:
    """The notices of the arms in ``rows`` whose results may not be published (``notice`` in their engine entry)."""
    eng = load_engines() if engines is None else engines
    present = {r.get("engine") for r in rows}
    return [eng[k].notice for k in sorted(present) if k in eng and eng[k].notice]


def drop_restricted(rows: list[Row], engines: dict[str, EngineSpec] | None = None) -> list[Row]:
    eng = load_engines() if engines is None else engines
    return [r for r in rows if not (r.get("engine") in eng and eng[r["engine"]].notice)]


def write_rows(rows: list[Row], path: Path, public: bool = False) -> None:
    """JSON lines; starts with the notice of every restricted arm present (--public drops those arms instead)."""
    rows = drop_restricted(rows) if public else rows
    with path.open("w") as fh:
        fh.writelines(n + "\n" for n in restricted_notices(rows))
        fh.writelines(json.dumps(r, default=float) + "\n" for r in rows)


def read_rows(path: Path) -> list[Row]:
    """Rows from write_rows output (notice lines are skipped)."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


__all__ = ["Row", "drop_restricted", "read_rows", "restricted_notices", "run_checker", "score_dir", "score_run",
           "transcript_numbers", "write_rows"]
