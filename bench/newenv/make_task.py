#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build one request's task directory and its request-specific prompt block.

The task directory is what a cold agent finds beside TASK.md in ``~/task`` (the harness also restores it before the
scored re-run): ``drive.py``, ``request.json`` (the request WITHOUT the hidden ``checks`` and ``checker`` fields) and
``script.py`` (``scripts/<checker>.py``: the harness's action laws, no thresholds).

    python3 make_task.py E06 DEST          # writes DEST/{drive.py,request.json,script.py}, prints the block

Example:
    >>> shape_text([]), shape_text([3]), shape_text(["V", 3])
    ('(N,)', '(N, 3)', '(N, V, 3)')
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from bench.newenv.requests import ConfigError, Request, load_requests

log = logging.getLogger("bench.newenv")
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
CHECKERS = HERE / "checkers"
HIDDEN_FIELDS = ("checks", "checker")
COUNT_WORDS = {"V": "vertices of the cloth or soft body", "P": "particles", "K": "points along the rope"}


class MissingScriptError(FileNotFoundError):
    """No ``scripts/<checker>.py`` for the request yet (the group agent has not written it)."""


def shape_text(trailing: list) -> str:
    return "(" + ", ".join(["N", *map(str, trailing)]) + ("," if not trailing else "") + ")"


def task_request(req: Request) -> dict:
    """The request.json an agent sees: the merged entry minus the hidden fields."""
    return {k: v for k, v in req.entry.items() if k not in HIDDEN_FIELDS}


def script_path(req: Request, scripts: Path = SCRIPTS) -> Path:
    return scripts / f"{req.checker}.py"


def missing_files(req: Request, scripts: Path = SCRIPTS, checkers: Path = CHECKERS) -> list[str]:
    """Names of the script / checker files this request still lacks (empty = ready to run)."""
    out = [f"scripts/{req.checker}.py"] if not script_path(req, scripts).is_file() else []
    return out + ([f"checkers/{req.checker}.py"] if not (checkers / f"{req.checker}.py").is_file() else [])


def probe_lines(req: Request) -> list[str]:
    lines = []
    for key, trailing in req.probe.items():
        counts = [COUNT_WORDS.get(d, "a count") for d in trailing if isinstance(d, str)]
        note = f"  (the count of {counts[0]} is yours to choose; keep it fixed within a run)" if counts else ""
        lines.append(f"- `{key}`: shape `{shape_text(trailing)}`{note}")
    return lines


def prompt_block(req: Request) -> str:
    """The request-specific text appended after the engine-neutral contract."""
    policy = ("**Yes.** This is a robot task: `env.py` must also provide `scripted_policy(env, obs)` returning the "
              "action `(N, act_dim)`; it should make the task succeed in at least 80 % of worlds within one episode."
              if req.policy else "No: this request needs no `scripted_policy`.")
    return "\n".join([
        "## The request", "", req.request, "",
        "## What is specific to this request", "",
        "- Actions: `act_dim` and what each action component means are given by the request above; the action value "
        "range [-1, 1] maps linearly to the stated limits (a symmetric range around zero).",
        "- `env.probe()` must return exactly these keys (SI units, world frame, z up, quaternions wxyz, read from the "
        "simulation, not computed from your own bookkeeping), each with the shape shown, `N` = `num_envs`:",
        *["  " + line for line in probe_lines(req)],
        f"- Episodes: `episode_steps * dt` must be at least {float(req.entry.get('episode_s', 5.0)):g} s "
        "(`episode_s` in `request.json`); the harness's action phases are sized to fit in one episode.",
        f"- `scripted_policy`: {policy}",
        f"- Request id: `{req.id}` (it is in `request.json`; `python drive.py --quick` uses it).", ""])


def write_task_dir(req: Request, dest: Path, scripts: Path = SCRIPTS, strict: bool = True) -> list[str]:
    """Write drive.py, request.json and script.py into ``dest``; returns the names written.

    With ``strict=False`` a missing script is skipped (a warning is logged) instead of raising.
    """
    src = script_path(req, scripts)
    if strict and not src.is_file():
        raise MissingScriptError(f"{req.id}: {src} does not exist; the group agent for checker "
                                 f"{req.checker!r} has not written it yet")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "drive.py", dest / "drive.py")
    (dest / "request.json").write_text(json.dumps(task_request(req), indent=1) + "\n")
    written = ["drive.py", "request.json"]
    if src.is_file():
        shutil.copy2(src, dest / "script.py")
        written.append("script.py")
    else:
        log.warning("%s: no %s; task directory has no script.py", req.id, src)
    return written


def find_request(request_id: str, requests_path: Path | None = None) -> Request:
    by_id = {r.id: r for r in load_requests(requests_path)}
    if request_id not in by_id:
        raise ConfigError(f"unknown request {request_id!r}; known: {', '.join(by_id)}")
    return by_id[request_id]


def make_task(request_id: str, dest: Path, requests_path: Path | None = None, scripts: Path = SCRIPTS,
              strict: bool = True) -> str:
    """Write the task directory for ``request_id`` into ``dest`` and return the prompt block."""
    req = find_request(request_id, requests_path)
    write_task_dir(req, dest, scripts, strict)
    return prompt_block(req)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("request")
    ap.add_argument("dest", type=Path)
    a = ap.parse_args(argv)
    try:
        sys.stdout.write(make_task(a.request, a.dest))
    except (ConfigError, MissingScriptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
