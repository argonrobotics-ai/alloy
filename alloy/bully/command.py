# SPDX-License-Identifier: Apache-2.0
"""Command findings: corpus entries whose repro is a pinned program run, not a bully trace (SPEC §3.11).

A crash during a legal-action run is an SB-7 S1 by definition (coordinator ruling on SCR-SC3-3), and it can come from
a runner outside the bully targets (a clip runner, say). Such a finding's ``finding.json`` has ``kind: "command"``
and a ``command`` block: the argv, the alloy commit and venv pins it was found at, the GPU type, and the output
markers that mean a crash. It has no ``trace.npz``; its scene hash is the hash of the argv (the dedup key).

One replay run is one full run of the command from a checkout (``workdir``). A run trips when the process dies on a
signal (segfault: rc -11, or 139 through a shell) or prints a crash marker. A run that exits non-zero without a
marker, or times out, is a setup error and stops the replay: it is neither a trip nor clean. Such crashes are
intermittent (SCR-SC3-3: 3 crashes in 8 runs), so a command replay counts at one crash or more, and ``--close``
still needs every run clean (10 by default).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from alloy.bully.types import ReplayResult
from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")

KIND = "command"
_SIGNAL_RC = (-11, -6, 134, 139)   # SIGSEGV / SIGABRT, direct or through a shell


@dataclasses.dataclass(frozen=True)
class CrashCommand:
    """A pinned program run. ``argv`` placeholders: ``{python}`` (this interpreter), ``{device}``, ``{out}`` (a
    fresh directory per run)."""

    argv: tuple[str, ...]
    commit: str                     # alloy commit the crash was found at (40 hex)
    pins: str                       # venv pin file at that commit
    gpu: str = ""                   # GPU type the counts hold for ("L4")
    markers: tuple[str, ...] = ()   # output text that means a crash
    timeout_s: float = 7200.0       # per run

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(rec: dict[str, Any]) -> CrashCommand:
        names = {f.name for f in dataclasses.fields(CrashCommand)}
        kw = {k: v for k, v in rec.items() if k in names}
        kw["argv"], kw["markers"] = tuple(kw["argv"]), tuple(kw.get("markers", ()))
        return CrashCommand(**kw)

    def hash(self) -> str:
        return hashlib.sha256(("command:" + " ".join(self.argv)).encode()).hexdigest()

    def resolve(self, device: str, out: Path) -> list[str]:
        sub = {"{python}": sys.executable, "{device}": device, "{out}": str(out)}
        return [sub.get(a, a) for a in self.argv]


def is_command(rec: dict[str, Any]) -> bool:
    return rec.get("kind") == KIND


def command_of(rec: dict[str, Any]) -> CrashCommand:
    if not is_command(rec) or "command" not in rec:
        raise BullyError(f"{rec.get('id', '?')} is not a command finding")
    return CrashCommand.from_json(rec["command"])


def _head(workdir: Path) -> str:
    res = subprocess.run(["git", "-C", str(workdir), "rev-parse", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else ""


def run_crashed(cmd: CrashCommand, workdir: Path, device: str) -> bool:
    """One full run of ``cmd`` in ``workdir``: True on a crash, False on a clean exit; BullyError otherwise."""
    with tempfile.TemporaryDirectory(prefix="bully_cmd_") as out:
        argv = cmd.resolve(device, Path(out))
        try:
            res = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, timeout=cmd.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise BullyError(f"command timed out after {cmd.timeout_s:.0f} s: {' '.join(argv)}") from exc
    text = res.stdout + res.stderr
    hit = [m for m in cmd.markers if m in text]
    if res.returncode in _SIGNAL_RC or hit:
        log.info("command run crashed: rc %d, markers %s", res.returncode, hit)
        return True
    if res.returncode != 0:
        raise BullyError(f"command failed without a crash (rc {res.returncode}): {text[-600:]}")
    return False


def replay_command(rec: dict[str, Any], device: str, runs: int, workdir: Path) -> tuple[ReplayResult, dict]:
    """Run a command finding ``runs`` times; returns the result and the extra replay-line fields (checkout head)."""
    cmd = command_of(rec)
    trips = 0
    for k in range(runs):
        hit = run_crashed(cmd, workdir, device)
        trips += int(hit)
        log.info("replay %s run %d/%d: %s", rec["id"], k + 1, runs, "crash" if hit else "clean")
    return ReplayResult(rec["id"], trips, runs, device, trips >= 1), {"commit": _head(workdir), "kind": KIND}
