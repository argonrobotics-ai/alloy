# SPDX-License-Identifier: Apache-2.0
"""Typed steering commands, applied at step boundaries, logged, and replayable from the log.

A command is a small dict ``{"id", "op", "args", "issuer"}``; :data:`OPS` is the whole list with each op's argument
names. The worker applies every pending command before the next training step (:meth:`CommandQueue.due`) and appends
one :class:`LogEntry` per command: wall time, step, issuer ("user" or "agent:<name>"), arguments and result. Replay
re-applies the logged commands at the same step boundaries (:class:`Replay`), so a run with commands repeats from its
log on any deterministic route (same seed, same device).

Example::

    cmd = make_command("set", {"name": "reward.rise", "value": 0.0}, issuer="agent:tuner")
    queue.push(cmd)
    for c in queue.due(step=1200):      # the worker's step boundary
        ...
"""
from __future__ import annotations

import itertools
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

OPS: dict[str, tuple[str, ...]] = {
    "pause": ("world",),            # world: int (global index on this worker) or None for the whole run
    "resume": ("world",),
    "set": ("name", "value"),       # a whitelisted setting (Settings.names())
    "push": ("world", "body", "force", "steps"),   # force (N, world frame) on a body for `steps` control steps
    "fork": ("world", "into"),      # into: a world index (branch from here) or "snapshot:<name>"
    "restore": ("snapshot", "world"),               # a saved snapshot into a world slot
    "reset": ("world",),
}
_ids = itertools.count(1)


class CommandError(ValueError):
    """A command with an unknown op, missing arguments or values out of range."""


def make_command(op: str, args: dict[str, Any] | None = None, *, issuer: str = "user",
                 cid: str | None = None) -> dict[str, Any]:
    """A validated command dict. Raises :class:`CommandError` naming the expected arguments."""
    if op not in OPS:
        raise CommandError(f"unknown op {op!r}; ops are {sorted(OPS)}")
    args = dict(args or {})
    missing = [a for a in OPS[op] if a not in args and not (a == "world" and op in ("pause", "resume"))]
    if missing:
        raise CommandError(f"{op} needs {list(OPS[op])}; missing {missing}")
    return {"id": cid or f"c{int(time.time() * 1e3)}-{next(_ids)}", "op": op, "args": args, "issuer": issuer}


@dataclass
class LogEntry:
    """One applied command, as written to the command log (JSON lines)."""
    id: str
    op: str
    args: dict[str, Any]
    issuer: str
    step: int                       # the step boundary it was applied at (before step `step` ran)
    wall: float                     # unix time it was applied
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class CommandQueue:
    """Pending commands in arrival order; :meth:`due` hands them out at the next step boundary."""

    def __init__(self) -> None:
        self._pending: list[dict[str, Any]] = []

    def push(self, cmd: dict[str, Any]) -> None:
        self._pending.append(cmd)

    def due(self, step: int) -> list[dict[str, Any]]:      # noqa: ARG002 - live commands are due at the next boundary
        out, self._pending = self._pending, []
        return out


class Replay(CommandQueue):
    """Commands from a log, each due at the step it was applied at in the original run."""

    def __init__(self, entries: Iterable[LogEntry | dict[str, Any]]) -> None:
        super().__init__()
        rows = [e if isinstance(e, dict) else asdict(e) for e in entries]
        self._by_step: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            cmd = {"id": r["id"], "op": r["op"], "args": r["args"], "issuer": r["issuer"]}
            self._by_step.setdefault(int(r["step"]), []).append(cmd)

    def due(self, step: int) -> list[dict[str, Any]]:
        return self._by_step.pop(step, [])

    @property
    def remaining(self) -> int:
        return sum(len(v) for v in self._by_step.values())


class CommandLog:
    """Append-only command log, in memory and (optionally) in a JSON-lines file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.entries: list[LogEntry] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: LogEntry) -> None:
        self.entries.append(entry)
        if self.path:
            with self.path.open("a") as f:
                f.write(entry.to_json() + "\n")

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self.entries)
