# SPDX-License-Identifier: Apache-2.0
"""Findings corpus ``bench/bully/corpus/BUL-nnnn/`` (SPEC §3.11, bench §4.7).

Each finding directory holds ``finding.json`` (the ``Finding`` fields plus ``target``, ``status`` and the schema
tag), ``trace.npz`` (legal actions per tick [T, A], the control period ``dt``, the seed) and ``replays.jsonl`` (one
line per replay). A command finding (``alloy.bully.command``: a pinned program run that crashed) has no trace. A
finding opens when its replay counts, closes when a replay after the fix no longer trips, and stays as a regression
test. A new finding with the same cause as a closed one (same scene hash and monitor) is
filed as a REPEAT of it, one severity higher; one that matches an open finding is the same finding and is not filed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt_mod
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from alloy.bully.command import KIND, CrashCommand, is_command, replay_command
from alloy.bully.types import BullyLimits, BullySeverity, Finding, ReplayResult
from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")

SCHEMA = "alloy.bully.finding/1"
_ID = re.compile(r"^BUL-(\d{4,})$")


def _finding_to_json(finding: Finding, target: str, status: str) -> dict[str, Any]:
    rec = dataclasses.asdict(finding)
    rec["severity"] = int(finding.severity)
    rec.update({"schema": SCHEMA, "target": target, "status": status})
    return rec


def _finding_from_json(rec: dict[str, Any]) -> Finding:
    names = {f.name for f in dataclasses.fields(Finding)}
    kwargs = {k: v for k, v in rec.items() if k in names}
    kwargs["severity"] = BullySeverity(int(kwargs["severity"]))
    return Finding(**kwargs)


class Corpus:
    """Directory-backed corpus; every write is a whole file (no partial updates)."""

    def __init__(self, root: Path = Path("bench/bully/corpus")) -> None:
        self.root = Path(root)

    # ---------------------------------------------------------------- listing
    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if _ID.match(p.name) and (p / "finding.json").is_file())

    def next_id(self) -> str:
        nums = [int(_ID.match(i).group(1)) for i in self.ids()]  # type: ignore[union-attr]
        return f"BUL-{(max(nums) + 1) if nums else 1:04d}"

    def record(self, finding_id: str) -> dict[str, Any]:
        path = self.root / finding_id / "finding.json"
        if not path.is_file():
            raise BullyError(f"no finding {finding_id} under {self.root}")
        return json.loads(path.read_text())

    def load(self, finding_id: str) -> tuple[Finding, np.ndarray, float, int]:
        """(finding, trace [T, A], control period dt [s], seed). Command findings have no trace."""
        rec = self.record(finding_id)
        if is_command(rec):
            raise BullyError(f"{finding_id} is a command finding: no trace (alloy.bully.command)")
        finding = _finding_from_json(rec)
        with np.load(self.root / finding_id / "trace.npz") as z:
            return finding, np.asarray(z["actions"]), float(z["dt"]), int(z["seed"])

    # ---------------------------------------------------------------- filing
    def _same_cause(self, finding: Finding) -> list[dict[str, Any]]:
        return [r for r in (self.record(i) for i in self.ids())
                if r["scene_hash"] == finding.scene_hash and r["monitor"] == finding.monitor]

    def _open_same(self, finding: Finding) -> Path | None:
        """The directory of an open finding with the same cause (it is the same finding), else None."""
        open_same = [r for r in self._same_cause(finding) if r["status"] == "open"]
        if not open_same:
            return None
        log.info("finding %s/%s already open as %s; not filed again", finding.scene_hash[:8], finding.monitor,
                 open_same[0]["id"])
        return self.root / open_same[0]["id"]

    def _new(self, finding: Finding) -> Finding:
        """Assign a fresh id; a closed finding with the same cause makes this a REPEAT one severity higher."""
        same = self._same_cause(finding)
        fid = self.next_id()
        finding = dataclasses.replace(finding, id=fid, repro=finding.repro.replace("<id>", fid))
        if same:  # catalogued cause, closed: REPEAT one severity higher (S1 stays S1)
            prior = same[-1]
            sev = BullySeverity(max(int(BullySeverity.S1), int(finding.severity) - 1))
            finding = dataclasses.replace(finding, severity=sev, repeat_of=prior["id"])
            log.warning("finding %s repeats closed %s: filed as %s", fid, prior["id"], sev.name)
        return finding

    def _write(self, finding: Finding, extra: dict[str, Any]) -> Path:
        out = self.root / finding.id
        out.mkdir(parents=True, exist_ok=False)
        rec = {**_finding_to_json(finding, finding.target, "open"), **extra}
        (out / "finding.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
        log.info("filed %s: %s %s = %.4g (threshold %.4g) on %s", finding.id, finding.severity.name, finding.monitor,
                 finding.value, finding.threshold, finding.target or finding.scene_hash[:12])
        return out

    def add(self, finding: Finding, trace: np.ndarray, dt: float, seed: int, *, target: str = "") -> Path:
        """File ``finding`` (a fresh id is assigned) with its trace; returns the finding directory."""
        finding = dataclasses.replace(finding, target=target or finding.target)
        same_open = self._open_same(finding)
        if same_open is not None:
            return same_open
        out = self._write(self._new(finding), {})
        np.savez(out / "trace.npz", actions=np.asarray(trace, dtype=np.float32), dt=float(dt), seed=int(seed))
        return out

    def add_command(self, finding: Finding, command: CrashCommand, source: dict[str, Any]) -> Path:
        """File a command finding (``alloy.bully.command``): no trace, the argv hash as its scene hash, ``source``
        says who measured it and under which conditions."""
        finding = dataclasses.replace(finding, scene_hash=command.hash(), target="")
        same_open = self._open_same(finding)
        if same_open is not None:
            return same_open
        return self._write(self._new(finding), {"kind": KIND, "command": command.to_json(), "source": source})

    def set_status(self, finding_id: str, status: str) -> None:
        if status not in ("open", "closed"):
            raise BullyError(f"status must be open or closed, got {status!r}")
        rec = self.record(finding_id)
        rec["status"] = status
        (self.root / finding_id / "finding.json").write_text(json.dumps(rec, indent=2, sort_keys=True))

    # ---------------------------------------------------------------- replay
    def replay(self, finding_id: str, device: str, runs: int = 10, limits: BullyLimits | None = None,
               workdir: Path = Path(".")) -> ReplayResult:
        """Replay a filed finding ``runs`` times: a trace on its registered target, or a command finding's program from
        the checkout ``workdir``. The result is appended to replays.jsonl."""
        rec = self.record(finding_id)
        if is_command(rec):
            result, extra = replay_command(rec, device, runs, Path(workdir))
            self.log_replay(finding_id, result, extra)
            return result
        from alloy.bully.replay import replay_trace
        from alloy.bully.targets import get_target

        finding, trace, _, _ = self.load(finding_id)
        target = get_target(rec["target"] or finding.target)
        result = replay_trace(target, trace, finding.monitor, device, limits or BullyLimits(), runs, finding_id)
        self.log_replay(finding_id, result)
        return result

    def log_replay(self, finding_id: str, result: ReplayResult, extra: dict[str, Any] | None = None) -> None:
        """Append one replay line; ``extra`` adds fields (a command replay's checkout head, who ran it)."""
        line = {**dataclasses.asdict(result), **(extra or {})}
        line.setdefault("date", dt_mod.datetime.now(dt_mod.timezone.utc).isoformat(timespec="seconds"))
        with open(self.root / finding_id / "replays.jsonl", "a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    def last_replay(self, finding_id: str) -> dict[str, Any] | None:
        path = self.root / finding_id / "replays.jsonl"
        if not path.is_file():
            return None
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None

    def regressions(self) -> list[str]:
        """Closed findings whose latest replay tripped again (BUL-2 counts these; target 0)."""
        out = []
        for fid in self.ids():
            last = self.last_replay(fid)
            if self.record(fid)["status"] == "closed" and last is not None and last["trips"] > 0:
                out.append(fid)
        return out
