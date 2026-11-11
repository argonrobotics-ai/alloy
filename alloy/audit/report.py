# SPDX-License-Identifier: Apache-2.0
"""What ``alloy.audit`` returns: findings by severity, each with how often it showed, how to replay it and what to
change. Plain JSON out (``AuditReport.to_json``).

A finding comes from one of two sources:

- ``check``: ``alloy.check`` over several rounds (a new seed each round, so randomized worlds differ); ``rate`` is
  the share of rounds it appeared in, ``replay`` the round's seed and settings;
- ``bully``: the bully agent (``alloy.bully``) driving the scene with random legal inputs; ``rate`` is the share
  of episodes, ``replay`` the handle that reproduces the episode.

Example::

    >>> r = AuditReport(scene="demo")
    >>> r.add(AuditFinding("check", "error", "overlap", "a and b overlap by 3 mm", "lift a by 3.5 mm", ("a", "b")))
    >>> r.ok, r.to_dict()["counts"]
    (False, {'error': 1, 'warn': 0, 'info': 0})
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA = "alloy.audit/1"
SEVERITIES = ("error", "warn", "info")


@dataclass
class AuditFinding:
    """One problem: ``source`` ("check" or "bully"), ``severity`` ("error", "warn", "info"), ``kind`` (the check or
    the bully class), what happened, the fix, the objects or bodies involved, ``rate`` (share of rounds or episodes
    it appeared in), ``first_time_s`` when first seen, numbers, and ``replay`` (how to see it again)."""

    source: str
    severity: str
    kind: str
    message: str
    fix: str
    objects: tuple[str, ...] = ()
    rate: float = 1.0
    seen: int = 1
    first_time_s: float | None = None
    value: float | None = None
    limit: float | None = None
    unit: str = ""
    replay: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity {self.severity!r}: use one of {SEVERITIES}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["objects"] = list(self.objects)
        for k in ("rate", "first_time_s", "value", "limit"):
            v = d[k]
            d[k] = None if v is None or not math.isfinite(v) else round(float(v), 6)
        return d

    def text(self) -> str:
        """One line: ``[error] check/overlap (3/3 rounds): ... -> fix``."""
        return f"[{self.severity}] {self.source}/{self.kind} ({self.rate:.0%}): {self.message} -> {self.fix}"


@dataclass
class AuditReport:
    """The audit of one scene. ``ok``: no error finding. ``runs``: what ran (rounds, episodes, device, whether the
    bully agent was available); ``timings_s``: wall time per stage."""

    scene: str
    findings: list[AuditFinding] = field(default_factory=list)
    runs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def add(self, finding: AuditFinding) -> None:
        self.findings.append(finding)

    def counts(self) -> dict[str, int]:
        return {s: sum(f.severity == s for f in self.findings) for s in SEVERITIES}

    def sorted(self) -> list[AuditFinding]:
        """Errors first, then the most frequent."""
        return sorted(self.findings, key=lambda f: (SEVERITIES.index(f.severity), -f.rate, f.source, f.kind))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "scene": self.scene, "ok": self.ok, "counts": self.counts(), "runs": self.runs,
                "findings": [f.to_dict() for f in self.sorted()], "notes": self.notes,
                "timings_s": {k: round(v, 2) for k, v in self.timings_s.items()}}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def summary(self) -> str:
        """A few lines for people: the verdict, then every error and warning with its fix."""
        c = self.counts()
        lines = [f"{self.scene}: {'OK' if self.ok else 'FAIL'} ({c['error']} errors, {c['warn']} warnings)"]
        lines += ["  " + f.text() for f in self.sorted() if f.severity != "info"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


__all__ = ["AuditFinding", "AuditReport", "SCHEMA", "SEVERITIES"]
