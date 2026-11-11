# SPDX-License-Identifier: Apache-2.0
"""Report types of the physics check: ``Finding`` (one problem and what to change) and ``CheckReport`` (JSON).

Every finding names the check that produced it, a level, what is wrong (with numbers and units) and the edit that
fixes it. ``suggestion`` carries the same edit as data when it is mechanical (a new pose for a placement), so a
scene builder can apply it without parsing the text.

Example::

    >>> f = Finding("overlap", "error", "mug overlaps table by 3.2 mm at start", "lift mug by 4 mm",
    ...             objects=("mug", "table"), value=3.2e-3, limit=1e-3, unit="m")
    >>> CheckReport(scene="kitchen", findings=[f]).ok
    False
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Level = Literal["error", "warn", "info"]
LEVELS: tuple[str, ...] = ("error", "warn", "info")
CHECKS: tuple[str, ...] = ("cards", "overlap", "rest_drift", "penetration", "health", "energy")
EXTRA_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")   # names of extra run checks (check(..., extra=[fn])), e.g. "soft"
SCHEMA = "alloy.check/1"


@dataclass(frozen=True)
class Finding:
    """One problem found by a check, and the change that fixes it.

    ``check`` is one of ``CHECKS``; ``level`` "error" means the scene is not usable as is, "warn" that it runs but
    a result may be wrong, "info" a measurement worth knowing. ``objects`` are placed object or robot names (a name
    placed twice is ``name#2``), ``field`` a card field path when one is at fault, ``worlds`` the affected worlds
    (empty: every world, or not world-specific).

    Example::

        >>> Finding("rest_drift", "warn", "mug slides 2.1 mm in 2 s", "lower mug.material.surface.mu_static ...",
        ...         objects=("mug",)).to_dict()["check"]
        'rest_drift'
    """

    check: str
    level: Level
    message: str
    fix: str
    objects: tuple[str, ...] = ()
    field: str | None = None
    value: float | None = None
    limit: float | None = None
    unit: str = ""
    worlds: tuple[int, ...] = ()
    suggestion: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.check not in CHECKS and not EXTRA_NAME.match(str(self.check)):
            raise ValueError(f"Finding.check {self.check!r} is not a check; use one of {CHECKS}, or a lowercase name "
                             f"(letters, digits, _) for an extra check such as 'soft'")
        if self.level not in LEVELS:
            raise ValueError(f"Finding.level {self.level!r} is not a level; use one of {LEVELS}")

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready data (non-finite numbers become None).

        Example::

            >>> Finding("overlap", "warn", "m", "f", value=float("nan")).to_dict()["value"] is None
            True
        """
        d = asdict(self)
        d["objects"], d["worlds"] = list(self.objects), list(self.worlds)
        d["value"], d["limit"] = _finite(self.value), _finite(self.limit)
        d["suggestion"] = _plain(self.suggestion) if self.suggestion is not None else None
        return d

    def text(self) -> str:
        """One line for people: ``[error] overlap: <message> -> <fix>``.

        Example::

            >>> Finding("overlap", "error", "mug overlaps table by 3.2 mm at start", "lift mug by 4 mm").text()
            '[error] overlap: mug overlaps table by 3.2 mm at start -> lift mug by 4 mm'
        """
        return f"[{self.level}] {self.check}: {self.message} -> {self.fix}"


@dataclass
class CheckReport:
    """What ``alloy.check.check`` found. ``ok`` is True when no finding is an error.

    ``checks`` maps each check to "pass", "warn", "fail" or "skipped: <reason>"; ``metrics`` holds the measured
    numbers (per object and per world maxima); ``settings`` the run (worlds, seconds, device); ``timings_s`` the
    wall time per stage.

    Example::

        >>> r = CheckReport(scene="empty")
        >>> r.ok, r.to_dict()["schema"]
        (True, 'alloy.check/1')
    """

    scene: str
    findings: list[Finding] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no finding is an error.

        Example::

            >>> CheckReport(scene="empty").ok
            True
        """
        return not self.errors()

    def errors(self) -> list[Finding]:
        """The error findings.

        Example::

            >>> f = Finding("overlap", "error", "m", "fix it")
            >>> CheckReport(scene="k", findings=[f]).errors() == [f]
            True
        """
        return [f for f in self.findings if f.level == "error"]

    def warnings(self) -> list[Finding]:
        """The warning findings.

        Example::

            >>> f = Finding("rest_drift", "warn", "m", "fix it")
            >>> CheckReport(scene="k", findings=[f]).warnings() == [f]
            True
        """
        return [f for f in self.findings if f.level == "warn"]

    def add(self, findings: list[Finding], check: str | None = None, ran: bool = True) -> None:
        """Append findings; with ``check``, also record that check's verdict from its findings.

        Example::

            >>> r = CheckReport(scene="k")
            >>> r.add([Finding("overlap", "error", "m", "fix it")], "overlap")
            >>> r.checks
            {'overlap': 'fail'}
        """
        self.findings.extend(findings)
        if check is not None and ran:
            self.checks[check] = verdict([f for f in self.findings if f.check == check])

    def skip(self, check: str, reason: str) -> None:
        """Record that ``check`` did not run, and why.

        Example::

            >>> r = CheckReport(scene="k")
            >>> r.skip("energy", "run=False")
            >>> r.checks
            {'energy': 'skipped: run=False'}
        """
        self.checks[check] = f"skipped: {reason}"

    def to_dict(self) -> dict[str, Any]:
        """The report as plain JSON-ready data, findings sorted errors first.

        Example::

            >>> d = CheckReport(scene="k").to_dict()
            >>> d["schema"], d["ok"], d["errors"]
            ('alloy.check/1', True, 0)
        """
        order = {lvl: i for i, lvl in enumerate(LEVELS)}
        found = sorted(self.findings, key=lambda f: (order[f.level], check_rank(f.check)))
        return {"schema": SCHEMA, "scene": self.scene, "ok": self.ok,
                "errors": len(self.errors()), "warnings": len(self.warnings()),
                "checks": dict(self.checks), "findings": [f.to_dict() for f in found],
                "metrics": _plain(self.metrics), "settings": _plain(self.settings),
                "timings_s": {k: round(v, 3) for k, v in self.timings_s.items()}}

    def to_json(self, indent: int | None = 2) -> str:
        """The report as JSON text (``json.loads`` gives ``to_dict()``).

        Example::

            >>> json.loads(CheckReport(scene="k").to_json())["scene"]
            'k'
        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def summary(self) -> str:
        """A few lines for people: verdict, per-check state, then every error and warning with its fix.

        Example::

            >>> CheckReport(scene="k").summary().splitlines()[0]
            'k: OK (0 errors, 0 warnings)'
        """
        head = f"{self.scene}: {'OK' if self.ok else 'FAIL'} ({len(self.errors())} errors, " \
               f"{len(self.warnings())} warnings)"
        lines = [head, "  " + ", ".join(f"{k}: {v}" for k, v in self.checks.items())]
        lines += ["  " + f.text() for f in self.findings if f.level != "info"]
        return "\n".join(lines)


def check_rank(name: str) -> int:
    """Sort key of a check name: the standard checks in ``CHECKS`` order, extra checks after them.

    Example::

        >>> check_rank("cards"), check_rank("soft")
        (0, 6)
    """
    return CHECKS.index(name) if name in CHECKS else len(CHECKS)


def verdict(findings: list[Finding]) -> str:
    """"fail" if any error, "warn" if any warning, else "pass".

    Example::

        >>> verdict([Finding("overlap", "warn", "m", "fix it")])
        'warn'
        >>> verdict([])
        'pass'
    """
    levels = {f.level for f in findings}
    return "fail" if "error" in levels else "warn" if "warn" in levels else "pass"


def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _plain(node: Any) -> Any:
    """JSON-safe copy: tuples to lists, numpy scalars to Python, non-finite floats to None."""
    if isinstance(node, Mapping):
        return {str(k): _plain(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(v) for v in node]
    if hasattr(node, "tolist"):
        return _plain(node.tolist())
    if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
        return node
    if isinstance(node, float):
        return _finite(node)
    return str(node)


__all__ = ["CHECKS", "EXTRA_NAME", "LEVELS", "SCHEMA", "CheckReport", "Finding", "Level", "check_rank", "verdict"]
