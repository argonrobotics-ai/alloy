# SPDX-License-Identifier: Apache-2.0
"""The harness report: one ``CheckResult`` per check, JSON in and out."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """One check. ``fix`` says what to change when it fails; ``skipped`` checks do not count.

    Example: ``CheckResult("determinism", "determinism", True, 0.0, 0.0, "two runs, digests equal")``.
    """

    name: str
    group: str            # contract | crash | determinism | independence | speed | lint | law
    passed: bool
    value: float | None = None
    bound: float | None = None
    detail: str = ""
    fix: str = ""
    skipped: bool = False

    @staticmethod
    def skip(name: str, group: str, why: str) -> CheckResult:
        """A check that did not run, and why."""
        return CheckResult(name, group, True, detail=why, skipped=True)


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


@dataclass
class CheckReport:
    """Everything the harness found for one plug-in. ``passed`` is True when no check that ran failed.

    Example::

        report = alloy.plugins.check("magnet")
        print(report.to_json())          # {"plugin": "magnet", "passed": true, "checks": [...]}
    """

    plugin: str
    kind: str
    target: str
    device: str
    checks: list[CheckResult] = field(default_factory=list)
    seconds: float = 0.0
    versions: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if not c.skipped) and any(not c.skipped for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and not c.skipped]

    def add(self, result: CheckResult | list[CheckResult]) -> None:
        self.checks.extend(result if isinstance(result, list) else [result])

    def to_dict(self) -> dict[str, Any]:
        rows = [{k: _clean(v) for k, v in dataclasses.asdict(c).items()} for c in self.checks]
        return {"plugin": self.plugin, "kind": self.kind, "target": self.target, "device": self.device,
                "passed": self.passed, "failed": [c.name for c in self.failures], "seconds": round(self.seconds, 2),
                "versions": self.versions, "checks": rows}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CheckReport:
        """Inverse of ``to_dict`` (the sandbox reads the child's report this way)."""
        fields = {f.name for f in dataclasses.fields(CheckResult)}
        checks = [CheckResult(**{k: v for k, v in c.items() if k in fields}) for c in d.get("checks", [])]
        return CheckReport(d["plugin"], d["kind"], d["target"], d["device"], checks, float(d.get("seconds", 0.0)),
                           dict(d.get("versions", {})))

    def summary(self) -> str:
        """Short text: one line per check, failures with their fix."""
        head = f"{self.plugin} ({self.kind}) on {self.device}: {'PASS' if self.passed else 'FAIL'}"
        lines = [head]
        for c in self.checks:
            mark = "skip" if c.skipped else ("ok" if c.passed else "FAIL")
            val = "" if c.value is None else f" value={c.value:.4g}"
            val += "" if c.value is None or c.bound is None else f" bound={c.bound:.4g}"
            lines.append(f"  [{mark}] {c.group}/{c.name}{val} {c.detail}".rstrip())
            if not c.passed and c.fix:
                lines.append(f"         fix: {c.fix}")
        return "\n".join(lines)


__all__ = ["CheckReport", "CheckResult"]
