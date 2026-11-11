# SPDX-License-Identifier: Apache-2.0
"""What ``alloy.create`` returns: the new card, its files, every physical value with its source, the list of things
to verify, and the drop test's verdict. ``to_json()`` is the command line's output.

Example::

    >>> r = CreateResult(name="mug", card="my_objects/objects/mug.yaml")
    >>> r.to_dict()["estimates"]
    []
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "alloy.create/1"


@dataclass
class CreateResult:
    """``ok``: the card is valid and passed the drop test (or the test was skipped). ``values``: one row per
    physical value (field, value, unit, tag USER / CITED / LIBRARY / MODEL / ESTIMATE, source, range, what to verify).
    ``verify``: what a person should check, most important first. ``check``: the drop test report (dict)."""

    name: str
    card: str
    root: str = ""
    files: dict[str, str] = field(default_factory=dict)
    values: list[dict[str, Any]] = field(default_factory=list)
    check: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    generator: dict[str, Any] | None = None
    objects: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.check is None or bool(self.check.get("ok"))

    @property
    def estimates(self) -> list[str]:
        return [v["field"] for v in self.values if v.get("tag") == "ESTIMATE"]

    @property
    def verify(self) -> list[str]:
        out = [f"{v['field']}: {v['verify']}" for v in self.values if v.get("verify")]
        if self.check is not None and not self.check.get("ok"):
            out.insert(0, "drop test failed: " + "; ".join(f.get("message", "") for f in self.check.get("findings", [])
                                                           if f.get("level") == "error")[:300])
        if self.generator is None and _moved(self.check):
            out.insert(0, "pose: it did not rest still in the pose it was made in; if it tipped over, pass up= (the "
                          "axis that is up in your mesh file: z, y, -y or x)")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "ok": self.ok, "name": self.name, "card": self.card, "root": self.root,
                "objects": self.objects or [self.name], "estimates": self.estimates, "verify": self.verify,
                "values": self.values, "files": self.files, "generator": self.generator, "notes": self.notes,
                "check": _check_summary(self.check), "timings_s": self.timings_s}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def summary(self) -> str:
        """A few lines for people."""
        lines = [f"{self.name}: {'OK' if self.ok else 'CHECK FAILED'} -> {self.card}"]
        lines += [f"  {v['field']}: {v['value']} {v.get('unit', '')} [{v['tag']}] {v['source']}" for v in self.values
                  if v["field"] not in ("material", "shape")]
        lines += [f"  verify: {s}" for s in self.verify]
        return "\n".join(lines)


def _moved(report: dict[str, Any] | None) -> bool:
    """True when the drop test saw the object slide or tip after landing (its card pose is not a resting pose)."""
    return report is not None and any(f.get("check") == "rest_drift" and f.get("level") in ("error", "warn")
                                      for f in report.get("findings", []))


def _check_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    keep = ("ok", "checks", "errors", "warnings")
    out = {k: report.get(k) for k in keep}
    out["findings"] = [{k: f.get(k) for k in ("check", "level", "message", "fix")} for f in report.get("findings", [])
                       if f.get("level") in ("error", "warn")]
    return out


__all__ = ["CreateResult", "SCHEMA"]
