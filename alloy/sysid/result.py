# SPDX-License-Identifier: Apache-2.0
"""The result of a fit and its file (``*.sysid.json``): what ``scene.add_robot(..., sysid=)`` reads.

File layout (``schema: alloy.sysid/1``)::

    {"schema": "alloy.sysid/1", "robot": "panda", "control_dt": 0.01,
     "joints": {"joint1": {"kp": 4210.0, "kd": 395.1, "damping": 0.88, "friction": 0.31, "delay": 0.012}, ...},
     "fit": {"variables": [...], "rms_before_deg": {...}, "rms_after_deg": {...}, "poorly_determined": {...},
             "sigma": {...}, "moved": {...}, "notes": [...], "seed": 0, "worlds": 64, "generations": 30}}

``joints`` holds only what the fit changed, in SI units (``delay`` in seconds); everything else stays as the robot
file has it. Only the ``schema`` and ``joints`` blocks are needed to apply a file, so one can be written by hand.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alloy.sysid.errors import SysidError
from alloy.sysid.model import FIELDS

SCHEMA = "alloy.sysid/1"
APPLIED = (*FIELDS, "delay")
MAX_APPLIED_DELAY = 0.5         # s


@dataclass
class SysidResult:
    """A fit: per joint the fitted ``values[field]`` (J,), the tracking error before and after, and what the
    recording did not determine."""

    robot: str
    joints: list[str]
    control_dt: float
    values: dict[str, np.ndarray]                    # field -> (J,): kp, kd, damping, friction, delay, ...
    variables: list[str] = field(default_factory=list)
    nominal: dict[str, np.ndarray] = field(default_factory=dict)
    rms_before: np.ndarray | None = None             # (J,) rad (m for a slide)
    rms_after: np.ndarray | None = None
    moved: np.ndarray | None = None                  # (J,) bool
    sigma: dict[str, np.ndarray] = field(default_factory=dict)        # variable -> (J,) relative 1-sigma
    determined: dict[str, np.ndarray] = field(default_factory=dict)   # variable -> (J,) bool
    hinge: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    curves: dict[str, Any] = field(default_factory=dict)    # for plots: t, measured, before, after (not saved)

    def poorly_determined(self) -> dict[str, list[str]]:
        """{joint: [variables the recording does not pin down]} (joints with none are left out)."""
        out: dict[str, list[str]] = {}
        for j, name in enumerate(self.joints):
            weak = [v for v in self.variables if v in self.determined and not bool(self.determined[v][j])]
            if weak:
                out[name] = weak
        return out

    def to_dict(self) -> dict[str, Any]:
        """The JSON form (module docstring). Example: ``result.to_dict()["joints"]["joint1"]["kp"]``."""
        def per_joint(arr: np.ndarray | None, scale: float = 1.0) -> dict[str, float]:
            return {} if arr is None else {n: _num(float(arr[j]) * scale) for j, n in enumerate(self.joints)}

        deg = [180.0 / math.pi if self.hinge is None or self.hinge[j] else 1.0 for j in range(len(self.joints))]
        joints = {n: {f: _num(float(v[j])) for f, v in self.values.items()} for j, n in enumerate(self.joints)}
        fit = {"variables": list(self.variables),
               "rms_before_deg": per_joint(None if self.rms_before is None else self.rms_before * deg),
               "rms_after_deg": per_joint(None if self.rms_after is None else self.rms_after * deg),
               "moved": {n: bool(self.moved[j]) for j, n in enumerate(self.joints)} if self.moved is not None else {},
               "poorly_determined": self.poorly_determined(),
               "sigma": {v: per_joint(s) for v, s in self.sigma.items()},
               "nominal": {n: {f: _num(float(v[j])) for f, v in self.nominal.items()}
                           for j, n in enumerate(self.joints)},
               "notes": list(self.notes), **self.meta}
        return {"schema": SCHEMA, "robot": self.robot, "control_dt": self.control_dt, "joints": joints, "fit": fit}

    def save(self, path: str | Path) -> Path:
        """Write the file. Example: ``result.save("my_robot.sysid.json")``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        return path


def _num(x: float) -> float | None:
    return x if math.isfinite(x) else None


def load(source: Any) -> dict[str, Any]:
    """A sysid file (path), its dict, or a ``SysidResult`` as the checked dict ``add_robot(sysid=)`` stores:
    ``{"schema", "robot", "joints": {joint: {field: value}}}``. Example: ``load("my_robot.sysid.json")["joints"]``."""
    if isinstance(source, SysidResult):
        data = source.to_dict()
    elif isinstance(source, dict):
        data = source
    else:
        path = Path(str(source)).expanduser()
        if not path.is_file():
            raise SysidError(f"sysid file {str(source)!r} does not exist",
                             "pass the file written by `python -m alloy.sysid fit ... --out my_robot.sysid.json`",
                             field="sysid")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise SysidError(f"sysid file {path.name} is not JSON ({exc})", "write it again with "
                             "`python -m alloy.sysid fit`", field="sysid") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or not isinstance(data.get("joints"), dict):
        raise SysidError(f"this is not a sysid file (it needs \"schema\": \"{SCHEMA}\" and a \"joints\" block)",
                         "pass the file written by `python -m alloy.sysid fit`", field="sysid")
    joints = {str(j): _checked(str(j), vals) for j, vals in data["joints"].items()}
    return {"schema": SCHEMA, "robot": str(data.get("robot", "")), "joints": joints}


def _checked(joint: str, vals: Any) -> dict[str, float]:
    if not isinstance(vals, dict):
        raise SysidError(f"sysid file: joint {joint!r} must map parameter names to numbers",
                         f"use any of {APPLIED}", field="sysid")
    out = {}
    for f, v in vals.items():
        if f not in APPLIED:
            raise SysidError(f"sysid file: joint {joint!r} has an unknown parameter {f!r}", f"use any of {APPLIED}",
                             field="sysid")
        if v is None:
            continue                                # "no limit" / not fitted: keep the robot file's value
        v = float(v)
        bad = not math.isfinite(v) or v < 0.0 or (f == "kp" and v == 0.0) or (f == "delay" and v > MAX_APPLIED_DELAY)
        if bad:
            raise SysidError(f"sysid file: {joint}.{f} = {v} is out of range", "kp > 0; every other value >= 0; "
                             f"delay in seconds, at most {MAX_APPLIED_DELAY}", field="sysid")
        out[str(f)] = v
    return out


__all__ = ["APPLIED", "SCHEMA", "SysidResult", "load"]
