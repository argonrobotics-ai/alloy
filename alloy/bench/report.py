# SPDX-License-Identifier: Apache-2.0
"""Report files, schema `alloy.bench.report/1` (bench §4.8), one per (law, solver, device).

Path: `<root>/<YYYYMMDD>-<alloy7>/<suite>/<law>.<solver>.<device>.json`. Non-finite floats are written as null and
read back as NaN, so every file is strict JSON. `read_report(write_report(r)) == r` up to NaN equality.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alloy.bench.provenance import commit_short
from alloy.bench.types import REPORT_SCHEMA, VERDICTS, LawResult, MetricVerdict, Quantity

SCHEMA = REPORT_SCHEMA
STATS = ("mean", "curve", "distribution", "rate")
_REQUIRED = ("schema", "law", "suite", "solver", "device", "command", "provenance", "config", "data", "metrics",
             "monitors", "wall_s", "fit_dt_only")


def _num(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def _unnum(x: float | None) -> float:
    return math.nan if x is None else float(x)


def _quantity(q: Quantity | None) -> dict[str, Any] | None:
    if q is None:
        return None
    return {"name": q.name, "unit": q.unit, "mean": _num(q.mean), "std": _num(q.std), "n": q.n, "u95": _num(q.u95)}


def _unquantity(d: Mapping[str, Any] | None) -> Quantity | None:
    if d is None:
        return None
    return Quantity(d["name"], d["unit"], _unnum(d["mean"]), _unnum(d["std"]), int(d["n"]), _unnum(d["u95"]))


def _verdict(v: MetricVerdict) -> dict[str, Any]:
    return {"name": v.name, "unit": v.unit, "real": _quantity(v.real), "sim": _quantity(v.sim),
            "error": _num(v.error), "bound": _num(v.bound), "tur": _num(v.tur), "stat": v.stat, "verdict": v.verdict}


def _unverdict(d: Mapping[str, Any]) -> MetricVerdict:
    sim = _unquantity(d["sim"])
    if sim is None:
        raise ValueError(f"metric {d.get('name')}: sim quantity missing")
    tur = None if d["tur"] is None else float(d["tur"])
    return MetricVerdict(d["name"], d["unit"], _unquantity(d["real"]), sim, _unnum(d["error"]), _unnum(d["bound"]),
                         tur, d["stat"], d["verdict"])


def _jsonable(obj: Any) -> Any:
    """Monitors may hold numpy scalars/arrays and non-finite floats; make them strict JSON."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    if isinstance(obj, float):
        return _num(obj)
    return obj


def to_dict(result: LawResult) -> dict[str, Any]:
    p = dict(result.provenance)
    return {
        "schema": SCHEMA, "law": result.law, "suite": p.get("suite", ""), "solver": result.solver,
        "device": result.device, "command": result.command, "provenance": p, "config": _jsonable(dict(result.config)),
        "data": dict(result.data),
        "metrics": [_verdict(v) for v in result.verdicts], "monitors": _jsonable(dict(result.monitors)),
        "wall_s": result.wall_s, "fit_dt_only": result.fit_dt_only,
    }


def from_dict(d: Mapping[str, Any]) -> LawResult:
    problems = validate(d)
    if problems:
        raise ValueError(f"not an {SCHEMA} report: {problems}")
    return LawResult(d["law"], d["solver"], d["device"], tuple(_unverdict(m) for m in d["metrics"]),
                     dict(d["monitors"]), dict(d["provenance"]), float(d["wall_s"]), bool(d["fit_dt_only"]),
                     command=str(d["command"]), config=dict(d["config"]), data=dict(d["data"]))


def validate(d: Mapping[str, Any]) -> list[str]:
    """Schema problems of a report dict (empty list = valid)."""
    out = [f"missing {k}" for k in _REQUIRED if k not in d]
    if d.get("schema") != SCHEMA:
        out.append(f"schema {d.get('schema')!r} != {SCHEMA}")
    for m in d.get("metrics", []):
        if m.get("verdict") not in VERDICTS:
            out.append(f"metric {m.get('name')}: verdict {m.get('verdict')!r}")
        if m.get("stat") not in STATS:
            out.append(f"metric {m.get('name')}: stat {m.get('stat')!r}")
        if m.get("sim") is None:
            out.append(f"metric {m.get('name')}: sim quantity missing")
    return out


def report_path(result: LawResult, root: Path) -> Path:
    suite = result.provenance.get("suite", "") or "unknown"
    day = time.strftime("%Y%m%d")
    ds = (result.data or {}).get("dataset_id", "")  # rig: one report per dataset (e.g. one per arm)
    safe = f"{result.law}.{result.solver}.{result.device}{'.' + ds if ds else ''}".replace("/", "_").replace(":", "")
    return Path(root) / f"{day}-{commit_short()}" / suite / f"{safe}.json"


def write_report(result: LawResult, root: Path = Path("bench/results")) -> Path:
    """Write one report file (schema alloy.bench.report/1) and return its path."""
    path = report_path(result, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(result), indent=1, sort_keys=False, allow_nan=False) + "\n")
    return path


def read_report(path: Path) -> LawResult:
    return from_dict(json.loads(Path(path).read_text()))


def is_report(path: Path) -> bool:
    try:
        return json.loads(Path(path).read_text()).get("schema") == SCHEMA
    except (OSError, ValueError, AttributeError):
        return False


__all__ = ["SCHEMA", "write_report", "read_report", "to_dict", "from_dict", "validate", "report_path", "is_report"]
