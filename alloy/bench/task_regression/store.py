# SPDX-License-Identifier: Apache-2.0
"""The task-regression suite's last recorded results: `last_results.json` beside this module.

One entry per task: the metric values, the build that produced them, and the run they came from. Changing an entry
needs a reason, recorded in `history` (the same rule as the analytic baseline, alloy.bench.baseline), so a re-baseline
after a ruled change is traceable and a silent drift is not possible. Tolerances are not stored here: they live in the
registry's code, where loosening one is a reviewed change.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alloy.bench.task_regression.types import LastResult

log = logging.getLogger("alloy.bench.task_regression")

SCHEMA = "alloy.bench.task_regression.last/1"
DEFAULT_PATH = Path(__file__).with_name("last_results.json")


def load(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"schema": SCHEMA, "tasks": {}, "history": []}
    data = json.loads(p.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {data.get('schema')!r} != {SCHEMA}")
    return data


def last_result(task_id: str, path: Path = DEFAULT_PATH) -> LastResult | None:
    entry = load(path)["tasks"].get(task_id)
    if entry is None:
        return None
    return LastResult({k: float(v) for k, v in entry["values"].items()}, entry["build"], entry["date"], entry["source"])


def record(task_id: str, values: Mapping[str, float], *, build: str, source: str, reason: str,
           path: Path = DEFAULT_PATH) -> None:
    """Set a task's last recorded result; replacing an existing one needs a reason (kept in `history`)."""
    if not reason:
        raise ValueError("recording a task's last result needs a reason")
    data = load(path)
    previous = data["tasks"].get(task_id)
    data["tasks"][task_id] = {"values": {k: float(v) for k, v in values.items()}, "build": build,
                              "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source": source}
    data["tasks"] = dict(sorted(data["tasks"].items()))
    data["history"].append({"date": data["tasks"][task_id]["date"], "task": task_id, "reason": reason,
                            "build": build, "replaced": None if previous is None else previous["values"]})
    Path(path).write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    log.info("task regression: %s last result %s on %s (%s)", task_id, dict(values), build, reason)


__all__ = ["DEFAULT_PATH", "SCHEMA", "last_result", "load", "record"]
