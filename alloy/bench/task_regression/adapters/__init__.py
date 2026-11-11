# SPDX-License-Identifier: Apache-2.0
"""Per-task adapters: each runs a task's small check through the task's own code and returns its metrics.

An adapter is `fn(seeds, device) -> dict[str, float]`; it imports the task's modules inside the call, so listing the
registry never imports a solver. Metric names carry their unit (`_mm`, `_N`, `_rate`)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

WORK_ENV = "ALLOY_TASKREG_WORK"    # set by run.measure: this task's folder in the run's output (report dir)


def keep(src: Path, name: str = "") -> None:
    """Copy a small result file (per-world rows) into this task's folder of the run, so the owner of a regression sees
    which worlds changed. Does nothing outside a suite run."""
    dest = os.environ.get(WORK_ENV)
    if not dest or not src.is_file():
        return
    Path(dest).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, Path(dest) / (name or src.name))


def keep_json(name: str, obj: Any) -> None:
    """Write per-world rows the adapter computed itself into this task's folder of the run (as `keep`)."""
    dest = os.environ.get(WORK_ENV)
    if not dest:
        return
    Path(dest).mkdir(parents=True, exist_ok=True)
    (Path(dest) / name).write_text(json.dumps(obj, indent=1, default=str))
