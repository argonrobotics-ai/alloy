# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the fit tests: quiet Warp, a results-row recorder (bench/out/fit/results.jsonl)."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pytest
import warp as wp

wp.config.quiet = True
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bench" / "out" / "fit" / "results.jsonl"


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        stamp = REPO / "COMMIT"          # GPU VMs get the tree without .git; the sync writes the commit here
        return stamp.read_text().strip() if stamp.exists() else "unknown"


@pytest.fixture
def record_row(request):
    """Append one metric row (id, value, verdict, test node, commit) for docs/notes/wp11-results.md."""
    def write(metric: str, value: float, verdict: str, **extra) -> None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        row = {"metric": metric, "value": value, "verdict": verdict, "test": request.node.nodeid,
               "commit": _commit(), "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return write
