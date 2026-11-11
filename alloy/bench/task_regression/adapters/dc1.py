# SPDX-License-Identifier: Apache-2.0
"""DC1: the Ethernet scene's ghost-hand sequence on CPU (bench/dc/eth/checks.py `scene_rows`, the S1-S6 rows).

task_dc_eth_swap through the assembler and build_simulation (HealthMonitor on), one world: the plug starts latched in
port 1, holds an 89 N pull, releases with the tab pressed, is inserted into port 3 (no slam), latches and is let go.
`scene_rows` returns the rows `checks.py scene` merges into bench/dc/eth/results.json; the adapter only reads them (no
merge, no tracked file written). About 18 CPU-min (one thread)."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ROW_METRICS = {"SCENE-start": "start_latched", "SCENE-hold": "hold_slip_mm", "SCENE-release": "withdraw_peak_N",
               "SCENE-insert": "insert_peak_N", "SCENE-latched3": "latched_port3", "SCENE-pen": "max_pen_mm",
               "SCENE-invalid": "invalid_worlds"}


def eth_scene(seeds: Sequence[int], device: str) -> dict[str, float]:
    from alloy.bench.task_regression.adapters import keep_json

    rows = _checks().scene_rows(device)
    keep_json("dc1_scene_rows.json", rows)
    out = {ROW_METRICS[r["id"]]: float(r["value"]) for r in rows if r["id"] in ROW_METRICS}
    out["rows_failed"] = float(sum(not r["pass"] for r in rows))
    out["insert_max_speed_mps"] = float(rows[0]["settings"]["metrics"].get("insert_max_speed_mps", 0.0))
    return out


def _checks():  # noqa: ANN202
    """bench/dc/eth/checks.py (bench/dc has no package __init__), loaded by path."""
    spec = importlib.util.spec_from_file_location("dc_eth_checks", REPO / "bench/dc/eth/checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
