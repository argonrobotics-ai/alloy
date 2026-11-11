# SPDX-License-Identifier: Apache-2.0
"""Test helpers for alloy.parallel (WP8): synthetic profiles, a stub assembler and a fake Simulation factory."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alloy.parallel.types import WorldProfile

GB = 10**9

T12_LIKE = dict(scene_hash="s" * 16, solver_config_hash="c" * 16, newton_commit="ac7f7830", gpu_model="L4", dt=0.02,
                nv=12, n_particles=0, island_nv_max=12, chain_len_max=1, rows_p99=40, contacts_p99=8, pairs_peak=20,
                t_floor_s=3.27e-3, c_w_s=27.9e-6, iters_p50=4.0, iters_p99=9.0, bytes_fixed=50_000_000,
                bytes_per_world=210_000)  # floor and per-world cost of T12 on an L4 (parallel §3)


def profile(**overrides: Any) -> WorldProfile:
    """A WorldProfile with T12-like numbers, fields overridden by keyword."""
    return WorldProfile(**{**T12_LIKE, **overrides})


@dataclass
class StubBuilt:
    """Stands in for BuiltScene in packer tests (only the fields WorldPacker.build reads)."""

    scene: Any
    device: str
    world_count: int
    world_id_offset: int
    capacities: Any = None
    world_ids: Any = None  # wp.array of global ids for repacked packs (SCR-WP8-2)


@dataclass
class StubAssembler:
    """Records every assemble() call and returns a StubBuilt."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def assemble(self, scene: Any, device: str, *, world_count: int | None = None, world_id_offset: int = 0,
                 capacities: Any = None, world_ids: Any = None) -> StubBuilt:
        import warp as wp

        self.calls.append(dict(scene=scene, device=device, world_count=world_count, world_id_offset=world_id_offset,
                               capacities=capacities, world_ids=world_ids))
        ids = None if world_ids is None else wp.array(list(world_ids), dtype=wp.int32, device=device)
        return StubBuilt(scene, device, world_count or 1, world_id_offset, capacities, ids)


def fake_sim_factory(built: Any) -> dict[str, Any]:
    """Simulation stand-in: remembers what it wraps."""
    return {"built": built}


def record_metric(metric: str, value: Any, status: str, command: str, **context: Any) -> Path:
    """Append one metric row to ``bench/results/wp8/metrics.jsonl`` (copied back by scripts/gpu_run.sh)."""
    import warp as wp

    row = {"metric": metric, "value": value, "status": status, "command": command,
           "commit": os.environ.get("ALLOY_COMMIT") or _git_head(), "host": platform.node(),
           "gpu": wp.get_device(os.environ.get("ALLOY_DEVICE", "cpu")).name,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **context}
    path = Path("bench/results/wp8/metrics.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return path


def _git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
