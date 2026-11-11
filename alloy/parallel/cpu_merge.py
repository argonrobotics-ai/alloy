# SPDX-License-Identifier: Apache-2.0
"""Splitting calls and joining results for CPU worlds in worker processes (``alloy.build(..., processes=N)``).

Each worker holds one consecutive block of global worlds (``alloy.parallel.multigpu.world_block``). Values given per
world (a leading axis of the scene's world count) are cut to each block; results with a leading world axis (state)
or a world axis after the frames (trajectories) are joined in block order; world ids in health and check results
are local in a worker and become global here. Pure functions, no processes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from alloy.api.errors import SimulationError
from alloy.parallel.multigpu import WorldBlock, world_block


def blocks(worlds: int, processes: int) -> list[WorldBlock]:
    """The world block of every process, in rank order. Example: ``blocks(5, 2)`` -> offsets 0 and 3."""
    return [world_block(rank, processes, worlds) for rank in range(processes)]


def owner(parts: Sequence[WorldBlock], world: int) -> tuple[int, int]:
    """(rank, local world) of global ``world``. Example: ``owner(blocks(5, 2), 4)`` -> ``(1, 1)``."""
    for rank, b in enumerate(parts):
        if b.offset <= world < b.offset + b.count:
            return rank, world - b.offset
    total = parts[-1].offset + parts[-1].count
    raise SimulationError(f"world {world} is outside 0..{total - 1}", "pass world=0 for the first", field="world")


def cut(value: Any, block: WorldBlock, worlds: int, what: str, *, row_ndim: int = 0) -> Any:
    """``value`` for one block: one row per world (``row_ndim`` + 1 dimensions, leading axis ``worlds``) is cut to
    the block; one row for every world (``row_ndim`` dimensions) passes unchanged.

    Example: ``cut(np.arange(4.0), WorldBlock(2, 2), 4, "q")`` -> ``[2., 3.]``; ``cut(0.5, ...)`` -> ``0.5``."""
    if value is None:
        return None
    a = np.asarray(value)
    if a.ndim == row_ndim:
        return value
    if a.ndim != row_ndim + 1 or a.shape[0] != worlds:
        raise SimulationError(f"{what}: got shape {a.shape} for {worlds} worlds",
                              f"pass one value for every world, or one per world (leading axis {worlds})", field=what)
    return a[block.offset:block.offset + block.count]


def cut_map(values: Mapping[str, Any], block: WorldBlock, worlds: int) -> dict[str, Any]:
    """Joint targets, controls: each a scalar for every world or a (worlds,) array, cut to ``block``."""
    return {k: cut(v, block, worlds, k) for k, v in values.items()}


def cut_ramps(targets: Mapping[str, Any] | None, block: WorldBlock, worlds: int) -> dict[str, Any] | None:
    """``run(targets=...)``: a hold value or a (start, end) ramp per joint, each end a scalar or (worlds,)."""
    if targets is None:
        return None
    out: dict[str, Any] = {}
    for k, v in targets.items():
        if isinstance(v, tuple) and len(v) == 2:
            out[k] = (cut(v[0], block, worlds, k), cut(v[1], block, worlds, k))
        else:
            out[k] = cut(v, block, worlds, k)
    return out


def join_state(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``sim.state()`` of every worker -> one, worlds in block order (``t`` from the first)."""
    out: dict[str, Any] = {"t": parts[0]["t"]} if "t" in parts[0] else {}
    for key in parts[0]:
        if key != "t":
            out[key] = np.concatenate([np.asarray(p[key]) for p in parts], axis=0)
    return out


def join_frames(parts: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Trajectory data of every worker -> one: ``t`` from the first, the world axis (axis 1) joined."""
    out = {"t": np.asarray(parts[0]["t"])}
    for key in parts[0]:
        if key != "t":
            out[key] = np.concatenate([np.asarray(p[key]) for p in parts], axis=1)
    return out


def join_per_world(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Dicts of per-world arrays (``sim.heat``) -> one: each key's leading world axis joined."""
    return {k: np.concatenate([np.atleast_1d(np.asarray(p[k])) for p in parts], axis=0) for k in parts[0]}


def _counts(parts: Sequence[Mapping[str, Any]], spans: Sequence[WorldBlock], key: str, what: str) -> list[int] | None:
    """Per-world counts ``key`` of every worker, in rank order (None when no worker has them); each worker's list has
    one entry per world of its block (WP8's review: a short list would shift every later world id)."""
    counts = [p.get(key) for p in parts]
    if all(c is None for c in counts):
        return None
    if any(c is None for c in counts):
        raise SimulationError(f"workers disagree on {what}", "report this: every worker builds the same scene")
    if any(len(c) != b.count for c, b in zip(counts, spans, strict=True)):
        raise SimulationError(f"a worker's {key} does not have one entry per world of its block",
                              "report this: every worker builds the same scene")
    return [int(n) for c in counts for n in c]


def _retired(parts: Sequence[Mapping[str, Any]], spans: Sequence[WorldBlock]) -> list[int] | None:
    return _counts(parts, spans, "retired_particles", "retired particles (some have a liquid, some not)")


def join_health(parts: Sequence[Mapping[str, Any]], spans: Sequence[WorldBlock]) -> dict[str, Any]:
    """``sim.health()`` of every worker -> one with global world ids."""
    out = dict(parts[0])
    out["worlds"] = [row for p in parts for row in p["worlds"]]
    for key in ("invalid_worlds", "degraded_worlds"):
        out[key] = [b.offset + int(w) for p, b in zip(parts, spans) for w in p.get(key, [])]
    out["ok"] = all(bool(p["ok"]) for p in parts)
    out["monitored"] = any(bool(p.get("monitored", False)) for p in parts)
    if any("retired_particles" in p for p in parts):
        out["retired_particles"] = _retired(parts, spans)
    if any("refined_periods" in p for p in parts):            # WP4: periods each world re-ran finer (coupling)
        out["refined_periods"] = _counts(parts, spans, "refined_periods", "refined periods (some couple, some not)")
    return out


def join_check(parts: Sequence[Mapping[str, Any]], spans: Sequence[WorldBlock]) -> dict[str, Any]:
    """``sim.check()`` of every worker -> one: problems with global world ids; each worker's full ``report`` (when
    ``alloy.check`` made it) kept in ``reports``, in rank order."""
    out = {k: v for k, v in parts[0].items() if k != "report"}
    out["ok"] = all(bool(p["ok"]) for p in parts)
    problems = []
    for p, b in zip(parts, spans):
        for row in p.get("problems", []):
            row = dict(row)
            if isinstance(row.get("world"), (int, np.integer)):
                row["world"] = b.offset + int(row["world"])
            problems.append(row)
    out["problems"] = problems
    if any("invalid_worlds" in p for p in parts):
        out["invalid_worlds"] = [b.offset + int(w) for p, b in zip(parts, spans) for w in p.get("invalid_worlds", [])]
    if any("retired_particles" in p for p in parts):
        out["retired_particles"] = _retired(parts, spans)
    if any("report" in p for p in parts):
        out["reports"] = [p.get("report") for p in parts]
    return out


def join_meta(parts: Sequence[Mapping[str, Any]], spans: Sequence[WorldBlock], wall_s: float) -> dict[str, Any]:
    """Trajectory ``meta`` of every worker -> one: the whole scene's worlds and health, the parent's wall time, and
    ``processes`` / ``worker_wall_s`` (each worker's own run time). ``physics_steps`` is the most any worker took;
    ``worker_physics_steps`` each one's (they differ when an adaptive clock follows a process's own worlds)."""
    out = dict(parts[0])
    steps = [int(p.get("physics_steps", 0)) for p in parts]
    out["physics_steps"] = max(steps)
    out["worker_physics_steps"] = steps
    out["worlds"] = sum(b.count for b in spans)
    out["wall_s"] = float(wall_s)
    out["health"] = join_health([p["health"] for p in parts], spans)
    out["processes"] = len(parts)
    out["worker_wall_s"] = [float(p.get("wall_s", float("nan"))) for p in parts]
    return out
