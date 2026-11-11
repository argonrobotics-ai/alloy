# SPDX-License-Identifier: Apache-2.0
"""Row 4 (time to train): the one record every framework runner writes, as JSON lines.

A runner writes a ``run`` header line first, then one ``point`` line per training iteration (or per logging
interval). The scorer (``score.py``) reads only these lines, so every framework is scored by the same code.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA = "alloy.score_train/1"


@dataclass(frozen=True)
class RunHeader:
    """What was trained, on what, with which settings. One per (framework, task, seed) run."""

    framework: str  # "alloy", "isaaclab_physx", "isaaclab_newton", "playground_mjx", "playground_warp", "mjlab"
    task: str  # protocol task key: "go2_flat", "franka_lift", "cube_reorient"
    task_id: str  # the framework's own registered id
    seed: int
    num_envs: int
    variant: str  # "matched" (the protocol's PPO settings) or "native" (the framework's recommended settings)
    gpu: str
    versions: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)  # PPO hyper-parameters as run
    internal_only: bool = False  # True for PhysX / Isaac Sim rows (NVIDIA licence): never published
    kind: str = "run"
    schema: str = SCHEMA


@dataclass(frozen=True)
class Point:
    """One reading during training."""

    iteration: int
    env_steps: int  # environment steps collected so far, summed over all envs
    wall_s: float  # seconds since the first training iteration started (excludes import, build, compile)
    wall_total_s: float  # seconds since process start (includes them)
    metrics: dict[str, float]  # protocol metric names (PROTOCOL.md), e.g. {"vel_err": 0.21, "reward": 14.2}
    kind: str = "point"


class RunWriter:
    """Appends a run's header and points to a JSON-lines file, flushing each line (a spot VM can vanish)."""

    def __init__(self, path: str | Path, header: RunHeader, process_start: float | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._t_process = time.perf_counter() if process_start is None else process_start
        self._t_train: float | None = None
        self._append(asdict(header))

    def start_training(self) -> None:
        """Call just before the first training iteration: ``wall_s`` counts from here."""
        self._t_train = time.perf_counter()

    def point(self, iteration: int, env_steps: int, metrics: dict[str, float]) -> Point:
        if self._t_train is None:
            raise RuntimeError("call start_training() before point()")
        now = time.perf_counter()
        clean = {k: float(v) for k, v in metrics.items()}
        p = Point(iteration, int(env_steps), now - self._t_train, now - self._t_process, clean)
        self._append(asdict(p))
        return p

    def _append(self, record: dict[str, Any]) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass(frozen=True)
class Run:
    header: RunHeader
    points: list[Point]


def read_runs(path: str | Path) -> Iterator[Run]:
    """Yield every run in a JSON-lines file (several runs may be concatenated). Skips a torn last line."""
    header: RunHeader | None = None
    points: list[Point] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip().strip("\x00")
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a preempted VM can leave a torn tail
        if rec.get("kind") == "run":
            if header is not None:
                yield Run(header, points)
            header, points = RunHeader(**rec), []
        elif rec.get("kind") == "point" and header is not None:
            points.append(Point(**rec))
    if header is not None:
        yield Run(header, points)
