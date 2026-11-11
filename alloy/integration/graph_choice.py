# SPDX-License-Identifier: Apache-2.0
"""Graph or eager per exchange shape (mixed-speed lever #3, [internal note] notes/mixed-speed-1107.md).

A captured period is not always the faster one. At 4 worlds on an L4, a rigid block on dug sand ran 9.7 s per period
from graphs against 2.3 s eagerly once its clock escalated, and rigid + paste + cloth 3.28 s against 1.40 s (20 ADMM
passes unrolled per exchange). ``GraphChoice`` times each graph key once both ways and keeps the faster:

- eager: the second exchange of the key's first period, which ``IntegratedSimulation`` runs eagerly anyway
  (``_warm_up``, SCR-WP10-3: the first carries the one-time costs; a one-exchange period times its only exchange,
  which can only overstate the eager time), or a whole eager period;
- graph: the key's first period whose graphs were captured and launched before it (a period that captures, or makes
  a graph's first launch, is not timed);
- times are per exchange; a key whose graph exchanges take more than ``margin`` times its eager one runs eagerly from
  then on.

Each timed period is bracketed by a device synchronize: two per key, once. With NATIVE solvers eager equals graphs
bitwise (D9, tests/integration/test_graph_gpu.py), so the choice changes nothing. In a non-NATIVE scene (MuJoCo on the
GPU) it moves results the way ``graph="off"`` does (dig_rigid's block: 0.024 mm against a 0.007 mm run-to-run floor),
and since the choice comes from wall time, two runs of such a scene can take different paths. ``graph="on"`` keeps
graphs on every key without timing (``wiring.build_simulation``); ``"auto"`` times.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import warp as wp

log = logging.getLogger("alloy.integration")

__all__ = ["GraphChoice"]


@dataclass
class GraphChoice:
    """Per graph key: one timed eager exchange, one timed graph period (per exchange), and the keys sent to eager."""

    margin: float = 1.5
    eager_s: dict[Any, float] = field(default_factory=dict)
    graph_s: dict[Any, float] = field(default_factory=dict)
    slow: dict[Any, float] = field(default_factory=dict)      # key -> graph / eager time, for keys sent to eager

    def eager(self, key: Any) -> bool:
        """``key`` was timed slower from graphs: run it eagerly."""
        return key in self.slow

    def what(self, key: Any, graph: Any) -> str | None:
        """What this period times for ``key``: "eager" (its first, eager period), "graph" (a replay of graphs
        launched before), or None."""
        if graph is None:
            return "eager" if key not in self.eager_s else None
        if key in self.eager_s and key not in self.graph_s and getattr(graph, "launches", 0) > 0 \
                and getattr(graph, "_graph", None) is not None:
            return "graph"
        return None

    @contextmanager
    def timed(self, key: Any, what: str | None, device: Any, exchanges: int = 1) -> Iterator[None]:
        """Time the block as ``what`` for ``key``, per exchange (``exchanges`` run inside it)."""
        if what is None:
            yield
            return
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        yield
        wp.synchronize_device(device)
        self._record(key, what, (time.perf_counter() - t0) / max(int(exchanges), 1))

    def _record(self, key: Any, what: str, seconds: float) -> None:
        if what == "eager":
            self.eager_s[key] = seconds
            return
        self.graph_s[key] = seconds
        ratio = seconds / max(self.eager_s[key], 1e-9)
        if ratio > self.margin:
            self.slow[key] = ratio
            log.warning("graph key %s: a graph exchange took %.3g s against %.3g s eagerly (%.2fx); this key runs "
                        "eagerly", key, seconds, self.eager_s[key], ratio)
        else:
            log.info("graph key %s: graphs %.3g s, eager %.3g s per exchange", key, seconds, self.eager_s[key])

    def stats(self) -> dict[str, Any]:
        return {"timed_keys": len(self.graph_s), "slow_graph_keys": sorted(round(r, 2) for r in self.slow.values())}
