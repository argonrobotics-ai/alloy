# SPDX-License-Identifier: Apache-2.0
"""Health's check schedule in captured period graphs (release red, coordinator ruling 2026-11-09, scr/WP7.md).

``HealthMonitor.update`` decides on the host, from the step index, whether a step runs its checks: the cheap ones every
``every`` steps, the full ones (joints, welds) every ``full_every``. A CUDA graph captures that decision once. Before
this rule ``IntegratedSimulation`` kept one graph per (key, chunk), so every replay ran the checks of the capture's
steps: under the fast preset (cheap checks every 10 steps, captured after the eager warm-up at step 2) no replayed
period checked anything, and a world kicked to 8 km read ``sim.check()["ok"]``; under the standard preset the full
checks ran at the capture's phase, not every ``full_every`` steps.

``pattern(health, start, chunk)`` is what the chunk's steps ``start .. start + chunk - 1`` run: per step 0 (no check),
1 (cheap) or 2 (full), exactly as ``HealthMonitor.update`` decides. It joins the graph's key, so a chunk replays a
graph captured with its own checks. The patterns repeat with the schedule (at most ``full_every`` per key and chunk;
two at one step per period under the fast preset: the check period and the others). Without health, or with its
checks off, the pattern is empty and nothing changes. Example: ``pattern(sim.health, sim.steps, 4)``.

Contract (WP5): ``_kind`` mirrors every host-side branch of ``HealthMonitor.update`` on ``step``; a new one there must
be added here (tests/integration/test_graph_checks.py runs update() eagerly against ``pattern`` over many steps).

``launch`` runs a period's chunks (WP0's review): a pattern graph is captured and its nodes counted before its first
launch, and one over ``graph_node_budget`` (a full-check chunk is the largest) is dropped, the key's per-exchange count
raised so later periods take smaller chunks, and that chunk runs eagerly; running out of device memory after the first
chunk runs the rest of the period eagerly. Each capture is logged with its nodes (graph memory grows with patterns).
"""

from __future__ import annotations

import logging
from typing import Any

from alloy.integration.graph_budget import graph_nodes, out_of_memory

log = logging.getLogger("alloy.integration")

__all__ = ["captured", "chunk_graph", "launch", "pattern", "send_eager", "timed_what"]


def send_eager(sim: Any, key: Any, nodes: int) -> None:
    """``key``'s one exchange is ``nodes`` graph nodes, over the budget: it runs eagerly from now on, and the stepper
    hears of it (``graph_over_budget``, opt-in ``ClockPolicy.graph_budget_rungs``: its clock may take a finer dt_x
    with fewer substeps instead). Both eager paths (``IntegratedSimulation._fits``, ``_sized`` at one exchange)."""
    sim.eager_keys[key] = nodes
    note = getattr(sim.stepper, "graph_over_budget", None)
    if callable(note):
        note(dict(sim.substeps), int(nodes))


def _kind(cfg: Any, health: Any, step: int) -> int:
    if not health.is_check_step(step):
        return 0
    full = cfg.level == "full" and step % max(cfg.full_every, 1) == 0 and "joints" not in cfg.skip
    return 2 if full else 1


def pattern(health: Any, start: int, chunk: int) -> tuple[int, ...]:
    """The check kinds of steps ``start .. start + chunk - 1`` (module docstring); () without checks."""
    cfg = getattr(health, "cfg", None)
    if cfg is None or cfg.level == "off" or not callable(getattr(health, "is_check_step", None)):
        return ()
    return tuple(_kind(cfg, health, start + i) for i in range(chunk))


def chunk_graph(sim: Any, key: Any, chunk: int, dt_x: float) -> Any:
    """``sim``'s (an IntegratedSimulation's) graph of ``chunk`` exchanges from its current step, keyed with their
    checks: ``(key, chunk, pattern)``, captured around ``sim._period_body`` on first use."""
    return sim._graphs.get((key, chunk, pattern(sim.health, sim.steps, chunk)), lambda: sim._period_body(dt_x, chunk))


def captured(sim: Any, key: Any, chunk: int, n: int) -> bool:
    """Every graph the next ``n`` exchanges of ``sim`` launch (``n // chunk`` chunks from its current step) is
    captured already: lever #3 (``graph_choice``) times a period only then, never one that captures a new pattern."""
    graphs = getattr(sim._graphs, "_graphs", {}) if sim._graphs is not None else {}
    for i in range(max(n // max(chunk, 1), 0)):
        g = graphs.get((key, chunk, pattern(sim.health, sim.steps + i * chunk, chunk)))
        if g is None or getattr(g, "_graph", None) is None:
            return False
    return True


def timed_what(sim: Any, key: Any, graph: Any, chunk: int, n: int, done: int) -> str | None:
    """What lever #3 (``sim.graph_choice``) times this period for ``key`` (``GraphChoice.what``), or None: never a
    period whose graphs are not all captured yet (``captured``), since a capture would read as a slow graph."""
    choice = sim.graph_choice
    if choice is None or done or key in sim.eager_keys:
        return None
    return choice.what(key, graph) if graph is None or captured(sim, key, chunk, n - done) else None


def _eager(sim: Any, n: int, dt_x: float) -> None:
    for _ in range(n):
        sim._exchange(dt_x)


def _sized(sim: Any, key: Any, chunk: int, graph: Any) -> bool:
    """Capture a fresh ``graph`` and count its nodes; False (graph dropped, the key's per-exchange count raised, or
    the key sent eager at one exchange) when it is over ``sim.graph_node_budget``. An out-of-memory capture raises."""
    if graph._graph is not None:
        return True
    steps = sim.steps
    try:
        graph._capture()                      # WP8's PeriodGraph: no launch (as IntegratedSimulation._fits)
    finally:
        sim.steps = steps
    nodes = graph_nodes(graph._graph)
    log.info("graph key %s chunk %d: a check pattern captured, %s nodes", key, chunk, nodes)
    if nodes is None or nodes <= sim.graph_node_budget:
        return True
    graph.invalidate()
    if chunk == 1:
        send_eager(sim, key, nodes)
    sim.exchange_nodes[key] = max(sim.exchange_nodes.get(key) or 0, -(-nodes // chunk))
    log.warning("graph key %s: a %d-exchange graph with its checks is %d nodes (budget %d); it runs eagerly, later "
                "periods take smaller chunks", key, chunk, nodes, sim.graph_node_budget)
    return False


def launch(sim: Any, graph: Any, key: Any, n: int, chunk: int, dt_x: float) -> bool:
    """``sim``'s period as ``n // chunk`` graphs, the first ``graph``, each the one for its steps' checks (module
    docstring). False when the first cannot be made or launched for lack of device memory: nothing ran, the caller
    runs the period eagerly."""
    for i in range(n // chunk):
        start = sim.steps                       # the capture runs the Python body (and its counter) once
        g = graph if not i else chunk_graph(sim, key, chunk, dt_x)
        try:
            if not _sized(sim, key, chunk, g):
                _eager(sim, chunk, dt_x)
                continue
            g.launch()
        except Exception as exc:                # noqa: BLE001 - only an allocation failure is handled
            if not out_of_memory(exc):
                raise
            sim.steps = start
            sim._out_of_memory(key, chunk, g, exc)
            if not i:
                return False
            _eager(sim, n - i * chunk, dt_x)
            return True
        sim.steps = start + chunk
    return True

