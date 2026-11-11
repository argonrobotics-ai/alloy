# SPDX-License-Identifier: Apache-2.0
"""Graph-size budget for captured periods (WP5's MuJoCo + VBD segfault, SCR-DC5-4): the chunk arithmetic; a CPU
period whose chunk comes from a (patched) node count equals eager bitwise, and so does a key whose one exchange is
over the budget (dropped unlaunched, run eagerly)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.integration import graph_budget as B
from alloy.integration import simulation as sim_mod
from alloy.testing import integration as I
from alloy.testing import scene as S

L4_CRASH_NODES = 1_194_978     # WP5's 16-exchange ADMM graph: SIGSEGV in cudaGraphLaunch (L4, driver 580)
SHARED_L4_CRASH_NODES = 448_117   # 6 exchanges of it: launched on an idle L4, SIGSEGV on a shared one (WP5)


def test_pow2_floor():
    assert [B.pow2_floor(x) for x in (0, 1, 2, 3, 6, 64, 65)] == [1, 1, 2, 2, 4, 64, 64]


def test_chunk_for_stays_within_the_budget():
    per_exchange = L4_CRASH_NODES // 16
    assert B.GRAPH_NODE_BUDGET * 4 < SHARED_L4_CRASH_NODES < L4_CRASH_NODES
    assert B.chunk_for(per_exchange, 16) == 1                        # 74,686 nodes: one exchange per graph
    assert B.chunk_for(per_exchange // 3, 16) == 2 and 2 * (per_exchange // 3) <= B.GRAPH_NODE_BUDGET
    assert B.chunk_for(B.GRAPH_NODE_BUDGET + 1, 16) == 1           # over budget: chunk 1 (that key then runs eager)
    assert B.chunk_for(1_000, 64) == 64 and B.chunk_for(1_000, 6) == 4
    assert B.chunk_for(None, 64) == 64 and B.chunk_for(None, 6) == 4  # no count (CPU graph): the cap


def test_no_count_on_cpu():
    assert B.graph_nodes(None) is None and not B.countable("cpu")


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_cpu_budget_chunk_equals_eager(monkeypatch):
    """With a (patched) node count that allows 2 exchanges per graph: the first replayed period runs one-exchange
    graphs, counted once, the next runs 2-exchange graphs, and the result equals eager bitwise."""
    counted = []

    def nodes(graph):
        counted.append(graph)
        return B.GRAPH_NODE_BUDGET // 3                      # 2 exchanges fit, 4 do not

    periods = 3
    eager, graph, same = _eager_and_graph(monkeypatch, nodes, periods)
    n, stats = graph.clock.steps_per_period, graph.graph_stats()
    assert n > 2 and stats["chunks"] == [1, 2] and len(counted) == 1 and counted[0] is not None
    assert stats["sizes"] == [(2, B.GRAPH_NODE_BUDGET // 3)] and stats["eager_keys"] == []
    assert stats["launches"] == n + (periods - 2) * (n // 2)     # period 1: n one-exchange graphs; then n / 2
    assert graph.steps == eager.steps and same


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_cpu_over_budget_key_runs_eager(monkeypatch):
    """One exchange over the (patched) budget: the key's one-exchange graph is captured and counted but never
    launched (WP5's ordered-VBD exchange segfaulted in its first launch); the key runs eagerly from then on and the
    result equals eager bitwise."""
    over = B.GRAPH_NODE_BUDGET + 1
    eager, graph, same = _eager_and_graph(monkeypatch, lambda graph: over, periods=3)
    stats = graph.graph_stats()
    assert stats["captures"] == 1 and stats["launches"] == 0 and stats["eager_keys"] == [over]
    assert graph.host_syncs == 0 and graph.steps == eager.steps and same



def test_eager_scene_reason():
    """SCR-WP11-5: the accurate cloth profile's effect names a reason to skip graphs; other effects do not."""
    from alloy.solvers.vbd.cloth_profile import ACCURATE_EFFECT, ACCURATE_ITERATIONS

    assert str(ACCURATE_ITERATIONS) in B.eager_scene_reason(["rk4", ACCURATE_EFFECT])
    assert B.eager_scene_reason(["rk4"]) is None and B.eager_scene_reason([]) is None


@pytest.mark.parametrize("cloth_profile, graphs", [("default", True), ("accurate", False)])
def test_accurate_cloth_scene_skips_graphs(caplog, cloth_profile, graphs):
    """A scene carrying the accurate cloth profile runs eagerly even under graph="on" (its exchange is known to be over
    the node budget, so the capture attempt is skipped, with a warning); the default profile keeps its graphs."""
    from alloy.integration import build_simulation
    from alloy.route.router import Router
    from alloy.scene.assembler import SceneAssembler

    card, lib = S.scene([S.cloth()], [S.place(S.cloth().card.name, z=0.3)], worlds=2)
    with isolated_registry(empty=True), caplog.at_level("INFO", logger="alloy.integration"):
        built = SceneAssembler(lib, Router(), cloth_profile=cloth_profile).assemble(card, "cpu")
        sim = build_simulation(built, "standard", graph="on")
    assert sim.use_graph is graphs
    warned = [r for r in caplog.records if "graphs off for this scene" in r.getMessage()]
    assert [r.levelname for r in warned] == ([] if graphs else ["WARNING"])

def _eager_and_graph(monkeypatch, nodes, periods: int):
    """Eager and graph runs of the CPU cloth with ``nodes`` as the graph node count; (eager, graph, bitwise equal)."""
    from alloy.integration import build_simulation

    card, lib = _cloth()
    monkeypatch.setattr(sim_mod, "countable", lambda device: True)
    monkeypatch.setattr(sim_mod, "graph_nodes", nodes)
    out = {}
    with isolated_registry(empty=True):
        for graph in ("off", "on"):
            sim = build_simulation(I.assemble(card, lib, "cpu", 2), "standard", graph=graph)
            for _ in range(periods):
                sim.step_period()
            out[graph] = (sim, I.digests(sim))
    (eager, d_eager), (graph, d_graph) = out["off"], out["on"]
    return eager, graph, np.array_equal(d_eager, d_graph)


def _cloth():
    card, lib = {n: (c, lb) for n, c, lb in S.mixed_scenes()}["cloth_on_table"]
    return dataclasses.replace(card, control_dt=0.005), lib        # the CPU budget of test_graph_eager
