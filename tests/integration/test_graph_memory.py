# SPDX-License-Identifier: Apache-2.0
"""A period graph that runs out of device memory (WP13's combination sweep, F10: MPM coupled to another entry at 64
worlds failed creating its graph on an L4) shrinks to fewer exchanges per graph, and a one-exchange graph that does
not fit runs the key eagerly; the results equal eager bitwise. CPU, the graph budget tests' cloth scene, with the
allocation failure raised from the captured body."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alloy.integration import graph_budget as B
from alloy.integration import simulation as sim_mod
from tests.integration.test_graph_budget import _eager_and_graph

OOM = "Graph creation error: Warp CUDA error 2: out of memory"


def _fail_above(monkeypatch, limit: int, where: str) -> None:
    """Graphs of more than ``limit`` exchanges raise the L4's allocation failure: inside the captured body
    (``where`` "body"), or after the whole body was recorded, where the L4 failed (``wp_cuda_graph_create_exec``,
    "instantiate"; the body's host side ran once, as in a capture that succeeds)."""
    from alloy.core.errors import GraphCaptureError  # noqa: PLC0415
    from alloy.parallel.graphs import PeriodGraph  # noqa: PLC0415

    body, capture, seen = sim_mod.IntegratedSimulation._period_body, PeriodGraph._capture, []

    def period_body(self, dt, n):
        seen.append(n)
        if where == "body" and n > limit:
            raise RuntimeError(OOM)
        return body(self, dt, n)

    def instantiate(graph):
        capture(graph)
        if where == "instantiate" and seen and seen[-1] > limit:
            graph._graph = None
            raise GraphCaptureError(f"capturing the control period on cpu failed: {OOM}")

    monkeypatch.setattr(sim_mod.IntegratedSimulation, "_period_body", period_body)
    monkeypatch.setattr(PeriodGraph, "_capture", instantiate)


def test_out_of_memory_reads_through_the_cause_chain() -> None:
    try:
        try:
            raise RuntimeError(OOM)
        except RuntimeError as inner:
            raise ValueError("capturing the control period on cuda:0 failed") from inner
    except ValueError as exc:
        assert B.out_of_memory(exc)
    assert not B.out_of_memory(RuntimeError("host sync inside a captured control period")) and not B.out_of_memory(None)


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
@pytest.mark.parametrize("where", ["body", "instantiate"])
def test_a_graph_out_of_memory_halves_its_chunk(monkeypatch, caplog, where):
    """Nodes allow 4 exchanges per graph; graphs of more than 2 run out of memory: period 3's 4-exchange graph fails
    before running, the period runs eagerly, and period 4 runs 2-exchange graphs. Same result as eager."""
    _fail_above(monkeypatch, 2, where)
    eager, graph, same = _eager_and_graph(monkeypatch, lambda g: B.GRAPH_NODE_BUDGET // 5, periods=4)
    n, stats = graph.clock.steps_per_period, graph.graph_stats()
    assert n >= 4 and stats["memory_chunks"] == [2] and stats["memory_eager"] == 0
    assert stats["launches"] == n + n // 2                     # period 2: one-exchange graphs; period 4: pairs
    assert graph.steps == eager.steps and same
    assert sum("does not fit in device memory" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
@pytest.mark.parametrize("where", ["body", "instantiate"])
def test_a_one_exchange_graph_out_of_memory_runs_the_key_eagerly(monkeypatch, where):
    _fail_above(monkeypatch, 0, where)
    eager, graph, same = _eager_and_graph(monkeypatch, lambda g: B.GRAPH_NODE_BUDGET // 5, periods=3)
    stats = graph.graph_stats()
    assert stats["memory_eager"] == 1 and stats["launches"] == 0 and stats["memory_chunks"] == []
    assert graph.steps == eager.steps and same


class _Graph:
    dropped = 0

    def invalidate(self) -> None:
        self.dropped += 1


def test_release_hands_the_device_back_what_the_process_holds_unused(monkeypatch) -> None:
    """On CUDA: the failed graph and the simulation's other graphs are dropped, dropped simulations are collected, the
    graph pool is trimmed, the free memory is read with the memory pool's unused blocks handed back, and the error the
    failed allocation leaves for the next CUB native is absorbed (SCR-OSS22-9). On the CPU: the drop only."""
    import alloy.parallel.counters as counters  # noqa: PLC0415

    done: list[str] = []
    cuda = SimpleNamespace(is_cuda=True, __str__=lambda s: "cuda:0")
    monkeypatch.setattr(B.wp, "get_device", lambda d: cuda if d == "cuda:0" else SimpleNamespace(is_cuda=False))
    monkeypatch.setattr(B.gc, "collect", lambda: done.append("collect"))
    monkeypatch.setattr(counters, "trim_graph_pool", lambda d: done.append("trim graph pool"))
    monkeypatch.setattr(B, "free_bytes", lambda d: done.append("trim pool, read") or 7)
    monkeypatch.setattr(B, "absorb_stale_error", lambda d: done.append("absorb the stale CUB error"))
    g, cache = _Graph(), SimpleNamespace(invalidate=lambda: done.append("drop the other graphs"))
    assert B.release("cuda:0", g, cache) == 7 and g.dropped == 1
    assert done == ["drop the other graphs", "collect", "trim graph pool", "trim pool, read",
                    "absorb the stale CUB error"]
    assert B.release("cuda:0", g) == 7 and len(done) == 9           # no cache: the failed graph only

    def launch(graph):                                              # the frame that holds the graph when it fails
        raise RuntimeError(OOM)

    try:
        try:
            launch(object())
        except RuntimeError as inner:
            raise ValueError("capturing the control period on cuda:0 failed") from inner
    except ValueError as exc:
        assert exc.__cause__.__traceback__.tb_next.tb_frame.f_locals       # launch's frame still holds its graph
        B.release("cuda:0", g, None, exc)
        assert not exc.__cause__.__traceback__.tb_next.tb_frame.f_locals   # cleared: the graph can be freed
    assert B.release("cpu", g, cache) is None and len(done) == 13


def test_the_step_down_asks_for_room_to_step_eagerly(monkeypatch) -> None:
    """``_out_of_memory`` records the step down, then asks whether the eager period that follows has room, with the
    free bytes ``release`` read; a refusal there leaves the record in place."""
    import alloy  # noqa: PLC0415
    from alloy.integration.mpm_memory import MPMMemoryError  # noqa: PLC0415

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("b", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.0505), mass=0.2)
    eng, asked = alloy.build(scene, device="cpu").engine, []
    monkeypatch.setattr(sim_mod, "release", lambda device, graph, cache, exc: 5 if cache is eng._graphs else None)
    monkeypatch.setattr(sim_mod, "room_to_step", lambda built, device, free: asked.append((built, str(device), free)))
    with pytest.warns(RuntimeWarning, match="step graph"):
        eng._out_of_memory("k", 4, _Graph(), RuntimeError(OOM))
    assert asked == [(eng.built, str(eng.model.device), 5)] and eng.memory_chunk == {"k": 2}

    def refuse(built, device, free):
        raise MPMMemoryError("no room", need=2, free=1, fits=0)

    monkeypatch.setattr(sim_mod, "room_to_step", refuse)
    with pytest.raises(MPMMemoryError):
        eng._out_of_memory("k", 1, _Graph(), RuntimeError(OOM))
    assert "k" in eng.memory_eager and len(eng.memory_notes) == 2
