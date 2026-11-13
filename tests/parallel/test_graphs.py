# SPDX-License-Identifier: Apache-2.0
"""PeriodGraph / GraphCache on CPU (eager and Warp CPU graphs). The CUDA host-sync test is test_graph_syncs_gpu.py;
CPU graphs by default, on a contact scene and with the host-read guard: test_cpu_graphs.py."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.core.errors import GraphCaptureError
from alloy.parallel.graphs import GraphCache, PeriodGraph


@wp.kernel
def _axpy(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i] + 1.0


class PingPong:
    """Two buffers and a period of ``k`` substeps, each reading one buffer and writing the other, then swapping."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.a = wp.zeros(8, dtype=wp.float32, device="cpu")
        self.b = wp.zeros(8, dtype=wp.float32, device="cpu")

    def period(self) -> None:
        for _ in range(self.k):
            wp.launch(_axpy, dim=8, inputs=[self.a, self.b], device="cpu")
            self.a, self.b = self.b, self.a

    def buffers(self):
        return (self.a, self.b)


def test_eager_runs_every_launch():
    pp = PingPong(2)
    g = PeriodGraph(pp.period, "cpu", use_graph=False)  # the CPU default is a graph since SCR-WP8-5 (test_cpu_graphs)
    assert not g.use_graph
    for _ in range(5):
        g.launch()
    assert g.mode == "eager" and g.launches == 5 and g.captures == 0
    np.testing.assert_array_equal(pp.a.numpy(), np.full(8, 10.0))


def test_cpu_graph_replays_and_matches_eager():
    eager, graph = PingPong(4), PingPong(4)
    ge = PeriodGraph(eager.period, "cpu", use_graph=False)
    gg = PeriodGraph(graph.period, "cpu", use_graph=True, buffers=graph.buffers)
    for _ in range(6):
        ge.launch()
        gg.launch()
    assert gg.mode == "graph" and gg.captures == 1 and gg.host_syncs == 0
    np.testing.assert_array_equal(graph.a.numpy(), eager.a.numpy())
    np.testing.assert_array_equal(graph.a.numpy(), np.full(8, 24.0))


def test_odd_swaps_are_refused():
    pp = PingPong(3)
    g = PeriodGraph(pp.period, "cpu", use_graph=True, buffers=pp.buffers)
    with pytest.raises(GraphCaptureError, match="odd number"):
        g.launch()


def test_a_zero_length_buffer_does_not_break_the_capture():
    """S3-001 (Gemini audit of 0.1.2): a scene whose robots have no degrees of freedom has zero-length joint arrays,
    whose ``ptr`` is None; the swap check read int(None) and the first GPU step raised a TypeError. Here on the CPU,
    the same capture path (an empty buffer beside two that swap): it captures, replays and keeps the result."""
    pp = PingPong(4)
    empty = wp.zeros(0, dtype=wp.float32, device="cpu")
    assert empty.ptr is None
    g = PeriodGraph(pp.period, "cpu", use_graph=True, buffers=lambda: (*pp.buffers(), empty, None))
    for _ in range(3):
        g.launch()
    assert g.mode == "graph" and g.captures == 1
    np.testing.assert_array_equal(pp.a.numpy(), np.full(8, 12.0))


def test_invalidate_recaptures():
    pp = PingPong(2)
    g = PeriodGraph(pp.period, "cpu", use_graph=True, buffers=pp.buffers)
    g.launch()
    g.invalidate()
    assert g.mode == "eager"
    g.launch()
    assert g.captures == 2 and g.launches == 2


def test_graph_cache_one_graph_per_key():
    pp = PingPong(2)
    cache = GraphCache("cpu", use_graph=True, buffers=pp.buffers)
    g1 = cache.get((1.25e-3, 1), pp.period)
    assert cache.get((1.25e-3, 1), pp.period) is g1
    g2 = cache.get((6.25e-4, 2), pp.period)
    assert g2 is not g1 and len(cache) == 2
    g1.launch()
    g2.launch()
    cache.invalidate()
    assert g1.mode == g2.mode == "eager" and cache.host_syncs == 0
