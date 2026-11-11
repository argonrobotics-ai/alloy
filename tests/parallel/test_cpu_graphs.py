# SPDX-License-Identifier: Apache-2.0
"""CPU graph replay by default (SCR-WP8-5 Ask 1): PeriodGraph and the probe on Warp CPU.

- ``cpu_graphs_supported`` (one trial capture per process) turns ``use_graph=None`` on for the CPU; without it the
  default stays eager.
- A raw box pile (contacts from the first step) replays bitwise equal to eager over several periods, also after an
  invalidate and recapture.
- The odd-swap guard and the CPU host-read guard (``array.numpy()`` / ``wp.synchronize*`` while recording) refuse the
  capture; the device is not left capturing.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.core.errors import GraphCaptureError
from alloy.parallel import graphs
from alloy.parallel.graphs import PeriodGraph, cpu_graphs_supported, graphs_by_default
from alloy.parallel.probe import StepTarget, TimingOptions, _graph_on
from alloy.parallel.rungs import RawConfig, RawRung

PERIODS = 4
PILE = RawConfig(dt=2.5e-3)  # 8 substeps per 20 ms period


@wp.kernel
def _inc(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i] + 1.0


class PingPong:
    def __init__(self, k: int) -> None:
        self.k = k
        self.a = wp.zeros(4, dtype=wp.float32, device="cpu")
        self.b = wp.zeros(4, dtype=wp.float32, device="cpu")

    def period(self) -> None:
        for _ in range(self.k):
            wp.launch(_inc, dim=4, inputs=[self.a, self.b], device="cpu")
            self.a, self.b = self.b, self.a

    def buffers(self):
        return (self.a, self.b)


def test_warp_records_cpu_graphs():
    assert cpu_graphs_supported()  # the installed Warp 1.18 records CPU graphs (APIC)
    assert graphs_by_default("cpu")


def test_cpu_default_follows_support(monkeypatch):
    pp = PingPong(2)
    assert PeriodGraph(pp.period, "cpu").use_graph
    monkeypatch.setitem(graphs._CPU_GRAPHS, "ok", False)
    g = PeriodGraph(pp.period, "cpu")
    assert not g.use_graph
    g.launch()
    assert g.mode == "eager" and pp.a.numpy().tolist() == [2.0] * 4


def test_probe_defaults_to_cpu_graphs_for_graph_safe_targets():
    target = StepTarget(step=lambda: None, nworld=1, sim_dt=0.02, device="cpu")
    assert _graph_on(target, TimingOptions())
    assert not _graph_on(StepTarget(step=lambda: None, nworld=1, sim_dt=0.02, device="cpu", graph_safe=False),
                         TimingOptions())
    assert not _graph_on(target, TimingOptions(use_graph=False))


def test_default_odd_swaps_are_refused():
    pp = PingPong(3)
    g = PeriodGraph(pp.period, "cpu", buffers=pp.buffers)
    with pytest.raises(GraphCaptureError, match="odd number"):
        g.launch()
    assert not wp.get_device("cpu").is_capturing


@pytest.mark.parametrize("read", ["numpy", "sync"])
def test_host_read_inside_cpu_capture_is_refused(read):
    pp = PingPong(2)
    numpy = wp.array.numpy

    def period():
        pp.period()
        if read == "numpy":
            pp.a.numpy()
        else:
            wp.synchronize_device("cpu")

    g = PeriodGraph(period, "cpu", buffers=pp.buffers)
    with pytest.raises(GraphCaptureError, match="host read"):
        g.launch()
    assert g.host_syncs == 1 and g.mode == "eager" and not wp.get_device("cpu").is_capturing
    assert wp.array.numpy is numpy and pp.a.numpy().tolist() == [0.0] * 4  # recorded only, nothing ran


def _pile() -> RawRung:
    return RawRung("box_pile", 4, 2, "cpu", PILE)


def _state(rung: RawRung) -> list[np.ndarray]:
    d = rung.solver.mjw_data
    return [a.numpy().copy() for a in (rung.s0.body_q, rung.s0.body_qd, d.qpos, d.qvel, d.qacc_warmstart)]


def _assert_bitwise(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), f"array {i} differs"


def test_pile_graph_replay_is_bitwise_equal_to_eager():
    eager, graph = _pile(), _pile()
    ge = PeriodGraph(eager.step, "cpu", use_graph=False)
    gg = PeriodGraph(graph.step, "cpu", buffers=lambda: (graph.s0.body_q, graph.s1.body_q))
    assert gg.use_graph
    for p in range(PERIODS):
        ge.launch()
        gg.launch()
        _assert_bitwise(_state(graph), _state(eager))
        if p == 1:  # invalidate + recapture mid-run keeps the bits
            gg.invalidate()
            assert gg.mode == "eager"
    assert gg.mode == "graph" and gg.captures == 2 and gg.host_syncs == 0 and gg.launches == PERIODS
    assert int(graph.solver.mjw_data.nacon.numpy()[0]) > 0  # the pile is in contact
    assert int(graph.step_count.numpy()[0]) == PERIODS * PILE.substeps  # the device counter ran on every replay
