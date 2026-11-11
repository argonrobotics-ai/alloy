# SPDX-License-Identifier: Apache-2.0
"""Sub-knee packs sharing one GPU on separate streams (parallel §4.3 step 4; the planner's Σu <= 1 assumption).

Two identical packs below their knee, each period a captured graph, run back to back on one stream and then
concurrently through ``PackRunner`` (one stream each). Reported as information next to PAR-10: the concurrent period
should cost less than the sequential one when the packs leave the GPU idle.
"""

from __future__ import annotations

import time

import pytest
import warp as wp

from alloy.parallel.graphs import PeriodGraph
from alloy.parallel.planning import PRIORITY_NORMAL
from alloy.parallel.rungs import RawRung
from alloy.parallel.streams import PackRunner
from alloy.parallel.types import Pack, PackSpec
from alloy.testing.parallel import record_metric

pytestmark = [pytest.mark.gpu, pytest.mark.nightly]
N, PERIODS = 64, 50


class GraphSim:
    """SimulationLike stand-in: one captured raw-rung period per step_period."""

    def __init__(self, rung: RawRung) -> None:
        self.rung = rung
        self.graph = PeriodGraph(rung.step, rung.device, buffers=lambda: (rung.s0.body_q, rung.s1.body_q))

    def step_period(self) -> None:
        self.graph.launch()


def make_pack(device: str, first: int) -> Pack:
    rung = RawRung("box_pile", 4, N, device)
    spec = PackSpec(device=device, world_ids=tuple(range(first, first + N)), n=N, stream_priority=PRIORITY_NORMAL,
                    kind="throughput", rigid_contact_max=0, njmax=0, nconmax=0, shape_pairs_max=0, predicted_bytes=0)
    return Pack(spec=spec, built=None, sim=GraphSim(rung), world_ids=wp.array(spec.world_ids, dtype=wp.int32,
                                                                                device=device))


def timed(fn, device: str) -> float:
    fn()  # capture / warm
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(PERIODS):
        fn()
    wp.synchronize_device(device)
    return (time.perf_counter() - t0) / PERIODS


def test_two_subknee_packs_overlap_on_streams(device):
    if not device.startswith("cuda"):
        pytest.skip("CUDA streams")
    packs = [make_pack(device, 0), make_pack(device, N)]
    runner = PackRunner(packs)
    one = timed(lambda: packs[0].sim.step_period(), device)
    concurrent = timed(runner.step_period, device)

    def sequential() -> None:
        with wp.ScopedStream(runner.streams[0]):
            packs[0].sim.step_period()
            packs[1].sim.step_period()

    seq = timed(sequential, device)
    ratio = concurrent / seq
    record_metric("WP8-streams", ratio, "info", "pytest tests/parallel/test_streams_gpu.py", worlds_per_pack=N,
                  one_pack_s=one, sequential_s=seq, concurrent_s=concurrent)
    assert all(p.sim.graph.host_syncs == 0 for p in packs)
    assert concurrent < 1.5 * seq  # streams never make it much worse; the speed-up itself is reported
