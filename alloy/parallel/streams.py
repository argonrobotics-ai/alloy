# SPDX-License-Identifier: Apache-2.0
"""Run several packs of one device concurrently, one CUDA stream per pack (parallel §4.3 steps 4-5).

The planner lets packs below their knee share a device (Σu <= 1) and gives latency packs a high-priority stream;
``PackRunner`` makes that real: each pack's control period is issued on its own ``wp.Stream`` (priority from
``PackSpec.stream_priority``: -1 high, 0 normal; warp context.py:5596), then the host waits for all of them. On CPU
the packs simply run one after another. The runner owns no stepping logic: it calls ``Simulation.step_period``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import warp as wp

from alloy.parallel.types import Pack

log = logging.getLogger("alloy.parallel")


class PackRunner:
    """Issue one control period of every pack, each on its own stream, then synchronise once."""

    def __init__(self, packs: Sequence[Pack]) -> None:
        if not packs:
            raise ValueError("PackRunner needs at least one pack")
        self.packs = list(packs)
        self.streams: list[wp.Stream | None] = [self._stream(p) for p in self.packs]
        self.periods = 0

    @staticmethod
    def _stream(pack: Pack) -> wp.Stream | None:
        dev = wp.get_device(pack.spec.device)
        if not dev.is_cuda:
            return None
        return wp.Stream(dev, priority=-1 if pack.spec.stream_priority < 0 else 0)

    def step_period(self) -> None:
        """One control period of every pack, concurrently on CUDA."""
        for pack, stream in zip(self.packs, self.streams, strict=True):
            if stream is None:
                pack.sim.step_period()
                continue
            with wp.ScopedStream(stream):
                pack.sim.step_period()
        self.synchronize()
        self.periods += 1

    def synchronize(self) -> None:
        for stream in self.streams:
            if stream is not None:
                wp.synchronize_stream(stream)
