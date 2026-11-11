# SPDX-License-Identifier: Apache-2.0
"""Laws on CUDA (``-m gpu``) for a step graph that runs out of device memory (SCR-OSS22-9, tag blocker 2 of
rc1-1110: on one L4 sand at 64 worlds late in a speed sweep, and T09 at 64, 256 and 1,024 worlds, ran out creating a
64-exchange graph, and the eager period that followed died with "CUDA error 101: invalid device ordinal" in a prefix
sum).

- After a CUDA allocation failure the next Warp native built on CUB fails once, whatever memory is free again
  (``native_guard``'s module docstring). ``absorb_stale_error`` takes that failure: a prefix sum, a radix sort and an
  array sum after it are right.
- A sand scene whose step graph hits a real allocation failure while it is captured steps on without graphs, with a
  clean health check, and says so in ``graph_stats()["memory_notes"]``.
- Simulations dropped earlier in the process (a sweep's last world count) do not count against an MPM scene: the
  memory plan reads the free memory after collecting them.
The allocation failure is real on any GPU and takes no memory: one array far larger than the device.

    PYTHONPATH=. python -m pytest tests/integration/test_graph_memory_gpu.py -q -m gpu -s
"""

from __future__ import annotations

import gc
import logging
import os

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.integration import native_guard as G

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.timeout(1800)]

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
GB = 2**30


@pytest.fixture(autouse=True)
def _cuda():
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")


def _fail_an_allocation() -> RuntimeError:
    """A real CUDA allocation failure that takes no memory: 200 GiB in one array."""
    with pytest.raises(RuntimeError, match="allocate") as e:
        wp.empty((200, 1 << 30), dtype=wp.uint8, device=DEV)
    return e.value


def sand(worlds: int) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True)
    s.add_box("sand", size=(0.08, 0.08, 0.03), pos=(0, 0, 0.116), material="soil_sand_dry")
    return s


def test_cub_natives_are_right_after_an_absorbed_allocation_failure() -> None:
    n = 4096
    ones = wp.ones(n, dtype=wp.int32, device=DEV)
    out = wp.zeros(n, dtype=wp.int32, device=DEV)
    wp.utils.array_scan(ones, out, inclusive=True)                  # CUB's own setup, before any failure
    _fail_an_allocation()
    assert G.absorb_stale_error(DEV) is True and DEV not in G._STALE
    out.zero_()
    wp.utils.array_scan(ones, out, inclusive=True)
    assert np.array_equal(out.numpy(), np.arange(1, n + 1))
    _fail_an_allocation()
    G.absorb_stale_error(DEV)
    keys = wp.array(np.concatenate([np.arange(n, 0, -1), np.zeros(n)]), dtype=wp.int32, device=DEV)
    vals = wp.array(np.concatenate([np.arange(n), np.zeros(n)]), dtype=wp.int32, device=DEV)
    wp.utils.radix_sort_pairs(keys, vals, n)
    assert np.array_equal(keys.numpy()[:n], np.arange(1, n + 1))    # unabsorbed, the sort returns its keys unsorted
    _fail_an_allocation()
    G.absorb_stale_error(DEV)
    assert wp.utils.array_sum(wp.ones(n, dtype=wp.float32, device=DEV)) == float(n)


def test_a_sand_scene_steps_on_after_its_step_graph_ran_out_of_memory(monkeypatch) -> None:
    """The red: the allocation failure is raised for real inside the first graph capture; before the fix the eager
    period's first prefix sum then raised CUDA error 101."""
    from alloy.core.errors import GraphCaptureError
    from alloy.parallel.graphs import PeriodGraph

    capture, fired = PeriodGraph._capture, []

    def failing(graph) -> None:
        if not fired:
            fired.append(1)
            try:
                wp.empty((200, 1 << 30), dtype=wp.uint8, device=DEV)
            except RuntimeError as exc:
                raise GraphCaptureError(f"capturing the control period on {DEV} failed: {exc}") from exc
        capture(graph)

    monkeypatch.setattr(PeriodGraph, "_capture", failing)
    sim = alloy.build(sand(4), device=DEV, seed=0)
    assert sim.engine.use_graph
    with pytest.warns(RuntimeWarning, match="step graph"):
        sim.step(3)
    stats = sim.engine.graph_stats()
    assert fired and len(stats["memory_notes"]) == 1 and stats["memory_eager"] == 1, stats
    assert sim.check()["ok"] and np.isfinite(sim.state()["sand.points"]).all()
    z = sim.state()["sand.points"][..., 2]
    assert 0.09 < float(z.min()) and float(z.max()) < 0.14           # the sand is still a pile on the table


def test_dropped_simulations_do_not_count_against_an_mpm_scene(caplog) -> None:
    """A sweep's leftovers: simulations dropped without a collection keep their arrays and graphs. The plan collects
    them before it reads the free memory, so the next MPM scene is judged against the whole device."""
    from alloy.integration import mpm_memory as M

    dev = wp.get_device(DEV)
    gc.collect()
    idle = M.free_bytes(dev)
    was = gc.isenabled()
    gc.disable()                                                    # the collector "rarely runs": here, not at all
    try:
        for _ in range(2):
            old = alloy.build(sand(8), device=DEV, seed=0)
            old.step(2)
            del old
        held = idle - M.free_bytes(dev)
        with caplog.at_level(logging.INFO, logger="alloy.integration"):
            sim = alloy.build(sand(2), device=DEV, seed=0)
    finally:
        if was:
            gc.enable()
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("MPM memory: ") and "free" in r.
                getMessage() and "more free" not in r.getMessage())
    free_seen = float(line.split(" GB free")[0].rsplit(" ", 1)[-1]) * GB
    print(f"\nidle {idle / GB:.2f} GB free; two dropped simulations held {held / GB:.2f} GB; the plan read "
          f"{free_seen / GB:.2f} GB free")
    assert held > 0.05 * GB, "the dropped simulations held nothing: the law has no leftovers to collect"
    assert free_seen > idle - 0.25 * held - 0.3 * GB                 # all but the new model's own arrays came back
    sim.step(1)
    assert sim.check()["ok"]
