# SPDX-License-Identifier: Apache-2.0
"""CPU mode A's split and join rules (``alloy.parallel.cpu_merge``): per-world values cut to each process's block,
results joined in world order with global world ids. Pure functions, no processes."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.api.errors import BuildError, SimulationError
from alloy.parallel import cpu_merge as M
from alloy.parallel.cpu_worlds import process_count
from alloy.parallel.multigpu import WorldBlock


def test_blocks_cover_every_world_once_in_order():
    for worlds, procs in ((5, 2), (7, 3), (16, 16), (64, 16)):
        spans = M.blocks(worlds, procs)
        ids = [b.offset + i for b in spans for i in range(b.count)]
        assert ids == list(range(worlds)) and max(b.count for b in spans) - min(b.count for b in spans) <= 1
    spans = M.blocks(5, 2)
    assert [M.owner(spans, w) for w in range(5)] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    with pytest.raises(SimulationError, match="outside 0..4"):
        M.owner(spans, 5)


def test_per_world_values_are_cut_and_shared_values_pass():
    b = WorldBlock(2, 2)
    np.testing.assert_array_equal(M.cut(np.arange(4.0), b, 4, "q"), [2.0, 3.0])
    assert M.cut(0.5, b, 4, "q") == 0.5 and M.cut(None, b, 4, "q") is None
    pos = np.arange(12.0).reshape(4, 3)
    np.testing.assert_array_equal(M.cut(pos, b, 4, "pos", row_ndim=1), pos[2:])
    assert M.cut((0.0, 0.0, 1.0), b, 4, "pos", row_ndim=1) == (0.0, 0.0, 1.0)     # one position for every world
    with pytest.raises(SimulationError, match=r"shape \(3,\) for 4 worlds"):
        M.cut(np.zeros(3), b, 4, "q")
    ramps = M.cut_ramps({"a": (0.0, np.arange(4.0)), "b": np.ones(4)}, b, 4)
    assert ramps["a"][0] == 0.0
    np.testing.assert_array_equal(ramps["a"][1], [2.0, 3.0])
    np.testing.assert_array_equal(ramps["b"], [1.0, 1.0])
    assert M.cut_ramps(None, b, 4) is None


def test_state_and_frames_join_in_world_order():
    a = {"t": 0.5, "x.pos": np.zeros((2, 3))}
    b = {"t": 0.5, "x.pos": np.ones((1, 3))}
    s = M.join_state([a, b])
    assert s["t"] == 0.5 and s["x.pos"].shape == (3, 3) and s["x.pos"][2, 0] == 1.0
    fa = {"t": np.arange(4.0), "x.pos": np.zeros((4, 2, 3))}
    fb = {"t": np.arange(4.0), "x.pos": np.ones((4, 1, 3))}
    f = M.join_frames([fa, fb])
    assert f["x.pos"].shape == (4, 3, 3) and f["x.pos"][:, 2].min() == 1.0
    h = M.join_per_world([{"mean_K": np.array([300.0, 301.0])}, {"mean_K": np.array([302.0])}])
    np.testing.assert_array_equal(h["mean_K"], [300.0, 301.0, 302.0])


def _row(sev: str) -> dict:
    return {"severity": sev, "flags": [] if sev == "OK" else ["PEN_FAIL"], "fixes": []}


def test_health_and_check_take_global_world_ids():
    spans = [WorldBlock(0, 2), WorldBlock(2, 2)]
    h0 = {"ok": True, "monitored": True, "invalid_worlds": [], "degraded_worlds": [1], "worlds": [_row("OK"),
          _row("DEGRADED")], "retired_particles": [0, 0], "refined_periods": [0, 2]}
    h1 = {"ok": False, "monitored": True, "invalid_worlds": [1], "degraded_worlds": [], "worlds": [_row("OK"),
          _row("INVALID")], "retired_particles": [3, 0], "refined_periods": [1, 0]}
    h = M.join_health([h0, h1], spans)
    assert not h["ok"] and h["invalid_worlds"] == [3] and h["degraded_worlds"] == [1]
    assert [r["severity"] for r in h["worlds"]] == ["OK", "DEGRADED", "OK", "INVALID"]
    assert h["retired_particles"] == [0, 0, 3, 0]
    assert h["refined_periods"] == [0, 2, 1, 0]                      # every worker's worlds, not worker 0's
    with pytest.raises(SimulationError, match="one entry per world"):  # a short list would shift the world ids
        M.join_health([h0, {**h1, "refined_periods": [1]}], spans)
    c0 = {"ok": True, "source": "health", "problems": [], "report": {"r": 0}}
    c1 = {"ok": False, "source": "health", "problems": [{"world": 1, "what": "sank"}], "invalid_worlds": [1],
          "report": {"r": 1}}
    c = M.join_check([c0, c1], spans)
    assert not c["ok"] and c["problems"] == [{"world": 3, "what": "sank"}] and c["invalid_worlds"] == [3]
    assert c["reports"] == [{"r": 0}, {"r": 1}] and "report" not in c
    meta = M.join_meta([{"dt": 0.01, "worlds": 2, "wall_s": 1.0, "health": h0},
                        {"dt": 0.01, "worlds": 2, "wall_s": 2.0, "health": h1}], spans, 2.5)
    assert meta["worlds"] == 4 and meta["wall_s"] == 2.5 and meta["worker_wall_s"] == [1.0, 2.0]
    assert meta["processes"] == 2 and meta["health"]["invalid_worlds"] == [3]
    steps = M.join_meta([{"physics_steps": 40, "health": h0}, {"physics_steps": 48, "health": h1}], spans, 1.0)
    assert steps["physics_steps"] == 48 and steps["worker_physics_steps"] == [40, 48]


def test_parted_clocks_warn_once(caplog):
    from alloy.parallel.cpu_worlds import SplitSim

    sim = SplitSim.__new__(SplitSim)
    sim._warned_clock = False
    sim._check_clocks([40, 40])
    sim._check_clocks([40, 48])
    sim._check_clocks([48, 56])
    msgs = [r.getMessage() for r in caplog.records if "physics steps" in r.getMessage()]
    assert len(msgs) == 1 and "[40, 48]" in msgs[0] and "processes=None" in msgs[0], msgs


def test_process_count():
    assert process_count(None, 8, "cpu") == 1
    assert process_count(4, 3, "cpu") == 3                       # at most one process per world
    assert 1 <= process_count("auto", 64, "cpu") <= 64
    assert process_count("auto", 64, "cuda:0") == 1
    for bad in (0, -2, 1.5, "many", True):
        with pytest.raises(BuildError, match="processes must be"):
            process_count(bad, 8, "cpu")
    with pytest.raises(BuildError, match="splits worlds over CPU processes"):
        process_count(4, 8, "cuda:0")
