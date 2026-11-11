# SPDX-License-Identifier: Apache-2.0
"""Probe (parallel §4.1): timing protocol and fit, SceneCard path through an assembler, cache, trace replay."""

from __future__ import annotations

import time

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.cards.types import SceneCard
from alloy.parallel.counters import ReplayLayout, TraceReplayer
from alloy.parallel.graphs import PeriodGraph
from alloy.parallel.probe import ProfileCache, ProfileMeta, StepTarget, TimingOptions, probe_scene, probe_targets
from alloy.testing.fakes import FakeAssembler, FakeSimulation, FakeStepper
from alloy.testing.parallel import profile

META = ProfileMeta("scene" * 4, "cfg" * 4, "ac7f7830", "cpu")


def sleeper(t_floor: float, c_w: float):
    """A pack whose step costs t_floor + n c_w of wall time (sleep), for testing the fit end to end. The sleep is host
    code, so the target is not graph-safe (a CPU graph would replay nothing)."""

    def make(n: int) -> StepTarget:
        return StepTarget(step=lambda: time.sleep(t_floor + n * c_w), nworld=n, sim_dt=0.02, device="cpu",
                          graph_safe=False)

    return make


def test_probe_targets_recovers_a_synthetic_cost_law():
    opts = TimingOptions(window_s=0.0, min_window_sim_s=0.2, windows=3, chunks=2, warmup_steps=1)
    prof, runs = probe_targets(sleeper(4e-3, 2.5e-4), (1, 8, 32), "cpu", META, opts)
    assert [r.n for r in runs] == [1, 8, 32] and all(r.windows == 3 and not r.graph for r in runs)
    # sleep overshoots by a roughly constant ~0.1 ms, which lands in t_floor
    assert prof.t_floor_s == pytest.approx(4e-3, rel=0.08) and prof.c_w_s == pytest.approx(2.5e-4, rel=0.05)
    assert prof.dt == 0.02 and prof.bytes_per_world == 0 and np.isnan(prof.iters_p50)


def test_probe_scene_through_fake_assembler_and_simulation():
    asm = FakeAssembler(solver="mujoco")

    def sim_factory(built):
        return FakeSimulation(built.model, FakeStepper(newton.solvers.SolverMuJoCo(built.model, njmax=32, nconmax=8)),
                              control_dt=built.control_dt)

    scene = SceneCard(robots=(), objects=(), control_dt=0.02)
    opts = TimingOptions(window_s=0.0, min_window_sim_s=0.04, windows=2, chunks=2, warmup_steps=1)
    prof = probe_scene(scene, None, "cpu", (1, 4), assembler=asm, sim_factory=sim_factory, opts=opts)
    assert asm.calls[:2] == [(1, 0), (4, 0)]
    assert prof.nv == 6 and prof.island_nv_max == 6 and prof.chain_len_max == 1 and prof.dt == 0.02
    assert prof.t_floor_s > 0.0 and prof.iters_p99 >= prof.iters_p50 >= 1.0 and prof.contacts_p99 >= 1
    assert prof.gpu_model == wp.get_device("cpu").name and len(prof.scene_hash) == 64


def test_profile_cache_roundtrip(tmp_path):
    cache = ProfileCache(tmp_path / "profiles.json")
    p = profile(gpu_model="cpu")
    cache.put(p)
    again = ProfileCache(tmp_path / "profiles.json")
    assert again.get(p.scene_hash, p.solver_config_hash, p.newton_commit, "cpu") == p
    assert again.get(p.scene_hash, "other", p.newton_commit, "cpu") is None


def test_trace_replayer_broadcast_and_per_world_rows():
    dst_a = wp.zeros(3 * 2, dtype=wp.float32, device="cpu")
    dst_b = wp.zeros(3 * 1, dtype=wp.float32, device="cpu")
    rows_a = np.arange(8, dtype=np.float32).reshape(4, 2)  # [T, n], same for every world
    rows_b = np.arange(4 * 3, dtype=np.float32).reshape(4, 3, 1) * 10  # [T, W, n]
    rep = TraceReplayer({"a": rows_a, "b": rows_b}, {"a": dst_a, "b": dst_b}, nworld=3, device="cpu")
    for k in range(6):  # past T the last row holds
        rep.apply()
        kk = min(k, 3)
        np.testing.assert_array_equal(dst_a.numpy(), np.tile(rows_a[kk], 3))
        np.testing.assert_array_equal(dst_b.numpy(), rows_b[kk, :, 0])
    with pytest.raises(ValueError):
        TraceReplayer({"a": rows_a}, {"a": dst_b}, nworld=3, device="cpu")


def test_trace_replayer_scatters_indexed_columns():
    """Columns land at ``w * stride + index[i]``: a few dofs of each world's block, or a vec2 array as float32."""
    dofs = wp.zeros(2 * 5, dtype=wp.float32, device="cpu")  # 2 worlds x 5 dofs, trace drives dofs 1 and 3
    pairs = wp.zeros((2, 3), dtype=wp.vec2, device="cpu")  # 2 worlds x 3 joints, trace drives joint 2
    rows_f = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rows_r = np.array([[[-5.0, 5.0]], [[-6.0, 6.0]]], dtype=np.float32)  # [T, 1, 2]
    rep = TraceReplayer({"f": rows_f, "r": rows_r.reshape(2, -1)}, {"f": dofs, "r": pairs.view(wp.float32)}, 2, "cpu",
                        layouts={"f": ReplayLayout(np.array([1, 3]), 5), "r": ReplayLayout(np.array([4, 5]), 6)})
    rep.apply()
    rep.apply()
    np.testing.assert_array_equal(dofs.numpy().reshape(2, 5), [[0, 3, 0, 4, 0]] * 2)
    np.testing.assert_array_equal(pairs.numpy()[:, 2], [[-6.0, 6.0]] * 2)
    assert not pairs.numpy()[:, :2].any()
    with pytest.raises(ValueError, match="stride"):
        TraceReplayer({"f": rows_f}, {"f": dofs}, 3, "cpu", layouts={"f": ReplayLayout(np.array([1, 3]), 5)})


def test_trace_replayer_is_graph_safe_on_cpu():
    dst = wp.zeros(2, dtype=wp.float32, device="cpu")
    rows = np.arange(10, dtype=np.float32).reshape(5, 2)
    rep = TraceReplayer({"x": rows}, {"x": dst}, nworld=1, device="cpu")
    g = PeriodGraph(rep.apply, "cpu", use_graph=True)
    for _ in range(3):
        g.launch()
    assert g.mode == "graph"
    np.testing.assert_array_equal(dst.numpy(), rows[2])


def test_overflow_bits_split_capacity_from_iteration_limit():
    from types import SimpleNamespace

    from alloy._vendor.mujoco_warp import OverflowType as O

    from alloy.parallel.counters import _overflow_worlds

    bits = [0, int(O.ITERATIONS), int(O.NEFC | O.ITERATIONS), int(O.NARROWPHASE), int(O.LS_ITERATIONS)]
    data = SimpleNamespace(overflow=wp.array(np.array(bits, np.int32), dtype=wp.int32, device="cpu"))
    assert _overflow_worlds(data) == (2, 3)          # capacity: NEFC, NARROWPHASE; iteration cap: 3 worlds
    assert _overflow_worlds(SimpleNamespace()) == (0, 0)
