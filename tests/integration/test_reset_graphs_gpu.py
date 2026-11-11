# SPDX-License-Identifier: Apache-2.0
"""The kept-graph law on CUDA (``alloy.integration.reset_graphs``; one L4): a run with masked resets and one full
reset reads the same with the period graphs kept as with them dropped at every reset.

- A scene that repeats bit for bit on the GPU (the dropped run twice gives the same bytes) must match bit for bit,
  health records included.
- MuJoCo Warp's contacts do not repeat on a GPU (docs/known_gaps), so there the kept run must sit inside the dropped
  run's own repeat spread: at most ``SPREAD`` times the largest difference between two dropped runs.
Scenes: free cubes on the ground (MuJoCo Warp; the standard and the fast profile) and a cloth on a stand (VBD).
Each row also prints the cost of a masked reset plus a step, kept against dropped. A coupled scene (a ball under a
cloth, MuJoCo Warp + VBD under the Scheduler) still drops its graphs at a reset: it does not repeat on a GPU (two
dropped runs 15 mm apart after 24 periods), so no row of it can show the kept graphs are right; its test pins the
drop.

    PYTHONPATH=. python -m pytest tests/integration/test_reset_graphs_gpu.py -q -m gpu -s
"""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.integration import profiles as P

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.timeout(3600)]

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
SPREAD = 4.0


@pytest.fixture(autouse=True)
def _cuda_graphs(monkeypatch):
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    for k, v in list(P.PROFILES.items()):
        monkeypatch.setitem(P.PROFILES, k, dataclasses.replace(v, graph="on"))


def cubes(worlds: int) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground(friction=1.0)
    s.add_box("c", size=(0.2,) * 3, pos=np.array([[1.0 * (w % 8), 1.0 * (w // 8), 0.3] for w in range(worlds)]),
              density=500.0, friction=0.8)
    return s


def cloth(worlds: int) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("stand", size=(0.2, 0.2, 0.3), pos=(0, 0, 0.15), static=True)
    s.add_cloth("sheet", size=(0.4, 0.4), pos=(0, 0, 0.36), resolution=12, mass=0.1, self_contact=False)
    return s


def ball_under_cloth(worlds: int) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_sphere("ball", 0.05, pos=(0, 0, 0.05), density=500.0)
    s.add_cloth("sheet", size=(0.4, 0.4), pos=(0, 0, 0.14), resolution=12, mass=0.1, self_contact=False)
    return s


def _mask(worlds: int, picked: list[int]) -> wp.array:
    m = np.zeros(worlds, dtype=np.int32)
    m[picked] = 1
    return wp.array(m, dtype=wp.int32, device=DEV)


def _run(scene: alloy.Scene, keep: bool, profile: str, periods: int = 24) -> tuple[list[dict], bytes, dict, float]:
    """(a state snapshot per period, the health records at the end, graph stats, seconds per masked reset + step)."""
    sim = alloy.build(scene, device=DEV, seed=0, profile=profile)
    eng, worlds = sim.engine, sim.worlds
    assert eng.use_graph
    eng.reset_keeps_graphs = keep
    snaps, spent, resets = [], 0.0, 0
    for i in range(periods):
        if (i >= 10 and i % 2 == 0) or i == 21:    # from period 10, every second period (and 21): a reset, then a step
            wp.synchronize_device(DEV)
            t0 = time.perf_counter()
            eng.reset(None if i == 16 else _mask(worlds, [(3 * i) % worlds, (3 * i + 1) % worlds]))
            sim.step(1)
            wp.synchronize_device(DEV)
            spent, resets = spent + time.perf_counter() - t0, resets + 1
        else:
            sim.step(1)
        snaps.append({k: np.array(v, copy=True) for k, v in sim.state().items()})
    assert sim.check()["ok"]
    return snaps, eng.health.record.numpy().tobytes(), eng.graph_stats(), spent / resets


def _gap(a: list[dict], b: list[dict]) -> float:
    return max(float(np.abs(x[k].astype(np.float64) - y[k].astype(np.float64)).max()) for x, y in zip(a, b) for k in x)


@pytest.mark.parametrize(("make", "worlds", "profile"), [(cubes, 64, "standard"), (cloth, 8, "standard"),
                                                         (cubes, 64, "fast")])
def test_kept_graphs_read_as_the_dropped_run(make, worlds, profile) -> None:
    """The fast profile checks every 10 steps (the check pattern is in the graph key, period snapshots are saved
    after check steps only): its resets fall right before and right after a check step."""
    kept, rec_k, stats_k, t_k = _run(make(worlds), True, profile)
    dropped, rec_d, stats_d, t_d = _run(make(worlds), False, profile)
    again, rec_a, _, _ = _run(make(worlds), False, profile)
    floor, gap = _gap(dropped, again), _gap(kept, dropped)
    print(f"\n{make.__name__} x{worlds} {profile}: kept vs dropped {gap:.3e}, dropped vs dropped {floor:.3e}; a masked "
          f"reset + a step {t_k * 1e3:.1f} ms kept, {t_d * 1e3:.1f} ms dropped; captures {stats_k['captures']} kept, "
          f"{stats_d['captures']} dropped; resets kept {stats_k['reset_kept']} dropped {stats_k['reset_dropped']}")
    assert stats_k["reset_dropped"] == 0 and stats_k["reset_kept"] == stats_d["reset_dropped"] > 0
    assert stats_k["captures"] < stats_d["captures"]
    if floor == 0.0 and rec_d == rec_a:            # the scene repeats bit for bit: the law is bit for bit
        assert gap == 0.0 and rec_k == rec_d
    else:
        assert gap <= SPREAD * floor, (gap, floor)


def test_a_coupled_scene_still_drops_its_graphs() -> None:
    """More than one entry solver: dropped at a reset, as before (reset_graphs' module docstring says what row lifts
    this)."""
    sim = alloy.build(ball_under_cloth(2), device=DEV, seed=0)
    sim.step(2)
    sim.engine.reset(_mask(2, [1]))
    assert sim.engine.reset_graphs == {"kept": 0, "dropped": 1}
