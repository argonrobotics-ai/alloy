# SPDX-License-Identifier: Apache-2.0
"""Speed ladder on CPU (SPEC §7 WP8 acceptance: L0, L1 with 1-64 objects and L4 run on CPU at N <= 16).

The CI subset is small (Warp CPU is ~0.01x real time); the full acceptance set is marked ``slow`` and its run is
recorded in docs/notes/wp8-results.md. Speed numbers from CPU are plumbing checks, not results.
"""

from __future__ import annotations

import pytest

from alloy.parallel.ladder_run import CapacityCounts, fit_row, sweep
from alloy.parallel.probe import TimingOptions
from alloy.parallel.rungs import LADDER, RAW_SCENES, RawRung

QUICK = TimingOptions(window_s=0.0, min_window_sim_s=0.1, windows=2, chunks=2, warmup_steps=1)
SETTLE = TimingOptions(window_s=0.0, min_window_sim_s=0.25, windows=2, chunks=2, warmup_steps=1)  # cable settles ~0.3 s
FULL = TimingOptions(window_s=0.0, min_window_sim_s=0.2, windows=2, chunks=2, warmup_steps=2)


def check_sweep(rows, worlds):
    points, fit = rows[:-1], rows[-1]
    assert [r["nworld"] for r in points] == sorted(worlds)
    for r in points:
        assert r["status"] == "ok", r["error"]
        assert r["quality_gate"]["pass"], r["quality_gate"]
        assert r["sim_seconds_per_gpu_second"] > 0 and r["iters_p99"] >= 1 and r["substeps"] == 16
        assert r["kernels_per_substep"] is None and r["overflow"] == 0  # CPU: no CUDA kernel timing
    assert fit["kind"] == "fit" and fit["status"] == "ok"
    # CPU graph replay (SCR-WP8-5) leaves little fixed cost per period: t_floor may fit to 0 (the clamp), c_w carries it
    assert fit["t_floor_s"] >= 0 and fit["c_w_s"] >= 0 and fit["t_floor_s"] + fit["c_w_s"] > 0
    return fit


@pytest.mark.parametrize("scene,param", [("arm", 0), ("box_pile", 1), ("box_pile", 16), ("cable_chain", 20)])
def test_ladder_cpu_quick(scene, param):
    opts = SETTLE if scene == "cable_chain" else QUICK
    check_sweep(sweep(scene, param, (1, 4), "cpu", opts), (1, 4))


@pytest.mark.slow
@pytest.mark.timeout(900)  # 64 boxes x 16 worlds runs ~6 s per period on Warp CPU
@pytest.mark.parametrize("scene,param", [("arm", 0), ("box_pile", 1), ("box_pile", 4), ("box_pile", 16),
                                         ("box_pile", 64), ("cable_chain", 20)])
def test_ladder_cpu_acceptance(scene, param):
    check_sweep(sweep(scene, param, (1, 4, 16), "cpu", FULL), (1, 4, 16))


def test_ladder_table_covers_l0_to_l7():
    ids = sorted({r.id for r in LADDER})
    assert ids == [f"L{k}" for k in range(8)]
    assert {r.scene for r in LADDER} >= set(RAW_SCENES)
    l1 = next(r for r in LADDER if r.id == "L1")
    assert l1.sweep == {"objects": (1, 4, 16, 64, 100)} and l1.worlds[-1] == 4096


def test_fit_row_needs_two_gate_passing_points():
    rows = [{"status": "ok", "nworld": 1, "ms_per_period": 2.0, "quality_gate": {"pass": True}},
            {"status": "ok", "nworld": 4, "ms_per_period": 5.0, "quality_gate": {"pass": False}}]
    assert fit_row(rows)["status"] == "not_run" and fit_row(rows)["excluded"] == [4]
    rows[1]["quality_gate"]["pass"] = True
    fit = fit_row(rows)
    assert fit["t_floor_s"] == pytest.approx(1e-3) and fit["c_w_s"] == pytest.approx(1e-3)


def test_failed_point_is_a_row_and_skips_larger_n(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(RawRung, "__init__", boom)
    rows = sweep("box_pile", 4, (1, 4), "cpu", QUICK)
    assert [r["status"] for r in rows] == ["error", "skipped", "not_run"]
    assert "synthetic build failure" in rows[0]["error"]


def check_task_rows(rows, worlds):
    for r in rows[:-1]:
        assert r["status"] == "ok", r["error"]
        assert r["quality_gate"]["pass"], r["quality_gate"]
        assert r["config"].startswith("alloy_stepper") and r["rung"] == "L3" and r["overflow"] == 0
        assert r["substeps"] * r["periods_per_call"] % 2 == 0  # even swaps per captured call
    assert [r["nworld"] for r in rows[:-1]] == sorted(worlds)


def test_ladder_l3_t12_cpu():
    """L3 T12: assembled WP13 scene, Alloy entry stepper, trace replayed on the device; gate = every world within
    1 mm of the recording over 0.2 s (CPU 2026-11-01: 0.100 mm, as WP13 measured)."""
    rows = sweep("t12", 0, (1, 2), "cpu", QUICK)
    check_task_rows(rows, (1, 2))
    assert rows[0]["periods_per_call"] == 2 and rows[0]["dt"] == pytest.approx(1 / 1200)
    assert all(r["quality_gate"]["replay_point_err_max_m"] < 1e-3 for r in rows[:-1])


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_ladder_l3_t09_cpu():
    """The rung steps at T09's own clock, which its precision pair sets: the USB-A plug in its port at
    max_penetration 0.05 mm, design force 50 N, m_eff 17.9 g. The pair's static sink F / K_real <= max_penetration / 2
    fails at 0.625 ms (43.7 µm against 25 µm) and holds at 0.3125 ms, so 64 steps per 20 ms period. Why this test
    saw 32 on 2026-10-31 is not identified: the cards, the L2 rule and the pair's mass rule give 64 today; every
    nightly on record, from 2026-11-09T19:01Z on, failed on the pin."""
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    rows = sweep("t09", 0, (1, 2), "cpu", TimingOptions(window_s=0.0, min_window_sim_s=0.02, windows=1, chunks=1,
                                                         warmup_steps=1))
    check_task_rows(rows, (1, 2))
    lib = CardLibrary()
    built = SceneAssembler(lib).assemble(lib.scene("task_t09"), "cpu", world_count=1)
    (entry,) = built.entries                                          # one MuJoCo entry: no MPM or SPH limit
    assert built.precision_pairs and entry.target_dt < built.rigid_dt  # the pair, not the rigid step, sets the clock
    k = round(built.control_dt / entry.target_dt)
    assert rows[0]["substeps"] == k == 64 and rows[0]["periods_per_call"] == 1, (rows[0]["substeps"], k)


def test_ladder_l3_t12_wp8_capacities_cpu():
    """WP8-sized capacities (1.5x the probed p99: 96 contacts, 402 rows per world) instead of the router's 2,056 /
    12,336; the replay gate and the overflow check still pass."""
    rows = sweep("t12", 0, (2,), "cpu", QUICK, counts=CapacityCounts(96.0, 402.0))
    check_task_rows(rows, (2,))
    caps = rows[0]["capacities"]
    assert rows[0]["config"] == "alloy_stepper_wp8cap" and (caps["nconmax"], caps["njmax"]) == (144, 603)
