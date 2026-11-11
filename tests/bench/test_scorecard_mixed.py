# SPDX-License-Identifier: Apache-2.0
"""The mixed-batch scorecard harness (bench/scorecard/mixed): its pure parts and that it loads. No simulation."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

MIXED = Path(__file__).resolve().parents[2] / "bench" / "scorecard" / "mixed"


@pytest.fixture(scope="module")
def mods():
    path, before = list(sys.path), set(sys.modules)
    sys.path.insert(0, str(MIXED))
    try:
        yield importlib.import_module("mixed"), importlib.import_module("build")
    finally:
        sys.path[:] = path                          # the harness modules have short names (build, tasks, ...):
        for name in set(sys.modules) - before:      # leave none of them behind for other tests
            if str(MIXED) in str(getattr(sys.modules[name], "__file__", "") or ""):
                del sys.modules[name]


def test_grid_is_centred_and_spaced(mods):
    mixed, _ = mods
    pts = [mixed.grid_pos(i, 10, 2.0) for i in range(10)]
    assert len(set(pts)) == 10
    assert max(abs(c) for p in pts for c in p) <= 3.0           # health flags objects over 4 m from the origin
    assert min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in pts for b in pts if a != b) == pytest.approx(2.0)


def test_harness_loads_every_task(mods):
    mixed, _ = mods
    _, tasks, alloy_hold, newton_hold = mixed.load_pxs(mixed.DEFAULT_PXS)
    assert {"franka_lift", "go2_flat", "g1_flat", "hand_cube", "fr3_reach", "h1_2_flat"} <= set(tasks.TASKS)
    for task in tasks.TASKS.values():
        assert (tasks.LIBRARY / "robots" / task.pack / "scene.xml").is_file(), task.name
    assert callable(alloy_hold.read_watch) and callable(newton_hold.ctrl_bank)


def test_upper_bound_matches_the_rule_of_three(mods):
    _, build = mods
    assert build.up95(0, 1024) == pytest.approx(1 - 0.05 ** (1 / 1024), rel=1e-6, abs=0)
    assert build.up95(5, 5) == 1.0
    assert build.up95(3, 100) > 0.03


def _point(path: Path, value: float, mem: float) -> None:
    path.write_text(json.dumps({"engine": "alloy", "form": "multi", "tasks": ["a", "b"], "task_types": 2, "gpu": "x",
                                "solver_steps_per_env_step": [4.0, 4.0], "group_end": [{"health": {"invalid": 0}}] * 2,
                                "speed": {"env_steps_per_s": value, "block_walls_s": [1.0], "group_wall_s": [],
                                          "memory": {"nvidia_smi_used_mb": mem}}}))


def test_build_tables_best_batch_and_failed_point(mods, tmp_path):
    _, build = mods
    _point(tmp_path / "mix2_alloy_multi_speed_512.json", 100.0, 10.0)
    _point(tmp_path / "mix2_alloy_multi_speed_1024.json", 150.0, 20.0)
    (tmp_path / "mix2_alloy_multi_speed_2048.json").write_text(json.dumps({"error": "RuntimeError: Failed to allocate"}))
    (tmp_path / "mix2_alloy_multi_quality_2048.json").write_text(json.dumps({
        "engine": "alloy", "form": "multi", "tasks": ["a", "b"], "group_end": [{"health": {"invalid": 0}}],
        "quality": [{"task": "a", "worlds": 1024, "score": {"s1": 0, "episodes": 1024, "causes": {}, "held": 1024,
                                                           "kind": "stand", "drift_mm": {"p95": 1.0, "max": 2.0}}}]}))
    rows = build.load(tmp_path)
    assert sorted(r["metric_name"] for r in rows) == ["env_steps_per_s", "env_steps_per_s", "failed", "quality"]
    text = build.tables(rows)
    assert "| Alloy as shipped, one sim per type | 150 | 1,024 | 20 | - | 2,048 |" in text
    assert "0/1024 (<= 0.29 %)" in text and "1024/1024 (stand)" in text


def test_frozen_worlds_fail_the_point_and_count_as_glitches(mods, tmp_path):
    _, build = mods
    _point(tmp_path / "mix2_alloy_multi_speed_512.json", 100.0, 10.0)
    d = json.loads((tmp_path / "mix2_alloy_multi_speed_512.json").read_text())
    d["group_end"][1]["health"] = {"invalid": 30, "n": 256}                     # 12 % of one type frozen
    (tmp_path / "mix2_alloy_multi_speed_512.json").write_text(json.dumps(d))
    (tmp_path / "mix2_alloy_multi_quality_2048.json").write_text(json.dumps({
        "engine": "alloy", "form": "multi", "tasks": ["a"], "group_end": [{"health": {"invalid": 7, "n": 1024}}],
        "quality": [{"task": "a", "worlds": 1024, "score": {"s1": 0, "episodes": 1024, "causes": {}, "held": 1024,
                                                           "kind": "stand", "drift_mm": {"p95": 1.0, "max": 2.0}}}]}))
    rows = {r["metric_name"]: r for r in build.load(tmp_path)}
    assert set(rows) == {"failed", "quality"} and "frozen" in rows["failed"]["error"]
    assert rows["quality"]["s1"] == 7 and rows["quality"]["s1_outside_rule"] == 0


def test_a_type_with_every_world_non_finite_still_renders(mods, tmp_path):
    _, build = mods
    (tmp_path / "mix1_mjws_sub2_quality_1024.json").write_text(json.dumps({
        "engine": "newton_mjw", "form": "multi", "tasks": ["a"], "group_end": [{}],
        "quality": [{"task": "a", "worlds": 1024, "score": {"s1": 1024, "episodes": 1024, "causes": {"nonfinite": 1024},
                                                           "held": 0, "kind": "stand",
                                                           "drift_mm": {"p95": None, "max": None}}}]}))
    assert "| - / - |" in build.tables(build.load(tmp_path))


def test_sweep_script_parses():
    assert subprocess.run(["bash", "-n", str(MIXED / "sweep.sh")], check=False).returncode == 0
