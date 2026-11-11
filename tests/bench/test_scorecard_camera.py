# SPDX-License-Identifier: Apache-2.0
"""Row 5 (camera throughput) harness: cell grid, table maker, row format, repo hygiene, one tiny CPU cell."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROW5 = Path(__file__).resolve().parents[2] / "bench" / "scorecard" / "row5"
PROTOCOL = (ROW5 / "PROTOCOL.md").read_text()


@pytest.fixture(scope="module")
def row5():
    """The row 5 modules, which import each other by bare name (they run as scripts)."""
    sys.path.insert(0, str(ROW5))
    try:
        import alloy_cam
        import cam_scenes
        import cam_util
        import make_table
        import run_cells
        yield argparse.Namespace(alloy_cam=alloy_cam, scenes=cam_scenes, util=cam_util, table=make_table, cells=run_cells)
    finally:
        sys.path.remove(str(ROW5))


def test_grid_matches_protocol(row5):
    s = row5.scenes
    assert list(s.SIZES) == [int(x) for x in re.findall(r"(\d+)x\d+", PROTOCOL)]
    assert list(s.WORLDS) == [int(x.replace(",", "")) for x in re.search(r"Worlds: ([\d, ]+)\.", PROTOCOL).group(1).split(", ")]
    assert all(f"`{o}`" in PROTOCOL for o in s.OUTPUTS)
    assert set(s.SCENE_NAMES) == {"cartpole", "franka_cube"}
    assert all(f"`{n}`" in PROTOCOL for n in s.SCENE_NAMES)


def test_cell_lists(row5):
    alloy = list(row5.cells.alloy_cells(False, False))
    mjw = list(row5.cells.mujoco_warp_cells(False))
    assert len(mjw) == 3 * 2 * 3 * 3
    assert len(alloy) == 2 * 54 + 3  # fast and default, plus three t12 extra rows
    assert len({c.key for c in alloy}) == len(alloy) and len({c.key for c in mjw}) == len(mjw)
    assert {c.setting for c in alloy} == {"fast", "default"}
    assert {c.setting for c in row5.cells.alloy_cells(False, True)} == {"fast", "default", "fast_optix"}
    assert all(c.outputs == "rgb" and c.size == 128 for c in alloy if c.scene == "t12")
    assert len(list(row5.cells.alloy_cells(True, False))) == 2 * 3 * 2 + 1


def test_scene_inputs(row5):
    s = row5.scenes
    for name, (joints, hold, noise, cam) in s.SCENES.items():
        pool = s.noise_pool(name, 4)
        assert pool.shape == (16, 4, len(joints)) and pool.dtype == np.float32 and np.isfinite(pool).all()
        assert len(hold) == len(joints) and len(cam) == 2
    assert "<mujoco" in s.CARTPOLE_MJCF.format(floor="")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_make_table_renders_statuses(row5, tmp_path):
    base = {"scene": "cartpole", "width": 128, "worlds": 1024}
    rows = [{**base, "outputs": "rgb", "status": "ok", "images_per_s": 12345.6, "gpu_mem_mib": 2048, "setting": "fast"},
            {**base, "outputs": "rgb+depth", "status": "oom", "setting": "fast"},
            {**base, "outputs": "rgb+depth+seg", "status": "unsupported", "setting": "fast"}]
    _write_rows(tmp_path / "alloy_g4.jsonl", rows)
    out = tmp_path / "TABLE.md"
    row5.table.main(["--raw", str(tmp_path), "--out", str(out)])
    text = out.read_text()
    assert "12,346" in text and "out of memory" in text and "no such output" in text and "not measured" in text
    assert "Alloy, 1 spp (Warp)" in text
    for absent in ("MuJoCo Warp", "Madrona", "ManiSkill"):
        assert absent not in text


def test_make_table_extra_rows_only_by_argument(row5, tmp_path):
    _write_rows(tmp_path / "other.jsonl", [{"scene": "franka_cube", "outputs": "rgb", "width": 128, "worlds": 1024,
                                            "status": "ok", "images_per_s": 7.0}])
    out = tmp_path / "T.md"
    row5.table.main(["--raw", str(tmp_path), "--out", str(out), "--extra", f"Other engine={tmp_path / 'other.jsonl'}"])
    assert "Other engine" in out.read_text()


def test_row_fields_are_the_protocol_fields(row5):
    listed = re.search(r"Row format.*?\n(\{.*?\})", PROTOCOL, re.S).group(1)
    assert tuple(re.findall(r'"(\w+)"', listed)) == row5.util.ROW_FIELDS


def test_repo_hygiene():
    # Generic patterns only: home paths, private IPs, the private notes folder, licence-held engine names.
    banned = re.compile(r"/home/[a-z0-9_-]+/|\b10\.\d+\.\d+\.\d+\b|alloy[-_]refs|isaac|physx", re.I)
    files = [p for p in ROW5.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert files
    assert [p.name for p in files if banned.search(p.read_text(errors="ignore"))] == []


@pytest.mark.compile_heavy
def test_tiny_alloy_cell_on_cpu(row5, tmp_path):
    pytest.importorskip("torch")
    a = argparse.Namespace(scene="cartpole", size=64, worlds=2, setting="fast", outputs="rgb", out=str(tmp_path / "r.jsonl"),
                           png_dir=str(tmp_path / "png"), warmup=5, repeats=1, loops=2, commit="test", backend_t12="optix")
    row = row5.alloy_cam.run_to_row(a)
    assert row["status"] == "ok", row
    row5.util.write_row(a.out, row)  # adds the utc stamp
    row = json.loads(Path(a.out).read_text())
    assert set(row5.util.ROW_FIELDS) <= set(row)
    assert row["images_per_s"] > 0 and row["worlds"] == 2
    from PIL import Image
    grid = np.asarray(Image.open(tmp_path / "png" / "alloy_cartpole_fast_rgb_64px_2w.png"))
    assert grid.std() > 1.0
