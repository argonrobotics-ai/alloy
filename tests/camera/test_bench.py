# SPDX-License-Identifier: Apache-2.0
"""Camera speed bench on the CPU: a pack keeps its own Cycles settings, rows name the resolved backend."""
from __future__ import annotations

import json

import numpy as np
import pytest

from alloy.camera import bench
from tests.camera.packs import principled_material, write_pack_v2


def test_pack_bench_keeps_the_packs_settings(tmp_path):
    root = write_pack_v2(tmp_path / "p", principled_material())
    meta = json.loads((root / "pack.json").read_text())
    meta["settings"]["cycles"]["max_bounces"] = 3
    (root / "pack.json").write_text(json.dumps(meta))
    cfg = bench.BenchConfig(scene=f"pack:{root}@0", worlds=2, width=8, height=6, spp=1, physics=False,
                            backend="auto", iters=1, device="cpu")
    _, render, cam, _ = bench._pack_loop(cfg)
    render()
    assert cam._bound[1].st.max_bounces == 3 and cam._bound[1].st.backend == "auto"
    row = bench.run(cfg)
    assert row["backend"] == "warp" and row["worlds"] == 2 and row["images_per_s"] > 0


def test_v1_pack_bench_moves_its_meshes(tmp_path):
    from tests.camera.packs import write_pack_v1
    cfg = bench.BenchConfig(scene=f"pack:{write_pack_v1(tmp_path / 'v1')}@0", worlds=2, width=8, height=6, spp=1,
                            physics=True, iters=1, device="cpu")
    row = bench.run(cfg)
    assert row["backend"] == "warp" and row["images_per_s"] > 0 and row["ms"]["physics"] >= 0


def test_t12_scene_comes_from_the_shipped_library(tmp_path, monkeypatch):
    """AB-camera-1: the T12 file is found through the library root (an installed package keeps it under
    ``alloy/_data/library``), not at a path relative to a source checkout."""
    assert bench.t12_xml().is_file()
    monkeypatch.setenv("ALLOY_LIBRARY", str(tmp_path))
    assert bench.t12_xml() == tmp_path.resolve() / "robots" / "task_t12" / "scene.xml"


def test_t12_row_times_physics_that_fits_its_buffers():
    """AB-camera-2: T12 at rest needs several hundred constraint rows per world; MuJoCo Warp's default of 64
    overflowed (blocks fell through the table), so the bench sizes the buffers and a row reports how full they were."""
    cfg = bench.BenchConfig(scene="t12", worlds=1, width=16, height=12, spp=1, iters=20, device="cpu")
    fill = bench.run(cfg)["mujoco"]
    assert 64 < fill["constraint_rows"] <= fill["njmax"] == bench.T12_CAPACITY["njmax"]
    assert 0 < fill["contacts"] <= fill["naconmax"]


def test_mujoco_fill_refuses_an_overflow():
    class _Arr:
        def __init__(self, v):
            self.v = np.array(v)

        def numpy(self):
            return self.v

    class _Solver:
        class mjw_data:
            nefc, njmax, nacon, naconmax = _Arr([70, 12]), 64, _Arr([5]), 96

    with pytest.raises(RuntimeError, match="overflowed"):
        bench.mujoco_fill(_Solver())


def test_claim_sets_the_gated_setting_of_the_size():
    """AB-inv-reports-11: a speed row at a gate scene's claim runs that claim's samples, denoiser, tta and weights."""
    from alloy.camera.gate_claims import claim_for
    for res in ("128x128", "640x480"):
        w, h = (int(v) for v in res.split("x"))
        cfg = bench.with_claim(bench.BenchConfig(width=w, height=h, spp=5, denoise="learned", tta=2), "t12")
        claim = claim_for("t12", res)
        assert (cfg.spp, cfg.denoise, cfg.tta) == (claim.spp, claim.kind, claim.tta)
        assert cfg.claim == f"t12: {claim.label()}"
        assert (cfg.weights is None) == (claim.kind != "learned")
    assert bench.with_claim(bench.BenchConfig(), "t12").weights == claim_for("t12", "128x128").weights
    with pytest.raises(ValueError, match="no gate claim"):
        bench.with_claim(bench.BenchConfig(width=16, height=12), "t12")
