# SPDX-License-Identifier: Apache-2.0
"""Gate renderer helpers on the CPU: converged accumulation and the command's error path."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.camera import Camera, load_pack
from alloy.camera import gate_render as gr
from tests.camera.packs import write_pack_v1


def test_converged_equals_one_long_render(tmp_path):
    scene = load_pack(write_pack_v1(tmp_path / "v1"), device="cpu")
    lin, alb, nrm = gr._converged(Camera("g", width=8, height=6, spp=1), scene, 8, chunk=4)
    one = Camera("one", width=8, height=6, spp=8).render(scene, output="all")
    assert np.allclose(lin, one.linear.numpy().reshape(lin.shape), atol=1e-5)
    assert np.allclose(nrm, one.normal.numpy().reshape(nrm.shape), atol=1e-5)


def test_main_fails_with_a_reason(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(gr, "png_converter", lambda: (lambda lin, view: np.zeros((1, 1, 3), np.uint8)))
    rc = gr.main(["--scene", "c3", "--res", "8x8", "--frames", "0", "--pack", str(tmp_path / "missing"),
                  "--out", str(tmp_path / "root"), "--device", "cpu"])
    assert rc == 1 and "gate_render c3 8x8" in capsys.readouterr().err


def test_speed_row_halves_worlds_until_they_fit(monkeypatch):
    calls = []

    def fake(pack, frame, res, worlds, gpu, opts):
        calls.append(worlds)
        if worlds > 2:
            raise RuntimeError(f"Failed to allocate {worlds << 30} bytes on device 'cuda:0'")
        return {"res": res, "worlds": worlds, "images_per_s": 10.0}

    monkeypatch.setattr(gr, "_speed_row_at", fake)
    row = gr.speed_row(None, 0, "640x480", 8, "l4", {})
    assert calls == [8, 4, 2] and row["worlds"] == 2 and row["worlds_requested"] == 8 and not row["fits_requested"]
    monkeypatch.setattr(gr, "_speed_row_at", lambda *a: (_ for _ in ()).throw(RuntimeError("kernel launch failed")))
    with pytest.raises(RuntimeError, match="kernel launch"):
        gr.speed_row(None, 0, "640x480", 8, "l4", {})


def test_a_failed_speed_row_does_not_fail_the_quality_images(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(gr, "png_converter", lambda: (lambda lin, view: np.zeros((1, 1, 3), np.uint8)))
    monkeypatch.setattr(gr, "render_frame", lambda *a, **k: [{"engine": "alloy", "res": "8x8", "spp": 1}])
    def no_memory(*a, **k):
        raise RuntimeError("Failed to allocate 1 bytes")

    monkeypatch.setattr(gr, "speed_row", no_memory)
    rc = gr.main(["--scene", "t12", "--res", "8x8", "--frames", "0", "--pack", str(write_pack_v1(tmp_path / "v1")),
                  "--out", str(tmp_path / "root"), "--device", "cpu", "--rows", str(tmp_path / "rows.jsonl")])
    err = capsys.readouterr().err
    assert rc == 0 and "no speed row" in err and not (tmp_path / "rows.jsonl").exists()
