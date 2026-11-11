# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.render_gates import matrix, score
from bench.render_gates.metrics import Scorer

from .conftest import make_frame


@pytest.fixture
def spec() -> dict:
    s = matrix.load_spec()
    s["scenes"] = {"c3": {"title": "C3", "frames": ["10"]}}
    s["sizes"] = ["32x32"]
    s["claims"] = {"32x32": {"spp": 16, "den": "on"}}
    s["speed"] = [{"name": "bar", "gpu": "rtx_pro_6000", "match": {"worlds": 1024}, "min_images_per_s": 1000},
                  {"name": "reported", "match": {"worlds": 1024}}]
    return s


def test_shipped_spec_is_complete() -> None:
    s = matrix.load_spec()
    assert list(s["scenes"]) == ["c2", "c3", "c4", "dc5", "t12"]
    assert s["sizes"] == ["128x128", "640x480", "1280x720"]
    assert set(s["claims"]) == set(s["sizes"])


def test_missing_key_names_it(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"scenes": {}}))
    with pytest.raises(ValueError, match="lacks 'sizes'"):
        matrix.load_spec(p)


def test_cells_pass_fail_and_pending(tmp_path: Path, scorer: Scorer, spec: dict) -> None:
    root = tmp_path / "root"
    make_frame(root, "c3", "10", {"cycles_s1": 1.0, "cycles_s2": 1.0, "alloy": 1.0, "alloy_e2e": 0.5})
    rep = score.score_root(root, scorer, gpu="l4")
    rows = [{"gpu": "rtx_pro_6000", "scene": "c3", "res": "32x32", "worlds": 1024, "spp": 16, "images_per_s": 800}]
    m = matrix.build_matrix(spec, {"l4": rep, "rtx_pro_6000": rep}, rows)
    by = {c["gpu"]: c for c in m["cells"]}
    assert by["l4"]["status"] == "PASS", by["l4"]
    assert by["rtx_pro_6000"]["status"] == "FAIL"  # speed bar 800 < 1000
    assert by["l4"]["g1"]["setting"] == "16spp_on"
    none = matrix.build_matrix(spec, {}, [])
    assert none["counts"] == {"PASS": 0, "FAIL": 0, "PENDING": 2}


def test_markdown_has_one_row_per_scene_and_size(spec: dict) -> None:
    md = matrix.to_markdown(matrix.build_matrix(spec, {}, []), spec)
    assert md.splitlines()[2] == "| C3 | 32x32 | PENDING | PENDING |"


def test_render_failure_fails_the_cell(spec: dict) -> None:
    rep = {"cells": {}, "meta": {"renders": [{"scene": "c3", "res": "32x32", "rc": 1,
                                              "stderr_tail": "Traceback\nNotImplementedError: glass: Transmission is used\n"}]}}
    m = matrix.build_matrix(spec, {"l4": rep}, [])
    cell = next(c for c in m["cells"] if c["gpu"] == "l4")
    assert cell["status"] == "FAIL" and "Transmission" in cell["render"]["reason"]
    assert "render: NotImplementedError: glass" in matrix.to_markdown(m, spec)


def test_per_scene_claim_overrides_the_size_default(tmp_path: Path, scorer: Scorer, spec: dict) -> None:
    root = tmp_path / "root"
    make_frame(root, "c3", "10", {"cycles_s1": 1.0, "cycles_s2": 1.0, "alloy": 1.0, "alloy_e2e": 0.5})
    rep = score.score_root(root, scorer)
    spec["scenes"]["c3"]["claims"] = {"32x32": {"spp": 64, "den": "on"}}
    rows = [{"gpu": "rtx_pro_6000", "scene": "c3", "res": "32x32", "worlds": 1024, "spp": 16, "images_per_s": 5000},
            {"gpu": "rtx_pro_6000", "scene": "c3", "res": "32x32", "worlds": 1024, "spp": 64, "images_per_s": 700}]
    m = matrix.build_matrix(spec, {"l4": rep, "rtx_pro_6000": rep}, rows)
    by = {c["gpu"]: c for c in m["cells"]}
    assert by["l4"]["g1"]["setting"] == "64spp_on"
    gated = [s for s in by["rtx_pro_6000"]["speed"] if s["gated"]]
    assert gated[0]["value"] == 700 and by["rtx_pro_6000"]["status"] == "FAIL"  # the 16 spp row is not the claim


def test_shipped_claims_follow_the_ruling() -> None:
    s = matrix.load_spec()
    assert matrix.claim_for(s, "t12", "640x480")["spp"] == 64
    assert matrix.claim_for(s, "t12", "128x128") == {"spp": 40, "den": "on", "denoiser": "learned t12_128, tta 1",
                                                     "weights": "t12_128"}
    assert matrix.claim_for(s, "c3", "1280x720") == {"spp": 16, "den": "on", "denoiser": "learned c3_1280, tta 1",
                                                     "weights": "c3_1280"}
    assert matrix.claim_for(s, "c3", "640x480")["spp"] == 12
