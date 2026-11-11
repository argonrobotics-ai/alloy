# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from bench.render_gates import score
from bench.render_gates.layout import image_name
from bench.render_gates.metrics import Scorer

from .conftest import fake_metric, write


def test_cycles_like_engine_passes_and_noisy_engine_fails(gate_root: Path, scorer: Scorer) -> None:
    rep = score.score_root(gate_root, scorer, gpu="l4")
    cell = rep["cells"]["c3/32x32"]
    assert cell["alloy"]["frames"] == 2
    assert cell["alloy"]["parity"]["status"] == "PASS"
    assert cell["alloy"]["equal_spp"]["status"] == "PASS"
    assert cell["bad"]["equal_spp"]["status"] == "FAIL"
    assert cell["bad"]["equal_spp"]["worst_ratio"] > 1.25
    assert set(cell["_controls"]) == {"cycles_s1", "cycles_s2", "cycles_s3"}
    assert "16spp_on" in cell["alloy"]["g1"]
    json.dumps(rep)  # the report is JSON-serialisable


def test_only_filters_engines(gate_root: Path, scorer: Scorer) -> None:
    rep = score.score_root(gate_root, scorer, only={"alloy"})
    assert set(rep["cells"]["c3/32x32"]) == {"alloy", "_controls"}


def test_size_mismatch_is_reported_not_scored(gate_root: Path, scorer: Scorer) -> None:
    import numpy as np

    write(gate_root / "c3" / "10" / "small" / image_name("32x32", 4, "on"), np.zeros((16, 16, 3), np.uint8))
    rep = score.score_root(gate_root, scorer)
    blk = rep["frames"]["c3/10"]["32x32"]["engines"]
    assert "reference size" in blk["small"]["errors"][0]  # reported with what to change, never scored
    assert blk["small"]["parity"]["status"] == "n/a"
    assert rep["cells"]["c3/32x32"]["alloy"]["parity"]["status"] == "PASS"


def test_disk_cache_reuses_rows(gate_root: Path, tmp_path: Path) -> None:
    sc = Scorer(tmp_path / "cache", metric_fn=fake_metric)
    score.score_root(gate_root, sc)
    first = sc.computed
    sc2 = Scorer(tmp_path / "cache", metric_fn=fake_metric)
    score.score_root(gate_root, sc2)
    assert first > 0 and sc2.computed == 0 and sc2.hits == first
