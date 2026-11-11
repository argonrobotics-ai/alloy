# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bench.render_gates import cli, metrics, nightly

from .conftest import fake_metric


@pytest.fixture(autouse=True)
def cheap_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "pair_metrics", fake_metric)
    orig = metrics.Scorer.__init__

    def init(self, cache_dir=None, metric_fn=None):
        orig(self, cache_dir, metric_fn or fake_metric)

    monkeypatch.setattr(metrics.Scorer, "__init__", init)


def test_spec_prints_json(capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["spec"]) == 0
    assert json.loads(capsys.readouterr().out)["sizes"] == ["128x128", "640x480", "1280x720"]


def test_score_then_matrix(gate_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rep = tmp_path / "rep.json"
    assert cli.main(["score", str(gate_root), "--gpu", "l4", "--out", str(rep), "--cache", ""]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cells"]["c3/32x32"]["alloy"]["parity"] == "PASS"
    m = tmp_path / "m.json"
    assert cli.main(["matrix", "--report", f"l4={rep}", "--out", str(m), "--md", str(tmp_path / "m.md")]) == 0
    assert json.loads(m.read_text())["counts"]["PENDING"] > 0  # the test root only covers c3 at 32x32
    with pytest.raises(SystemExit, match="gpu one of"):
        cli.main(["matrix", "--report", f"h100={rep}"])


def test_nightly_runs_the_render_template(gate_root: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "root").parent.mkdir(parents=True, exist_ok=True)
    (work / "root").symlink_to(gate_root, target_is_directory=True)
    marker = tmp_path / "calls.txt"
    tmpl = f"{sys.executable} -c \"open('{marker}', 'a').write('{{scene}} {{res}} {{frames}} {{pack}}\\n')\""
    cfg = nightly.NightlyConfig(work, "l4", tmpl, scenes=["c3"], sizes=["128x128"], skip_fetch=True)
    m = nightly.run_nightly(cfg)
    assert marker.read_text().split() == ["c3", "128x128", "420,4400,8800,11439,15500", str(work / "packs" / "c3")]
    assert {c["gpu"] for c in m["cells"]} == {"l4"}
    assert (work / "matrix_l4.md").exists() and (work / "report_l4.json").exists()
    assert nightly.exit_code(m, strict=True) == 1
