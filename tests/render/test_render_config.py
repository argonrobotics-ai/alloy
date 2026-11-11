# SPDX-License-Identifier: Apache-2.0
"""Render cloud settings come from the environment or a config file, never from built-in internal defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from alloy.render import cli
from alloy.render.staging import stage_prefix
from alloy.render.vm import CmdResult, GcloudBackend, RenderConfig, VmError

NO_FILE = Path("/nonexistent/render.json")


def test_environment_wins_over_the_config_file(tmp_path: Path) -> None:
    f = tmp_path / "render.json"
    f.write_text(json.dumps({"project": "p-file", "bucket": "b-file", "labels": "team=x", "role": "r-file"}))
    cfg = RenderConfig.load(env={"ALLOY_RENDER_BUCKET": "gs://b-env/", "ALLOY_RENDER_CONFIG": str(f)})
    assert (cfg.project, cfg.bucket, cfg.labels, cfg.role) == ("p-file", "b-env", "team=x", "r-file")


def test_nothing_configured_means_no_bucket_and_the_error_names_the_variable() -> None:
    cfg = RenderConfig.load(env={}, path=NO_FILE)
    assert cfg.project is None and cfg.bucket is None and cfg.vm_labels() == "role=alloy-render"
    with pytest.raises(VmError, match="ALLOY_RENDER_BUCKET"):
        cfg.bucket_uri("stage")


def test_bad_config_file_names_the_file(tmp_path: Path) -> None:
    f = tmp_path / "render.json"
    f.write_text("{not json")
    with pytest.raises(VmError, match="render.json"):
        RenderConfig.load(env={"ALLOY_RENDER_CONFIG": str(f)})


def test_labels_keep_one_role() -> None:
    cfg = RenderConfig(labels="purpose=sim, role=old ,team=a", role="farm")
    assert cfg.vm_labels() == "purpose=sim,team=a,role=farm"


def test_stage_prefix_uses_the_explicit_base_or_the_configured_bucket(caplog: pytest.LogCaptureFixture) -> None:
    assert stage_prefix("gs://x/stage/", "job") == "gs://x/stage/job"
    assert stage_prefix(None, "job", RenderConfig(bucket="b")) == "gs://b/stage/job"
    assert stage_prefix(None, "job", RenderConfig()) == ""  # staging off: rsync, and the log names the variable
    assert "ALLOY_RENDER_BUCKET" in caplog.text


def _recording_exec(calls: list[list[str]]):
    def run(argv: list[str], timeout_s: float = 0) -> CmdResult:
        calls.append(list(argv))
        return CmdResult(0, "[]")
    return run


def test_backend_uses_the_configured_project_labels_and_role() -> None:
    calls: list[list[str]] = []
    cfg = RenderConfig(project="proj", labels="team=a", role="farm")
    be = GcloudBackend(user="u", exec_fn=_recording_exec(calls), config=cfg)
    assert be.project == "proj" and calls == []  # no gcloud lookup when the project is configured
    be.list_owned()
    assert "labels.role=farm" in calls[-1] and calls[-1][-2:] == ["--project", "proj"]


def test_backend_without_any_project_says_what_to_set() -> None:
    def no_project(argv: list[str], timeout_s: float = 0) -> CmdResult:
        return CmdResult(0, "(unset)")
    with pytest.raises(VmError, match="ALLOY_RENDER_PROJECT"):
        GcloudBackend(user="u", exec_fn=no_project, config=RenderConfig())


def test_frames_auto_needs_a_bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = argparse.Namespace(frames_gcs="auto", out=str(tmp_path / "clip1"))
    monkeypatch.setenv("ALLOY_RENDER_CONFIG", str(NO_FILE))
    monkeypatch.delenv("ALLOY_RENDER_BUCKET", raising=False)
    with pytest.raises(SystemExit, match="ALLOY_RENDER_BUCKET"):
        cli._frames_gcs(args)
    monkeypatch.setenv("ALLOY_RENDER_BUCKET", "renders")
    assert cli._frames_gcs(args) == "gs://renders/frames/clip1/frames"
