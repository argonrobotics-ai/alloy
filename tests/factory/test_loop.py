# SPDX-License-Identifier: Apache-2.0
"""factory_loop end to end on CPU with the scripted fake runner (SPEC §7 WP7 acceptance), plus the sandbox."""

from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
from pathlib import Path

import pytest

from alloy.bully.types import SearchConfig
from alloy.factory.loop import LoopConfig, factory_loop, round_brief
from alloy.factory.runner import ShellAgentRunner, repo_root
from alloy.testing.factory import STRAY, TAMPERED, CopyTemplateRunner

CFG = LoopConfig(attack=SearchConfig(worlds=8, generations=1, horizon_s=0.5, budget_gpu_h=0.01), replay_runs=2,
                 eval_timeout_s=170.0)


@pytest.fixture
def brief(tmp_path: Path) -> Path:
    p = tmp_path / "spec.md"
    p.write_text("Write a particle solver that rests on the ground plane without gaining energy.\n")
    return p


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _worktrees() -> set[str]:
    """Worktree paths of the repository (compared as sets: other runs may prune unrelated stale entries)."""
    out = subprocess.run(["git", "-C", str(repo_root()), "worktree", "list", "--porcelain"], capture_output=True,
                         text=True, check=True).stdout
    return {line[len("worktree "):] for line in out.splitlines() if line.startswith("worktree ")}


@pytest.mark.compile_heavy
@pytest.mark.timeout(180)
def test_loop_end_to_end_with_copy_template_runner(brief, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # one round keeps this inside the compile_heavy budget on a loaded host; round feedback is tested separately
    verdicts = factory_loop(brief, CopyTemplateRunner(), "tfake", rounds=1,
                            cfg=dataclasses.replace(CFG, scratch_parent=scratch))
    v = verdicts[-1]
    assert len(verdicts) == 1
    assert v.plugin == "tfake" and v.contract_ok, v.reasons
    assert v.findings == ()
    assert [r.law for r in v.law_results] == ["FAC.tfake.freefall"]
    expected = "pass" if _has("alloy.bench.law") else "not_run"  # the law runs once the bench library (WP6) is in
    assert v.law_results[0].verdict == expected, v.law_results[0].monitors
    assert all("not_run" in r for r in v.reasons)  # only missing producers (WP5 lint, WP6 bench) block shipping
    assert v.ship == (not v.reasons)
    # our scratch worktrees are gone (other runs share the worktree list, so only ours are checked)
    assert not [w for w in _worktrees() if w.startswith(str(scratch))]


@pytest.mark.compile_heavy
@pytest.mark.timeout(180)
def test_writes_outside_the_package_are_reverted_and_reported(brief):
    verdicts = factory_loop(brief, CopyTemplateRunner(stray_write=True), "tfake", rounds=1, cfg=CFG)
    assert any(r.startswith("sandbox") and str(STRAY) in r for r in verdicts[0].reasons)
    assert not verdicts[0].ship
    assert not (repo_root() / STRAY).exists()


def test_agent_never_sees_the_hidden_valid_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOY_HIDDEN_VALID", "/secret/rigs")
    out = tmp_path / "pkg"
    out.mkdir()
    runner = ShellAgentRunner("sh -c 'env > {out}/env.txt'")
    assert runner.run(tmp_path / "brief.md", tmp_path, out, timeout_s=10) == 0
    env = (out / "env.txt").read_text()
    assert "ALLOY_HIDDEN_VALID" not in env and "/secret/rigs" not in env


def test_round_brief_carries_feedback():
    from alloy.factory.types import FactoryVerdict

    prev = FactoryVerdict("p", False, (), (), False, ("contract no_alloc_in_step: 3 allocations",))
    text = round_brief("spec text", "p", prev, 2)
    assert "round 2" in text and "spec text" in text and "no_alloc_in_step" in text and "contrib/p/" in text


@pytest.mark.compile_heavy
@pytest.mark.timeout(180)
def test_committed_tampering_is_reported_and_never_evaluated(brief):
    verdicts = factory_loop(brief, CopyTemplateRunner(commit_outside=True), "tfake", rounds=1, cfg=CFG)
    v = verdicts[0]
    assert any(r.startswith("sandbox") and str(TAMPERED) in r for r in v.reasons), v.reasons
    assert v.contract_ok and not v.ship  # evaluated from the pristine tree with the package alone
    assert "tampered" not in (repo_root() / TAMPERED).read_text()


def test_manifest_must_name_the_agent_package(brief):
    verdicts = factory_loop(brief, CopyTemplateRunner(template_manifest=True), "tfake", rounds=1, cfg=CFG)
    assert not verdicts[0].ship and not verdicts[0].contract_ok
    assert any("manifest module is 'alloy.factory.template'" in r for r in verdicts[0].reasons), verdicts[0].reasons
