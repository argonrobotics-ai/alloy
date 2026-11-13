# SPDX-License-Identifier: Apache-2.0
"""audit: bully findings (any of the field spellings) become severity-ranked findings with fixes and replay
handles, merged per class and body with their episode rate; without the bully call the audit says so; the
alloy.check rounds run on CPU with a replay command per finding; the report and CLI are JSON."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from alloy.audit import bully as B
from alloy.audit import cli
from alloy.audit.report import AuditFinding, AuditReport
from alloy.audit.run import audit
from alloy.cards import types as ct

TABLE = ct.Placement("table_top_plywood", (0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0))


def _scene(z: float = 0.0155) -> ct.SceneCard:
    cube = ct.Placement("cube_beech_30", (0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0))
    return ct.SceneCard(robots=(), objects=(TABLE, cube), worlds=1)


def test_bully_findings_convert_and_merge():
    raw = [{"cls": "pen_fail", "severity": "S1", "first_tick": 40, "body": "cube_beech_30", "value": 0.012,
            "threshold": 0.01, "replay": {"episode": 3, "seed": 7}},
           SimpleNamespace(monitor="pen_fail", severity=SimpleNamespace(name="S1"), first_step=55,
                           body="cube_beech_30", value=0.02, threshold=0.01, replay="BUL-0003"),
           {"class": "energy_jump", "severity": "S2", "first_tick": 10}]
    res = B.run_bully(_scene(), episodes=8, seed=0, device="cpu", runner=lambda c, **k: raw, control_dt=0.02)
    assert res.available and len(res.findings) == 2
    pen = next(f for f in res.findings if f.kind == "pen_fail")
    assert pen.severity == "error" and pen.seen == 2 and pen.rate == pytest.approx(0.25)
    assert pen.first_time_s == pytest.approx(0.8) and pen.replay == {"episode": 3, "seed": 7}
    assert "penetration" in pen.fix
    energy = next(f for f in res.findings if f.kind == "energy_jump")
    assert energy.severity == "warn" and "energy" in energy.fix
    # alloy.bully's own Finding spelling: monitor, IntEnum severity, first_step, repro, id
    real = SimpleNamespace(id="BUL-0001", monitor="nonfinite", severity=SimpleNamespace(name="S1"), value=1.0,
                           threshold=0.0, first_step=12, repro="python -m alloy.bully replay BUL-0001")
    f = B.convert(real, 0.02)
    assert (f.kind, f.severity, f.first_time_s) == ("nonfinite", "error", pytest.approx(0.24))
    assert f.replay == {"handle": "python -m alloy.bully replay BUL-0001", "first_tick": 12, "id": "BUL-0001"}
    assert B.convert({"cause": "pen_s1_ratio", "severity": "S1", "first_tick": 0}).fix.startswith(
        "it tripped at the first tick")                                  # a bad start pose, not the inputs


def test_no_bully_call_is_a_note_and_a_crash_is_a_finding(monkeypatch):
    monkeypatch.setattr(B, "_call", lambda: None)
    res = B.run_bully(_scene(), episodes=4, seed=0, device="cpu")
    assert not res.available and "not in this install" in res.note

    def boom(card, **kw):
        raise RuntimeError("engine fell over")

    crash = B.run_bully(_scene(), episodes=4, seed=0, device="cpu", runner=boom)
    assert crash.findings[0].severity == "error" and "engine fell over" in crash.findings[0].message


def test_check_rounds_and_bully_merge_into_one_report():
    rep = audit(_scene(0.012), rounds=2, episodes=4, device="cpu", worlds=1, seconds=0.1,
                runner=lambda c, **k: [{"cls": "teleport", "severity": "S2", "body": "cube_beech_30"}])
    d = rep.to_dict()
    assert not rep.ok and d["counts"]["error"] >= 1                     # the 3 mm start overlap
    over = next(f for f in d["findings"] if f["kind"] == "overlap")
    assert over["source"] == "check" and over["rate"] == 1.0 and over["seen"] == 2
    assert "--seed 0" in over["replay"]["command"] and "lift cube_beech_30" in over["fix"]
    assert any(f["source"] == "bully" and f["kind"] == "teleport" for f in d["findings"])
    assert d["runs"]["bully"] is True and d["schema"] == "alloy.audit/1"
    assert d["findings"][0]["severity"] == "error"


def test_report_json_and_limits():
    r = AuditReport("s")
    r.add(AuditFinding("check", "warn", "rest_drift", "m", "f", value=float("nan")))
    assert json.loads(r.to_json())["findings"][0]["value"] is None and r.ok
    with pytest.raises(ValueError):
        audit(_scene(), rounds=0)


def test_cli_exit_codes(tmp_path, capfd, monkeypatch):
    from alloy.cards.loader import dump_card

    (tmp_path / "s.yaml").write_text(dump_card(_scene()), encoding="utf-8")
    monkeypatch.setattr(B, "_call", lambda: None)
    code = cli.main([str(tmp_path / "s.yaml"), "--rounds", "1", "--device", "cpu", "--worlds", "1", "--seconds",
                     "0.1"])
    out = json.loads(capfd.readouterr().out)
    assert code == 0 and out["ok"] and out["runs"]["bully"] is False
    assert cli.main(["x", "--rounds", "0"]) == 2


def test_the_real_bully_call_runs_a_short_cpu_audit():
    from alloy.bully import audit as bully_audit

    if not callable(bully_audit):
        pytest.skip("this install has no alloy.bully.audit")
    report = audit(_scene(), rounds=1, episodes=2, device="cpu", worlds=1, seconds=0.2)
    assert report.runs["bully"] is True and report.runs["episodes"] == 2
    summary = report.runs["bully_summary"]
    assert summary["channel"].startswith("push") and sum(summary["counts"].values()) == 2      # the cube gets pushed
    assert report.ok, report.summary()
    json.loads(report.to_json())


ARM = ("<mujoco><worldbody><body name='b' pos='0 0.2 0.1'><joint name='j' type='hinge' axis='0 0 1' "
       "range='-1.5 1.5'/><geom type='box' size='0.15 0.02 0.02' mass='1.0'/></body></worldbody><actuator>"
       "<position joint='j' kp='5000' kv='1'/></actuator></mujoco>")
RECIPE = {"dt": 0.01, "objects": [
    {"add": "ground"},
    {"add": "box", "name": "block_a", "size": [0.1, 0.1, 0.1], "pos": [0.0, 0.0, 0.05], "mass": 0.5},
    {"add": "box", "name": "block_b", "size": [0.1, 0.1, 0.1], "pos": [0.02, 0.0, 0.05], "mass": 0.5},   # overlaps a
    {"add": "robot", "name": "arm", "mjcf": ARM}]}


def test_replays_of_an_audited_recipe_work_after_the_call(tmp_path, monkeypatch):
    """Gemini audit S2-016: a recipe was written to a temporary folder deleted before the reply, so the check
    replay command exited 2 and the bully handle named a deleted robot file. The cards now stay."""
    import subprocess
    import sys
    from pathlib import Path

    import alloy.compose
    from alloy.audit.ops import op_audit
    from alloy.bully import replay_finding

    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    res = op_audit(None, {"scene": RECIPE, "rounds": 1, "episodes": 2, "worlds": 1, "seconds": 0.05,
                          "device": "cpu"})
    cards = Path(res["cards"])
    assert cards.parent == (tmp_path / "cache" / "audit").resolve() and (cards / "scenes" / "scene.yaml").is_file()
    check = next(f for f in res["findings"] if f["source"] == "check" and f["kind"] == "overlap")
    cmd = check["replay"]["command"]
    assert cmd.startswith("python -m alloy.check ") and str(cards) in cmd
    proc = subprocess.run([sys.executable, *_argv(cmd)], capture_output=True, text=True, check=False)
    again = json.loads(proc.stdout)
    assert proc.returncode == 1 and "error" not in again                        # it ran: findings, not a usage error
    assert any(f["check"] == "overlap" for f in again["findings"])
    replay = next(f for f in res["findings"] if f["source"] == "bully")["replay"]
    assert replay["library"][0] == str(cards) and "replay.get('library'" in replay["how"]
    robots = [ln.split("path:")[1].strip() for ln in replay["handle"]["scene"].splitlines() if "path:" in ln]
    assert robots and all((cards / r).is_file() for r in robots)              # relative to the kept folder, or absolute
    assert replay_finding(replay["handle"], "cpu", library=alloy.compose.library(replay["library"])) is not None


def test_cards_names_the_folder_and_card_scenes_write_nothing(tmp_path, monkeypatch):
    from alloy.audit.ops import op_audit, scene_files

    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(B, "_call", lambda: None)
    res = op_audit(None, {"scene": RECIPE, "rounds": 1, "worlds": 1, "seconds": 0.05, "device": "cpu",
                          "cards": str(tmp_path / "kept")})
    assert res["cards"] == str((tmp_path / "kept").resolve()) and (tmp_path / "kept/scenes/scene.yaml").is_file()
    assert scene_files("demo_peg_d12", None) == ("demo_peg_d12", None, None) and not (tmp_path / "cache").exists()


def _argv(command: str) -> list[str]:
    import shlex

    return shlex.split(command)[1:]                                             # without the leading "python"
