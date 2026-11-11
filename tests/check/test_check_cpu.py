# SPDX-License-Identifier: Apache-2.0
"""OSS-7 end to end on CPU: a beech cube on a plywood table top (library cards). Load checks catch a start overlap
and say how far to lift; a short run of the resting cube is clean; the CLI prints strict JSON with exit codes; bad
inputs come back as findings or as errors that say what to pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import alloy.check
from alloy.cards import types as ct
from alloy.cards.loader import dump_card
from alloy.check import cli
from alloy.check.source import SceneInputError, load_source

TABLE = ct.Placement("table_top_plywood", (0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0))   # top face at z = 0


def _scene(cube_z: float, worlds: int = 1) -> ct.SceneCard:
    cube = ct.Placement("cube_beech_30", (0.0, 0.0, cube_z, 0.0, 0.0, 0.0, 1.0))    # 30 mm cube, half 15 mm
    return ct.SceneCard(robots=(), objects=(TABLE, cube), worlds=worlds)


def test_overlap_at_load_says_how_far_to_lift():
    report = alloy.check.check_load(_scene(0.012), worlds=2, device="cpu")       # 3 mm into the table
    assert report.checks["cards"] != "fail" and report.checks["overlap"] == "fail"
    (f,) = [f for f in report.findings if f.check == "overlap"]
    assert f.level == "error" and set(f.objects) == {"cube_beech_30", "table_top_plywood"}
    assert f.value == pytest.approx(0.003, abs=2e-4)
    assert "lift cube_beech_30 by 3.5 mm" in f.fix and "in every world" in f.message
    assert f.suggestion["field"] == "objects.1.pose" and f.suggestion["pose"][2] == pytest.approx(0.0155)
    assert report.checks["rest_drift"].startswith("skipped")


def test_apply_fixes_clears_the_overlap():
    card = _scene(0.012)
    fixed, applied = alloy.check.apply_fixes(card, alloy.check.check_load(card, device="cpu"))
    assert applied == ["objects.1.pose: lift cube_beech_30 by 3.5 mm (pose z 0.012 -> 0.0155 m)"]
    assert fixed.objects[1].pose[2] == pytest.approx(0.0155) and fixed.objects[0] == card.objects[0]
    assert alloy.check.check_load(fixed, device="cpu").checks["overlap"] == "pass"


def test_resting_cube_runs_clean():
    report = alloy.check(_scene(0.015), worlds=1, seconds=0.1, device="cpu")
    assert report.ok, report.summary()
    assert {k: v for k, v in report.checks.items() if k != "cards"} == {
        "overlap": "pass", "rest_drift": "pass", "health": "pass", "penetration": "pass", "energy": "pass"}
    assert report.metrics["run"]["periods"] == 5
    pen = report.metrics["penetration"]
    assert pen["end_pen_m"] is not None and pen["end_pen_m"] <= pen["max_pen_m"]       # end depth = sim.health()'s
    assert report.metrics["rest_drift"]["cube_beech_30"]["drift_m"] < 5e-4


def test_unknown_object_is_a_card_error_with_close_match():
    bad = ct.SceneCard(robots=(), objects=(TABLE, ct.Placement("cube_beech_3", (0, 0, 0.015, 0, 0, 0, 1))))
    report = alloy.check(bad, device="cpu")
    (f,) = report.errors()
    assert f.check == "cards" and "cube_beech_30" in f.fix and f.suggestion["replace_with"] == "cube_beech_30"
    assert report.checks["overlap"] == "skipped: card errors (fix them first)"
    fixed, applied = alloy.check.apply_fixes(bad, report)
    assert fixed.objects[1].object == "cube_beech_30" and len(applied) == 1


def test_scene_file_with_a_bad_key_is_a_finding(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(dump_card(_scene(0.015)) + "world_count: 3\n", encoding="utf-8")
    report = alloy.check(path, device="cpu", run=False)
    assert not report.ok and report.errors()[0].check == "cards" and "world_count" in report.errors()[0].message


def test_load_source_names_and_errors(tmp_path: Path):
    assert load_source("demo_peg_d12").label == "demo_peg_d12"
    with pytest.raises(SceneInputError, match="did you mean"):
        load_source("demo_peg_d13")
    with pytest.raises(SceneInputError, match="does not exist"):
        load_source(str(tmp_path / "missing.yaml"))
    with pytest.raises(SceneInputError, match="pass a scene YAML path"):
        load_source(42)


def test_card_root_next_to_scenes_is_searched(tmp_path: Path):
    root = tmp_path / "mycards"
    (root / "scenes").mkdir(parents=True)
    path = root / "scenes" / "s.yaml"
    path.write_text(dump_card(_scene(0.015)), encoding="utf-8")
    src = load_source(path)
    assert src.label == "s" and root.resolve() in src.lib.roots


def test_json_scene_text_and_stdin(monkeypatch, capsys):
    import io

    from alloy.cards.loader import to_data

    text = json.dumps(to_data(_scene(0.012)))
    assert load_source(text).card == _scene(0.012)
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    assert cli.main(["-", "--json", "--quick", "--device", "cpu"]) == 1
    assert json.loads(capsys.readouterr().out)["checks"]["overlap"] == "fail"


def test_cli_json_and_exit_codes(tmp_path: Path, capsys):
    path = tmp_path / "scenes" / "overlap.yaml"
    path.parent.mkdir()
    path.write_text(dump_card(_scene(0.012)), encoding="utf-8")
    code = cli.main([str(path), "--json", "--quick", "--device", "cpu", "--out", str(tmp_path / "r.json")])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["ok"] is False and out["checks"]["overlap"] == "fail"
    assert json.loads((tmp_path / "r.json").read_text()) == out
    assert cli.main(["no_such_scene_xyz", "--json"]) == 2
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False and "no scene file" in err["error"] and err["fix"]


def test_check_sim_on_a_running_simulation():
    from alloy.cards.library import CardLibrary
    from alloy.integration.wiring import build_simulation
    from alloy.scene.assembler import SceneAssembler

    lib = CardLibrary()
    # 7 mm into the table: the cube's contact takes the multi-point resting floor (damping ratio 0.7, WP3 2026-11-07),
    # so a 3 mm overlap now releases at 0.21 m/s without an energy jump; 7 mm still latches health
    engine = build_simulation(SceneAssembler(lib).assemble(_scene(0.008, worlds=2), "cpu", world_count=2))
    out = alloy.check.check_sim(engine)                       # before any step: the start overlap
    assert out["source"] == "alloy.check" and out["ok"] is False
    (p,) = [p for p in out["problems"] if p["check"] == "overlap"]
    assert p["level"] == "error" and "lift cube_beech_30" in p["fix"] and "at start" in p["what"]
    engine.step_period()                     # the overlap releases its energy: health latches both worlds
    out = alloy.check.check_sim(engine)
    assert out["report"]["checks"]["health"] == "fail" and out["report"]["settings"]["t"] == pytest.approx(0.02)
    (h,) = [p for p in out["problems"] if p["check"] == "health"]
    assert "ENERGY" not in h["what"] and "energy jumped" in h["what"] and "fix overlaps" in h["fix"]
    assert out["report"]["checks"]["rest_drift"].startswith("skipped")
    assert out["report"]["checks"]["overlap"].startswith("skipped: start poses only")


def test_check_sim_resting_contact_is_not_an_overlap():
    from alloy.cards.library import CardLibrary
    from alloy.integration.wiring import build_simulation
    from alloy.scene.assembler import SceneAssembler

    engine = build_simulation(SceneAssembler(CardLibrary()).assemble(_scene(0.03), "cpu", world_count=1))
    for _ in range(10):                                       # drops 15 mm and rests on the table
        engine.step_period()
    out = alloy.check.check_sim(engine)
    assert out["ok"], out["problems"]
    assert not [p for p in out["problems"] if p["check"] == "overlap"]


def test_check_object_rests_on_a_generated_floor():
    report = alloy.check.check_object("cube_beech_30", worlds=1, seconds=0.1, device="cpu")
    assert report.ok and report.scene == "drop test: cube_beech_30", report.summary()
    assert report.metrics["rest_drift"]["cube_beech_30"]["settle_m"] == pytest.approx(5e-4, abs=1e-4)   # the gap
    assert not any("alloy_check_floor" in f.objects for f in report.findings)
    with pytest.raises(ValueError, match="library object name"):
        alloy.check.check_object("no_such_object_xyz")


def test_extra_run_checks_join_the_report():
    from alloy.check.report import Finding

    def soft(res, built, index):               # an extra check on the same run (SCR-OSS22-2)
        assert res.periods == 5 and "cube_beech_30" in index.names and built.world_count == 1
        return [Finding("soft", "warn", "fill left its container", "lower the fill level")], {"kept": 0.9}

    def broken(res, built, index):
        raise RuntimeError("boom")

    broken.check_name = "workspace"
    report = alloy.check(_scene(0.015), worlds=1, seconds=0.1, device="cpu", extra=[soft, broken])
    assert report.checks["soft"] == "warn" and report.metrics["soft"] == {"kept": 0.9}
    assert report.checks["workspace"] == "warn"                 # a crashing extra check warns, the report stands
    assert any("raised RuntimeError: boom" in f.message for f in report.findings if f.check == "workspace")
    assert json.loads(report.to_json())["findings"][0]["check"] in ("soft", "workspace")
    quick = alloy.check.check_load(_scene(0.015), device="cpu", extra=[soft])
    assert quick.checks["soft"].startswith("skipped: run=False")
    with pytest.raises(ValueError, match="lowercase name"):
        Finding("Soft Check", "warn", "m", "f")


def test_check_object_wider_than_the_default_floor():
    import dataclasses

    import numpy as np

    from alloy.cards.library import CardLibrary

    # SCR-SOFT-1-3: a 0.5 m object needs a 1.5 m floor (size an np.float64), and a card built from numpy numbers
    plate = dataclasses.replace(CardLibrary().peek_object("cube_beech_30"), name="plate_wide_check",
                                shape=ct.Primitive("box", (np.float64(0.25), np.float64(0.25), np.float64(0.01))))
    report = alloy.check.check_object(plate, worlds=1, seconds=0.1, device="cpu")
    assert report.checks["cards"] != "fail" and report.checks["overlap"] == "pass", report.summary()
    assert report.metrics["scene"]["bodies"] == 1
