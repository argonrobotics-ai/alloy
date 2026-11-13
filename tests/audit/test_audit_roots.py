# SPDX-License-Identifier: Apache-2.0
"""audit finds what alloy.check finds (Gemini audit S2-032): the cards beside a saved scene, objects in a Hub fetch
cache with no import root, and a replay command that names the scene file by its full path."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys

import pytest

import alloy.compose
from alloy.audit import audit, library
from alloy.bully import replay_finding

BLOCK = ("schema: alloy.object/1\nname: {name}\nshape:\n  primitive:\n    kind: box\n"
         "    size: [{s}, {s}, {s}]\nmaterial: wood_beech\n")
SCENE = ("schema: alloy.scene/1\nrobots: []\nworlds: 1\ncontrol_dt: 0.02 s\nobjects:\n"
         "  - object: table_top_plywood\n    pose: [0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0]\n"
         "  - object: {obj}\n    pose: [0.0, 0.0, {z}, 0.0, 0.0, 0.0, 1.0]\n")
RUN = {"rounds": 1, "episodes": 1, "seconds": 0.04, "worlds": 1, "device": "cpu"}


def _folder(root, obj: str, half: float = 0.02, z: float = 0.0205, scene: str = "scene"):
    """A saved card folder: ``objects/<obj>.yaml`` (a wooden cube) and ``scenes/<scene>.yaml`` placing it on a table."""
    (root / "objects").mkdir(parents=True)
    (root / "scenes").mkdir()
    (root / "objects" / f"{obj}.yaml").write_text(BLOCK.format(name=obj, s=half), encoding="utf-8")
    path = root / "scenes" / f"{scene}.yaml"
    path.write_text(SCENE.format(obj=obj, z=z), encoding="utf-8")
    return path


def test_a_saved_scene_resolves_the_cards_beside_it_with_and_without_library(tmp_path):
    path = _folder(tmp_path / "my_scenes", "made_block")
    other = tmp_path / "my_objects"
    other.mkdir()
    for lib in (None, [str(other)]):
        report = audit(str(path), library=lib, **RUN)
        assert report.ok and not report.findings, report.summary()            # was: unknown object + a bully crash
        assert report.runs["bully"] is True


def test_an_object_in_a_fetch_cache_without_an_import_root_is_found(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "imports" / "ycb" / "objects"
    cache.mkdir(parents=True)
    (cache / "fetched_block.yaml").write_text(BLOCK.format(name="fetched_block", s=0.02), encoding="utf-8")
    scenes = tmp_path / "hub_scenes" / "scenes"
    scenes.mkdir(parents=True)
    (scenes / "scene.yaml").write_text(SCENE.format(obj="fetched_block", z=0.0205), encoding="utf-8")
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    assert library().has_object("fetched_block") and alloy.compose.library().has_object("fetched_block")
    report = audit(str(scenes / "scene.yaml"), **RUN)
    assert report.ok, report.summary()


def test_a_scene_folders_own_card_is_not_shadowed_by_a_fetch_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "imports" / "ycb" / "objects"
    cache.mkdir(parents=True)
    (cache / "made_block.yaml").write_text(BLOCK.format(name="made_block", s=0.3), encoding="utf-8")   # another size
    _folder(tmp_path / "my_scenes", "made_block")
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    for lib in (library([str(tmp_path / "my_scenes")]), alloy.compose.library([str(tmp_path / "my_scenes")])):
        assert lib.roots[0] == (tmp_path / "my_scenes").resolve()
        assert lib.object("made_block").shape.size[0] == pytest.approx(0.02)


def test_replays_name_the_scene_file_and_its_folder(tmp_path):
    path = _folder(tmp_path / "ov_lib", "made_block", z=0.010, scene="my_overlap")   # sunk 10 mm into the table
    report = audit(str(path), **RUN)
    check = next(f for f in report.findings if f.source == "check" and f.kind == "overlap")
    cmd = check.replay["command"]
    assert f"alloy.check {shlex.quote(str(path.resolve()))} " in cmd
    proc = subprocess.run([sys.executable, *shlex.split(cmd)[1:]], capture_output=True, text=True, check=False)
    again = json.loads(proc.stdout)
    assert proc.returncode == 1 and "error" not in again and any(f["check"] == "overlap" for f in again["findings"])
    bully = [f for f in report.findings if f.source == "bully" and "handle" in (f.replay or {})]
    for f in bully[:1]:
        assert f.replay["library"] == [str((tmp_path / "ov_lib").resolve())]
        assert "alloy.audit.library" in f.replay["how"]
        replay_finding(f.replay["handle"], "cpu", library=library(f.replay["library"]))      # resolves made_block


def test_a_scene_card_that_does_not_load_raises_a_scene_error_with_the_fix(tmp_path):
    """Gemini audit S2-034 (audit half): a raw CardError leaked out of audit."""
    import alloy

    path = _folder(tmp_path / "bad", "made_block")
    path.write_text("name: my_scene\n" + path.read_text(encoding="utf-8"), encoding="utf-8")     # not a SceneCard key
    with pytest.raises(alloy.SceneError) as e:
        audit(str(path), **RUN)
    assert "unknown key" in e.value.message and e.value.fix.startswith("edit ") and e.value.field == "scene"
