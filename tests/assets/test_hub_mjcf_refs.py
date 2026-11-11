# SPDX-License-Identifier: Apache-2.0
"""Objects whose MJCF model uses another object's files (RoboCasa's composed cabinets use a door panel's and a handle's
meshes): `alloy fetch` brings those objects too, and the on-first-use fetch does not take a cabinet cached without
them as complete. CPU, a toy library in RoboCasa's layout served by a fake Hub."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from alloy.assets.hub import build, card, fetch, lazy, mjcf, source
from alloy.assets.hub.layout import sha256_file

LIB = "toycasa"
CARD = "schema: alloy.object/1\nname: {name}\nshape:\n  mjcf:\n    path: {model}\n    body: {name}\n"
PANEL_XML = ('<mujoco model="panel_001"><asset><mesh name="m" file="visuals/panel.msh"/></asset>'
             '<worldbody><body name="panel_001"><geom type="mesh" mesh="m"/></body></worldbody></mujoco>')
CAB_XML = ('<mujoco model="cab_001"><asset><mesh name="v" file="../../../mjcf/panels/panel_001/visuals/panel.msh"/>'
           '{extra}</asset><worldbody><body name="cab_001"><geom type="box" size="0.3 0.3 0.3"/>'
           '<geom type="mesh" mesh="v"/></body></worldbody></mujoco>')


def _asset(root: Path, name: str, category: str, files: dict[str, str]) -> dict:
    """Write one asset's files (model, mesh) and card under ``root``; its RoboCasa-style manifest entry."""
    model = f"mjcf/{category}/{name}/model.xml"
    files = {model: files.pop("model.xml"), **{f"mjcf/{category}/{name}/{k}": v for k, v in files.items()},
             f"objects/{category}/{name}.yaml": CARD.format(name=name, model=model)}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return {"card": f"objects/{category}/{name}.yaml", "model": model, "kind": "fixture", "licence": "CC-BY-4.0",
            "attribution": f"{name} by someone, CC BY 4.0", "status": "ok",
            "files": {rel: {"sha256": sha256_file(root / rel)} for rel in files}}


@pytest.fixture
def casa(tmp_path, monkeypatch):
    """A fake Hub serving a staged toy library: panel_001 (a model and its mesh), cab_001 (uses panel_001's mesh),
    cab_002 (also names a mesh no object ships). Returns (downloaded paths, cache root)."""
    root = tmp_path / "imports" / LIB
    assets = {"panel_001": _asset(root, "panel_001", "panels", {"model.xml": PANEL_XML, "visuals/panel.msh": "msh"}),
              "cab_001": _asset(root, "cab_001", "composed", {"model.xml": CAB_XML.format(extra="")}),
              "cab_002": _asset(root, "cab_002", "composed", {"model.xml": CAB_XML.format(
                  extra='<texture name="t" type="2d" file="gone.png"/>')})}
    (root / "manifest.json").write_text(json.dumps({"assets": assets}))
    out, rows = tmp_path / "ds", tmp_path / "rows" / f"{LIB}.jsonl"
    assert not build.stage(LIB, root, tmp_path / "store", out, rows)["failed"]
    (tmp_path / "texts").mkdir()
    (tmp_path / "texts" / "CC-BY-4.0.txt").write_text("Attribution 4.0 International")
    card.write_top(rows.parent, tmp_path / "texts", out)
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    calls: list[str] = []

    def fake_download(path, local_dir, repo=None):
        calls.append(path)
        dst = Path(local_dir) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / path, dst)
        return dst

    monkeypatch.setattr(source, "download", fake_download)
    monkeypatch.setattr(fetch, "download", fake_download)
    return calls, tmp_path / "cache" / "imports" / LIB


def test_named_files_join_the_model_folder_and_compiler_folders(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.xml").write_text('<mujoco><compiler meshdir="../meshes" texturedir="tex"/><include '
                                          'file="b.xml"/><asset><mesh file="x.obj"/><texture file="t.png"/></asset>'
                                          '</mujoco>')
    (tmp_path / "m" / "b.xml").write_text('<mujoco><asset><hfield file="h.png"/></asset></mujoco>')
    (tmp_path / "m" / "c.xml").write_text("not xml <")
    assert mjcf.named_files(tmp_path, ["m/a.xml", "m/c.xml", "meshes/x.obj"]) == [
        "m/b.xml", "m/h.png", "m/tex/t.png", "meshes/x.obj"]


def test_fetch_brings_the_objects_a_model_uses_files_of(casa):
    calls, base = casa
    res = fetch.fetch(names=["cab_001"])
    assert res["required"] == ["panel_001"] and res["objects"] == 2 and res["unresolved"] == []
    assert (base / "mjcf/panels/panel_001/visuals/panel.msh").read_text() == "msh"
    assert (base / "objects/panels/panel_001.yaml").is_file()          # the whole panel object, not just its mesh
    assert mjcf.absent_files(base, ["mjcf/composed/cab_001/model.xml"]) == []
    calls.clear()
    again = fetch.fetch(names=["cab_001"])
    assert again["required"] == ["panel_001"] and again["downloaded"] == 0
    assert not any(c.startswith("objects/") for c in calls)              # nothing downloaded twice


def test_a_file_no_object_ships_is_reported_not_fatal(casa, caplog):
    with caplog.at_level(logging.WARNING, "alloy.assets.hub"):
        res = fetch.fetch(names=["cab_002"])
    assert res["required"] == ["panel_001"] and res["unresolved"] == [f"{LIB}/mjcf/composed/cab_002/gone.png"]
    assert "not in the object library" in caplog.text


def test_a_cabinet_cached_without_its_panel_is_fetched_again(casa):
    calls, base = casa
    fetch.fetch(names=["cab_001"])
    assert lazy._complete_root("cab_001") == base
    shutil.rmtree(base / "mjcf/panels")                                # an older fetch: the cabinet's files only
    assert lazy._complete_root("cab_001") is None

    class Lib:
        roots: tuple = ()

        def rescan(self):
            pass

    lazy.ensure(Lib(), "cab_001")
    assert (base / "mjcf/panels/panel_001/visuals/panel.msh").is_file()
