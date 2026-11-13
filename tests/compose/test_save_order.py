# SPDX-License-Identifier: Apache-2.0
"""ComposeResult.save and the compose command's ``saved`` list put the scene cards first, so the documented
``alloy.load(str(paths[0]))`` loads the composed scene. Before, generated object cards and copied robot files came
first: on a robot scene ``paths[0]`` loaded the bare robot on a ground plane (table and mug gone, no error), and on a
floor, fill, towel or rigid-copy scene it was an object card ("not a scene card")."""

from __future__ import annotations

import io
import json

import pytest

import alloy
from alloy.cards.library import CardLibrary, default_root
from alloy.cli import main as cli_main
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


def _rel(paths, root):
    return [p.relative_to(root).as_posix() for p in paths]


def test_robot_scene_paths0_is_the_scene_with_the_table_and_the_mug(cat, tmp_path):
    r = compose("a robot arm next to a mug on a table", variations=1, check="none", device="cpu", catalog=cat)
    paths = r.save(tmp_path)
    rel = _rel(paths, tmp_path)
    assert rel[:2] == [f"scenes/{s.name}.yaml" for s in r.scenes] and rel[-1] == "compose.json"
    assert any(p.startswith("robots/") for p in rel)                      # the robot copy is still saved
    sim = alloy.load(str(paths[0]), device="cpu")
    try:
        names = sim.names()
        assert "table_top_plywood" in names["static"] and "mug_stoneware" in names["bodies"]
    finally:
        sim.close()


def test_floor_scene_paths0_loads(cat, tmp_path):
    r = compose("a mug on the floor", check="none", device="cpu", catalog=cat)
    paths = r.save(tmp_path)
    rel = _rel(paths, tmp_path)
    assert rel[0] == "scenes/scene.yaml" and any(p.startswith("objects/") for p in rel)
    sim = alloy.load(str(paths[0]), device="cpu")
    try:
        assert "mug_stoneware" in sim.names()["bodies"]
    finally:
        sim.close()


def test_compose_command_saved0_is_the_scene(tmp_path):
    out = io.StringIO()
    rc = cli_main(["compose", "a mug on the floor", "--check", "none", "--device", "cpu", "--out", str(tmp_path)],
                  stdout=out)
    saved = json.loads(out.getvalue())["result"]["saved"]
    assert rc == 0 and saved[0] == str(tmp_path / "scenes" / "scene.yaml")
