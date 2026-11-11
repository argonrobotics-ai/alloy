# SPDX-License-Identifier: Apache-2.0
"""Scene arguments: library names, files, Scene JSON and recipes; recipe errors name the entry and the fix."""

from __future__ import annotations

import json

import pytest

import alloy.api as alloy
from alloy.cli.scenes import ADDERS, describe_adders, from_recipe, to_scene

RECIPE = {"worlds": 2, "dt": 0.005,
          "materials": [{"name": "rubber_pad", "density": 1100, "friction": 1.2}],
          "objects": [{"add": "ground"},
                      {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1], "pos": [0, 0, 0.5], "mass": 1.0,
                       "material": "rubber_pad"},
                      {"add": "sphere", "name": "ball", "radius": 0.05, "pos": [[0, 0, 1], [0.1, 0, 1]]},
                      {"add": "open_box", "name": "bin", "inner": [0.1, 0.1, 0.05]}],
          "friction": [{"a": "cube", "b": "ground", "mu": 0.9}],
          "randomize": [{"name": "cube", "field": "x", "lo": -0.01, "hi": 0.01}]}


def test_recipe_builds_the_same_scene_as_python():
    scene = from_recipe(RECIPE)
    py = alloy.Scene(worlds=2, dt=0.005)
    py.add_material("rubber_pad", density=1100, friction=1.2)
    py.add_ground()
    py.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5), mass=1.0, material="rubber_pad")
    py.add_sphere("ball", 0.05, pos=[[0, 0, 1], [0.1, 0, 1]])
    py.add_open_box("bin", inner=(0.1, 0.1, 0.05))
    py.set_friction("cube", "ground", 0.9)
    py.randomize("cube", "x", -0.01, 0.01)
    assert scene.to_dict() == py.to_dict()


@pytest.mark.parametrize("row,match,field", [
    ({"add": "cube", "name": "c"}, "unknown add 'cube'", "objects[0].add"),
    ({"add": "box", "size": [0.1, 0.1, 0.1]}, r"needs \['name'\]", "objects[0].name"),
    ({"add": "box", "name": "c", "size": [0.1, 0.1, 0.1], "colour": "red"}, "no argument", "objects[0].colour"),
    ({"add": "box", "name": "c", "size": [0.1, 0.1]}, "size needs 3", "objects[0].size"),
    ({"add": "sphere", "name": "s", "radius": 0.1, "material": "unobtainium"}, "unknown material",
     "objects[0].material"),
])
def test_recipe_errors_name_the_entry(row, match, field):
    with pytest.raises(alloy.SceneError, match=match) as info:
        from_recipe({"objects": [row]})
    assert info.value.field == field and info.value.fix


def test_recipe_top_level_errors():
    with pytest.raises(alloy.SceneError, match="unknown key"):
        from_recipe({"object": []})
    with pytest.raises(alloy.SceneError, match="list of objects"):
        from_recipe({"objects": {"add": "ground"}})
    with pytest.raises(alloy.SceneError, match="must be an int"):
        from_recipe({"worlds": 0})


def test_names_files_scene_json_and_base(tmp_path):
    lib = to_scene("demo_peg_d12", worlds=3)
    assert lib.worlds == 3 and "peg_d12" in lib.names()
    path = tmp_path / "scene.json"
    path.write_text(from_recipe(RECIPE).to_json())
    assert to_scene(str(path)).names() == ["ground", "cube", "ball", "bin_floor", "bin_wall0", "bin_wall1",
                                           "bin_wall2", "bin_wall3"]
    assert to_scene(json.loads(path.read_text())).worlds == 2
    based = from_recipe({"base": "demo_peg_d12", "dt": 0.01,
                         "objects": [{"add": "sphere", "name": "ball", "radius": 0.01, "pos": [0, 0, 0.2]}]})
    assert based.dt == 0.01 and based.names()[-1] == "ball"
    with pytest.raises(alloy.SceneError, match="no scene file or library scene"):
        to_scene("demo_peg_d1")
    with pytest.raises(alloy.SceneError, match="name, a path or a JSON object"):
        to_scene(3)


def test_library_roots_find_scenes(tmp_path):
    scene = from_recipe({"objects": [{"add": "ground"}, {"add": "box", "name": "b", "size": [0.1, 0.1, 0.1]}]})
    scene.save(tmp_path / "lib")
    (tmp_path / "lib" / "scenes" / "scene.yaml").rename(tmp_path / "lib" / "scenes" / "my_scene_xyz.yaml")
    found = to_scene("my_scene_xyz", library=[str(tmp_path / "lib")])
    assert "b" in found.names() and str(tmp_path / "lib") in found.roots


def test_adders_describe_their_arguments():
    kinds = describe_adders()
    assert set(kinds) == set(ADDERS)
    assert kinds["box"][:3] == ["name", "size", "half_extents"] and "mass" in kinds["box"] and "rpy" in kinds["box"]
    assert "path" in kinds["robot"] and "card" in kinds["object"]


def test_json_files_hold_recipes_or_scene_json(tmp_path):
    (tmp_path / "recipe.json").write_text(json.dumps(RECIPE))
    assert to_scene(str(tmp_path / "recipe.json")).to_dict() == from_recipe(RECIPE).to_dict()
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(alloy.SceneError, match="not valid JSON"):
        to_scene(str(tmp_path / "bad.json"))
    (tmp_path / "list.json").write_text("[1]")
    with pytest.raises(alloy.SceneError, match="not a scene"):
        to_scene(str(tmp_path / "list.json"))


def test_recipe_objects_come_from_library_folders(tmp_path):
    """SCR-OSS17-5: a recipe without "base" finds an object card in a --library folder (own cards, fetched ones)."""
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / "my_cup.yaml").write_text(
        "schema: alloy.object/1\nname: my_cup\nshape: {primitive: {kind: cylinder, size: [0.04, 0.05]}}\n"
        "material: abs\n")
    recipe = {"objects": [{"add": "ground"}, {"add": "object", "name": "cup", "card": "my_cup", "pos": [0, 0, 0.1]}]}
    with pytest.raises(alloy.SceneError, match="objects\\[1\\]"):
        from_recipe(recipe)
    scene = to_scene(recipe, library=[str(tmp_path)])
    assert "cup" in scene.names() and scene.roots[0] == str(tmp_path)
