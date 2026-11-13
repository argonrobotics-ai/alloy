# SPDX-License-Identifier: Apache-2.0
"""Several card library folders: a repeated ``--library`` flag keeps every folder, and export (command line and
``alloy.io.export_scene``) searches them all, as check and build do; an ``alloy.Scene`` exports with its own roots.
One hand-made mesh object card in a folder of its own, a scene card in another."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest

import alloy
from alloy.cli import main
from alloy.cli.registry import OPS
from alloy.cli.spec import add_arguments, flags_to_args
from alloy.io import export_scene

CUBE_OBJ = "".join(f"v {x} {y} {z}\n" for z in (0, 0.04) for x, y in ((-0.02, -0.02), (0.02, -0.02),
                                                                        (0.02, 0.02), (-0.02, 0.02)))
CUBE_OBJ += "".join(f"f {a} {b} {c}\n" for a, b, c in ((1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8), (1, 2, 6),
                                                       (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7),
                                                       (4, 1, 5), (4, 5, 8)))
RECIPE = {"objects": [{"add": "ground"}, {"add": "object", "name": "blk", "card": "my_block", "pos": [0, 0, 0]}]}


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    """``(folder with object card my_block and its mesh, the card of a scene saved elsewhere that places it)``."""
    root = tmp_path_factory.mktemp("libs")
    objects = root / "my_objects"
    mesh = objects / "meshes" / "my_block" / "physics.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text(CUBE_OBJ)
    (objects / "objects").mkdir()
    (objects / "objects" / "my_block.yaml").write_text(
        "schema: alloy.object/1\nname: my_block\nshape:\n  mesh:\n    path: meshes/my_block/physics.obj\n"
        f"    sha256: {hashlib.sha256(mesh.read_bytes()).hexdigest()}\n    collision: convex\nmaterial: abs\n")
    scene = alloy.Scene(roots=[str(objects)])
    scene.add_ground()
    scene.add_object("blk", "my_block", pos=(0, 0, 0))
    scenes = root / "my_scenes"
    card = scene.save(scenes)
    assert (scenes / "meshes" / "my_block" / "physics.obj").is_file()    # save copies what it needs from its roots
    shutil.rmtree(scenes / "meshes" / "my_block")      # a scene card that names a mesh in another folder
    return objects, card


def run(argv):
    out = io.StringIO()
    code = main(argv, stdout=out)
    return code, json.loads(out.getvalue())


def test_a_repeated_list_flag_keeps_every_value():
    parser = argparse.ArgumentParser()
    add_arguments(parser, OPS["check"])
    ns = parser.parse_args(["x.yaml", "--library", "a", "--library", "b,c", "--library", '["d"]'])
    assert flags_to_args(OPS["check"], ns)["library"] == ["a", "b", "c", "d"]


def test_check_and_export_search_every_library_folder(folders, tmp_path):
    objects, card = folders
    libs = ["--library", str(objects), "--library", str(card.parent.parent)]   # the mesh is in the first
    code, checked = run(["check", str(card), *libs, "--quick"])
    assert code == 0 and checked["result"]["ok"], checked
    code, exported = run(["export", str(card), *libs, "--formats", "mjcf", "--out", str(tmp_path / "yaml")])
    assert code == 0 and exported["ok"], exported
    code, exported = run(["export", json.dumps(RECIPE), "--library", str(objects), "--formats", "mjcf",
                          "--out", str(tmp_path / "recipe")])
    assert code == 0 and exported["ok"], exported
    assert Path(exported["result"]["files"]["mjcf"]).is_file()


def test_export_scene_uses_the_scene_roots_or_a_list_of_folders(folders, tmp_path):
    objects, card = folders
    scene = alloy.Scene(roots=[str(objects)])
    scene.add_object("blk", "my_block", pos=(0, 0, 0))
    assert export_scene(scene, tmp_path / "own", formats=("mjcf",)).files["mjcf"]
    plain = alloy.Scene()
    plain.add_ground()
    res = export_scene(str(card), tmp_path / "listed", formats=("mjcf",), library=[str(objects), card.parent.parent])
    assert res.files["mjcf"] and export_scene(plain, tmp_path / "plain", formats=("mjcf",)).files["mjcf"]
