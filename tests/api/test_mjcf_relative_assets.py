# SPDX-License-Identifier: Apache-2.0
"""``add_robot(mjcf=<text>)`` reads asset paths as MuJoCo reads an XML string: from the working directory.

Laws (CPU): from a folder holding ``cube.obj``, an MJCF string naming ``file="cube.obj"`` builds and steps, the stored
text naming the file by its absolute path; a compiler ``meshdir`` and an ``<include>`` (whose own mesh path is
relative) resolve from there too; a file that is not there raises ``SceneError`` naming the robot, the ``<mesh>``
element and the path it resolved to; ``scene.save()`` then ``Scene.load`` from another folder, and ``to_json`` /
``from_dict`` there, still build. Without trimesh (a base install), such a robot raises ``FeatureUnavailableError``
whose fix is ``pip install "alloy-engine[import]"``, and a shipped robot still builds. Before, the build wrote the text
into its own card folder, the importer looked for ``cube.obj`` there and trimesh raised a bare "string is not a file";
on a base install the error was a ``BuildError`` with no install command.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import alloy.api as alloy

CUBE = ("v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nv .1 .1 0\nv .1 0 .1\nv 0 .1 .1\nv .1 .1 .1\n"
        "f 1 3 5 2\nf 1 2 6 4\nf 1 4 7 3\nf 8 6 2 5\nf 8 5 3 7\nf 8 7 4 6\n")


def _robot(mesh: str) -> str:
    return (f'<mujoco><asset><mesh name="m" file="{mesh}"/></asset><worldbody><body name="b" pos="0 0 0.3"><freejoint/>'
            '<geom type="box" size=".05 .05 .05"/><geom type="mesh" mesh="m" contype="0" conaffinity="0"/></body>'
            '</worldbody></mujoco>')


def _builds(scene: alloy.Scene) -> None:
    sim = alloy.build(scene, device="cpu")
    sim.step()
    assert sim.check()["ok"]
    sim.close()


def _scene(text: str) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("r", mjcf=text)
    return scene


@pytest.mark.compile_heavy
def test_relative_paths_read_from_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cube.obj").write_text(CUBE)
    scene = _scene(_robot("cube.obj"))                      # was: string is not a file: .../robots/cube.obj
    assert f'file="{tmp_path / "cube.obj"}"' in scene.items["r"]["mjcf"]
    _builds(scene)
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "leg.obj").write_text(CUBE)
    (tmp_path / "legs.xml").write_text('<mujoco><mesh name="leg" file="leg.obj"/></mujoco>')   # relative, in parts/
    text = ('<mujoco><compiler meshdir="parts"/><asset><include file="legs.xml"/></asset><worldbody>'
            '<body name="b" pos="0 0 0.3"><freejoint/><geom type="box" size=".05 .05 .05"/>'
            '<geom type="mesh" mesh="leg" contype="0" conaffinity="0"/></body></worldbody></mujoco>')
    _builds(_scene(text))                                   # the include and its mesh from here, under meshdir


@pytest.mark.compile_heavy
def test_a_saved_scene_and_its_json_build_from_another_folder(tmp_path, monkeypatch):
    work, other = tmp_path / "work", tmp_path / "other"
    work.mkdir()
    other.mkdir()
    monkeypatch.chdir(work)
    (work / "cube.obj").write_text(CUBE)
    scene = _scene(_robot("cube.obj"))
    saved = scene.save(tmp_path / "saved")
    data = json.loads(scene.to_json())
    monkeypatch.chdir(other)
    _builds(alloy.Scene.load(saved))
    _builds(alloy.Scene.from_dict(data))


def test_a_missing_file_names_the_robot_the_element_and_the_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(alloy.SceneError) as err:
        _scene(_robot("missing.obj"))
    msg = err.value.message
    assert "'r'" in msg and '<mesh name="m" file="missing.obj">' in msg and str(tmp_path / "missing.obj") in msg
    assert err.value.field == "mjcf" and "working directory" in err.value.fix


@pytest.mark.compile_heavy
def test_a_base_install_names_the_import_extra(tmp_path, monkeypatch):
    from alloy.cards.library import default_root  # noqa: PLC0415

    monkeypatch.chdir(tmp_path)
    (tmp_path / "cube.obj").write_text(CUBE)
    monkeypatch.setitem(sys.modules, "trimesh", None)       # import trimesh fails, as without the import extra
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        alloy.build(_scene(_robot("cube.obj")), device="cpu")
    assert 'pip install "alloy-engine[import]"' in err.value.fix  # was a BuildError with no install command
    shipped = alloy.Scene(dt=0.01)
    shipped.add_robot("go2", Path(default_root()) / "robots" / "emb_e8_go2" / "scene.xml")
    _builds(shipped)                                        # shipped robots need no trimesh
