# SPDX-License-Identifier: Apache-2.0
"""``scene.save()`` of a scene that takes cards from its own card roots (``Scene(roots=[...])``: ``alloy.create`` and
``alloy.io.import_file`` folders, a folder saved before) writes a folder that loads on its own (CPU).

The saved folder named those cards but held none of them: a card placed under its own name was not written, and no
custom material card, mesh, pair card or look was copied, so ``alloy.load`` of the folder failed (no material, no
mesh file, or a Hub lookup of the user's own card). Law: the folder holds what its cards name from the scene's own
roots, at the same relative paths, and builds the same bodies after the roots are gone; library cards are not copied.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.api import catalog, meshes
from alloy.cards.units_yaml import yaml_dump, yaml_load

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link" pos="0 0 0.1">
  <joint name="hinge" type="hinge" axis="0 0 1" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><position joint="hinge" kp="100" kv="10"/></actuator></mujoco>"""


def _library_card(kind: str, name: str) -> dict:
    path = catalog.card_path(kind, name)
    return yaml_load(path.read_text(encoding="utf-8"), path.stem)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_dump(data, header="test card"), encoding="utf-8")


def _own_root(root: Path) -> Path:
    """A card root as alloy.create and import_file write them: a mesh card with its own material (a pair card and a
    look for it) in ``meshes/puck/`` beside its hulls and visual.glb, and a primitive card with its own material."""
    v, f = meshes.box((0.06, 0.06, 0.02))
    obj = "".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in v.tolist())
    obj += "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f.tolist())
    (root / "meshes" / "puck").mkdir(parents=True)
    (root / "meshes" / "puck" / "physics.obj").write_text(obj)
    (root / "meshes" / "puck" / "physics.hulls.obj").write_text("# hulls\n")
    (root / "meshes" / "puck" / "visual.glb").write_bytes(b"glb")
    _write_yaml(root / "objects" / "puck.yaml", {
        "schema": "alloy.object/1", "name": "puck", "material": "puck_eff",
        "shape": {"mesh": {"path": "meshes/puck/physics.obj", "sha256": hashlib.sha256(obj.encode()).hexdigest(),
                           "collision": "convex"}}})
    _write_yaml(root / "objects" / "block.yaml", {
        "schema": "alloy.object/1", "name": "block", "material": "block_mat",
        "shape": {"primitive": {"kind": "box", "size": [0.02, 0.02, 0.02]}}})
    for name, density in (("puck_eff", 900.0), ("block_mat", 1500.0)):
        data = {**_library_card("materials", "abs"), "name": name}
        data["density"] = {"value": density, "source": "model", "note": "test material"}
        _write_yaml(root / "materials" / f"{name}.yaml", data)
    _write_yaml(root / "pairs" / "abs__puck_eff.yaml", {**_library_card("pairs", "abs__abs"), "b": "puck_eff"})
    (root / "looks" / "materials").mkdir(parents=True)
    (root / "looks" / "materials" / "puck_eff.json").write_text(json.dumps(
        {"schema": "alloy.look/1", "material": "puck_eff", "preset": "abs_black"}))
    return root


def _scene(root: Path) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, roots=[str(root)])
    scene.add_ground()
    scene.add_object("puck", "puck", pos=(0.0, 0.0, 0.05))            # placed under its card's own name
    scene.add_object("b1", "block", pos=(0.2, 0.0, 0.05))             # renamed: the save writes objects/b1.yaml
    scene.add_box("cube", size=(0.04, 0.04, 0.04), pos=(-0.2, 0.0, 0.05), material="abs", mass=0.1)
    return scene


def test_save_copies_what_its_cards_name_from_the_scenes_own_roots(tmp_path: Path) -> None:
    own = _own_root(tmp_path / "my_objects")
    _scene(own).save(tmp_path / "saved")
    saved = tmp_path / "saved"
    for rel in ("objects/puck.yaml", "objects/b1.yaml", "materials/puck_eff.yaml", "materials/block_mat.yaml",
                "meshes/puck/physics.obj", "meshes/puck/physics.hulls.obj", "meshes/puck/visual.glb",
                "pairs/abs__puck_eff.yaml", "looks/materials/puck_eff.json"):
        assert (saved / rel).is_file(), rel
    assert (saved / "objects" / "puck.yaml").read_bytes() == (own / "objects" / "puck.yaml").read_bytes()
    assert not (saved / "materials" / "abs.yaml").exists()                 # the shipped library is not copied


def test_a_scene_loaded_from_a_saved_folder_saves_its_robot_file_again(tmp_path: Path) -> None:
    """A scene loaded from a saved folder names its mjcf= robot ``robots/arm.xml`` relative to that folder; saved
    elsewhere, the new folder lacked the file and failed to build."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(-0.3, 0.0, 0.0))
    first = scene.save(tmp_path / "first")
    alloy.Scene.load(first).save(tmp_path / "second")
    assert (tmp_path / "second" / "robots" / "arm.xml").read_text() == ARM


@pytest.mark.compile_heavy
def test_a_saved_folder_builds_the_same_bodies_after_its_own_roots_are_gone(tmp_path: Path) -> None:
    own = _own_root(tmp_path / "my_objects")
    scene = _scene(own)
    sim = alloy.build(scene, device="cpu")
    before = sim.engine.model.body_mass.numpy().copy()
    sim.step(30)
    pos = sim.state(["puck", "b1", "cube"], world=0)
    sim.close()
    scene.save(tmp_path / "saved")
    shutil.move(str(own), str(tmp_path / "gone"))                         # the card roots are not there any more
    shutil.move(str(tmp_path / "saved"), str(tmp_path / "moved"))
    catalog.refresh()
    loaded = alloy.load(tmp_path / "moved" / "scenes" / "scene.yaml", device="cpu")
    assert np.allclose(loaded.engine.model.body_mass.numpy(), before)
    loaded.step(30)
    again = loaded.state(["puck", "b1", "cube"], world=0)
    for key in ("puck.pos", "b1.pos", "cube.pos"):
        assert np.allclose(again[key], pos[key], atol=1e-6), key
    assert loaded.check()["ok"]
    loaded.close()
