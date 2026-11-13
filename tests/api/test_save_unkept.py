# SPDX-License-Identifier: Apache-2.0
"""``scene.save()`` says what its cards cannot keep: heat, solver settings, the robot physics the API adds and the
anchors of placed objects (CPU).

The card schema has no field for any of these, and ``save`` dropped them without a word: a scene loaded from the
cards ran cold (no ``"<name>.temperature"``, ``sim.heat`` raised), stepped cloth at the default accuracy, let bent
joints spring back, drew no robot randomization or pushes, and read an anchored object's ``.pos`` at its body
origin, so ``set_pose`` / ``randomize`` at the old values sank it (INVALID). Law: ``save`` names each kind, its
objects and the fix (``scene.to_json()`` / ``Scene.from_dict``, which keep all of them), and says nothing when the
scene has none; ``alloy.io.export_scene`` lists the robot physics and heat as unmapped.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.api import catalog, meshes

CLIP = """<mujoco model="clip"><compiler angle="radian"/><worldbody>
  <body name="base" pos="0 0 0.3"><geom type="box" size="0.04 0.04 0.04" mass="1.0"/>
  <body name="arm" pos="0 0 0.05">
  <joint name="hinge" type="hinge" axis="0 1 0" stiffness="1.0" springref="0.0" damping="0.15" range="-2 2"/>
  <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.018" mass="0.08"/></body></body></worldbody>
  <actuator><motor name="bend" joint="hinge" gear="1.0" ctrlrange="-5 5"/></actuator></mujoco>"""


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and r.name == "alloy.api"]


def _one(said: list[str], key: str) -> str:
    hits = [m for m in said if key in m]
    assert len(hits) == 1, (key, said)
    assert "scene.to_json() / Scene.from_dict" in hits[0]
    return hits[0]


def test_save_warns_that_heat_and_solver_settings_are_not_kept(tmp_path: Path, caplog) -> None:
    scene = alloy.Scene(dt=0.01, cloth_accuracy="fast", liquid_accuracy="super")
    scene.rigid_step = "fixed"
    scene.add_ground()
    scene.add_open_box("pan", inner=(0.08, 0.08, 0.04), wall=0.025, pos=(0.0, 0.0, 0.02))
    scene.add_liquid("water", size=(0.032, 0.032, 0.016), pos=(0.0, 0.0, 0.055), temperature=293.15)
    scene.add_heater("hotplate", size=(0.10, 0.10, 0.02), pos=(0.25, 0.0, 0.01), temperature=363.15)
    scene.set_heater("pan_floor", temperature=353.15)
    scene.set_heat(time_scale=60.0)
    scene.add_cloth("towel", size=(0.08, 0.08), pos=(-0.25, 0.0, 0.04), resolution=5, self_contact=False)
    caplog.set_level("WARNING", logger="alloy.api")
    card = scene.save(tmp_path / "cards")
    said = _warnings(caplog)
    heat = _one(said, "do not keep heat")
    assert "(hotplate, pan_floor, water, set_heat)" in heat and "sim.heat() raises" in heat
    settings = _one(said, "cloth_accuracy=")
    assert "cloth_accuracy='fast', liquid_accuracy='super', rigid_step='fixed'" in settings
    loaded = alloy.Scene.load(card)                                     # what the warnings say
    assert loaded.cloth_accuracy == "default" and not loaded.thermal["heaters"]
    kept = alloy.Scene.from_dict(scene.to_dict())                       # what they point to
    assert kept.cloth_accuracy == "fast" and kept.thermal == scene.thermal


def test_save_warns_for_yielding_joints_robot_randomization_and_pushes(tmp_path: Path, caplog) -> None:
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_robot("clip", mjcf=CLIP)
    scene.set_joint_yield("clip/hinge", torque=0.15)
    scene.randomize("clip", "mass_scale", 0.95, 1.05)
    scene.add_push("clip/arm", force=(0.01, 0.0, 0.0), every=(5.0, 10.0), duration=0.05)
    caplog.set_level("WARNING", logger="alloy.api")
    card = scene.save(tmp_path / "cards")
    said = _warnings(caplog)
    assert "spring back" in _one(said, "do not keep set_joint_yield (clip/hinge)")
    _one(said, "do not keep robot randomization (clip.mass_scale)")
    _one(said, "do not keep add_push (clip/arm)")
    loaded = alloy.Scene.load(card)
    assert not loaded.joint_yield and not loaded.robot_random and not loaded.pushes


def test_save_says_nothing_of_these_for_a_plain_scene(tmp_path: Path, caplog) -> None:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.05,) * 3, pos=(0.0, 0.0, 0.05))
    scene.add_object("cube2", "cube_beech_30", pos=(0.2, 0.0, 0.0))     # Alloy's own card: placed by its origin
    caplog.set_level("WARNING", logger="alloy.api")
    scene.save(tmp_path / "cards")
    assert not [m for m in _warnings(caplog) if "do not keep" in m or "anchor" in m]


@pytest.fixture
def scans(tmp_path, monkeypatch):
    """A fetched (imported) library holding 'scan_box', a box whose card origin sits 40 mm above its bottom, as a
    scan's origin sits off its mesh."""
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "cache" / "imports" / "scans"
    v, f = meshes.box((0.03, 0.03, 0.04))                               # bottom at z = 0
    v = v - np.array([0.0, 0.0, 0.04])                                   # the origin 40 mm above the bottom
    obj = "".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in v.tolist())
    obj += "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f.tolist())
    (root / "meshes").mkdir(parents=True)
    (root / "objects").mkdir(parents=True)
    (root / "meshes" / "scan_box.obj").write_text(obj)
    (root / "objects" / "scan_box.yaml").write_text(
        "schema: alloy.object/1\nname: scan_box\nshape:\n  mesh:\n    path: meshes/scan_box.obj\n"
        f"    sha256: {hashlib.sha256(obj.encode()).hexdigest()}\n    collision: convex\nmaterial: abs\n")
    catalog.refresh()
    yield root
    catalog.refresh()


def test_save_warns_that_anchored_objects_are_placed_by_their_body_origin(scans, tmp_path: Path, caplog) -> None:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_object("can", "scan_box", pos=(0.0, 0.08, 0.0))           # "auto": an imported scan stands on pos
    scene.add_object("same", "scan_box", pos=(0.0, -0.08, 0.04), anchor="origin")
    caplog.set_level("WARNING", logger="alloy.api")
    card = scene.save(tmp_path / "cards")
    said = _one(_warnings(caplog), "by the body origin")
    assert "can (origin [0.0, 0.0, 40.0] mm from its anchor)" in said and "same" not in said
    loaded = alloy.Scene.load(card)                                     # what the warning says: origin values
    assert loaded.items["can"]["raw_placement"]["pose"][:3] == pytest.approx([0.0, 0.08, 0.04])


@pytest.mark.compile_heavy
def test_export_scene_lists_the_robot_physics_its_files_do_not_carry(tmp_path: Path) -> None:
    import alloy.io

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("clip", mjcf=CLIP)
    scene.set_joint_yield("clip/hinge", torque=0.15)
    scene.randomize("clip", "mass_scale", 0.95, 1.05)
    scene.add_push("clip/arm", force=(0.01, 0.0, 0.0), every=(5.0, 10.0), duration=0.05)
    res = alloy.io.export_scene(scene, tmp_path / "out", formats=("mjcf",))
    items = {u.item for u in res.unmapped if u.format == "all"}
    assert {"set_joint_yield clip/hinge", "robot randomization clip.mass_scale", "add_push clip/arm"} <= items
