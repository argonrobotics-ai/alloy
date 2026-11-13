# SPDX-License-Identifier: Apache-2.0
"""``scene.save()``: what the saved cards keep, what they cannot keep (said, with the fix), and ``scene.randomize``
on a fixture's name (CPU).

- An ``mjcf=`` robot is written under the card folder and referenced relative to it, so a moved folder loads; a
  missing file's error names the file, not a material card.
- The card schema holds no per-world start rows, no kinematic fixture as one body and no control period apart from
  the engine's step: ``save`` warns for each, naming the objects and ``scene.to_json()``.
- ``randomize(<kinematic fixture>, "x" | "y" | "z", ...)`` moves the whole fixture; what a fixture cannot take is a
  ``SceneError`` when called, whose fix builds.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link" pos="0 0 0.1">
  <joint name="hinge" type="hinge" axis="0 0 1" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><position joint="hinge" kp="100" kv="10"/></actuator></mujoco>"""


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and r.name == "alloy.api"]


def _arm_scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(-0.3, 0.0, 0.0))
    return scene


# --- robots in a saved folder ------------------------------------------------------------------------------------

@pytest.mark.compile_heavy
def test_a_saved_folder_with_an_mjcf_robot_loads_after_it_moves(tmp_path: Path) -> None:
    """The card named the robot by an absolute path into the folder it was saved to, so a moved folder failed to
    load. Law: the path is relative to the card folder, and the moved folder builds the same robot."""
    import yaml

    card = _arm_scene().save(tmp_path / "saved")
    assert yaml.safe_load(card.read_text(encoding="utf-8"))["robots"][0]["path"] == "robots/arm.xml"
    shutil.move(str(tmp_path / "saved"), str(tmp_path / "moved"))
    sim = alloy.load(tmp_path / "moved" / "scenes" / "scene.yaml", device="cpu")
    assert "arm/hinge" in sim.names()["joints"]
    sim.set_targets({"hinge": 0.5})
    sim.step(50)
    assert float(np.ravel(sim.state(world=0)["arm/hinge.q"])[0]) == pytest.approx(0.5, abs=0.05)
    sim.close()


def test_a_missing_robot_file_names_the_file_not_a_material_card(tmp_path: Path) -> None:
    """A missing robot file failed with the fix for an unknown material ("add the card with Scene.add_material").
    Law: the BuildError names the file and how to keep a saved folder whole (it fails before any kernel compiles)."""
    card = _arm_scene().save(tmp_path / "saved")
    (tmp_path / "saved" / "robots" / "arm.xml").unlink()
    with pytest.raises(alloy.BuildError, match="robot file not found") as err:
        alloy.load(card, device="cpu")
    assert "add_material" not in err.value.fix and "add_robot" in err.value.fix and "folder whole" in err.value.fix


# --- what the cards cannot keep: said when saving --------------------------------------------------------------

def test_save_warns_that_per_world_starts_are_not_kept(tmp_path: Path, caplog) -> None:
    """Saved cards hold one start per object, so a scene loaded from them put every world at world 0's start, with
    no warning. Law: save names what starts per world (pos= rows, a cloth's randomized start) and the fix; an
    object's randomized x is a card field, kept, and not named."""
    scene = alloy.Scene(worlds=4, dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.05,) * 3, pos=[[0.1 * w, 0.0, 0.05] for w in range(4)])
    scene.add_box("drawn", size=(0.05,) * 3, pos=(0.0, 0.3, 0.05))
    scene.randomize("drawn", "x", -0.1, 0.1)
    scene.add_cloth("towel", size=(0.2, 0.2), pos=(0.0, -0.3, 0.2), resolution=4, mass=0.02)
    scene.randomize("towel", "y", -0.35, -0.25)
    caplog.set_level("WARNING", logger="alloy.api")
    scene.save(tmp_path / "cards")
    said = [m for m in _warnings(caplog) if "per-world starts" in m]
    assert len(said) == 1 and "of cube, towel;" in said[0] and "scene.to_json()" in said[0]
    loaded = alloy.Scene.load(tmp_path / "cards" / "scenes" / "scene.yaml")
    assert loaded.worlds == 4 and loaded.items["drawn"]["raw_placement"]["randomize"]      # the draw is kept


def test_save_warns_for_kinematic_fixtures_and_a_precise_control_period(tmp_path: Path, caplog) -> None:
    """A kinematic fixture loaded from cards as separate kinematic parts (no body to drive by its name, its floor
    driven alone), and a precise-fit scene's dt became its 0.156 ms rigid step, with no warning. Law: save names
    both, with the numbers and the fix; a static fixture and a scene without a precise fit say nothing of them."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("bin", inner=(0.12, 0.12, 0.10), wall=0.01, pos=(0.0, -0.18, 0.05), kinematic=True)
    scene.add_open_box("tray", inner=(0.12, 0.12, 0.05), wall=0.01, pos=(0.5, 0.5, 0.0))
    scene.add_block_with_hole("blk", (0.1, 0.1, 0.05), hole=(0.0102, 0.0102), depth=0.03, pos=(0.5, -0.5, 0.0))
    caplog.set_level("WARNING", logger="alloy.api")
    card = scene.save(tmp_path / "cards")
    said = _warnings(caplog)
    fixture = [m for m in said if "kinematic fixture" in m]
    assert len(fixture) == 1 and "fixture bin as 5 separate" in fixture[0] and "no 'bin' to sim.drive" in fixture[0]
    assert "tray" not in fixture[0] and "scene.to_json()" in fixture[0]
    period = [m for m in said if "precise-fit step" in m]
    assert len(period) == 1 and "of blk" in period[0] and "dt 0.00015625 s, not 0.01 s" in period[0]
    assert "64x" in period[0]
    assert alloy.Scene.load(card).dt == pytest.approx(0.01 / 64)        # what the warning says
    plain = alloy.Scene(dt=0.01)
    plain.add_open_box("tray", inner=(0.12, 0.12, 0.05), wall=0.01)
    caplog.clear()
    plain.save(tmp_path / "plain")
    assert not any("fixture" in m or "precise" in m or "per-world" in m for m in _warnings(caplog))


# --- randomize on a fixture's name -----------------------------------------------------------------------------

def test_randomize_on_a_fixture_name_draws_on_its_floor_or_says_what_to_do() -> None:
    """randomize("bin", "x", ...) on a kinematic open box raised "no object named 'bin'" with a fix listing its
    parts, and randomizing a wall's x (as listed) then failed the build. Law: a kinematic fixture's x / y / z draw is
    its floor's (z from the bottom face, ``pos``, to the floor's centre); a static fixture's start, a kinematic
    fixture's density and any other field on the fixture's name are SceneErrors when called, with a fix."""
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_open_box("bin", inner=(0.12, 0.12, 0.10), wall=0.01, pos=(0.0, 0.0, 0.05), kinematic=True)
    scene.add_open_box("tray", inner=(0.12, 0.12, 0.05), wall=0.01, pos=(0.5, 0.5, 0.0))
    scene.randomize("bin", "x", -0.2, 0.2)
    scene.randomize("bin", "z", 0.0, 0.1)
    assert scene.items["bin_floor"]["randomize"] == [{"field": "pose.0", "lo": -0.2, "hi": 0.2},
                                                     {"field": "pose.2", "lo": 0.005, "hi": pytest.approx(0.105)}]
    refused = {("bin", "density"): "drop the density draw", ("tray", "x"): "kinematic=True",
               ("tray", "friction"): "'tray_floor', 'tray_wall0'", ("tray", "density"): "'tray_floor', 'tray_wall0'"}
    for (name, field), fix in refused.items():
        with pytest.raises(alloy.SceneError) as err:
            scene.randomize(name, field, 0.1, 0.2)
        assert fix in err.value.fix, (name, field, err.value.fix)
    assert not any(it["randomize"] for n, it in scene.items.items() if n != "bin_floor")


@pytest.mark.compile_heavy
def test_a_randomized_kinematic_fixture_builds_and_moves_whole() -> None:
    """Law: following those fixes builds: the fixture's own x and z draws place its body per world inside the range
    (``bin.pos`` reads the floor's centre, half the wall above ``pos``), a part's friction or density drawn by its
    name builds (kinematic and static fixtures), and a ball dropped in the bin rests on its floor in every world."""
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_open_box("bin", inner=(0.12, 0.12, 0.10), wall=0.01, pos=(0.0, 0.0, 0.05), kinematic=True)
    scene.randomize("bin", "x", -0.02, 0.02)
    scene.randomize("bin", "z", 0.04, 0.06)
    scene.randomize("bin_wall1", "friction", 0.4, 0.9)
    scene.add_open_box("tray", inner=(0.12, 0.12, 0.05), wall=0.01, pos=(0.5, 0.5, 0.0))     # static: part by part
    scene.randomize("tray_floor", "density", 900.0, 1100.0)
    scene.randomize("tray_wall0", "friction", 0.4, 0.9)
    scene.add_sphere("ball", radius=0.02, pos=(0.0, 0.0, 0.15), mass=0.05)
    sim = alloy.build(scene, device="cpu", seed=3)
    pos = sim.state(["bin"])["bin.pos"]
    assert np.all(np.abs(pos[:, 0]) <= 0.02) and abs(pos[0, 0] - pos[1, 0]) > 1e-4           # drawn per world
    assert np.all((pos[:, 2] >= 0.045 - 1e-6) & (pos[:, 2] <= 0.065 + 1e-6))                 # bottom face + 5 mm
    sim.run(0.4)
    ball = sim.state(["ball"])["ball.pos"]
    assert np.allclose(ball[:, 2], pos[:, 2] + 0.005 + 0.02, atol=2e-3) and sim.check()["ok"]
    sim.close()
