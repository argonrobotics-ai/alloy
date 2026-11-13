# SPDX-License-Identifier: Apache-2.0
"""create from model files (Gemini audit S2-015): a jointed URDF and a welded URDF become object cards, an MJCF
object takes the name asked for and reports the file's mass, the card places with ``Scene.add_object`` under that
name, and an override the file would silently drop (material=, mass=) raises with the fix."""

from __future__ import annotations

import pytest

import alloy
from alloy.create import create
from alloy.create.request import CreateError

URDF_HINGED = """<robot name="arm_link">
  <link name="frame">
    <inertial><mass value="1.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><box size="0.04 0.2 0.3"/></geometry></visual>
    <collision><geometry><box size="0.04 0.2 0.3"/></geometry></collision>
  </link>
  <link name="door">
    <inertial><mass value="0.8"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><geometry><box size="0.02 0.18 0.28"/></geometry></collision>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="frame"/><child link="door"/>
    <origin xyz="0.04 0.1 0"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1.57" effort="0" velocity="2"/>
  </joint>
</robot>
"""
URDF_FIXED = """<robot name="fixed_fixture">
  <link name="a"><inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="b"><inertial><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial><collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link>
  <joint name="weld" type="fixed"><parent link="a"/><child link="b"/><origin xyz="0 0 0.075"/></joint>
</robot>
"""
MJCF_DRAWER = """<mujoco model="drawer">
  <worldbody>
    <body name="cabinet" pos="0 0 0.1">
      <geom type="box" size="0.15 0.15 0.1" mass="2.0"/>
      <body name="tray" pos="0 0 0.15">
        <joint name="slide" type="slide" axis="0 1 0" range="0 0.2"/>
        <geom type="box" size="0.12 0.12 0.04" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _file(tmp_path, name: str, text: str) -> str:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return str(tmp_path / name)


def _mass(result) -> float:
    return next(v["value"] for v in result.values if v["field"] == f"{result.name}.mass")


def test_a_jointed_urdf_becomes_an_object_card_that_places_and_builds(tmp_path):
    out = tmp_path / "my_objects"
    r = create(mesh=_file(tmp_path, "arm_link.urdf", URDF_HINGED), out=str(out), device="cpu")
    assert r.ok and r.objects == ["arm_link"] and _mass(r) == pytest.approx(2.3)
    assert any("passive joints" in n for n in r.notes)
    scene = alloy.Scene(dt=0.01, roots=[str(out)])
    scene.add_ground()
    scene.add_object("door", "arm_link", pos=(0, 0, 0.2))
    sim = alloy.build(scene, device="cpu")
    sim.step(5)
    assert sim.check()["ok"] and any("hinge" in j for j in sim.names()["joints"])


def test_a_urdf_of_welded_links_becomes_a_free_object(tmp_path):
    r = create(mesh=_file(tmp_path, "fixed_fixture.urdf", URDF_FIXED), out=str(tmp_path / "lib"), device="cpu")
    assert r.ok and r.objects == ["fixed_fixture"] and _mass(r) == pytest.approx(1.5)


def test_an_mjcf_object_takes_the_name_asked_for_and_reports_the_files_mass(tmp_path):
    out = tmp_path / "my_objects"
    r = create(mesh=_file(tmp_path, "drawer.xml", MJCF_DRAWER), name="my_drawer", out=str(out), check=False)
    assert r.name == "my_drawer" and r.objects == ["my_drawer"] and _mass(r) == pytest.approx(2.5)
    scene = alloy.Scene(dt=0.01, roots=[str(out)])
    scene.add_ground()
    scene.add_object("drawer", "my_drawer", pos=(0, 0, 0.15))                # raised SceneError before the fix
    assert "drawer" in scene.names()


def test_overrides_a_model_file_would_drop_raise_with_the_fix(tmp_path):
    src = _file(tmp_path, "drawer.xml", MJCF_DRAWER)
    with pytest.raises(CreateError) as e:
        create(mesh=src, name="my_drawer", material="wood_pine", mass=2.5, mass_source="kitchen scale",
               out=str(tmp_path / "lib"), check=False)
    assert "material=, mass=" in e.value.message and e.value.field == "material" and "in the file" in e.value.fix
    assert not (tmp_path / "lib").exists()                                   # refused before anything is written


def test_a_file_with_driven_joints_is_refused_as_a_robot(tmp_path):
    arm = ("<mujoco><worldbody><body name='base'><geom type='box' size='0.05 0.05 0.05' mass='2'/>"
           "<body name='b' pos='0.1 0 0'><joint name='j' type='hinge'/><geom type='box' size='0.1 0.02 0.02' "
           "mass='1'/></body></body></worldbody><actuator><position joint='j' kp='50'/></actuator></mujoco>")
    with pytest.raises(CreateError) as e:
        create(mesh=_file(tmp_path, "arm.xml", arm), out=str(tmp_path / "lib"), check=False)
    assert "driven joints" in e.value.message and "add_robot" in e.value.fix
