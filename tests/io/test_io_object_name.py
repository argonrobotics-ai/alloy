# SPDX-License-Identifier: Apache-2.0
"""import_file(as_object=True), the object reading alloy.create uses (Gemini audit S2-015): a jointed file with no
driven joint becomes one articulated object named exactly as asked, with the file's mass; a URDF whose joints are
all fixed is an object for every caller; and a URDF with a moving joint still loads as a robot by default."""

from __future__ import annotations

import alloy
from alloy.cards.library import default_root
from alloy.io import import_file
from alloy.io.urdf_read import read_urdf

LINK = """<link name="{n}"><inertial><mass value="{m}"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0"
  izz="0.01"/></inertial><collision><geometry><box size="{s}"/></geometry></collision></link>"""
HINGED = ('<robot name="arm_link">' + LINK.format(n="frame", m=1.5, s="0.04 0.2 0.3")
          + LINK.format(n="door", m=0.8, s="0.02 0.18 0.28")
          + '<joint name="hinge" type="revolute"><parent link="frame"/><child link="door"/><origin xyz="0.04 0.1 0"/>'
            '<axis xyz="0 0 1"/><limit lower="0" upper="1.57" effort="5" velocity="2"/></joint></robot>')
WELDED = ('<robot name="fixture">' + LINK.format(n="a", m=1.0, s="0.1 0.1 0.1")
          + LINK.format(n="b", m=0.5, s="0.05 0.05 0.05")
          + '<joint name="weld" type="fixed"><parent link="a"/><child link="b"/><origin xyz="0 0 0.075"/></joint>'
            '</robot>')


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_urdf_with_a_moving_joint_still_loads_as_a_robot_by_default(tmp_path):
    src = _write(tmp_path, "arm_link.urdf", HINGED)
    io = read_urdf(src)
    assert io.groups == {"arm_link": "robot"} and io.joints[0].kind == "fixed"       # fixed base, as before
    res = import_file(src, tmp_path / "lib")
    assert res.objects == [] and [r["name"] for r in res.robots] == ["arm_link"] and res.robots[0]["kind"] == "urdf"
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", str(src), pos=(0, 0, 0.5))
    sim = alloy.build(scene, device="cpu")
    assert "arm/hinge" in sim.names()["joints"]


def test_as_object_makes_the_jointed_urdf_one_object_with_its_name_and_mass(tmp_path):
    res = import_file(_write(tmp_path, "arm_link.urdf", HINGED), tmp_path / "lib", name="my_door", as_object=True)
    assert res.objects == ["my_door"] and res.robots == [] and abs(res.masses["my_door"] - 2.3) < 1e-9
    assert (tmp_path / "lib" / "objects" / "my_door.yaml").is_file()
    assert not [i for i in res.issues if i["level"] == "error"]


def test_a_urdf_of_welded_links_is_a_free_object_for_every_caller(tmp_path):
    src = _write(tmp_path, "fixture.urdf", WELDED)
    io = read_urdf(src)
    assert io.groups == {"fixture": "object"} and io.joints[0].kind == "free"
    res = import_file(src, tmp_path / "lib")
    assert res.objects == ["fixture"] and res.robots == [] and abs(res.masses["fixture"] - 1.5) < 1e-9


def test_without_as_object_names_keep_the_body_suffix_and_a_driven_file_stays_a_robot(tmp_path):
    box = default_root() / "mjcf" / "hinged_box.xml"
    assert import_file(box, tmp_path / "a", name="my_box").objects == ["my_box_box"]
    assert import_file(box, tmp_path / "b", name="my_box", as_object=True).objects == ["my_box"]
    arm = _write(tmp_path, "arm.xml", "<mujoco><worldbody><body name='base'>"
                 "<geom type='box' size='0.05 0.05 0.05' mass='2'/>"
                 "<body name='b' pos='0.1 0 0'><joint name='j' type='hinge'/>"
                 "<geom type='box' size='0.1 0.02 0.02' mass='1'/></body></body></worldbody><actuator>"
                 "<position joint='j' kp='50'/></actuator></mujoco>")
    res = import_file(arm, tmp_path / "c", as_object=True)
    assert res.objects == [] and len(res.robots) == 1
