# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the import/export findings of the 0.1.2 audit (S3-002, 005, 007, 013, 014, 015, 016):
each one is the auditor's repro, cut down, asserting the fixed behaviour."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest

import alloy
import alloy.io
from alloy.io.errors import ImportFileError
from alloy.io.geometry import read_obj, signed_volume
from alloy.io.mjcf_include import read_mjcf

BOX_OBJ = """v -0.03 -0.02 -0.04\nv 0.03 -0.02 -0.04\nv 0.03 0.02 -0.04\nv -0.03 0.02 -0.04
v -0.03 -0.02 0.04\nv 0.03 -0.02 0.04\nv 0.03 0.02 0.04\nv -0.03 0.02 0.04
f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\nf 3 4 8\nf 3 8 7\nf 2 3 7\nf 2 7 6\nf 4 1 5\nf 4 5 8
"""
DAMPRATIO_ROBOT = """<mujoco model="dampratio_bot"><compiler angle="radian"/><worldbody>
  <body name="pelvis" pos="0 0 0.5"><freejoint name="root"/>
    <geom name="pelvis_g" type="box" size="0.1 0.1 0.1" mass="5.0"/>
    <body name="leg" pos="0 0 -0.2">
      <joint name="hip_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      <geom name="leg_g" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04" mass="2.0"/>
    </body></body></worldbody>
  <actuator><position name="hip_pitch" joint="hip_pitch" kp="500" dampratio="1" forcerange="-88 88"/></actuator>
</mujoco>"""
LAMP = """<mujoco model="lamp_scene"><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="2 2 0.1"/>
  <body name="base" pos="0 0 0.05"><freejoint/>
    <geom name="base_g" type="box" size="0.05 0.05 0.05" mass="10.0"/>
    <body name="shade" pos="0.12 0 0.20">
      <joint name="neck" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="5.0" stiffness="25.0"/>
      <geom name="shade_g" type="box" size="0.05 0.05 0.05" mass="0.05"/>
    </body></body></worldbody></mujoco>"""
PENDULUM = """<mujoco model="damped_pendulum"><compiler angle="radian"/><worldbody>
  <body name="mount" pos="0 0 0.5"><geom type="sphere" size="0.03" mass="1.0"/>
    <body name="rod" pos="0 0 0"><joint name="pivot" type="hinge" axis="0 1 0" damping="2.5"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.5"/></body></body></worldbody></mujoco>"""
ARM = """<mujoco model="two_link"><compiler angle="radian"/><worldbody>
  <body name="link1" pos="0 0 0.2">
    <joint name="j1" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="1.0"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1.2"/>
    <body name="link2" pos="0.3 0 0"><joint name="j2" type="hinge" axis="0 1 0" range="-2.0 2.0" damping="0.5"/>
      <geom type="sphere" pos="0.2 0 0" size="0.04" mass="0.8"/></body></body></worldbody>
  <actuator><position name="act1" joint="j1" kp="100" kv="10"/><position name="act2" joint="j2" kp="80" kv="8"/>
  </actuator></mujoco>"""
URDF_FLOATING = """<robot name="floating_bot"><link name="world"/>
  <link name="base_link"><inertial><mass value="2.0"/>
    <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision></link>
  <joint name="root_free" type="floating"><parent link="world"/><child link="base_link"/>
    <origin xyz="0 0 0.5"/></joint></robot>"""
URDF_XACRO = """<robot name="husky"><link name="base">
  <inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  <collision><origin xyz="$(optenv HUSKY_TOP_PLATE_XYZ 0.0812 0 0.245)"/>
    <geometry><box size="0.5 0.3 0.2"/></geometry></collision></link></robot>"""
URDF_HINGE = """<robot name="g"><link name="palm"><inertial><mass value="0.5"/>
    <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
    <collision><geometry><box size="0.08 0.04 0.04"/></geometry></collision></link>
  <link name="finger"><inertial><mass value="0.05"/>
    <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/></inertial>
    <collision><geometry><box size="0.015 0.02 0.05"/></geometry></collision></link>
  <joint name="finger_joint" type="revolute"><parent link="palm"/><child link="finger"/>
    <origin xyz="-0.03 0 0.04"/><axis xyz="0 1 0"/><limit lower="0" upper="0.8" effort="20" velocity="2"/></joint>
</robot>"""


def _write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text)
    return path


def _robot_scene(path: Path, name: str = "r") -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot(name, path)
    return scene


def _errors(res) -> list[dict]:
    return [i for i in res.issues if i["level"] == "error"]


# --- S3-005: dampratio servos and <include> files -----------------------------------------------------------------

def test_include_scene_with_dampratio_servo_imports(tmp_path):
    src = tmp_path / "src"
    _write(src, "robot.xml", DAMPRATIO_ROBOT)
    scene = _write(src, "scene.xml", '<mujoco><include file="robot.xml"/><worldbody>'
                                     '<geom name="floor" type="plane" size="0 0 0.05"/></worldbody></mujoco>')
    root = read_mjcf(scene)
    assert root.find(".//include") is None and len(root.findall("worldbody")) == 1
    assert [b.get("name") for b in root.find("worldbody") if b.tag == "body"] == ["pelvis"]
    res = alloy.io.import_file(scene, tmp_path / "cards")
    assert len(res.robots) == 1 and not _errors(res)
    m = mujoco.MjModel.from_xml_path(res.robots[0]["path"])          # the robot cut from the file, servo as written
    assert m.nu == 1 and 'dampratio="1"' in Path(res.robots[0]["path"]).read_text()


def test_generated_robot_file_keeps_a_damping_ratio(tmp_path):
    from alloy.io.groups import split_groups
    from alloy.io.importer import read_file
    from alloy.io.mjcf_write import write_group_mjcf

    io = read_file(_write(tmp_path, "robot.xml", DAMPRATIO_ROBOT))
    hip = next(j for j in io.joints if j.name == "hip_pitch")
    assert hip.stiffness == pytest.approx(500.0) and hip.drive_damping == 0.0 and hip.drive_dampratio == 1.0
    out = write_group_mjcf(io, split_groups(io)[0], tmp_path / "gen.xml", robot=True)     # loads in MuJoCo
    assert 'dampratio="1.0"' in out.read_text() and "kv=" not in out.read_text()


def test_import_errors_are_api_errors_with_a_fix(tmp_path):
    with pytest.raises(alloy.AlloyAPIError) as err:
        alloy.io.import_file(_write(tmp_path, "part.step", "x"), tmp_path / "cards")
    assert isinstance(err.value, ImportFileError) and "unknown suffix" in err.value.message and err.value.fix


# --- S3-007: URDF floating joints, and the fix text for a joint without a servo -----------------------------------

def test_urdf_floating_joint_on_an_empty_root_is_a_free_base(tmp_path):
    path = _write(tmp_path, "floating.urdf", URDF_FLOATING)
    sim = alloy.build(_robot_scene(path, "bot"), device="cpu")
    z0 = float(sim.state(world=0)["bot/base_link.pos"][2])
    sim.step(10)
    assert z0 == pytest.approx(0.5, abs=1e-6) and float(sim.state(world=0)["bot/base_link.pos"][2]) < z0 - 0.02
    io = alloy.io.importer.read_file(path)
    assert [b.name for b in io.bodies] == ["base_link"] and io.bodies[0].pose[2] == pytest.approx(0.5)
    assert io.joint_of("base_link").kind == "free"


def test_urdf_floating_joint_elsewhere_is_refused_with_a_fix(tmp_path):
    bad = URDF_FLOATING.replace('<link name="world"/>', '<link name="world"><collision><geometry>'
                                '<box size="1 1 0.1"/></geometry></collision></link>')
    path = _write(tmp_path, "bad.urdf", bad)
    with pytest.raises(alloy.BuildError, match="floating joint") as err:
        alloy.build(_robot_scene(path), device="cpu")
    assert "empty root link" in err.value.fix
    with pytest.raises(ImportFileError, match="floating joint"):
        alloy.io.import_file(path, tmp_path / "cards")


def test_target_on_a_urdf_joint_names_what_each_format_needs(tmp_path):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("g", _write(tmp_path, "g.urdf", URDF_HINGE), pos=(0, 0, 0.2))
    sim = alloy.build(scene, device="cpu")
    with pytest.raises(alloy.SimulationError) as err:
        sim.set_targets({"g/finger_joint": 0.5})
    assert "URDF has no servo gains" in err.value.fix and "PhysicsDriveAPI" in err.value.fix


# --- S3-016: static-only robot files and unexpanded xacro ---------------------------------------------------------

def test_static_only_robot_file_is_refused_with_a_fix(tmp_path):
    path = _write(tmp_path, "static_table.xml", '<mujoco model="static_table"><worldbody><geom name="table" '
                                                'type="box" size="0.4 0.3 0.35" pos="0 0 0.35"/></worldbody></mujoco>')
    for build in (lambda: alloy.load(path, device="cpu"), lambda: alloy.build(_robot_scene(path, "t"), device="cpu")):
        with pytest.raises(alloy.BuildError, match="only static geometry") as err:
            build()
        assert "add an object that moves" in err.value.fix
    scene = _robot_scene(path, "t")                 # beside a moving object the same file loads
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.9), mass=0.2)
    sim = alloy.build(scene, device="cpu")
    sim.step(60)
    assert sim.check()["ok"] and float(sim.state(world=0)["block.pos"][2]) == pytest.approx(0.725, abs=3e-3)


def test_scene_of_free_bodies_without_a_robot_file_still_builds():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("a", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1), mass=0.2)
    scene.add_sphere("b", 0.03, pos=(0.2, 0, 0.1), mass=0.1)
    sim = alloy.build(scene, device="cpu")
    sim.step(20)
    assert sim.check()["ok"]
    from alloy.io.robot_file import require_joint

    require_joint(0, [])                # no robot file: never refused, whatever the joint count
    require_joint(3, ["arm.xml"])       # a joint somewhere: not refused


def test_unexpanded_xacro_urdf_is_refused_with_the_command(tmp_path):
    path = _write(tmp_path, "husky.urdf", URDF_XACRO)
    with pytest.raises(alloy.BuildError, match="unexpanded xacro") as err:
        alloy.build(_robot_scene(path, "h"), device="cpu")
    assert "xacro husky.urdf" in err.value.fix
    with pytest.raises(ImportFileError, match="unexpanded xacro") as err2:
        alloy.io.import_file(path, tmp_path / "cards")
    assert "xacro husky.urdf" in err2.value.fix


# --- S3-014: per-body masses, passive damping and springs ---------------------------------------------------------

def test_import_keeps_each_body_mass_and_the_joint_spring(tmp_path):
    res = alloy.io.import_file(_write(tmp_path, "lamp_scene.xml", LAMP), tmp_path / "cards")
    m = mujoco.MjModel.from_xml_path(str(next((tmp_path / "cards").rglob("*.xml"))))
    np.testing.assert_allclose(m.body_mass[1:], [10.0, 0.05], rtol=1e-5)       # was the average, 5.025 each
    assert float(m.dof_damping[6]) == pytest.approx(5.0) and float(m.jnt_stiffness[1]) == pytest.approx(25.0)
    sim = alloy.load(res.scene, device="cpu")
    sim.run(1.0, record=[])
    st = sim.state(world=0)
    assert sim.check()["ok"] and abs(float(st["lamp_scene_base/neck.q"])) < 0.02       # it toppled to -0.81 rad
    assert float(st["lamp_scene_base/shade.pos"][2]) == pytest.approx(0.25, abs=5e-3)


def test_export_writes_passive_joint_damping(tmp_path):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("pend", mjcf=PENDULUM)
    files = alloy.io.export_scene(scene, tmp_path / "out").to_json()["files"]
    assert '<dynamics damping="2.5"' in (Path(files["urdf"]).parent / "pend.urdf").read_text()
    assert "newton:damping" in Path(files["usd"]).read_text()
    pytest.importorskip("pxr")
    pytest.importorskip("newton_usd_schemas")
    ref = alloy.build(scene, device="cpu")
    ref.run(1.0, record=[])
    sim = alloy.load(files["usd"], device="cpu")
    sim.run(1.0, record=[])
    assert float(sim.state(world=0)["scene/pivot.q"]) == pytest.approx(float(ref.state(world=0)["pend/pivot.q"]),
                                                                       abs=2e-3)


# --- S3-015: mirror-scaled meshes ---------------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["mjcf", "urdf"])
def test_mirror_scaled_mesh_imports_with_outward_winding(tmp_path, fmt):
    _write(tmp_path, "part.obj", BOX_OBJ)
    text = {"mjcf": '<mujoco model="mir"><asset><mesh name="m" file="part.obj" scale="-1 1 1"/></asset><worldbody>'
                    '<body name="part" pos="0 0 0.05"><freejoint/><geom type="mesh" mesh="m" mass="0.5"/></body>'
                    '</worldbody></mujoco>',
            "urdf": '<robot name="part"><link name="base"><inertial><mass value="0.5"/><inertia ixx="0.001" ixy="0" '
                    'ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial><collision><geometry>'
                    '<mesh filename="part.obj" scale="-1 1 1"/></geometry></collision></link></robot>'}[fmt]
    res = alloy.io.import_file(_write(tmp_path, f"mir.{'xml' if fmt == 'mjcf' else 'urdf'}", text), tmp_path / "cards")
    mesh = read_obj(next((tmp_path / "cards").rglob("*.obj")))
    assert signed_volume(mesh) == pytest.approx(0.06 * 0.04 * 0.08, rel=1e-4) and not _errors(res)
    assert alloy.load(res.scene, device="cpu").check()["ok"]


# --- S3-002: actuators in exported USD, the URDF index, and a card folder that can move ---------------------------

def _arm_export(tmp_path) -> dict:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(0.5, 0.0, 0.0))
    scene.add_box("cube", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.1), mass=0.2)
    return alloy.io.export_scene(scene, tmp_path / "exported").to_json()["files"]


def test_exported_usd_keeps_the_actuators(tmp_path):
    files = _arm_export(tmp_path)
    text = Path(files["usd"]).read_text()
    assert text.count("def MjcActuator") == 2 and "rel mjc:target = </scene/Joints/j1>" in text
    pytest.importorskip("pxr")
    pytest.importorskip("newton_usd_schemas")
    sim = alloy.load(files["usd"], device="cpu")
    assert {"scene/j1_act", "scene/j2_act"} <= set(sim.names()["actuators"])       # were []
    sim.set_targets({"scene/j1": 0.3})
    sim.step(150)
    assert float(sim.state(world=0)["scene/j1.q"]) == pytest.approx(0.3, abs=0.05) and sim.check()["ok"]


def test_urdf_index_imports_and_load_names_the_way(tmp_path):
    files = _arm_export(tmp_path)
    with pytest.raises(alloy.SceneError, match="URDF index") as err:
        alloy.load(files["urdf"], device="cpu")
    assert "alloy.io.import_file" in err.value.fix
    res = alloy.io.import_file(files["urdf"], tmp_path / "cards")
    assert len(res.robots) == 1 and any("cube" in name for name in res.objects) and not _errors(res)
    sim = alloy.load(res.scene, device="cpu")
    sim.step(5)                                     # URDF has no servo gains: the arm is limp, so a short run only
    assert sim.check()["ok"] and {"scene_arm/j1", "scene_arm/j2"} <= set(sim.names()["joints"])
    with pytest.raises(ImportFileError, match="not a file alloy.io imports"):
        alloy.io.import_file(_write(tmp_path, "other.json", json.dumps({"a": 1})), tmp_path / "cards2")


@pytest.mark.parametrize("fmt", ["mjcf", "urdf"])
def test_imported_card_folder_loads_after_the_source_is_gone(tmp_path, fmt):
    src = tmp_path / "src"
    _write(src / "meshes", "part.obj", BOX_OBJ)
    if fmt == "mjcf":
        text = ARM.replace("<worldbody>", '<asset><mesh name="m" file="meshes/part.obj"/></asset><worldbody>') \
                  .replace('<geom type="sphere" pos="0.2 0 0" size="0.04" mass="0.8"/>',
                           '<geom type="mesh" mesh="m" pos="0.2 0 0" mass="0.8"/>')
    else:
        text = URDF_HINGE.replace('<box size="0.015 0.02 0.05"/>', '<mesh filename="meshes/part.obj"/>')
    res = alloy.io.import_file(_write(src, f"robot.{'xml' if fmt == 'mjcf' else 'urdf'}", text), tmp_path / "cards")
    shutil.rmtree(src)
    shutil.move(str(tmp_path / "cards"), str(tmp_path / "moved"))
    scene = tmp_path / "moved" / "scenes" / f"{res.name}.yaml"
    assert "path: assets/" in scene.read_text() and str(tmp_path) not in scene.read_text()
    sim = alloy.load(str(scene), device="cpu")
    sim.step(5)
    assert sim.check()["ok"]
