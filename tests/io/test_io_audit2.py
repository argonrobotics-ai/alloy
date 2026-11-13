# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the second batch of import/export audit findings: each is the auditor's case, cut down,
asserting the fixed behaviour (bounce and stiffness in exported files, pair friction, kinematic bodies, welds and
connects, fixed-base and driven one-link robots, materials of jointed objects, gravity compensation and motors, URDF
mesh paths and free bases, MJCF compiler settings, rods, unreadable USD, bodies with several joints)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

import alloy
import alloy.io
from alloy.io.errors import ExportError, ImportFileError
from alloy.io.importer import read_file

HOPPER = """<mujoco model="hopper"><compiler angle="radian"/><worldbody>
  <body name="torso" pos="0 0 0.4"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="2"/>
    <body name="leg" pos="0 0 -0.1"><joint name="knee" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.02" mass="0.5"/></body></body></worldbody>
  <actuator><position name="a_knee" joint="knee" kp="50" kv="2"/></actuator></mujoco>"""
GANTRY = """<mujoco model="gantry"><worldbody><geom name="floor" type="plane" size="2 2 0.05"/>
  <body name="carriage" pos="0 0 0.4"><joint name="rail_x" type="slide" axis="1 0 0" range="-0.5 0.5" damping="2"/>
    <geom name="c_geom" type="box" size="0.06 0.06 0.04" mass="2"/></body></worldbody>
  <actuator><position name="drive_x" joint="rail_x" kp="100" kv="10"/></actuator></mujoco>"""
CABINET = """<mujoco model="cabinet"><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="2 2 0.05"/>
  <body name="frame" pos="0 0 0.45"><geom name="f_geom" type="box" size="0.15 0.1 0.2" mass="5"/>
    <body name="door" pos="0.16 0 0"><joint name="hinge" type="hinge" axis="0 0 1" range="0 1.5" damping="0.2"/>
      <geom name="d_geom" type="box" size="0.005 0.1 0.2" mass="1"/></body></body></worldbody></mujoco>"""
SLED = """<mujoco model="sled"><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="3 3 0.05" friction="0.3"/>
  <body name="hub" pos="0 0 0.1"><freejoint/><geom name="handle" type="box" size="0.05 0.05 0.02" mass="1"
      friction="0.85"/>
    <body name="ski" pos="0 0 -0.05"><joint name="flex" type="hinge" axis="0 1 0" range="-0.1 0.1"/>
      <geom name="ski_g" type="box" size="0.15 0.03 0.01" mass="0.5" friction="0.05"/></body></body></worldbody>
</mujoco>"""
GRAV_ARM = """<mujoco model="arm"><compiler angle="radian"/><worldbody>
  <body name="base" pos="0 0 0.5"><geom type="box" size="0.05 0.05 0.05" mass="2"/>
    <body name="grav_arm" pos="0 -0.14 0.06" gravcomp="1">
      <joint name="j_grav" type="hinge" axis="0 1 0" damping="0.3" range="-2.5 2.5"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1.5"/></body>
    <body name="mot_arm" pos="0 0.14 0.06" gravcomp="1">
      <joint name="j_mot" type="hinge" axis="0 1 0" damping="0.5" range="-2.5 2.5"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.018" mass="0.5"/></body></body></worldbody>
  <actuator><position name="a_grav" joint="j_grav" kp="2" kv="0.4"/>
    <motor name="a_mot" joint="j_mot" gear="1" ctrlrange="-5 5"/></actuator></mujoco>"""
TOTALMASS = """<mujoco model="tm"><compiler angle="radian" settotalmass="12.0"/><worldbody>
  <body name="b1" pos="0 0 0.5"><freejoint/><geom type="sphere" size="0.1"/>
    <body name="b2" pos="0 0 0.2"><joint name="j" type="hinge" axis="0 1 0"/><geom type="sphere" size="0.07"/>
    </body></body></worldbody></mujoco>"""
URDF_MESH = """<robot name="pedestal"><link name="base"><inertial><mass value="1"/>
  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  <collision><geometry><mesh filename="{ref}"/></geometry></collision></link>
  <link name="arm"><inertial><mass value="0.2"/>
  <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link>
  <joint name="j" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="2"/></joint></robot>"""
BOX_OBJ = """v -0.1 -0.1 0\nv 0.1 -0.1 0\nv 0.1 0.1 0\nv -0.1 0.1 0\nv -0.1 -0.1 0.2\nv 0.1 -0.1 0.2\nv 0.1 0.1 0.2
v -0.1 0.1 0.2\nf 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\nf 3 4 8\nf 3 8 7\nf 2 3 7\nf 2 7 6
f 4 1 5\nf 4 5 8
"""
usd = pytest.mark.usefixtures("usd_packages")


@pytest.fixture
def usd_packages():
    pytest.importorskip("pxr")
    pytest.importorskip("newton_usd_schemas")


def _write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text)
    return path


def _z(sim, key: str) -> float:
    return float(sim.state(world=0)[key][2])


def _reasons(res, fmt: str | None = None) -> str:
    return " | ".join(u.reason for u in res.unmapped if fmt is None or u.format == fmt)


# --- bounce and contact stiffness in exported files ---------------------------------------------------------------

def _balls() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_material("superball", density=800.0, friction=0.3, restitution=0.85)
    scene.add_sphere("abs_ball", radius=0.04, pos=(0.0, -0.15, 0.5))
    scene.add_sphere("super_ball", radius=0.04, pos=(0.0, 0.15, 0.5), material="superball")
    return scene


def test_imported_mjcf_export_keeps_the_bounce(tmp_path):
    scene = _balls()
    ref = alloy.build(scene, device="cpu")
    ref.run(0.5, record=[])
    files = alloy.io.export_scene(scene, tmp_path / "out").to_json()["files"]
    res = alloy.io.import_file(files["mjcf"], tmp_path / "cards")
    assert "pairs:" in Path(res.scene).read_text()                  # the file's explicit contact pair
    sim = alloy.load(res.scene, device="cpu")
    sim.run(0.5, record=[])
    for ball in ("abs_ball", "super_ball"):                         # both sat dead on the floor at z = 0.04
        assert _z(sim, f"scene_{ball}.pos") == pytest.approx(_z(ref, f"{ball}.pos"), abs=0.02)


@usd
def test_exported_usd_reloads_with_one_floor_and_stiff_contacts(tmp_path):
    files = alloy.io.export_scene(_balls(), tmp_path / "out").to_json()["files"]
    assert "mjc:solref" in Path(files["usd"]).read_text()
    sim = alloy.load(files["usd"], device="cpu")
    labels = [str(x) for x in sim.built.model.shape_label]
    assert len([x for x in labels if "ground" in x]) == 1           # alloy.load added a second, coplanar one
    sim.run(0.5, record=[])
    assert sim.check()["ok"] and _z(sim, "scene/super_ball.pos") > 0.25      # sank 20 mm into the floor: INVALID


# --- pair friction and kinematic bodies ---------------------------------------------------------------------------

def _ramp() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("ramp", size=(1.0, 0.4, 0.02), pos=(0, 0, 0.3), rpy=(0.0, 0.26, 0.0), static=True)
    scene.add_box("puck", size=(0.06, 0.06, 0.02), pos=(-0.2, 0, 0.375), rpy=(0.0, 0.26, 0.0), mass=0.2)
    scene.set_friction("puck", "ramp", 0.02)
    scene.add_box("pusher", size=(0.05, 0.05, 0.05), pos=(0.6, 0.6, 0.025), kinematic=True)
    return scene


def test_pair_friction_and_kinematic_bodies_survive_export_and_import(tmp_path):
    scene = _ramp()
    ref = alloy.build(scene, device="cpu")
    x0 = float(ref.state(world=0)["puck.pos"][0])
    ref.run(0.4, record=[])
    slide = float(ref.state(world=0)["puck.pos"][0]) - x0
    exp = alloy.io.export_scene(scene, tmp_path / "out")
    assert "pair rows" in _reasons(exp, "usd") and "pair rows" in _reasons(exp, "urdf")
    assert "kinematic object" in _reasons(exp, "all")
    text = Path(exp.files["mjcf"]).read_text()
    assert 'mocap="true"' in text and mujoco.MjModel.from_xml_path(exp.files["mjcf"]).nmocap == 1
    res = alloy.io.import_file(exp.files["mjcf"], tmp_path / "cards")
    assert "motion: kinematic" in (tmp_path / "cards" / "objects" / "scene_pusher.yaml").read_text()
    sim = alloy.load(res.scene, device="cpu")
    x1 = float(sim.state(world=0)["scene_puck.pos"][0])
    sim.run(0.4, record=[])
    assert slide > 0.1 and float(sim.state(world=0)["scene_puck.pos"][0]) - x1 == pytest.approx(slide, rel=0.1)
    sim.drive("scene_pusher", velocity=(0.1, 0.0, 0.0), duration=0.2)       # was a static card: drive raised
    sim.step(20)
    assert float(sim.state(world=0)["scene_pusher.pos"][0]) == pytest.approx(0.62, abs=5e-3)


# --- welds, connects, and bodies welded to a robot ----------------------------------------------------------------

def test_connect_and_robot_welded_bodies_stay_in_the_file(tmp_path):
    pend = _write(tmp_path, "pend.xml", '<mujoco model="pend"><worldbody><geom name="floor" type="plane" '
                  'size="2 2 0.05"/><body name="bob" pos="0 0 0.3"><freejoint/><geom name="g" type="sphere" '
                  'size="0.03" mass="0.5"/></body></worldbody><equality><connect body1="bob" anchor="0 0 0.15"/>'
                  "</equality></mujoco>")
    res = alloy.io.import_file(pend, tmp_path / "c1")
    assert len(res.robots) == 1 and res.objects == ["pend_floor"]
    sim = alloy.load(res.scene, device="cpu")
    sim.step(40)
    assert sim.state(world=0)["pend_bob/bob.pos"][2] > 0.25             # fell to the floor (0.03)
    tool = _write(tmp_path, "tool.xml", '<mujoco model="t"><worldbody><geom name="floor" type="plane" size="2 2 '
                  '0.05"/><body name="base" pos="0 0 0.4"><geom type="box" size="0.05 0.05 0.05" mass="2"/><body '
                  'name="carrier" pos="0.1 0 0"><joint name="slide_x" type="slide" axis="1 0 0" damping="1"/><geom '
                  'type="box" size="0.04 0.04 0.04" mass="1"/></body></body><body name="welded_tool" pos="0.25 0 '
                  '0.4"><freejoint/><geom type="box" size="0.03 0.03 0.03" mass="0.5"/></body></worldbody><equality>'
                  '<weld body1="carrier" body2="welded_tool"/></equality><actuator><position name="act_x" '
                  'joint="slide_x" kp="100" kv="10"/></actuator></mujoco>')
    res = alloy.io.import_file(tool, tmp_path / "c2")
    m = mujoco.MjModel.from_xml_path(res.robots[0]["path"])
    assert m.neq == 1 and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "welded_tool") > 0     # it was erased


@usd
def test_exported_usd_reloads_a_weld_between_two_bodies(tmp_path):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("r", mjcf='<mujoco><worldbody><body name="a" pos="0 0 0.3"><freejoint/><geom type="box" '
                    'size="0.03 0.03 0.03" mass="0.5"/></body><body name="b" pos="0.1 0 0.3"><freejoint/><geom '
                    'type="box" size="0.03 0.03 0.03" mass="0.5"/></body></worldbody><equality><weld body1="a" '
                    'body2="b"/></equality></mujoco>')
    files = alloy.io.export_scene(scene, tmp_path / "out", formats=["usd"]).to_json()["files"]
    assert "MjcEqualityWeldAPI" in Path(files["usd"]).read_text()
    sim = alloy.load(files["usd"], device="cpu")            # raised "joint(s) not belonging to any articulation"
    sim.step(30)
    st = sim.state(world=0)
    assert float(st["scene/b.pos"][0] - st["scene/a.pos"][0]) == pytest.approx(0.1, abs=5e-3)


# --- fixed-base and driven one-link robots, materials of jointed objects ------------------------------------------

def test_driven_one_link_robot_and_wall_cabinet_stay_mounted(tmp_path):
    res = alloy.io.import_file(_write(tmp_path, "gantry.xml", GANTRY), tmp_path / "c1")
    assert len(res.robots) == 1                                     # was an object card with no actuator
    sim = alloy.load(res.scene, device="cpu")
    name = res.robots[0]["name"]
    sim.run(0.3, targets={f"{name}/rail_x": 0.25}, record=[])
    pos = sim.state(world=0)[f"{name}/carriage.pos"]
    assert pos[2] == pytest.approx(0.4, abs=1e-3) and pos[0] > 0.15         # it fell off its rail to z = 0.04
    res = alloy.io.import_file(_write(tmp_path, "cabinet.xml", CABINET), tmp_path / "c2")
    assert len(res.robots) == 1 and res.objects == ["cabinet_floor"]
    sim = alloy.load(res.scene, device="cpu")
    sim.step(50)
    name = res.robots[0]["name"]
    assert _z(sim, f"{name}/frame.pos") == pytest.approx(0.45, abs=1e-3)    # the frame dropped to the floor


def test_jointed_object_with_several_surfaces_keeps_each_friction(tmp_path):
    res = alloy.io.import_file(_write(tmp_path, "sled.xml", SLED), tmp_path / "cards")
    assert len(res.robots) == 1                                     # one card material replaced the ski's 0.05
    sim = alloy.load(res.scene, device="cpu")
    mus = sorted(round(float(x), 3) for x in sim.built.model.shape_material_mu.numpy())
    assert 0.05 in mus and 0.85 in mus


# --- gravity compensation and motors ------------------------------------------------------------------------------

def test_gravity_compensation_and_motors_are_written(tmp_path):
    from alloy.io.groups import split_groups
    from alloy.io.mjcf_write import write_group_mjcf
    from alloy.io.usd_write import usd_text

    io = read_file(_write(tmp_path, "arm.xml", GRAV_ARM))
    assert [b.gravcomp for b in io.bodies] == [0.0, 1.0, 1.0]
    mot = next(j for j in io.joints if j.name == "j_mot")
    assert mot.motor_gear == 1.0 and mot.ctrl_range == (-5.0, 5.0) and mot.stiffness == 0.0
    out = write_group_mjcf(io, split_groups(io)[0], tmp_path / "gen.xml", robot=True)
    m = mujoco.MjModel.from_xml_path(str(out))
    np.testing.assert_allclose(m.body_gravcomp[1:], [0.0, 1.0, 1.0])
    assert m.nu == 2 and '<motor name="j_mot_act"' in out.read_text()
    text = usd_text(io)
    assert text.count("float mjc:gravcomp = 1") == 2 and 'mjc:biasType = "none"' in text


@usd
def test_exported_usd_holds_a_gravity_compensated_arm(tmp_path):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=GRAV_ARM)
    files = alloy.io.export_scene(scene, tmp_path / "out", formats=["usd"]).to_json()["files"]
    sim = alloy.load(files["usd"], device="cpu")
    sim.step(50)
    assert abs(float(sim.state(world=0)["scene/j_grav.q"])) < 0.05          # collapsed to 0.89 rad


# --- URDF mesh paths and free bases -------------------------------------------------------------------------------

def test_urdf_mesh_uris_resolve_and_a_missing_mesh_is_refused(tmp_path):
    mesh = _write(tmp_path / "pkg" / "pedestal_pkg", "base.obj", BOX_OBJ)
    counts = {}
    for tag, ref in (("relative", "pedestal_pkg/base.obj"), ("file", mesh.as_uri()),
                     ("package", "package://pedestal_pkg/base.obj"), ("model", "model://pedestal_pkg/base.obj")):
        scene = alloy.Scene(dt=0.01)
        scene.add_ground()
        scene.add_robot("r", _write(tmp_path / "pkg", f"{tag}.urdf", URDF_MESH.format(ref=ref)))
        counts[tag] = alloy.build(scene, device="cpu").built.model.shape_count
    assert len(set(counts.values())) == 1, counts                   # URI forms loaded without the base's mesh
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("r", _write(tmp_path / "pkg", "missing.urdf", URDF_MESH.format(ref="nowhere.stl")))
    with pytest.raises(alloy.BuildError, match="not found") as err:
        alloy.build(scene, device="cpu")
    assert "mesh path" in err.value.fix


def test_floating_flag_and_free_base_urdf(tmp_path):
    src = _write(tmp_path, "hopper.xml", HOPPER)
    res = alloy.io.import_file(src, tmp_path / "fixed", floating=False)
    assert "floating= changes the root joint" in _reasons(res)
    sim = alloy.load(res.scene, device="cpu")
    sim.step(30)
    name = res.robots[0]["name"]
    assert _z(sim, f"{name}/torso.pos") == pytest.approx(0.4, abs=1e-3)     # the flag was ignored: it fell
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("hopper", src)
    files = alloy.io.export_scene(scene, tmp_path / "out", formats=["urdf"]).to_json()["files"]
    urdf = Path(files["urdf"]).parent / "hopper.urdf"
    assert 'type="floating"' in urdf.read_text()
    again = alloy.Scene(dt=0.01)
    again.add_ground()
    again.add_robot("hopper", urdf)
    sim = alloy.build(again, device="cpu")
    z0 = _z(sim, "hopper/torso.pos")
    sim.step(20)
    assert z0 == pytest.approx(0.4, abs=1e-3) and _z(sim, "hopper/torso.pos") < z0 - 0.05   # was welded in mid-air


# --- MJCF compiler settings, rods, unreadable USD, bodies with several joints -------------------------------------

def test_settotalmass_is_applied_and_global_coordinates_refused(tmp_path):
    path = _write(tmp_path, "tm.xml", TOTALMASS)
    want = mujoco.MjModel.from_xml_path(str(path)).body_mass[1:]
    io = read_file(path)
    np.testing.assert_allclose([b.mass for b in io.bodies], want, rtol=1e-4)        # 3.1 kg in all, not 12
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("r", path)
    assert float(alloy.build(scene, device="cpu").built.model.body_mass.numpy().sum()) == pytest.approx(12.0, 1e-4)
    glob = _write(tmp_path, "glob.xml", TOTALMASS.replace('settotalmass="12.0"', 'coordinate="global"'))
    with pytest.raises(ImportFileError, match="coordinate") as err:
        alloy.io.import_file(glob, tmp_path / "cards")
    assert "relative to their parent" in err.value.fix
    with pytest.raises(alloy.BuildError, match="coordinate"):
        alloy.load(glob, device="cpu")


def test_export_leaves_a_clamped_rod_out_and_says_so(tmp_path):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.3, 0, 0.025), mass=0.2)
    scene.add_rod("hose", length=0.36, radius=0.008, segments=6, pos=(0, 0, 0.3), clamp="start")
    res = alloy.io.export_scene(scene, tmp_path / "out")            # raised: "Joint type 7 is not supported yet"
    assert "elastic rod" in _reasons(res, "all")
    assert sorted(p.name for p in (tmp_path / "out" / "urdf").glob("*.urdf")) == ["block.urdf", "ground.urdf"]


@usd
def test_unreadable_usd_is_a_build_error_with_a_fix(tmp_path):
    bad = _write(tmp_path, "broken.usda", '#usda 1.0\ndef Xform "Bot" { double size = 0.1 }\n')
    with pytest.raises(alloy.BuildError, match="OpenUSD") as err:
        alloy.load(bad, device="cpu")
    assert "usdchecker" in err.value.fix


def test_body_with_several_joints_is_never_written_free(tmp_path):
    xml = ('<mujoco model="stage"><worldbody><geom name="floor" type="plane" size="2 2 0.05"/><body name="xy" '
           'pos="0 0 0.3"><joint name="x" type="slide" axis="1 0 0"/><joint name="y" type="slide" axis="0 1 0"/>'
           '<geom type="box" size="0.05 0.05 0.02" mass="1"/></body></worldbody></mujoco>')
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("stage", mjcf=xml)
    with pytest.raises(ExportError, match="cannot write these joints") as err:
        alloy.io.export_scene(scene, tmp_path / "out")
    assert 'formats=["mjcf"]' in err.value.fix
    assert "mjcf" in alloy.io.export_scene(scene, tmp_path / "mj", formats=["mjcf"]).to_json()["files"]
    res = alloy.io.import_file(_write(tmp_path, "stage.xml", xml), tmp_path / "cards")
    sim = alloy.load(res.scene, device="cpu")                       # the stage stays a robot file, on its joints
    sim.step(30)
    name = res.robots[0]["name"]
    assert _z(sim, f"{name}/xy.pos") == pytest.approx(0.3, abs=1e-3)
