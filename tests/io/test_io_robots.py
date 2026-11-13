# SPDX-License-Identifier: Apache-2.0
"""Robots in imports and exports: a robot inside a larger MJCF is cut out by pruning the file (excludes, couplings,
actuators and keyframes stay), closed loops survive the generated-MJCF path, and a base on a joint to the world is
written to URDF and USD."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from alloy.cards.library import default_root
from alloy.core.registry import isolated_registry
from alloy.io import ImportFileError, import_file
from alloy.io.groups import split_groups
from alloy.io.mjcf_prune import PruneError, prune_mjcf
from alloy.io.mjcf_write import write_group_mjcf
from alloy.io.model import IOBody, IOJoint, IOModel, IOShape
from alloy.io.urdf_write import write_urdf
from alloy.io.usd_write import usd_text
from alloy.testing.io import arm_model

mujoco = pytest.importorskip("mujoco")

SCENE = """<mujoco model="cell">
  <compiler angle="radian" meshdir="parts"/>
  <asset><mesh name="tri" file="tri.obj"/><mesh name="crate" file="crate.obj"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0.1 0.2 0">
      <geom name="base_g" type="box" size="0.1 0.1 0.05"/>
      <body name="arm" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom name="arm_g" type="mesh" mesh="tri" mass="0.5"/>
        <site name="tip" pos="0.3 0 0"/>
        <body name="finger" pos="0.3 0 0">
          <joint name="grip" type="slide" axis="1 0 0" range="0 0.04"/>
          <geom name="finger_g" type="box" size="0.01 0.01 0.02" mass="0.05"/>
        </body>
      </body>
    </body>
    <body name="crate" pos="0.6 0 0.1">
      <freejoint/>
      <geom name="crate_g" type="mesh" mesh="crate" mass="1"/>
      <site name="crate_top" pos="0 0 0.1"/>
    </body>
  </worldbody>
  <tendon><fixed name="t_grip"><joint joint="grip" coef="1"/></fixed></tendon>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="100"/>
    <general name="a_grip" tendon="t_grip" dyntype="filter" dynprm="0.1" gainprm="50"/>
    <motor name="a_crate" site="crate_top" gear="0 0 1 0 0 0"/>
  </actuator>
  <equality><weld name="hold" body1="crate" body2="finger" active="false"/></equality>
  <contact>EXCLUDES<exclude body1="base" body2="arm"/><pair geom1="crate_g" geom2="floor"/></contact>
  <sensor><jointpos name="s_grip" joint="grip"/><framepos name="s_crate" objtype="site" objname="crate_top"/></sensor>
  <keyframe><key name="init" qpos="0.5 0.02 0.6 0 0.1 1 0 0 0" ctrl="0.5 0.02 0" act="0.01"/></keyframe>
</mujoco>
"""
TET = "v 0 0 0\nv 0.3 0 0\nv 0 0.05 0\nv 0 0 0.05\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 1 4 3\n"


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def _scene_file(tmp_path, excludes=""):
    (tmp_path / "parts").mkdir()
    for name in ("tri", "crate"):
        (tmp_path / "parts" / f"{name}.obj").write_text(TET)
    path = tmp_path / "cell.xml"
    path.write_text(SCENE.replace("EXCLUDES", excludes))
    return path


def test_prune_keeps_the_robot_as_written(tmp_path):
    src = _scene_file(tmp_path)
    rep = prune_mjcf(src, {"base"}, tmp_path / "out" / "robot.xml")
    m = mujoco.MjModel.from_xml_path(str(rep.path))
    assert (m.nq, m.nu, m.na, m.ntendon, m.nexclude, m.neq, m.nsensor, m.nmesh) == (2, 2, 1, 1, 1, 0, 1, 1)
    assert rep.removed == {"worldbody": 2, "equality": 1, "actuator": 1, "contact": 1, "sensor": 1}
    assert not rep.kept_world and m.ngeom == 3                               # the floor went with the crate
    np.testing.assert_allclose(m.key_qpos[0], [0.5, 0.02])                   # keyframe cut to the kept joints
    np.testing.assert_allclose(m.key_ctrl[0], [0.5, 0.02])
    np.testing.assert_allclose(m.key_act[0], [0.01])
    mesh = ET.parse(rep.path).getroot().find("asset/mesh")
    assert mesh.get("file") == "files/tri.obj" and (tmp_path / "out" / "files" / "tri.obj").is_file()   # copied
    assert m.body_pos[1] == pytest.approx([0.1, 0.2, 0.0])                   # the base keeps its place in the world


def test_prune_keeps_world_geoms_the_robot_is_excluded_from(tmp_path):
    src = _scene_file(tmp_path, '<exclude body1="world" body2="base"/>')
    rep = prune_mjcf(src, {"base"}, tmp_path / "robot.xml")
    m = mujoco.MjModel.from_xml_path(str(rep.path))
    assert rep.kept_world and m.ngeom == 4 and m.nexclude == 2
    with pytest.raises(PruneError, match="no top-level body"):
        prune_mjcf(src, {"arm"}, tmp_path / "bad.xml")


def test_import_cuts_the_robot_and_cards_the_rest(tmp_path):
    res = import_file(_scene_file(tmp_path), tmp_path / "lib")
    assert [r["name"] for r in res.robots] == ["cell_robots"] and res.robots[0]["base_pose"][:3] == [0.0, 0.0, 0.0]
    assert sorted(res.objects) == ["cell_floor"]        # the crate is welded to the finger: it stays in the file
    m = mujoco.MjModel.from_xml_path(res.robots[0]["path"])
    assert (m.nu, m.ntendon, m.nexclude, m.neq) == (3, 1, 1, 1)
    assert not [u for u in res.unmapped if "generated" in u.reason]


@pytest.mark.parametrize("pack,counts", [("emb_e3_ur10e_2f85", (7, 3, 6)), ("emb_e2_panda", (8, 1, 2)),
                                         ("emb_e5_fr3_duo", (21, 2, 9))])
def test_shipped_robot_scenes_import_with_couplings_and_excludes(tmp_path, pack, counts):
    """AUDIT-STACK TOOLS-7 (UR10e + 2F-85: closed loops) and TOOLS-8 (Panda, FR3 duo: excludes lost, links
    overlapping at the start)."""
    res = import_file(default_root() / "robots" / pack / "scene.xml", tmp_path / "lib")
    assert len(res.robots) == 1 and not [i for i in res.issues if i["level"] == "error"]
    m = mujoco.MjModel.from_xml_path(res.robots[0]["path"])
    assert (m.nu, m.neq, m.nexclude) == counts and m.nkey == 1
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)                                   # qpos0: where Alloy starts a robot
    assert min((c.dist for c in d.contact), default=0.0) > -1e-3
    shown = int(((m.geom_contype == 0) & (m.geom_conaffinity == 0) & (m.geom_group == 2)).sum())
    assert (shown > 0) == (default_root() / "robots" / pack / "visuals").is_dir()   # the pack's display meshes
    floor_card = any(name.endswith("_floor") for name in res.objects)
    assert floor_card == (pack != "emb_e5_fr3_duo")           # the duo excludes world contacts: its floor stays


def test_generated_mjcf_closes_loops_with_equalities(tmp_path):
    io = arm_model()
    io.joints.append(IOJoint("loop", "ball", "base", "link2", (0.0, 0.0, 0.3, 0, 0, 0, 1)))
    io.joints.append(IOJoint("loop_slide", "prismatic", "base", "link2"))
    path = write_group_mjcf(io, split_groups(io)[0], tmp_path / "arm.xml", robot=True)
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.nbody == 4 and m.neq == 1 and m.eq_type[0] == mujoco.mjtEq.mjEQ_CONNECT
    assert any(u.item == "loop_slide" for u in io.unmapped)


def _hinged_base() -> IOModel:
    shape = IOShape("link_g", "capsule", (0.02, 0.15))
    body = IOBody("link1", (0.0, 0.0, 0.5, 0, 0, 0, 1), 1.0, (0.0, 0.0, 0.0), ((0.01, 0, 0), (0, 0.01, 0), (0, 0, 1e-4)),
                  (shape,), group="arm")
    hinge = IOJoint("shoulder", "revolute", None, "link1", (0.0, 0.0, 0.5, 0, 0, 0, 1), axis=(0.0, 1.0, 0.0),
                    lower=-2.0, upper=2.0, damping=2.0, stiffness=200.0, actuated=True)
    return IOModel("arm", [body], [], [hinge], [], {}, {"arm": "robot"})


def test_base_joint_to_the_world_reaches_urdf_usd_and_mjcf(tmp_path):
    io = _hinged_base()
    index = write_urdf(io, tmp_path / "urdf")
    g = index["groups"][0]
    assert g["fixed_base"] and g["articulated"] and g["base_pose"][:3] == [0.0, 0.0, 0.0]
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "urdf" / g["file"]))
    assert m.njnt == 1 and m.jnt_range[0] == pytest.approx([-2.0, 2.0]) and m.body_mass.sum() == pytest.approx(1.0)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    assert d.xpos[-1] == pytest.approx([0.0, 0.0, 0.5])       # the file alone puts the link where the scene had it
    assert "PhysicsRevoluteJoint" in usd_text(io)
    path = write_group_mjcf(io, split_groups(io)[0], tmp_path / "arm.xml", robot=True)
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.njnt == 1 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_HINGE and m.nu == 1


def test_usd_import_without_usd_packages_names_the_fix(tmp_path, monkeypatch):
    (tmp_path / "x.usda").write_text("#usda 1.0\n")
    monkeypatch.setitem(sys.modules, "pxr", None)             # as on an install without usd-core
    with pytest.raises(ImportFileError, match="pip install") as err:
        import_file(tmp_path / "x.usda", tmp_path / "lib")
    assert "usd" in err.value.fix and "usd-core" in str(err.value)
