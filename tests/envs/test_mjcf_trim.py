# SPDX-License-Identifier: Apache-2.0
"""Gemini audit T2-023: trimmed and motorized robot files stay valid and never share a cache across checkouts.

``without_bodies`` left equality, actuator, sensor and contact rows that named things inside the dropped bodies (a
trimmed bimanual FR3 failed to build: an equality on the dropped right finger), and ignored the compiler's
``assetdir`` / ``texturedir`` (textures and assets resolved beside the file). ``motorized`` keyed its cache on the XML
text alone while writing absolute paths of the file's own folder, so a second checkout got the first one's copy. The
shipped files trimmed today produce the same bytes as before (physics condition 3). CPU, MuJoCo's own parser."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from alloy.cards.library import default_root
from alloy.envs.actuators import motorized
from alloy.envs.errors import EnvError
from alloy.envs.mjcf import without_bodies

PROPS = ["box", "cylinder", "mug"]
GOLDEN = {   # sha256 of the trimmed file on main a3719c304 (before the fix), the library root written as <LIB>
    ("emb_e2_panda", tuple(PROPS)): "c103f99eee3c273d96681b3d4743b244b3c4b9ec4d91b1d288a0c6536d41ab49",
    ("emb_e8_go2", ("obstacle",)): "91a9575393baae1389424a22a5db0f1a56f6faec1320b2dd038a088c82a03e3b",
    ("emb_e1_fr3", tuple(PROPS)): "282a87c39956abf7557b8edcb31738c7d83502254003002db774e7717860db0a",
    ("emb_e3_ur10e_2f85", tuple(PROPS)): "31ea8acc06b2de13c1c1fcf104a7f090801e47037ea0c0f75bb14b8858a6437a",
    # the same bytes plus the file's own <exclude body1="world" body2="left_base"/> and "right_base" rows, which the
    # trim dropped as naming a missing body (both bases are static: MuJoCo steps the two trims to the same state);
    # since 5f4fb3ff8 the pack lists its two link5 | link7 excludes in the generator's sorted place (the same rows:
    # MuJoCo compiles both orders to identical model arrays; was 16dbc1336a94a446e97a482818ca3b4b30f691eb9ec55548768b8)
    ("emb_e4_fr3_bimanual", tuple(PROPS)): "7566b134512c41d5f2273f3d1d4ea21260e2e22f6171ad2bc0d599a83f2bc39d",
}


@pytest.fixture(autouse=True)
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))


@pytest.mark.parametrize(("robot", "bodies"), sorted(GOLDEN))
def test_shipped_trims_are_byte_identical_to_before(robot, bodies):
    """The bytes of main a3719c304's trims (the bimanual one keeps two world rows: ``GOLDEN``)."""
    src = (default_root() / "robots" / robot / "scene.xml").resolve()
    lib = str(default_root().resolve())                              # meshes sit in robots/meshes, beside the robot
    text = without_bodies(src, list(bodies)).read_text(encoding="utf-8").replace(lib, "<LIB>")
    assert hashlib.sha256(text.encode()).hexdigest() == GOLDEN[(robot, bodies)]


def test_a_trimmed_bimanual_arm_loads_in_mujoco():
    mujoco = pytest.importorskip("mujoco")
    src = default_root() / "robots" / "emb_e4_fr3_bimanual" / "scene.xml"
    out = without_bodies(src, ["right_base", "table", *PROPS])
    model = mujoco.MjModel.from_xml_path(str(out))                   # was: "references unknown joint ..."
    assert model.nbody > 0 and "right_finger_joint1" not in out.read_text(encoding="utf-8")


def test_the_h1_2_obstacle_box_trims_as_a_bare_geom():
    """The H1-2 file's obstacle is a geom directly under <worldbody>, not a body: naming it raised EnvError."""
    mujoco = pytest.importorskip("mujoco")
    src = default_root() / "robots" / "emb_e7_h1_2" / "scene.xml"
    full, trimmed = (mujoco.MjModel.from_xml_path(str(p)) for p in (src, without_bodies(src, ["obstacle"])))
    names = [mujoco.mj_id2name(trimmed, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(trimmed.ngeom)]
    assert "obstacle" not in names and trimmed.ngeom == full.ngeom - 1               # only the box went
    assert (trimmed.nbody, trimmed.nq, trimmed.nu, trimmed.nexclude) == (full.nbody, full.nq, full.nu, full.nexclude)


POST = """<mujoco><worldbody><geom name="post" type="box" size="0.1 0.1 0.1" pos="1 0 0.1"/>
    <body name="arm"><joint name="j" type="hinge"/><geom name="arm_g" type="capsule" size="0.02 0.1"/></body>
    <geom name="wall" type="box" size="0.1 1 0.5" pos="-1 0 0.5"/></worldbody>
  <contact><pair geom1="post" geom2="arm_g"/><pair geom1="wall" geom2="arm_g"/></contact>
  <sensor><framepos objtype="geom" objname="post"/><framepos objtype="geom" objname="wall"/></sensor></mujoco>"""


def test_a_bare_geom_goes_with_the_rows_that_name_it(tmp_path):
    src = tmp_path / "post.xml"
    src.write_text(POST, encoding="utf-8")
    root = ET.parse(without_bodies(src, ["post"])).getroot()
    assert [g.get("name") for g in root.find("worldbody").iter("geom")] == ["arm_g", "wall"]
    assert [r.get("geom1") for r in root.find("contact")] == ["wall"]
    assert [r.get("objname") for r in root.find("sensor")] == ["wall"]
    with pytest.raises(EnvError, match="bodies or geoms"):
        without_bodies(src, ["arm_g"])                                    # a geom inside a body is not top-level


ROBOT = """<mujoco><compiler angle="radian" assetdir="assets" texturedir="textures"/>
  <asset><mesh name="m" file="part.obj"/><texture name="t" type="2d" file="wood.png"/></asset>
  <worldbody>
    <body name="arm"><joint name="j" type="hinge"/><geom name="arm_g" type="mesh" mesh="m"/><site name="tip"/></body>
    <body name="mug"><joint name="mug_j" type="hinge"/><geom name="mug_g" type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
  <contact><pair geom1="arm_g" geom2="mug_g"/><exclude body1="arm" body2="mug"/></contact>
  <equality><joint joint1="j" joint2="mug_j"/></equality>
  <tendon><fixed name="both"><joint joint="j" coef="1"/><joint joint="mug_j" coef="1"/></fixed></tendon>
  <actuator><position joint="j"/><position joint="mug_j"/><motor tendon="both"/></actuator>
  <sensor><jointpos joint="mug_j"/><framepos objtype="body" objname="mug"/><jointpos joint="j"/></sensor>
</mujoco>"""


def test_rows_naming_dropped_parts_go_and_asset_folders_resolve(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "textures").mkdir()
    (tmp_path / "assets" / "part.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\n")
    (tmp_path / "textures" / "wood.png").write_bytes(b"png")
    src = tmp_path / "robot.xml"
    src.write_text(ROBOT, encoding="utf-8")
    root = ET.parse(without_bodies(src, ["mug"])).getroot()
    for section in ("contact", "equality", "tendon", "sensor"):                   # each named something in the mug
        rows = [r.tag for r in root.find(section)]
        assert rows == ({"sensor": ["jointpos"]}.get(section, [])), (section, rows)
    assert [a.get("joint") for a in root.find("actuator")] == ["j"]               # mug_j's servo and the tendon motor
    files = {el.tag: Path(el.get("file")) for el in root.iter() if el.get("file")}
    assert files["mesh"] == (tmp_path / "assets" / "part.obj").resolve()          # was: beside the file
    assert files["texture"] == (tmp_path / "textures" / "wood.png").resolve()


SCENE = """<mujoco><include file="arm.xml"/><worldbody>
    <body name="cup"><freejoint/><geom name="cup_g" type="box" size="0.05 0.05 0.05"/></body>
    <body name="mug"><freejoint/><geom name="mug_g" type="box" size="0.05 0.05 0.05"/></body></worldbody>
  <equality><weld body1="cup" body2="world"/><weld body1="cup" body2="link"/>
    <weld body1="mug" body2="world"/></equality>
  <contact><exclude body1="world" body2="cup"/><exclude body1="link" body2="cup"/></contact>
  <sensor><framepos objtype="body" objname="world"/><framepos objtype="body" objname="link"/></sensor>
</mujoco>"""
ARM = """<mujocoinclude><worldbody><body name="link"><joint name="j" type="hinge"/>
  <geom type="capsule" size="0.02 0.1"/></body></worldbody></mujocoinclude>"""


def test_rows_naming_the_world_or_an_included_part_stay(tmp_path):
    (tmp_path / "arm.xml").write_text(ARM, encoding="utf-8")
    src = tmp_path / "scene.xml"
    src.write_text(SCENE, encoding="utf-8")
    root = ET.parse(without_bodies(src, ["mug"])).getroot()
    rows = {s: [(r.get("body1"), r.get("body2")) for r in root.find(s)] for s in ("equality", "contact")}
    assert rows["equality"] == [("cup", "world"), ("cup", "link")]              # was: both gone with the mug's weld
    assert rows["contact"] == [("world", "cup"), ("link", "cup")]
    assert [r.get("objname") for r in root.find("sensor")] == ["world", "link"]
    mujoco = pytest.importorskip("mujoco")
    assert mujoco.MjModel.from_xml_path(str(without_bodies(src, ["mug"]))).neq == 2


def test_a_copy_an_older_trimmer_wrote_is_not_reused(tmp_path, monkeypatch):
    from alloy.envs import mjcf

    (tmp_path / "arm.xml").write_text(ARM, encoding="utf-8")
    src = tmp_path / "scene.xml"
    src.write_text(SCENE, encoding="utf-8")
    first = without_bodies(src, ["mug"])
    monkeypatch.setattr(mjcf, "TRIM_VERSION", mjcf.TRIM_VERSION + "-next")
    assert without_bodies(src, ["mug"]) != first


SERVO_ARM = """<mujoco><asset><mesh name="m" file="part.obj"/></asset><worldbody>
  <body name="arm"><joint name="j" type="hinge"/><geom type="mesh" mesh="m" mass="0.1"/></body>
</worldbody><actuator><position joint="j" kp="10"/></actuator></mujoco>"""


def test_motorized_copies_of_one_file_in_two_checkouts_do_not_share_a_cache(tmp_path):
    outs = {}
    for name in ("checkout_a", "checkout_b"):                   # the same file, byte for byte, in two folders
        d = tmp_path / name
        d.mkdir()
        (d / "part.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\n")
        (d / "robot.xml").write_text(SERVO_ARM, encoding="utf-8")
        outs[name] = Path(motorized(d / "robot.xml"))
    a, b = outs["checkout_a"], outs["checkout_b"]
    assert a != b                                                                 # was: b got a's cached file
    text_b = b.read_text(encoding="utf-8")
    assert str(tmp_path / "checkout_b" / "part.obj") in text_b and "checkout_a" not in text_b
