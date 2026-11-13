# SPDX-License-Identifier: Apache-2.0
"""A robot file compose cannot use is an error that names it (field "robots"), raised before any object is placed and
in every check mode. Before, a missing file was blamed on each object in turn under the load checks (every object
dropped, "no object on this install matches"), passed with ok=True under check="none" (an unloadable scene card), and
a malformed MJCF leaked a raw XML ParseError."""

from __future__ import annotations

import pytest

from alloy.cards.library import CardLibrary, default_root
from alloy.cli.command import Session, respond
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog
from alloy.compose.spec import ComposeError

ARM = """<mujoco model="arm">
  <worldbody>
    <body name="base">
      <geom type="cylinder" size="0.06 0.02" pos="0 0 0.02" mass="2"/>
      <body name="upper" pos="0 0 0.04">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="5"/>
        <geom type="capsule" size="0.025 0.15" pos="0 0 0.17" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


@pytest.mark.parametrize("check", ["none", "load"])
def test_a_missing_robot_file_is_named(cat, check):
    with pytest.raises(ComposeError) as e:
        compose("a mug and a wooden cube on a table", robots=["missing_arm.xml"], check=check, device="cpu",
                catalog=cat)
    assert e.value.field == "robots" and "'missing_arm.xml' is not found" in e.value.message
    assert "robots/emb_e2_panda/scene.xml" in e.value.fix


def test_the_compose_command_names_the_robot_not_the_objects():
    resp = respond(Session(), {"op": "compose", "sentence": "a mug and a wooden cube on a table",
                               "robots": ["missing_arm.xml"], "check": "load", "device": "cpu"})
    assert resp["ok"] is False and resp["error"]["field"] == "robots"
    assert "missing_arm.xml" in resp["error"]["message"] and "mug" not in resp["error"]["message"]


def test_a_malformed_robot_file_is_a_compose_error(cat, tmp_path):
    bad = tmp_path / "bad_robot.xml"
    bad.write_text("<mujoco><worldbody><body name='arm'>", encoding="utf-8")
    with pytest.raises(ComposeError) as e:
        compose("a mug on a table", robots=[str(bad)], check="none", device="cpu", catalog=cat)
    assert e.value.field == "robots" and "is not readable XML" in e.value.message


@pytest.mark.parametrize("text", [
    "<scene/>",                                                                 # XML, but not a robot
    '<mujoco><asset><mesh name="m" file="nope.stl"/></asset><worldbody><body name="arm">'
    '<joint name="j" type="hinge"/><geom type="mesh" mesh="m" mass="1"/></body></worldbody></mujoco>',   # no mesh
])
def test_a_robot_the_loader_cannot_read_is_named(cat, tmp_path, text):
    arm = tmp_path / "arm.xml"
    arm.write_text(text, encoding="utf-8")
    with pytest.raises(ComposeError) as e:
        compose("a mug on a table", robots=[str(arm)], check="load", device="cpu", catalog=cat)
    assert e.value.field == "robots" and str(arm) in e.value.message and "does not load" in e.value.message


def test_a_robot_file_in_the_current_folder_is_used(cat, tmp_path, monkeypatch):
    (tmp_path / "arm.xml").write_text(ARM, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = compose("two cubes on a table", robots=["arm.xml"], check="load", device="cpu", catalog=cat)
    assert r.ok, r.summary()
    assert r.scenes[0].card.robots[0].path == str(tmp_path / "arm.xml")
