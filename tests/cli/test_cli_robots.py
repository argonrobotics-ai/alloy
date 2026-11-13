# SPDX-License-Identifier: Apache-2.0
"""Library robots without Python: ``alloy list robots`` lists the shipped robots (not the task scenes beside them),
and a recipe's robot path takes a robot name or a library-relative path as well as a file, from the shipped library
or a ``--library`` folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from alloy.cli.registry import respond
from alloy.cli.session import Session

SHIPPED = ("emb_e1_fr3", "emb_e2_panda", "emb_e3_ur10e_2f85", "emb_e4_fr3_bimanual", "emb_e5_fr3_duo", "emb_e6_g1",
           "emb_e7_h1_2", "emb_e8_go2", "emb_e9_leap")
ARM = """<mujoco><worldbody><body name="link1" pos="0 0 0.1"><joint name="j" type="hinge" axis="0 1 0"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody></mujoco>"""


def listed(**args):
    reply = respond(Session(), {"op": "list", "kind": "robots", "limit": 500, **args})
    assert reply["ok"], reply
    return {row["name"]: row for row in reply["result"]["items"]}


def test_list_robots_gives_the_shipped_robots_with_their_files():
    rows = listed()
    assert set(SHIPPED) <= set(rows) and not [n for n in rows if n.startswith("task_")], sorted(rows)
    for name in SHIPPED:
        assert rows[name]["title"] and Path(rows[name]["path"]).is_file(), rows[name]


@pytest.mark.parametrize("path", ["emb_e2_panda", "robots/emb_e2_panda/scene.xml"])
def test_a_recipe_robot_takes_a_library_name_or_path(path):
    reply = respond(Session(), {"op": "scene", "scene": {"objects": [{"add": "robot", "name": "arm", "path": path}]}})
    assert reply["ok"] and reply["result"]["names"] == ["arm"], reply


def test_a_library_folder_adds_its_own_robots(tmp_path):
    (tmp_path / "robots" / "my_arm").mkdir(parents=True)
    (tmp_path / "robots" / "my_arm" / "arm.xml").write_text(ARM)
    assert listed(library=[str(tmp_path)])["my_arm"]["path"].endswith("arm.xml")
    reply = respond(Session(), {"op": "scene", "library": [str(tmp_path)],
                                "scene": {"objects": [{"add": "robot", "name": "arm", "path": "my_arm"}]}})
    assert reply["ok"], reply


def test_an_unknown_robot_says_where_to_find_names():
    reply = respond(Session(), {"op": "scene", "scene": {"objects": [{"add": "robot", "name": "arm",
                                                                      "path": "emb_e0_nope"}]}})
    assert not reply["ok"] and reply["error"]["field"] == "objects[0].path", reply
    assert "alloy list robots" in reply["error"]["fix"]
