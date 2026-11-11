# SPDX-License-Identifier: Apache-2.0
"""OSS-19 robots in composed scenes: a robot file stands at the table's near edge facing the objects, nothing is placed
within its clear radius, everything lands within its reach, the scene card carries the robot, and the full check
runs with it (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.cards.library import CardLibrary, default_root
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog
from alloy.compose.robots import place_robots, reach_area
from alloy.compose.spec import ComposeError, RobotSpec, SceneSpec

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
  <actuator>
    <position joint="shoulder" kp="200" kv="20"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


def test_robots_stand_at_the_table_ends_and_their_reach_bounds_the_area():
    lo, hi = np.array([-0.58, -0.38]), np.array([0.58, 0.38])
    a, b = place_robots((RobotSpec("a.xml", name="a"), RobotSpec("b.xml", name="b")), lo, hi)
    assert a.pose == pytest.approx((-0.43, 0.0, 0.0, 0.0)) and b.pose == pytest.approx((0.43, 0.0, 0.0, math.pi))
    rlo, rhi = reach_area([a], lo, hi)
    assert rlo[0] == pytest.approx(-0.43 + 0.2) and rhi[0] == pytest.approx(-0.43 + 0.8)
    with pytest.raises(ComposeError):
        SceneSpec.from_dict({"items": ["cube"], "robots": ["a.xml", "b.xml"]})       # both named "robot"


def test_objects_keep_clear_of_the_robot_and_within_reach(cat, tmp_path):
    arm = tmp_path / "arm.xml"
    arm.write_text(ARM, encoding="utf-8")
    robot = {"path": str(arm), "name": "arm", "reach": 0.6}
    r = compose("four cubes on a table", robots=[robot], check="load", device="cpu", catalog=cat)
    assert r.ok, r.summary()
    card = r.scenes[0].card
    assert [rb.name for rb in card.robots] == ["arm"] and card.robots[0].base_pose[:3] == pytest.approx(
        (-0.43, 0.0, 0.0))
    for o in r.scenes[0].objects:
        d = math.hypot(o["pose"][0] + 0.43, o["pose"][1])
        assert 0.15 < d < 0.6 + 0.05, o
    assert r.to_dict()["spec"]["robots"][0]["name"] == "arm"


def test_full_check_runs_with_the_robot(cat, tmp_path):
    arm = tmp_path / "arm.xml"
    arm.write_text(ARM, encoding="utf-8")
    r = compose("two cubes on a tidy table", robots=[str(arm)], check="full", device="cpu", worlds=1,
                seconds=0.2, catalog=cat)
    assert r.scenes[0].report is not None and r.ok, r.summary()


def test_area_limits_where_objects_go(cat):
    r = compose("three cubes on a table", area=(0.1, 0.1, 0.4, 0.3), check="none", catalog=cat)
    for o in r.scenes[0].objects:
        assert 0.1 - 0.05 < o["pose"][0] < 0.4 + 0.05 and 0.1 - 0.05 < o["pose"][1] < 0.3 + 0.05, o
    with pytest.raises(ComposeError):
        compose("a cube", area=(0.4, 0.1, 0.1, 0.3), check="none", catalog=cat)
