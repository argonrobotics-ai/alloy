# SPDX-License-Identifier: Apache-2.0
"""A start overlap names a robot's parts by the name the robot was added with (CPU).

Health named shapes by the model's labels, which start with the robot file's model name: two robots from one file
(two shipped whole-scene files, each with its floor) read alike, e.g. "rig/worldbody/floor and
rig/worldbody/crate/crate_box", which looks like one robot overlapping itself and names neither robot as added.
Now a robot's shape reads "<robot>/<body>" as ``sim.names()`` names its bodies, or "<robot>/<shape>" for a shape
fixed to the world, and the penetration problem the overlap causes points at it.
"""

from __future__ import annotations

import pytest

import alloy.api as alloy

# a whole-scene file: its own floor plane and a free crate standing on it
RIG = """<mujoco model="rig"><worldbody>
  <geom name="floor" type="plane" size="0 0 0.05"/>
  <body name="crate" pos="0 0 0.05"><freejoint/><geom name="crate_box" type="box" size="0.05 0.05 0.05" mass="1"/>
  </body></worldbody></mujoco>"""


@pytest.mark.compile_heavy
def test_two_robots_from_one_file_are_told_apart_in_a_start_overlap():
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("rig_a", mjcf=RIG, pos=(0.0, 0.0, 0.0))
    scene.add_robot("rig_b", mjcf=RIG, pos=(1.0, 0.0, 0.08))      # its floor 8 cm up: through rig_a's crate
    sim = alloy.build(scene, device="cpu")
    sim.step(1)
    s = sim.health()["worlds"][0]["start_overlap"]
    assert {s["a"], s["b"]} == {"rig_b/floor", "rig_a/crate"} and s["level"] == "fail"
    assert s["depth_mm"] == pytest.approx(80.0, abs=5.0)
    problems = sim.check()["problems"]
    start = [p for p in problems if p.get("check") == "start_overlap"]
    assert len(start) == 1 and "rig_b/floor" in start[0]["what"] and "rig_a/crate" in start[0]["what"]
    assert "rig_a/crate" in sim.names()["bodies"]                  # the part reads as sim.names() lists it
    pen = [p for p in problems if p.get("check") == "penetration"]  # "between rig_a and rig_b": the overlap's
    assert pen and all(p["fix"].startswith("caused by the start overlap") for p in pen)
