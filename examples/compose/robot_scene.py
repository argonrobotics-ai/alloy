# SPDX-License-Identifier: Apache-2.0
"""A robot at the table and objects within its reach: compose a scene around your own robot file.

The arm stands at the middle of the table's near edge facing the objects; nothing is placed within ``clear`` of its
base and everything lands within ``reach``. The load checks after each placement include the arm's links, so an
object in its way is moved. A second robot would stand at the far edge (two arms across a table); a ``pose``
(x, y, z, yaw) puts a robot anywhere.

It searches the objects shipped with Alloy (``Catalog(imports=False)``), so it builds the same scenes on every
machine with nothing to download; ``--all-libraries`` also searches imported and fetched libraries (scanned mugs,
RoboCasa and Objaverse objects), as ``alloy.compose.compose`` does by default.

Run:

    python examples/compose/robot_scene.py
    python examples/compose/robot_scene.py --all-libraries
    python -m alloy.compose "three cubes and a mug on a table" --robot my_arm.xml --out my_scenes   # every library
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import alloy.compose
from alloy.compose.catalog import Catalog

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
  <actuator><position joint="shoulder" kp="200" kv="20"/></actuator>
</mujoco>
"""


def main() -> int:
    arm = Path(tempfile.mkdtemp()) / "arm.xml"              # your robot's MJCF (or URDF / USD with kind=...)
    arm.write_text(ARM, encoding="utf-8")
    robot = {"path": str(arm), "name": "arm", "clear": 0.15, "reach": 0.6}
    catalog = Catalog() if "--all-libraries" in sys.argv else Catalog(imports=False)
    result = alloy.compose.compose("three cubes and a mug on a table", robots=[robot], variations=1, catalog=catalog)
    print(result.summary())
    for o in result.scenes[0].objects:
        print(f"  {o['name']:20s} at x {o['pose'][0]:+.3f} y {o['pose'][1]:+.3f}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
