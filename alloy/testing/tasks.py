# SPDX-License-Identifier: Apache-2.0
"""Test helpers for the task-scene package (WP13): repo paths and a small MuJoCo model using every part the
converter writes."""

from __future__ import annotations

from pathlib import Path
from alloy.cards.library import default_root

REPO = Path(__file__).resolve().parents[2]
ROBOTS = default_root() / "robots"
TRACES = REPO / "bench" / "traces"
TRACE_NAMES = ("t12", "t12pre", "t14", "t11", "h1")

# Wedge start orientations. Tilted, it lands on a hull vertex, where MuJoCo-Warp and CPU MuJoCo mesh-box contacts
# differ: the Newton replay deviates by 24 mm at 0.125 s (0.5 mm level, 0.18 mm as a box; 2026-11-01). Level, it lands
# face down, which makes CPU replays sensitive to float32 vertex rounding (8e-5 m vs 1e-8 tilted).
WEDGE_TILTED = "0.98 0.1 0.1 0.1"
WEDGE_LEVEL = "1 0 0 0"

SYNTH_XML_TEMPLATE = """
<mujoco model="synth">
  <option timestep="0.0008333333333" integrator="implicitfast" cone="elliptic" impratio="100" noslip_iterations="5"/>
  <asset>
    <mesh name="wedge" vertex="0 0 0  0.05 0 0  0 0.03 0  0 0 0.02  0.05 0.03 0.02  0.013 0.007 0.011"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01" friction="0.8 0.01 0.001"/>
    <body name="ledge" pos="0.3 0 0.05"><geom name="ledge_box" type="box" size="0.1 0.1 0.05" margin="0.0001"/></body>
    <body name="arm" pos="0 0 0.3" quat="0.9238795 0 0 0.3826834">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.5" armature="0.01"
             frictionloss="0.02" actuatorfrcrange="-5 5"/>
      <geom name="upper" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="finger_a" pos="0.2 0 0">
        <joint name="fa" type="slide" axis="0 0 1" range="0 0.03"/>
        <geom name="pad_a" type="box" size="0.01 0.01 0.01" mass="0.05"/>
      </body>
      <body name="finger_b" pos="0.2 0.03 0">
        <joint name="fb" type="slide" axis="0 0 1" range="-0.03 0"/>
        <geom name="pad_b" type="box" size="0.01 0.01 0.01" mass="0.05"/>
      </body>
    </body>
    <body name="block" pos="0.3 0.02 0.13">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box" size="0.0375 0.0125 0.0075" density="700" friction="0.4 0.005 0.0001"/>
    </body>
    <body name="wedge_obj" pos="0.25 -0.05 0.12" quat="{wedge_quat}">
      <freejoint/>
      <geom type="mesh" mesh="wedge" density="1200"/>
    </body>
  </worldbody>
  <actuator>
    <position name="m_shoulder" joint="shoulder" kp="60" kv="5"/>
    <motor name="m_fa" joint="fa" ctrlrange="-10 10" gear="2"/>
  </actuator>
  <equality><joint name="couple" joint1="fb" joint2="fa" polycoef="0 -1 0 0 0"/></equality>
  <contact>
    <pair geom1="block_geom" geom2="ledge_box" condim="4" friction="0.5 0.5 0.01 0.0001 0.0001" margin="0.0002"/>
    <exclude body1="arm" body2="finger_a"/>
  </contact>
</mujoco>
"""


def synth_xml(wedge_quat: str = WEDGE_TILTED) -> str:
    return SYNTH_XML_TEMPLATE.replace("{wedge_quat}", wedge_quat)
