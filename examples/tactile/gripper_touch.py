# SPDX-License-Identifier: Apache-2.0
"""A two-finger gripper squeezes a thin rod; a GelSight sensor on the left finger shows the touch, in every world.

The left finger's inner face carries the gel: the sensor pose is in the finger's frame, +Z out of the gel (toward the
rod). The pads use a soft contact (solref) so the rod sinks into the gel about as far as a real gel lets it.
Run:  python examples/tactile/gripper_touch.py --out tactile_gripper [--worlds 4] [--preset digit]
Prints one JSON line (peak press per world [mm], images/s, PNG path). Works on CPU (slowly); a GPU is much faster.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import alloy
from alloy.camera import write_png
from alloy.camera.tactile import Augment, TactileSensor

GRIPPER = """<mujoco>
  <worldbody>
    <body name="palm">
      <geom type="box" size="0.04 0.06 0.01" mass="0.3"/>
      <body name="finger_left" pos="0 0.035 -0.04">
        <joint name="close_left" type="slide" axis="0 -1 0" range="0 0.03" damping="5"/>
        <geom type="box" size="0.012 0.005 0.01" mass="0.03" friction="1.0" solref="0.03 1"/>
      </body>
      <body name="finger_right" pos="0 -0.035 -0.04">
        <joint name="close_right" type="slide" axis="0 1 0" range="0 0.03" damping="5"/>
        <geom type="box" size="0.012 0.005 0.01" mass="0.03" friction="1.0" solref="0.03 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="close_left" kp="400" kv="10"/>
    <position joint="close_right" kp="400" kv="10"/>
  </actuator>
</mujoco>"""
GEL_ON_LEFT_FINGER = ((0.0, -0.005, 0.0), (0.7071068, 0.0, 0.0, 0.7071068))  # inner face, +Z -> finger -Y


def build(worlds: int):
    """Gripper (palm welded to the world) around a 5 mm-radius rod standing on the ground, in every world."""
    scene = alloy.Scene(worlds=worlds, dt=0.005)
    scene.add_ground()
    scene.add_cylinder("rod", 0.005, 0.04, pos=(0.0, 0.0, 0.02), mass=0.02, friction=1.0)
    scene.add_robot("gripper", mjcf=GRIPPER, pos=(0.0, 0.0, 0.065))  # no base joint: welded to the world
    return alloy.build(scene)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tactile_gripper")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--preset", default="gelsight", choices=["gelsight", "digit"])
    args = ap.parse_args()
    sim = build(args.worlds)
    sensor = TactileSensor("left_gel", attach_to="gripper/finger_left", pose=GEL_ON_LEFT_FINGER, preset=args.preset,
                           width=320, augment=Augment.strong(seed=0))
    frames = []
    sim.run(0.3)  # settle
    for close in (0.01, 0.02, 0.03):
        sim.run(0.4, targets={"close_left": close, "close_right": close})
        t0 = time.perf_counter()
        out = sensor.render(sim, output="all")
        rate = args.worlds / (time.perf_counter() - t0)
        frames.append(out.rgb.numpy()[0].copy())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = write_png(out_dir / "left_gel.png", np.concatenate(frames, axis=1))
    print(json.dumps({"peak_press_mm": [round(float(v), 3) for v in out.press.numpy().max(axis=(1, 2))],
                      "images_per_s_last": round(rate, 1), "png": str(png), "preset": args.preset}))


if __name__ == "__main__":
    main()
