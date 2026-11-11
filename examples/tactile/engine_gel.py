# SPDX-License-Identifier: Apache-2.0
"""The physics side's gel sensor (engine penetration + marker fields) rendered as GelSight images.

A 5 mm, 50 g ball rests on a gel pad (the GelSensor plug-in's own test scene); after each control period the plug-in
writes the gel depth field and marker displacements, and EngineTactile turns them into images for every world.
Run:  python examples/tactile/engine_gel.py --out tactile_engine [--worlds 4]
Prints one JSON line (peak gel depth [mm], image shape, PNG path). Works on CPU; a GPU is faster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alloy.camera import write_png
from alloy.camera.tactile import Augment
from alloy.camera.tactile.engine import EngineTactile
from alloy.plugins.harness.scene import build_simulation
from alloy.plugins.tactile import GelSensor
from alloy.plugins.tactile.gel_laws import press_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tactile_engine")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    scene = press_scene()
    gel = GelSensor(GelSensor.Config(bodies=("pad",), markers=(7, 9)))  # gel on the pad's top face, 7 x 9 markers
    sim = build_simulation(scene, [gel], args.device, worlds=args.worlds)
    view = EngineTactile.from_gel_sensor(gel, width=320, augment=Augment.strong(seed=0))
    frames = []
    for period in range(scene.periods):
        sim.step_period()
        if period % 8 == 0 or period == scene.periods - 1:
            frames.append(view.render(gel.outputs()).numpy()[0].copy())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = write_png(out / "engine_gel.png", np.concatenate(frames, axis=1))
    depth = gel.outputs()["depth"].numpy()
    print(json.dumps({"peak_depth_mm": round(float(depth.max() * 1e3), 3), "images": list(view.renderer.rgb.shape),
                      "png": str(png)}))


if __name__ == "__main__":
    main()
