# SPDX-License-Identifier: Apache-2.0
"""Camera quickstart: an Alloy scene in 4 parallel worlds, one snapshot, then a camera batch every physics step.

Run:  python examples/camera/quickstart.py --out camera_quickstart
Prints one JSON line (images/s, image shape, PNG paths). Works on CPU too (slowly); a GPU is much faster.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from alloy.api import Scene, build as build_sim
from alloy.camera import Camera, write_png


def build(worlds: int):
    """A ground, a cube and a ball in every world."""
    scene = Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.16, 0.12, 0.08), pos=(0.0, 0.0, 0.3), rpy=(0.3, 0.2, 0.1), mass=0.5)
    scene.add_sphere("ball", 0.05, pos=(0.15, 0.1, 0.5), mass=0.2)
    return build_sim(scene)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="camera_quickstart")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    sim = build(args.worlds)
    sim.render(width=320, height=240, spp=32, path=out / "snapshot.png")  # one framed image of world 0
    cam = Camera("front", look_at=((0.7, -0.7, 0.5), (0.0, 0.0, 0.1)), width=128, height=128, fov=45, spp=16)
    t_render, n_img = 0.0, 0
    for _ in range(args.steps):
        sim.step()
        t0 = time.perf_counter()
        batch = cam.render(sim)  # every world at once; cam.render_torch(sim) gives a torch tensor for training
        frames = batch.numpy()  # copies to the host (and waits for the GPU)
        t_render += time.perf_counter() - t0
        n_img += frames.shape[0]
    paths = [str(write_png(out / f"world{w}.png", frames[w])) for w in range(min(2, len(frames)))]
    print(json.dumps({"images_per_s": round(n_img / t_render, 1), "shape": list(frames.shape),
                      "pngs": [str(out / "snapshot.png")] + paths}))


if __name__ == "__main__":
    main()
