# SPDX-License-Identifier: Apache-2.0
"""Depth, segmentation, instance, class and optical-flow images: 4 worlds seen from above, saved as one PNG
(RGB | depth | segmentation | instances | flow).

Run:  python examples/camera/depth_segmentation.py            (add --device cpu to force the CPU)
Prints the depth range, the name of every segmentation, instance and class id seen, and the flow of the block after
it is moved 2 cm. Writes depth_segmentation.png (world 0).
"""
from __future__ import annotations

import argparse

import numpy as np

import alloy
from alloy.camera import Camera, write_png
from alloy.camera.aov import instance_labels, segment_labels, set_semantics

PALETTE = np.array([(0, 0, 0), (110, 110, 110), (230, 80, 60), (60, 160, 230), (240, 200, 60), (90, 200, 110)], np.uint8)


def build(worlds: int, device: str | None):
    """A ground, a static table, and a free block and ball on it (the ball's x differs per world)."""
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_box("block", size=(0.1, 0.1, 0.05), pos=(-0.1, 0, 0.43), mass=0.2)
    scene.add_sphere("ball", 0.04, pos=(0.12, 0, 0.46), mass=0.1)
    scene.randomize("ball", "x", 0.08, 0.16)
    return alloy.build(scene, device=device, seed=0)


def depth_to_grey(depth: np.ndarray) -> np.ndarray:
    """Near = bright, far = dark, no hit (inf) = black; (H, W) uint8."""
    hit = np.isfinite(depth)
    grey = np.zeros(depth.shape, np.uint8)
    if hit.any():
        lo, hi = depth[hit].min(), depth[hit].max()
        grey[hit] = (255 - 200 * (depth[hit] - lo) / max(hi - lo, 1e-6)).astype(np.uint8)
    return grey


def flow_to_rgb(flow: np.ndarray, full: float = 3.0) -> np.ndarray:
    """Right = red, down = green, at ``full`` pixels or more; still = grey, not tracked (NaN) = black; uint8."""
    valid = np.isfinite(flow[..., 0])
    rgb = np.zeros(flow.shape[:2] + (3,), np.uint8)
    scaled = np.clip(np.nan_to_num(flow) / full, -1.0, 1.0)
    rgb[..., 0] = 128 + 127 * scaled[..., 0]
    rgb[..., 1] = 128 + 127 * scaled[..., 1]
    rgb[..., 2] = 128
    rgb[~valid] = 0
    return rgb


def ids_and_flow(sim, cam) -> tuple[np.ndarray, np.ndarray]:
    """Print the instance and class names seen, move the block 2 cm along +x, and return world 0's instance
    image and the flow image of that move."""
    shapes = instance_labels(sim)
    classes = set_semantics(sim, {"block": "toy", "ball": "toy", "table": "furniture", "ground": "floor"})
    out = sim.render(cam, output=("instance_id", "semantic", "flow"))  # the first flow render reads 0
    print("instances:", {int(i): shapes[i] for i in np.unique(out["instance_id"])})
    print("classes:", {int(i): classes[i] for i in np.unique(out["semantic"])})
    pos = sim.state()["block.pos"]  # (worlds, 3)
    sim.set_pose("block", pos=pos + np.array([0.02, 0.0, 0.0]))
    out = sim.render(cam, output=("instance_id", "flow"))
    block = out["instance_id"] == shapes.index("block")
    dx, dy = np.median(out["flow"][block], axis=0)
    print(f"flow of the block after a 2 cm move: {dx:.2f} px right, {dy:.2f} px down")
    return out["instance_id"][0], out["flow"][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cpu or cuda:0 (default: the GPU if there is one)")
    ap.add_argument("--worlds", type=int, default=4)
    args = ap.parse_args()
    sim = build(args.worlds, args.device)
    sim.step(30)  # let the block and ball settle on the table
    cam = Camera("top", look_at=((0, 0, 1.4), (0, 0, 0), (0, 1, 0)), width=128, height=128, fov=40, spp=8)
    out = sim.render(cam, output=("rgb", "depth", "segmentation"))
    rgb, depth, seg = out["rgb"], out["depth"], out["segmentation"]
    print("shapes:", rgb.shape, depth.shape, seg.shape)
    print(f"depth range: {depth[np.isfinite(depth)].min():.3f} .. {depth[np.isfinite(depth)].max():.3f} m")
    names = segment_labels(sim)
    for i in np.unique(seg):
        print(f"id {i}: {names[i]} ({(seg == i).sum()} pixels over all worlds)")
    inst, flow = ids_and_flow(sim, cam)
    panels = [rgb[0], np.repeat(depth_to_grey(depth[0])[..., None], 3, axis=2), PALETTE[seg[0] % len(PALETTE)],
              PALETTE[inst % len(PALETTE)], flow_to_rgb(flow)]
    print("wrote", write_png("depth_segmentation.png", np.concatenate(panels, axis=1)))


if __name__ == "__main__":
    main()
