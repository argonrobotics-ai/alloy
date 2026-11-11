# SPDX-License-Identifier: Apache-2.0
"""Tactile image throughput: images/s for N worlds, each a gel pad touched by a ball and a cube (JSON out).

    python -m alloy.camera.tactile.bench --worlds 1024 --sizes 160,320,640 --preset gelsight --out tactile_speed.json

Every frame moves the touching objects (a fresh press in every world) and renders the whole pipeline: the press ray
cast, gel spreading, colour table, stick-slip, markers, augmentation and uint8 output. The device is synchronised
around each timed frame; the first frames (kernel compiles) are not timed. Prints one JSON line per size.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

PAD_HALF_MM = (12.0, 10.0, 2.0)


def bench_model(num_worlds: int, device: str, seed: int = 0):
    """N worlds of: a pad (gel on its +Z face at z = 2 mm), a 4 mm ball and a 3 mm cube near the gel.

    Example: ``model, base_q = bench_model(4, "cpu")``; body order per world is pad, ball, cube.
    """
    from alloy._vendor import newton
    w = newton.ModelBuilder()
    pad = w.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), label="pad")
    w.add_shape_box(pad, hx=PAD_HALF_MM[0] * 1e-3, hy=PAD_HALF_MM[1] * 1e-3, hz=PAD_HALF_MM[2] * 1e-3)
    ball = w.add_body(xform=wp.transform((-0.004, 0.0, 0.0035), wp.quat_identity()), label="ball")
    w.add_shape_sphere(ball, radius=0.002)
    cube = w.add_body(xform=wp.transform((0.004, 0.0, 0.0035), wp.quat_identity()), label="cube")
    w.add_shape_box(cube, hx=0.0015, hy=0.0015, hz=0.0015)
    b = newton.ModelBuilder()
    b.replicate(w, copy_count=num_worlds)
    model = b.finalize(device=device)
    return model, model.state().body_q.numpy().copy()


def frame_poses(base_q: np.ndarray, frame: int, seed: int = 0) -> np.ndarray:
    """Body poses of one frame: ball and cube pressed 0.1-1.0 mm at per-world, per-frame offsets."""
    rng = np.random.default_rng(seed * 100003 + frame)
    q = base_q.copy().reshape(-1, 3, 7)
    n = q.shape[0]
    q[:, 1, 0] = -0.004 + rng.uniform(-2e-3, 2e-3, n)
    q[:, 1, 1] = rng.uniform(-3e-3, 3e-3, n)
    q[:, 1, 2] = 0.002 + 0.002 - rng.uniform(0.1e-3, 1.0e-3, n)
    q[:, 2, 0] = 0.004 + rng.uniform(-2e-3, 2e-3, n)
    q[:, 2, 1] = rng.uniform(-3e-3, 3e-3, n)
    q[:, 2, 2] = 0.002 + 0.0015 - rng.uniform(0.1e-3, 0.6e-3, n)
    return q.reshape(-1, 7)


def run(num_worlds: int, width: int, preset: str, frames: int, device: str, augment: bool) -> dict:
    """Time ``frames`` renders of every world at ``width``; returns a JSON-ready row."""
    from .augment import Augment
    from .sensor import TactileSensor
    model, base_q = bench_model(num_worlds, device)
    state = model.state()
    sensor = TactileSensor("bench", attach_to="pad", pose=((0, 0, 0.002), (0, 0, 0, 1)), preset=preset, width=width,
                           markers=True, augment=Augment.strong(0) if augment else None)
    poses = [wp.array(frame_poses(base_q, f), dtype=wp.transform, device=device) for f in range(4)]
    for f in range(3):  # compile + warm up
        wp.copy(state.body_q, poses[f % 4])
        sensor.render(model, state)
    wp.synchronize_device(device)
    t = 0.0
    for f in range(frames):
        wp.copy(state.body_q, poses[f % 4])
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        img = sensor.render(model, state)
        wp.synchronize_device(device)
        t += time.perf_counter() - t0
    h, w = sensor.size
    touched = float((img.numpy().reshape(num_worlds, -1).std(axis=1) > 0).mean())
    return {"preset": preset, "worlds": num_worlds, "width": w, "height": h, "frames": frames,
            "ms_per_frame": 1e3 * t / frames, "images_per_s": num_worlds * frames / t, "augment": augment,
            "device": wp.get_device(device).name, "frac_worlds_rendered": touched}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="tactile image throughput (JSON lines)")
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--sizes", default="160,320,640", help="image widths, comma separated")
    ap.add_argument("--preset", default="gelsight")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--augment", action="store_true", help="render with Augment.strong()")
    ap.add_argument("--out", default=None, help="append the rows to this JSON-lines file")
    args = ap.parse_args(argv)
    dev = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    for width in (int(s) for s in args.sizes.split(",")):
        row = run(args.worlds, width, args.preset, args.frames, dev, args.augment)
        print(json.dumps(row), flush=True)
        if args.out:
            with Path(args.out).open("a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
