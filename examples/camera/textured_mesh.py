# SPDX-License-Identifier: Apache-2.0
"""Textured meshes without a pack: a checker-textured cube on a ground, rendered to a PNG with ``mesh_scene``.

Run:  python examples/camera/textured_mesh.py --out textured_mesh.png
Works on CPU too (small image, a minute the first time); a GPU is much faster. Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from alloy.camera import Camera, TexturedMesh, default_device, mesh_scene, write_png

FACES = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))  # corners of the cube


def checker(cells: int = 4, block: int = 16) -> np.ndarray:
    """A black / orange checker, (cells * block)^2 pixels, row 0 = top."""
    on = (np.add.outer(np.arange(cells), np.arange(cells)) % 2).astype(bool)
    img = np.where(on[..., None], np.array([0.95, 0.45, 0.1]), np.array([0.08, 0.08, 0.1]))
    return np.kron(img, np.ones((block, block, 1))).astype(np.float32)


def cube(half: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points (24, 3), triangles (12, 3), per-vertex uv (24, 2)): every face shows the whole texture."""
    corners = np.array([[x, y, z] for z in (-half, half) for y in (-half, half) for x in (-half, half)], np.float32)
    pts = np.concatenate([corners[list(f)] for f in FACES])
    tris = np.concatenate([[[4 * k, 4 * k + 1, 4 * k + 2], [4 * k, 4 * k + 2, 4 * k + 3]] for k in range(6)])
    return pts, tris.astype(np.int32), np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (6, 1)).astype(np.float32)


def build() -> list[TexturedMesh]:
    """A textured cube resting on a grey ground quad."""
    pts, tris, uv = cube()
    ground = np.array([[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]], np.float32)
    return [TexturedMesh("cube", pts, tris, uv=uv, texture=checker(), roughness=0.4, pose=(0, 0, 0.5, 0, 0, 0.26, 0.97)),
            TexturedMesh("ground", ground, np.array([[0, 1, 2], [0, 2, 3]]), color=(0.55, 0.55, 0.6), roughness=0.8)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="textured_mesh.png")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--spp", type=int, default=32)
    args = ap.parse_args()
    scene = mesh_scene(build(), device=default_device())
    cam = Camera("view", look_at=((2.2, -2.6, 1.8), (0.0, 0.0, 0.5)), width=args.size, height=args.size, fov=40,
                 spp=args.spp)
    t0 = time.perf_counter()
    img = cam.render(scene).numpy()[0]
    path = write_png(Path(args.out), img)
    print(json.dumps({"png": str(path), "shape": list(img.shape), "seconds": round(time.perf_counter() - t0, 2)}))


if __name__ == "__main__":
    main()
