# SPDX-License-Identifier: Apache-2.0
"""A pen block slides across a sheet of paper and leaves an ink track; saves the ink map as a PGM image.

Run: python examples/plugins/ink_demo.py [out.pgm]     (CPU, ~15 s)
"""

import sys

from alloy._vendor import newton
import numpy as np

import alloy.plugins as ap
from alloy.plugins.harness import TestScene, body_xform, build_simulation


def desk(b: newton.ModelBuilder) -> None:
    paper = newton.ModelBuilder.ShapeConfig(mu=0.2)
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.001), hx=0.1, hy=0.1, hz=0.001, cfg=paper, label="paper")
    pen = b.add_body(xform=body_xform(-0.05, 0.0, 0.012), label="pen")
    b.add_shape_box(pen, hx=0.01, hy=0.01, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.2))


def push(sim) -> None:
    qd = sim.state.joint_qd.numpy()
    qd[0], qd[1] = 0.4, 0.15                                   # slide diagonally (vx, vy) [m/s]
    sim.state.joint_qd.assign(qd)
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)


def main(out: str) -> None:
    ink = ap.create("ink", pen="pen", tip=(0.0, 0.0, -0.01), paper="paper", cells=(200, 200), brush_radius=2e-3)
    sim = build_simulation(TestScene(build=desk, worlds=1, init=push), [ink], device="cpu")
    for _ in range(25):
        sim.step_period()
    r = sim.plugins.read()["ink"]
    grid = r["ink"][0]
    print(f"ink laid {r['total'][0]:.3f} µL, left in pen {r['reservoir'][0]:.3f} µL, inked cells {(grid > 0).sum()}")
    img = (255 - 255 * grid / max(grid.max(), 1e-12)).astype(np.uint8).T[::-1]      # y up
    with open(out, "wb") as fh:
        fh.write(f"P5 {img.shape[1]} {img.shape[0]} 255\n".encode() + img.tobytes())
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ink_map.pgm")
