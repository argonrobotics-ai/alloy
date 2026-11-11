# SPDX-License-Identifier: Apache-2.0
"""Two NdFeB-like magnet cubes on a slippery table attract and snap together.

Run: python examples/plugins/magnet_demo.py        (CPU, ~15 s; prints the gap and the magnetic force)

MuJoCo contacts are soft: at rest the ~4.6 N pull presses the cubes about 1 mm into each other (negative gap).
"""

from alloy._vendor import newton
import numpy as np

import alloy.plugins as ap
from alloy.plugins.harness import TestScene, body_xform, build_simulation


def table(b: newton.ModelBuilder) -> None:
    cfg = newton.ModelBuilder.ShapeConfig(density=7500.0, mu=0.1)       # 2 cm cubes, 60 g
    for label, x in (("a", 0.0), ("b", 0.05)):
        body = b.add_body(xform=body_xform(x, 0.0, 0.0101), label=label)
        b.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01, cfg=cfg)


def main() -> None:
    magnets = ap.create("magnet", bodies=("a", "b"), moments=(1.0, 1.0), axes=((1, 0, 0), (1, 0, 0)))
    sim = build_simulation(TestScene(build=table, worlds=1, ground_mu=0.1), [magnets], device="cpu")
    for _ in range(20):
        sim.step_period()
        q = sim.state.body_q.numpy()
        f = sim.plugins.forces.body_f.numpy()
        gap = float(np.linalg.norm(q[1, :3] - q[0, :3])) - 0.02
        print(f"t={sim.clock.t:.2f} s  gap={gap * 1e3:6.2f} mm  pull on b={-f[1, 0]:6.3f} N")


if __name__ == "__main__":
    main()
