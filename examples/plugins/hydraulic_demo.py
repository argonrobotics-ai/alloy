# SPDX-License-Identifier: Apache-2.0
"""A hydraulic cylinder lifts a 0.5 kg plate: pump on for 0.6 s, then off (the fluid holds the load).

Run: python examples/plugins/hydraulic_demo.py      (CPU, ~15 s; prints pressure and rod extension)
"""

import warp as wp

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.harness import TestScene, build_simulation


def lift(b: newton.ModelBuilder) -> None:
    ap.create("hydraulic_cylinder", bore=0.02, stroke=0.1).build(b, wp.transform_identity(), "lift")


def main() -> None:
    hyd = ap.create("hydraulic", barrel="lift_barrel", rod="lift_rod", bore=0.02)
    sim = build_simulation(TestScene(build=lift, worlds=1), [hyd], device="cpu")
    for k in range(50):
        hyd.set_input("pump_flow", 1.0e-5 if k < 30 else 0.0)          # 10 mL/s, then off
        sim.step_period()
        r = hyd.read()
        if k % 5 == 4:
            print(f"t={sim.clock.t:.2f} s  pump={'on ' if k < 30 else 'off'}  P={r['pressure'][0] / 1e3:7.2f} kPa  "
                  f"x={r['extension'][0] * 1e3:6.2f} mm  fluid={r['volume'][0] * 1e6:6.3f} mL")


if __name__ == "__main__":
    main()
