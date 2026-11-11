# SPDX-License-Identifier: Apache-2.0
"""Watch a simulation in a viewer: a block and a ball drop onto a table, then rest.

    pip install "viser>=1.1.1"
    python examples/viewer/01_view_scene.py                   # prints a URL: open it in a browser (remote: forward the port)
    python examples/viewer/01_view_scene.py --backend null --seconds 0.2   # no window, no extra package (CI)

``alloy.viewer.launch`` reads the simulation and draws it; the results are the same with the viewer on or off.
``view.run()`` steps one control period per frame until you close the viewer (or ``--seconds`` of simulated time).
"""

from __future__ import annotations

import argparse

import alloy
import alloy.viewer


def build_scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)         # top at z = 0.4
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.6), mass=0.2)     # full edges (m), kg
    scene.add_sphere("ball", 0.03, pos=(0.12, 0.0, 0.9), mass=0.1, material="steel_carbon")
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="viser", help="viser | gl | rerun | usd | null")
    parser.add_argument("--seconds", type=float, default=None, help="simulated seconds to run (default: until closed)")
    parser.add_argument("--port", type=int, default=8080, help="viser: port of the browser viewer")
    parser.add_argument("--device", default=None, help="cpu or cuda:0 (default: GPU if present)")
    args = parser.parse_args()

    options = {"port": args.port} if args.backend == "viser" else {}
    sim = alloy.build(build_scene(), device=args.device)
    view = alloy.viewer.launch(sim, backend=args.backend, **options)
    print("open:", view.url)                       # viser: http://localhost:<port>; other backends: None
    view.run(seconds=args.seconds)                 # blocking; Ctrl-C or closing the viewer ends it
    print(f"stopped at t={sim.t:.2f} s, block z={sim.state(world=0)['block.pos'][2]:.3f} m, ok={sim.check()['ok']}")
    view.close()


if __name__ == "__main__":
    main()
