# SPDX-License-Identifier: Apache-2.0
"""Drag a box with the mouse: grab a body in the viewer and pull it with a spring; let go and it falls.

    pip install "viser>=1.1.1"
    python examples/viewer/04_drag.py --port 8803        # open the URL; tick "Enable Picking", click a box, drag the handle
    python examples/viewer/04_drag.py --backend gl       # desktop window: hold the right mouse button on a box and move
    python examples/viewer/04_drag.py --backend null --seconds 0.5   # no window: grabs a box by program (CI)

The pull is ``sim.apply_wrench`` (a spring of 2 Hz, capped at 50 weights), set only while you hold a body.
"""

from __future__ import annotations

import argparse

import alloy
import alloy.viewer


def build_scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)         # top at z = 0.4
    for i, x in enumerate((-0.15, 0.0, 0.15)):
        scene.add_box(f"box{i}", size=(0.06, 0.06, 0.06), pos=(x, 0.0, 0.43), mass=0.2)
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="viser", help="viser | gl | null")
    parser.add_argument("--seconds", type=float, default=None, help="simulated seconds to run (default: until closed)")
    parser.add_argument("--port", type=int, default=8080, help="viser: port of the browser viewer")
    parser.add_argument("--device", default=None, help="cpu or cuda:0 (default: GPU if present)")
    args = parser.parse_args()

    sim = alloy.build(build_scene(), device=args.device)
    options = {"port": args.port} if args.backend == "viser" else {}
    with alloy.viewer.launch(sim, backend=args.backend, drag=True, **options) as view:
        print("open:", view.url)
        print('how: viser: tick "Enable Picking", click a box, drag the handle; gl: hold the right mouse button on a box')
        drag = alloy.viewer.MouseDrag.of(view)
        if args.backend == "null":
            drag.grab(view, "box1", world=0, target_xyz=(0.0, 0.0, 0.8))            # what a mouse drag does
        z0 = float(sim.state(world=0)["box1.pos"][2])
        view.run(seconds=args.seconds)
        z = float(sim.state(world=0)["box1.pos"][2])
        lifted = args.backend != "null" or z > z0 + 0.005                          # the scripted grab lifted it
        print(f"box1 z={z:.3f} m (from {z0:.3f} m), ok={sim.check()['ok'] and lifted}")


if __name__ == "__main__":
    main()
