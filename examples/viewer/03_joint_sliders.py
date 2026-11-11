# SPDX-License-Identifier: Apache-2.0
"""Joint sliders: move the Franka Panda's joints from the viewer's panel.

    pip install "viser>=1.1.1"
    python examples/viewer/03_joint_sliders.py                           # prints a URL: open it in a browser
    python examples/viewer/03_joint_sliders.py --backend null --seconds 0.2   # headless (CI)

``joints=True`` adds one slider per joint that a position actuator drives (the seven arm joints and the fingers).
Moving a slider sets that joint's target in every world, like ``sim.set_targets({"joint": value})``; targets stay as
set. A viewer whose sliders are left alone steps the same simulation, bit for bit, as one without them.
"""

from __future__ import annotations

import argparse

import alloy
import alloy.viewer
from alloy.cards.library import default_root


def build_scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)                   # the Panda's own scene file has its floor: no add_ground
    scene.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")
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
    view = alloy.viewer.launch(sim, backend=args.backend, joints=True, **options)
    print("open:", view.url)                       # viser: http://localhost:<port>; other backends: None
    view.run(seconds=args.seconds)                 # blocking; Ctrl-C or closing the viewer ends it
    print("ok" if sim.check()["ok"] else "problems")
    view.close()


if __name__ == "__main__":
    main()
