# SPDX-License-Identifier: Apache-2.0
"""Pause, step by program, pick worlds and read contacts: four worlds, a block dropped on a table in each.

    pip install "viser>=1.1.1"
    python examples/viewer/02_pause_step_contacts.py                      # prints a URL: open it in a browser
    python examples/viewer/02_pause_step_contacts.py --backend null --seconds 0.2   # headless (CI)

Worlds 0 and 2 are shown; all four are simulated. ``alloy.viewer.contacts`` returns the rigid contacts of one world:
a block at rest on the table presses with its weight, m g.
"""

from __future__ import annotations

import argparse

import numpy as np

import alloy
import alloy.viewer

MASS, G = 0.2, 9.81


def build_scene(worlds: int) -> alloy.Scene:
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    starts = np.stack([np.linspace(-0.15, 0.15, worlds), np.zeros(worlds), np.full(worlds, 0.5)], axis=1)  # one per world
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=starts, mass=MASS)
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="viser", help="viser | gl | rerun | usd | null")
    parser.add_argument("--seconds", type=float, default=5.0, help="simulated seconds to run after the stepping")
    parser.add_argument("--port", type=int, default=8080, help="viser: port of the browser viewer")
    parser.add_argument("--device", default=None, help="cpu or cuda:0 (default: GPU if present)")
    args = parser.parse_args()

    options = {"port": args.port} if args.backend == "viser" else {}
    sim = alloy.build(build_scene(4), device=args.device)
    view = alloy.viewer.launch(sim, backend=args.backend, worlds=[0, 2], paused=True, **options)
    print("open:", view.url, "| showing worlds", view.worlds)

    view.step(100)                                     # 1 s of simulated time, by program, while paused
    c = alloy.viewer.contacts(sim, world=0)
    force = float(alloy.viewer.force_on(sim, "block", world=0)[2])   # net vertical contact force on the block (N)
    print(f"world 0: {len(c['depth'])} contacts, summed normal force {force:.3f} N, m g = {MASS * G:.3f} N")
    print("bodies in contact:", sorted({f"{a}-{b}" for a, b in zip(c["body_a"], c["body_b"])}))

    view.resume()
    view.run(seconds=args.seconds)                     # the loop runs `--seconds` more simulated seconds
    print(f"t={sim.t:.2f} s, ok={sim.check()['ok']}")
    view.close()


if __name__ == "__main__":
    main()
