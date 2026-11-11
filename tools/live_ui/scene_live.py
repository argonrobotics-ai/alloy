# SPDX-License-Identifier: Apache-2.0
"""Stream a soft-object scene's 3D frames to the live hub, without an env or training (checks the 3D path on cloth
and soft bodies: export -> hub -> page).

    python tools/live_ui/scene_live.py --hub 127.0.0.1:47110 --run soft-cpu --worlds 2 --steps 300 --device cpu

Scene: a static table, a cotton sheet dropped onto it (2 mm thick: the page draws its shells 1 mm each side of the
exported mid-surface) and a small silicone cube (a VBD soft body: the page draws the outer faces of its active tets).
Every world is exported; the hub caches each world's topology for the page.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import alloy  # noqa: E402
from alloy.live.frames import PickedFrames  # noqa: E402
from alloy.live.sender import Link  # noqa: E402
from alloy.live.wire import Kind, pack_arrays  # noqa: E402


def scene(worlds: int) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.3, 0.3, 0.1), pos=(0, 0, 0.05), static=True)
    s.add_cloth("sheet", size=(0.2, 0.2), pos=(0.0, 0.0, 0.13), resolution=10, mass=0.02, self_contact=False)
    s.add_box("jelly", size=(0.04, 0.04, 0.04), pos=(0.22, 0.0, 0.05), material="silicone_40a", rigid=False)
    return s


def numbers(worlds: int) -> tuple[dict, bytes]:
    """A minimal per-world NUMBERS frame, so the overview grid has data."""
    z = np.zeros(worlds, np.float32)
    meta, blob = pack_arrays({k: z for k in ("reward", "ret", "success", "done", "health", "ep_ret", "ep_success",
                                             "paused")})
    return meta, blob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hub", default="127.0.0.1:47110")
    p.add_argument("--run", default="soft-cpu")
    p.add_argument("--worlds", type=int, default=2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    sim = alloy.build(scene(a.worlds), device=a.device)
    print({k: r["route"] for k, r in sim.routes().items()}, flush=True)
    host, port = a.hub.rsplit(":", 1)
    link = Link(host, int(port), {"worker": "scene", "run": a.run, "worlds": a.worlds, "dt": sim.dt, "settings": [],
                                  "bodies": list(sim.index.bodies), "robots": []})
    frames = PickedFrames(sim)
    t0 = time.perf_counter()
    for k in range(a.steps):
        sim.step(1)
        frames.capture(list(range(min(a.worlds, 8))))
        link.offer(Kind.STATE, {"t": time.time()}, frames.packer())
        if k % 10 == 0:
            meta, blob = numbers(a.worlds)
            link.offer(Kind.NUMBERS, {"step": k + 1, "t": time.time(), "steps_per_s": 0.0, **meta}, blob)
            print(f"step {k} {time.perf_counter() - t0:.1f} s {frames.stats}", flush=True)
    link.close()


if __name__ == "__main__":
    main()
