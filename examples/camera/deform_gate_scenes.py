# SPDX-License-Identifier: Apache-2.0
"""The camera's deformable gate scenes, simulated and written as scene packs for Cycles references:

- ``towel``: a 70 x 70 cm linen towel (VBD cloth) dropped onto a 50 cm table, draping over two of its edges;
- ``heap``: a block of dry sand (``soil_sand_dry``, MPM grains) collapsing on a table, caught after 0.25 s.

Each pack holds world 0 at the end of the run, seen from a three-quarter camera, lit by one area light and the studio
world. ``--render`` also saves this camera's own image of the live sim (full materials) next to the pack.

Run (GPU recommended; MPM on the CPU takes minutes per second of sim)::

    python examples/camera/deform_gate_scenes.py --out packs --scenes towel,heap --render
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import alloy
from alloy.camera.pack_write import write_sim_pack

log = logging.getLogger("deform_gate_scenes")

LOOK = {
    "materials": {
        "towel_linen": {"preset": "linen", "texture": None, "tex_tint": False, "base_color": [0.55, 0.12, 0.10]},
        "sand_dry": {"preset": "plastic_matte", "base_color": [0.42, 0.31, 0.19], "roughness": 0.95},
        "table_top": {"preset": "plastic_matte", "base_color": [0.30, 0.33, 0.36], "roughness": 0.55},
        "floor": {"preset": "plaster_paint"},
    },
    "assign": [{"match": "towel", "material": "towel_linen"}, {"match": "sand", "material": "sand_dry"},
               {"match": "table", "material": "table_top"}, {"match": "ground", "material": "floor"}],
    "lights": [{"type": "area", "eye": [0.6, -0.9, 1.4], "target": [0.0, 0.0, 0.3], "watts": 120.0, "size_m": 0.6,
                "kelvin": 5200}],
    "world": {"strength": 0.8},
}
VIEWS = {"towel": ((0.85, -0.85, 0.7), (0.0, 0.0, 0.2)), "heap": ((0.32, -0.32, 0.42), (0.0, 0.0, 0.24))}


def towel_scene(worlds: int = 1) -> alloy.Scene:
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_box("table", size=(0.5, 0.5, 0.3), pos=(0.0, 0.0, 0.15), static=True)
    s.add_cloth("towel", size=(0.7, 0.7), pos=(0.12, 0.08, 0.42), resolution=56, mass=0.15, material="cotton_woven")
    return s


def heap_scene(worlds: int = 1) -> alloy.Scene:
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_box("table", size=(0.5, 0.5, 0.2), pos=(0.0, 0.0, 0.1), static=True)
    s.add_box("sand", size=(0.08, 0.08, 0.06), pos=(0.0, 0.0, 0.235), material="soil_sand_dry")
    return s


SCENES = {"towel": (towel_scene, 2.0), "heap": (heap_scene, 0.25)}


def build_one(name: str, out: Path, render: bool) -> Path:
    make, seconds = SCENES[name]
    sim = alloy.build(make(), seed=0)
    t0 = time.perf_counter()
    sim.run(seconds)
    log.info("%s: %.1f s simulated in %.0f s; health ok: %s", name, seconds, time.perf_counter() - t0,
             sim.check().get("ok"))
    pack = write_sim_pack(sim, out / name, VIEWS[name], look=LOOK)
    if render:
        from alloy.camera import Camera, camera_scene
        sc = camera_scene(sim, look=LOOK)
        eye, target = VIEWS[name]
        img = Camera("gate", look_at=(eye, target), width=640, height=480, fov=39.6, fit="AUTO", spp=128).render(
            sc, sim.engine.state).numpy()[0]
        from alloy.camera.view import write_png
        write_png(out / f"{name}_live.png", img)
    return pack


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("deform_packs"))
    ap.add_argument("--scenes", default="towel,heap")
    ap.add_argument("--render", action="store_true", help="also save the camera's live image of each sim")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for name in a.scenes.split(","):
        log.info("wrote %s", build_one(name, a.out, a.render))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
