# SPDX-License-Identifier: Apache-2.0
"""The camera's liquid gate scene (RENDER-GA-1), simulated and written as a scene pack for Cycles references:

- ``pour``: 120 mL of water poured from 17 cm into a cup on a table (SPH, 4 mm particles), caught mid-pour with the
  stream and the filling cup in view.

The pack holds world 0 at the end of the run, seen from a three-quarter camera, lit by one area light and the studio
world; the water is the camera's own liquid look (a dielectric with Beer-Lambert absorption, ``liquid_looks``) on the
surface it builds from the particles (``liquid_gpu``). ``--render`` also saves the camera's live image of the sim.

Run (GPU recommended)::

    python examples/camera/liquid_gate_scene.py --out packs --render
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import alloy
from alloy.camera.pack_write import write_sim_pack

log = logging.getLogger("liquid_gate_scene")

LOOK = {
    "materials": {
        "cup_glaze": {"preset": "plastic_matte", "base_color": [0.86, 0.84, 0.80], "roughness": 0.35},
        "table_top": {"preset": "plastic_matte", "base_color": [0.30, 0.33, 0.36], "roughness": 0.55},
        "floor": {"preset": "plaster_paint"},
    },
    "assign": [{"match": "cup", "material": "cup_glaze"}, {"match": "table", "material": "table_top"},
               {"match": "ground", "material": "floor"}],
    "lights": [{"type": "area", "eye": [0.5, -0.8, 1.3], "target": [0.0, 0.0, 0.35], "watts": 120.0, "size_m": 0.6,
                "kelvin": 5200}],
    "world": {"strength": 0.8},
}
VIEW = ((0.30, -0.30, 0.52), (0.0, 0.0, 0.37))
SECONDS = 0.45


def pour_scene(worlds: int = 1, liquid: bool = True) -> alloy.Scene:
    """The pour; ``liquid=False``: the same scene without it (the speed rows' no-liquid baseline)."""
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_box("table", size=(0.5, 0.5, 0.3), pos=(0.0, 0.0, 0.15), static=True)
    s.add_open_cylinder("cup", inner_radius=0.035, depth=0.07, wall=0.004, pos=(0.0, 0.0, 0.3))
    if liquid:
        s.add_pour("water", 1.2e-4, pos=(0.0, 0.0, 0.47), rate=2.0e-4, speed=0.6)
    return s


def build(out: Path, render: bool, worlds: int = 1) -> Path:
    sim = alloy.build(pour_scene(worlds), seed=0)
    t0 = time.perf_counter()
    sim.run(SECONDS)
    log.info("pour: %.2f s simulated in %.0f s; health ok: %s", SECONDS, time.perf_counter() - t0, sim.check().get("ok"))
    pack = write_sim_pack(sim, out / "pour", VIEW, look=LOOK)
    if render:
        from alloy.camera import Camera, camera_scene
        from alloy.camera.view import write_png
        sc = camera_scene(sim, look=LOOK)
        img = Camera("gate", look_at=VIEW, width=640, height=480, fov=39.6, fit="AUTO", spp=128).render(
            sc, sim.engine.state).numpy()[0]
        write_png(out / "pour_live.png", img)
    return pack


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("liquid_packs"))
    ap.add_argument("--render", action="store_true", help="also save the camera's live image of the sim")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    log.info("wrote %s", build(a.out, a.render))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
