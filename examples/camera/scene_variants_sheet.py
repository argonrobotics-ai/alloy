# SPDX-License-Identifier: Apache-2.0
"""Scene variants in the camera: n variants of one scene (``alloy.scene.variants.vary``: objects moved, textures
swapped within their family, colours tinted), each built and rendered with plain ``sim.render()``, which draws every
item's own look, tiled into one contact sheet.

    python examples/camera/scene_variants_sheet.py --out sheet.png --fetch    # 8 variants, 320x240, 64 spp
    python examples/camera/scene_variants_sheet.py --n 4 --size 160x120 --spp 16

Texture swaps show when their images are in the local render asset cache: ``--fetch`` downloads every texture the
variants use first (Poly Haven / ambientCG, CC0). Without it, a texture the cache lacks falls back to the object's
own library texture, else is left out (the log names it). Images show through the AgX view, so bright textures roll
off rather than clip (``--exposure`` adds stops). Each variant's materials compile the full-material tracer once
(about a minute on a GPU, several on a CPU).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import alloy
from alloy.camera.view import write_png
from alloy.scene.variants import vary
from alloy.scene.variants.sheet import tile

log = logging.getLogger("variants_sheet")
VIEW = ((0.42, -0.5, 0.75), (0.0, 0.0, 0.42))   # eye, target: the table top from the front right


def table_scene() -> alloy.Scene:
    """A pine table with wooden, bamboo, steel and plastic blocks and a ball on it."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.5, 0.4, 0.4), pos=(0, 0, 0.2), static=True, material="wood_pine")
    s.add_box("beech", size=(0.07, 0.07, 0.07), pos=(-0.12, 0.06, 0.435), material="wood_beech")
    s.add_box("bamboo", size=(0.09, 0.05, 0.04), pos=(0.0, 0.09, 0.42), material="bamboo")
    s.add_box("steel", size=(0.06, 0.06, 0.05), pos=(0.12, 0.05, 0.425), material="steel_carbon")
    s.add_box("plastic", size=(0.08, 0.06, 0.03), pos=(-0.08, -0.08, 0.415), material="abs")
    s.add_sphere("ball", 0.03, pos=(0.1, -0.08, 0.43), material="abs")
    return s


def fetch_textures(scenes) -> int:
    """Download (once) every texture the scenes' item looks use, at the resolution they ask for; returns the count."""
    from alloy.camera.looks.spec import resolve_spec
    from alloy.render.blender.assets import AssetCache
    want = set()
    for scene in scenes:
        for name, item in scene.items.items():
            for key, spec in ((item.get("look") or {}).get("materials") or {}).items():
                eff = resolve_spec(f"{name}:{key}", spec)
                if eff.get("texture"):
                    want.add((eff["texture"], str(eff.get("res", "2k"))))
    cache = AssetCache()
    for ref, res in sorted(want):
        cache.texture(ref, res)
    return len(want)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="sheet.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fetch", action="store_true", help="download the textures the variants use first")
    ap.add_argument("--exposure", type=float, default=0.0, help="extra stops on top of the look's (AgX view)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    w, h = (int(v) for v in a.size.split("x"))
    variants = vary(table_scene(), a.n, pos=0.03, yaw_deg=25.0, textures="same-family", seed=a.seed, tint=0.08)
    if a.fetch:
        log.info("textures in the cache: %d", fetch_textures(variants))
    images = []
    for k, v in enumerate(variants):
        sim = alloy.build(v, device=a.device)
        images.append(sim.render(look_at=VIEW, width=w, height=h, spp=a.spp, fov=40.0, exposure=a.exposure))
        log.info("variant %d rendered", k)
    out = write_png(Path(a.out), tile(images))
    log.info("contact sheet: %s (%d variants)", out, len(images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
