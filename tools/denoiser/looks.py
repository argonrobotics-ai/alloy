#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Seed-stable look variants for denoiser teacher sets: one alloy look JSON in, K variant look JSONs out.

    python tools/denoiser/looks.py LOOK.json --k 40 --seed 1 --out DIR [--p-swap 0.35] [--p-dark 0.3] [--p-hdri 0.6]

Per variant (rng = hash of seed and variant index; the same inputs always give the same files):

- world: rotation uniform over the circle, strength x log-uniform(0.5, 2); with ``p_hdri`` another CC0 HDRI from
  ``hdris.json`` (studio 50 %, indoor 35 %, outdoor 15 %), shown as the background half of the time;
- lights: watts x log-uniform(0.5, 2), 3,500-6,500 K, eye moved by +-15 % of its distance to the target;
- non-hero materials (tables, walls, mats; ``NON_HERO``): colour HSV jitter (untextured only), roughness +-0.15;
- material swaps: each material, with ``p_swap``, becomes a preset from one family (glass, liquid, metal, dark);
  transmissive materials only swap to other transmissive presets, so glass and liquid scenes stay glass and liquid;
- dark backdrop with ``p_dark`` (HDRI hidden, near-black background, world strength x 0.3-1);
- cameras: one common orbit of every key about its target (yaw +-35 deg, pitch +-12 deg, elevation kept in
  5-75 deg, distance x 0.75-1.3), lens x log-uniform(0.75, 1.6).

The parameters are recorded under the variant's ``"randomize"`` key (the renderer ignores it).
"""
from __future__ import annotations

import argparse
import colorsys
import copy
import hashlib
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alloy" / "render" / "blender"))
import presets as P

log = logging.getLogger("denoiser.looks")
NON_HERO = r"walnut|plywood|marble|linen|plaster|silicone_mat|worktop|bench|wall|backdrop|room|floor|table|mat$"
FAMILIES = {
    "glass": ["glass_clear", "plastic_clear", "polycarbonate_clear", "pp_clear", "pp_frosted"],
    "liquid": ["water", "coolant", "tea", "honey"],
    "metal": ["steel_polished", "steel_brushed", "aluminum_bare", "copper", "brass", "gold_plated", "zinc_plated",
              "steel_black_oxide", "stainless_steel_brushed"],
    "dark": ["robot_painted_dark", "rubber_black", "powder_coat_black", "abs_black", "aluminum_anodized_black"],
}
WEIGHTS = {"glass": 0.35, "liquid": 0.25, "metal": 0.25, "dark": 0.15}
TRANSMISSIVE = set(FAMILIES["glass"]) | set(FAMILIES["liquid"])
HDRI_MIX = {"studio": 0.5, "indoor": 0.35, "outdoor": 0.15}


def rng_for(seed: int, index: int) -> random.Random:
    """Deterministic RNG per (seed, variant). Example: ``rng_for(1, 0).random()`` is always the same."""
    return random.Random(int(hashlib.sha256(f"{seed}:{index}:dn".encode()).hexdigest()[:16], 16))


def logu(rng: random.Random, lo: float, hi: float) -> float:
    """Log-uniform sample in [lo, hi]. Example: ``logu(rng, 0.5, 2.0)``."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def jitter_world(world: dict, rng: random.Random, hdris: dict, p_hdri: float) -> dict:
    """Rotate / rescale the world and maybe swap its HDRI. Returns the record."""
    world["rotation_deg"] = round(rng.uniform(0, 360), 2)
    mult = logu(rng, 0.5, 2.0)
    world["strength"] = round(float(world.get("strength", 1.0)) * mult, 4)
    rec = {"rotation_deg": world["rotation_deg"], "strength_x": round(mult, 3)}
    if hdris and rng.random() < p_hdri:
        kind = rng.choices(list(HDRI_MIX), list(HDRI_MIX.values()))[0]
        world["hdri"] = "polyhaven:" + rng.choice(hdris[kind])
        world["show_hdri"] = rng.random() < 0.5
        rec.update(hdri=world["hdri"], show_hdri=world["show_hdri"])
    return rec


def shift(eye: list[float], target: list[float], rng: random.Random, frac: float) -> list[float]:
    """Move ``eye`` by up to ``frac`` of its distance to ``target`` per axis."""
    d = math.dist(eye, target)
    return [eye[i] + rng.uniform(-frac, frac) * d for i in range(3)]


def jitter_lights(lights: list, rng: random.Random) -> list:
    """Scale watts, change kelvin, move each light's eye. Returns the records."""
    out = []
    for lt in lights:
        mult = logu(rng, 0.5, 2.0)
        lt["watts"] = round(float(lt.get("watts", 50.0)) * mult, 4)
        lt["kelvin"] = round(rng.uniform(3500, 6500))
        if "eye" in lt:
            lt["eye"] = [round(v, 5) for v in shift(lt["eye"], lt.get("target", [0, 0, 0]), rng, 0.15)]
        out.append({"watts_x": round(mult, 3), "kelvin": lt["kelvin"]})
    return out


def preset_value(entry: dict, name: str, key: str):
    """A material entry's value for ``key``, falling back to its preset, then the preset defaults."""
    if key in entry:
        return entry[key]
    return P.PRESETS.get(entry.get("preset", name), {}).get(key, P.DEFAULTS[key])


def jitter_material(entry: dict, name: str, rng: random.Random) -> dict:
    """HSV / roughness jitter of one non-hero material entry (in place). Returns the record."""
    rough = min(1.0, max(0.02, float(preset_value(entry, name, "roughness")) + rng.uniform(-0.15, 0.15)))
    entry["roughness"] = round(rough, 3)
    rec = {"roughness": entry["roughness"]}
    if not entry.get("texture"):
        h, s, v = colorsys.rgb_to_hsv(*list(preset_value(entry, name, "base_color"))[:3])
        h = (h + rng.uniform(-0.04, 0.04)) % 1.0
        s = min(1.0, s * rng.uniform(0.7, 1.3) + rng.uniform(0.0, 0.08))
        v = min(1.0, v * rng.uniform(0.8, 1.25))
        entry["base_color"] = [round(c, 4) for c in colorsys.hsv_to_rgb(h, s, v)]
        rec["base_color"] = entry["base_color"]
    return rec


def jitter_materials(look: dict, rng: random.Random, non_hero: re.Pattern) -> dict:
    """Jitter every non-hero material. Returns {name: record}."""
    mats = look.get("materials", {})
    return {n: jitter_material(mats[n], n, rng) for n in sorted(mats) if non_hero.search(n)}


def swap_materials(look: dict, rng: random.Random, p: float) -> dict:
    """Replace materials by family presets with probability ``p`` each. Returns {name: new preset}."""
    out = {}
    fams = list(WEIGHTS)
    mats = look.get("materials", {})
    for name in sorted(mats):
        if rng.random() >= p:
            continue
        clear = mats[name].get("preset", name) in TRANSMISSIVE
        fam = rng.choice(["glass", "liquid"]) if clear else rng.choices(fams, [WEIGHTS[f] for f in fams])[0]
        mats[name] = {"preset": rng.choice(FAMILIES[fam])}
        out[name] = mats[name]["preset"]
    return out


def dark_backdrop(look: dict, rng: random.Random) -> dict:
    """Hide the HDRI behind a near-black background (it still lights the scene). Returns the record."""
    w = look.setdefault("world", {})
    base = rng.uniform(0.004, 0.04)
    w["show_hdri"] = False
    w["background_color"] = [round(base * rng.uniform(0.85, 1.15), 4) for _ in range(3)]
    w["strength"] = round(float(w.get("strength", 1.0)) * rng.uniform(0.3, 1.0), 4)
    return {"background_color": w["background_color"], "strength": w["strength"]}


def orbit(eye: list[float], target: list[float], yaw: float, pitch: float, scale: float) -> list[float]:
    """Rotate ``eye`` about ``target`` (z up) by ``yaw`` / ``pitch`` radians and scale its distance; the elevation
    stays within 5-75 deg. Example: ``orbit([1, 0, 0], [0, 0, 0], math.pi / 2, 0, 1)`` is about ``[0, 1, 0]``."""
    d = [eye[i] - target[i] for i in range(3)]
    r = math.sqrt(sum(v * v for v in d)) or 1e-9
    az = math.atan2(d[1], d[0]) + yaw
    el = math.asin(max(-1.0, min(1.0, d[2] / r))) + pitch
    el = max(math.radians(5), min(math.radians(75), el))
    r *= scale
    return [target[0] + r * math.cos(el) * math.cos(az), target[1] + r * math.cos(el) * math.sin(az),
            target[2] + r * math.sin(el)]


def move_cameras(look: dict, rng: random.Random) -> dict:
    """One common orbit + zoom for every key of every camera. Returns the record."""
    yaw, pitch = math.radians(rng.uniform(-35, 35)), math.radians(rng.uniform(-12, 12))
    scale, zoom = rng.uniform(0.75, 1.3), logu(rng, 0.75, 1.6)
    for cam in look.get("cameras", {}).values():
        for k in cam.get("keys", []):
            k["eye"] = [round(v, 5) for v in orbit(k["eye"], k["target"], yaw, pitch, scale)]
        if "lens_mm" in cam:
            cam["lens_mm"] = round(float(cam["lens_mm"]) * zoom, 3)
    return {"yaw_deg": round(math.degrees(yaw), 2), "pitch_deg": round(math.degrees(pitch), 2),
            "dist_x": round(scale, 3), "zoom": round(zoom, 3)}


def variant(look: dict, seed: int, index: int, hdris: dict | None = None, p_swap: float = 0.35,
            p_dark: float = 0.3, p_hdri: float = 0.6, non_hero: str = NON_HERO) -> dict:
    """One randomised copy of ``look``; the parameters are recorded under ``"randomize"``.

    Example: ``v = variant(json.load(open("look.json")), seed=1, index=0, hdris=load_hdris())``."""
    v = copy.deepcopy(look)
    rng = rng_for(seed, index)
    rec: dict = {"seed": seed, "index": index}
    rec["world"] = jitter_world(v.setdefault("world", {}), rng, hdris or {}, p_hdri)
    rec["lights"] = jitter_lights(v.get("lights", []), rng)
    rec["materials"] = jitter_materials(v, rng, re.compile(non_hero))
    rec["swaps"] = swap_materials(v, rng, p_swap)
    if rng.random() < p_dark:
        rec["dark"] = dark_backdrop(v, rng)
    rec["cameras"] = move_cameras(v, rng)
    v["randomize"] = rec
    return v


def load_hdris(path: Path | None = None) -> dict:
    """The CC0 HDRI ids by kind (studio, indoor, outdoor). Example: ``len(load_hdris()["studio"])``."""
    return json.loads((path or Path(__file__).with_name("hdris.json")).read_text())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("look")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--p-swap", dest="p_swap", type=float, default=0.35)
    p.add_argument("--p-dark", dest="p_dark", type=float, default=0.3)
    p.add_argument("--p-hdri", dest="p_hdri", type=float, default=0.6)
    a = p.parse_args(argv)
    look = json.loads(Path(a.look).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hdris = load_hdris()
    for i in range(a.k):
        v = variant(look, a.seed, i, hdris, a.p_swap, a.p_dark, a.p_hdri)
        (out / f"variant{i:03d}.json").write_text(json.dumps(v, indent=1))
    print(json.dumps({"variants": a.k, "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
