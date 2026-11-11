# SPDX-License-Identifier: Apache-2.0
"""How the camera draws SPH liquids: a dielectric with Beer-Lambert absorption on the full-material tracer, a tinted
Lambert colour on the fast tracer.

Presets (by the liquid's card material; a name with "wine" picks a wine, other unknown liquids are drawn as water):

- ``water``: the look preset ``water`` (IOR 1.333; absorption 0.24 / 0.07 / 0.05 per metre, close to pure water's
  0.34 / 0.06 / 0.01 per metre at 650 / 550 / 450 nm, Pope and Fry 1997).
- ``wine_red``, ``wine_rose``, ``wine_white``: IOR 1.34; decadic absorbance per cm (red wine A620 0.8, A520 3.6,
  A420 3.3, Glories colour indices; rose 0.08 / 0.6 / 0.35 and white 0.01 / 0.03 / 0.10, estimates) as natural-log
  coefficients ``A * ln(10) * 100`` per metre on R / G / B, as GA's wine render worker does.
- ``honey``, ``tea``, ``coolant``: the look presets of those names (estimates).

Cycles' Volume Absorption node absorbs ``sigma = Density * (1 - Color)`` per metre (``pbr/volume.py``), so a
coefficient triple becomes ``Density = max(sigma)`` and ``Color = 1 - sigma / Density``.

Example::

    key = liquid_key("water")                   # "water"
    mat = liquid_material(key)                  # pack-v2 material with a Volume Absorption node
    fast_color(liquid_key("red_wine"))          # (0.3, 0.02, 0.05)
"""
from __future__ import annotations

import math

LN10_PER_CM = math.log(10.0) * 100.0  # decadic absorbance per cm -> natural-log coefficient per metre
_WINE = {"wine_red": (0.8, 3.6, 3.3), "wine_rose": (0.08, 0.6, 0.35), "wine_white": (0.01, 0.03, 0.10)}
PRESETS = {
    "water": {"preset": "water"},
    **{k: {"preset": "water", "ior": 1.34, "roughness": 0.0, "sigma_a": tuple(a * LN10_PER_CM for a in v)}
       for k, v in _WINE.items()},
    "honey": {"preset": "honey"},
    "tea": {"preset": "tea"},
    "coolant": {"preset": "coolant"},
}
FAST = {  # linear albedo of the fast tracer's tinted look (it draws Lambert surfaces only)
    "water": (0.55, 0.68, 0.80), "wine_red": (0.30, 0.02, 0.05), "wine_rose": (0.75, 0.30, 0.35),
    "wine_white": (0.80, 0.76, 0.50), "honey": (0.60, 0.33, 0.05), "tea": (0.35, 0.14, 0.04),
    "coolant": (0.20, 0.50, 0.75),
}


def liquid_key(material: str | None) -> str:
    """The preset of a liquid card material: its own name, a wine by name, else ``water``."""
    m = str(material or "water").lower()
    if m in PRESETS:
        return m
    if "wine" in m:
        return "wine_rose" if "rose" in m or "rosé" in m else "wine_white" if "white" in m else "wine_red"
    return "water"


def fast_color(key: str) -> tuple:
    return FAST.get(key, FAST["water"])


def _volume_node(sigma) -> dict:
    density = max(float(s) for s in sigma)
    color = [1.0 - float(s) / density if density > 0.0 else 1.0 for s in sigma] + [1.0]
    sock = [("Color", "NodeSocketColor", color, True), ("Density", "NodeSocketFloat", density, True),
            ("Weight", "NodeSocketFloat", 0.0, False)]
    return {"name": "Volume Absorption", "type": "ShaderNodeVolumeAbsorption", "props": {},
            "inputs": [{"id": n, "name": n, "type": t, "linked": False, "enabled": e, "value": v} for n, t, v, e in sock]}


def _sigma_of(spec: dict) -> tuple | None:
    """Absorption per metre of a resolved look spec (``sigma_a``, else the preset's absorption colour and distance)."""
    if spec.get("sigma_a") is not None:
        return tuple(float(s) for s in spec["sigma_a"])
    if not spec.get("absorption_color") or not spec.get("absorption_m"):
        return None
    from alloy.render.blender import motion
    color, density = motion.absorption_density(spec["absorption_color"], float(spec["absorption_m"]))
    return tuple(float(density * (1.0 - c)) for c in color)


def liquid_material(key: str) -> dict:
    """Pack-v2 material ``liquid_<key>``: the preset's Principled BSDF (transmission 1) and, when the liquid absorbs,
    a Volume Absorption node on the Material Output's Volume socket."""
    from .looks import material_tree
    from .looks.spec import resolve_spec
    entry = dict(PRESETS.get(key, PRESETS["water"]))
    sigma = entry.pop("sigma_a", None)
    name = f"liquid_{key}"
    mat = material_tree(name, {**entry, "hard": False})
    sigma = sigma if sigma is not None else _sigma_of(resolve_spec(name, entry))
    if sigma is not None and max(sigma) > 0.0:
        tree = mat["tree"]
        tree["nodes"].append(_volume_node(sigma))
        out = next(n for n in tree["nodes"] if n["type"] == "ShaderNodeOutputMaterial")
        next(i for i in out["inputs"] if i["id"] == "Volume")["linked"] = True
        tree["links"].append({"from": "Volume Absorption", "from_id": "Volume", "to": out["name"], "to_id": "Volume",
                              "muted": False})
    return mat
