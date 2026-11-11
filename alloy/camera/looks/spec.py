# SPDX-License-Identifier: Apache-2.0
"""Look-material specs: preset plus overrides resolved to full Principled values (mirror of ``Look.material_spec``).

Example::

    spec = resolve_spec("housing", {"preset": "abs_black", "roughness": 0.5})
    spec["roughness"], spec["metallic"], spec["_overrides"]      # (0.5, 0.0, ['roughness'])
"""
from __future__ import annotations

from alloy.render.blender import presets

from ..errors import CameraError

ALLOWED_KEYS = frozenset(presets.OVERRIDE_KEYS) | frozenset(presets.EXTRA_KEYS)
UNSUPPORTED_DETAILS = {"crust": "MapRange/Mix/Noise ramps", "crumb": "Voronoi", "scratches": "Map Range"}


def resolve_spec(name: str, entry: dict | str | None) -> dict:
    """Preset values merged with the look overrides of ``entry`` (a dict, a bare preset name, or None for ``name``).

    Raises CameraError for an unknown preset or an override key the look format does not have.
    """
    if entry is None:
        entry = {"preset": name}
    elif isinstance(entry, str):
        entry = {"preset": entry}
    preset = entry.get("preset", presets.FALLBACK_MATERIAL)
    if preset not in presets.PRESETS:
        raise CameraError(f"material {name!r} has unknown preset {preset!r}",
                          f"use one of {sorted(presets.PRESETS)}")
    bad = sorted(set(entry) - ALLOWED_KEYS)
    if bad:
        raise CameraError(f"material {name!r} has unknown keys {bad}", f"use only {sorted(ALLOWED_KEYS)}")
    spec = {**presets.DEFAULTS, **presets.PRESETS[preset]}
    overrides = {k: v for k, v in entry.items() if k != "preset"}
    spec.update(overrides)
    spec["_name"] = name
    spec["_overrides"] = sorted(overrides)
    return spec


def check_supported(name: str, spec: dict) -> None:
    """Raise CameraError for look features the Warp generator cannot evaluate (so a render never silently differs)."""
    detail = spec.get("detail")
    features = {
        f"detail {detail!r}": detail in UNSUPPORTED_DETAILS,
        "texture_alpha (decal)": bool(spec.get("texture_alpha")),
        "tex_tint": bool(spec.get("tex_tint")),
        "imperfection": bool(spec.get("imperfection")),
        "mapping 'planar'": spec.get("mapping") == "planar",
    }
    used = [f for f, on in features.items() if on]
    if used:
        raise CameraError(f"material {name!r} uses {', '.join(used)}, which the path tracer's node generator does not support",
                          "drop that key (override it to None/False) or pick another preset")
