# SPDX-License-Identifier: Apache-2.0
"""Library looks for shapes the scene's own look does not cover: a card's object look, else its material's default
look (``alloy.looks.fallback_look``), instead of a flat palette colour or grey.

Shapes of library objects carry labels ``<card>``, ``<card>/<part>`` or ``<card>#k``; labels that name no card get
nothing here and keep their colour. Both camera paths use it: the full-material path builds the look's material, the
fast diffuse path takes its base colour.

Example::

    from alloy.camera.card_looks import fallback_rules, look_colors
    rules = fallback_rules(["rubber_ball_40mm/hull_0"])     # {"materials": {...}, "assign": [...]}
    look_colors(["rubber_ball_40mm/hull_0", "my_box"])      # {0: (0.72, 0.1, 0.08)}
"""
from __future__ import annotations

import logging
import re

from .errors import CameraError

log = logging.getLogger(__name__)


def assigned(look: dict, labels: list[str]) -> str | None:
    """First assign rule of ``look`` whose regex matches any of ``labels`` (as alloy.render's Look.assigned)."""
    for i, rule in enumerate(look.get("assign", [])):
        try:
            rx = re.compile(rule["match"])
        except (re.error, KeyError, TypeError) as exc:
            raise CameraError(f"look assign[{i}] is {rule!r}",
                              "give each rule a valid regex 'match' and a 'material'") from exc
        if any(lb and rx.search(lb) for lb in labels):
            return rule["material"]
    return None


def fallback_rules(labels) -> dict:
    """``alloy.looks.fallback_look`` for these labels, or {} when no label names a card or the card library cannot
    be read (the camera then keeps shape colours)."""
    names = sorted({lb for lb in labels if lb})
    if not names:
        return {}
    try:
        from alloy.looks.fallback import fallback_look  # noqa: PLC0415  (reads the card library on first use)
        return fallback_look(names)
    except Exception as exc:  # noqa: BLE001  (library missing or unreadable: render without library looks)
        log.info("camera: no library looks for this scene (%s)", exc)
        return {}


def look_over_scan(labels: list[str], card: str | None, look: dict | None = None,
                   items: dict | None = None) -> tuple[str, dict] | None:
    """(material name, look spec) that replaces a scanned object's own photo texture on these labels, else None (draw
    the texture). ``labels``: the scan part's shape and body labels (``<item>/visual``, ``<item>``); ``card``: the
    item's object card; ``look``: the caller's look; ``items``: the scene's item-look layer (``item_looks``).

    First match wins, in the camera's usual order: the caller's look rules, the item looks, then the card's OWN object
    look (``looks/objects/<card>.json``). A card's material-default look and the grey fallback do not count: they
    describe the material in general, and the scan's texture is the object's real look.

    Example: ``look_over_scan(["mug1/visual", "mug1"], "ycb_025_mug")`` is None unless a look names the mug."""
    for layer, materials in ((look, (look or {}).get("materials", {})), (items, (items or {}).get("materials", {}))):
        name = assigned(layer, labels) if layer else None
        if name is not None:
            return name, materials.get(name, name)
    return _object_look_material(labels, card) if card else None


def _object_look_material(labels: list[str], card: str) -> tuple[str, dict] | None:
    """The card's own object look for the part these labels name (``<item>/<part>``), or None without one."""
    try:
        from alloy.looks.resolve import object_look  # noqa: PLC0415  (reads the card library's look index)
        own = object_look(card)
    except Exception as exc:  # noqa: BLE001  (library missing or unreadable: the scan keeps its texture)
        log.info("camera: no object look for %s (%s)", card, exc)
        return None
    if own is None:
        return None
    parts = [lb.split("/", 1)[1] for lb in labels if "/" in lb]
    key = next((r["material"] for r in own.assign if any(re.search(r["match"], p) for p in parts)), own.default)
    return f"{card}.{key}", dict(own.materials[key])


def look_colors(labels: list[str]) -> dict[int, tuple[float, float, float]]:
    """{index into ``labels``: linear base colour} for the labels a library look covers (the fast diffuse path)."""
    rules = fallback_rules(labels)
    if not rules:
        return {}
    from .looks.spec import resolve_spec  # noqa: PLC0415
    out = {}
    for i, lb in enumerate(labels):
        name = assigned(rules, [lb])
        if name is None:
            continue
        base = resolve_spec(name, rules["materials"][name]).get("base_color")
        if base is not None:
            out[i] = tuple(float(c) for c in base[:3])
    return out
