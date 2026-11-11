# SPDX-License-Identifier: Apache-2.0
"""Scene item looks: the look an ``alloy.Scene`` item carries in ``item["look"]`` (a scene variant's texture swap and
tint from ``alloy.scene.variants.vary``), as a camera look layer.

Each item look becomes materials ``<item>:<key>`` and rules on the item's shape labels (``<item>``, ``<item>/<part>``,
``<item>#k``; ``alloy.scene.variants.scene_look``). The camera consults the layer after a caller's own ``look`` and
before the card library's looks (``card_looks``): an item with a look renders with it, the others as before. A
texture the local render asset cache lacks falls back to the item's own library look for that material (its card's
object look, else its material's look, through ``alloy.looks``) when that texture is cached, else it is dropped (the
material keeps its preset colour; fill the cache with ``python -m alloy.looks --fetch-textures``); details the camera
does not draw are turned off. Both are logged. Lights and world are not part of the layer: a variant renders lit like
its base scene.

Example::

    from alloy.camera.item_looks import items_look
    layer = items_look(sim)          # None when no item of sim.scene has a look
    img = alloy.camera.render(sim)   # uses the layer itself
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

_MEMO: dict[str, dict] = {}
_MEMO_MAX = 64


def scene_of(source):
    """The ``alloy.Scene`` behind ``source`` (an ``alloy.Sim``'s ``scene``, or a Scene itself), else None."""
    scene = getattr(source, "scene", source)
    items = getattr(scene, "items", None)
    return scene if isinstance(items, dict) else None


def items_look(source) -> dict | None:
    """The camera look layer of ``source``'s item looks (module docstring), or None when no item has one.

    Example: ``items_look(alloy.Scene.from_dict(variant_json))["materials"]``."""
    scene = scene_of(source)
    if scene is None or not any(isinstance(it, dict) and it.get("look") for it in scene.items.values()):
        return None
    from alloy.scene.variants.looks import scene_look  # noqa: PLC0415  (scene package: only when items have looks)

    raw = scene_look(scene)
    key = json.dumps(raw, sort_keys=True, default=str)
    if key not in _MEMO:
        if len(_MEMO) >= _MEMO_MAX:
            _MEMO.clear()
        _MEMO[key] = _drawable(_with_cached_textures(raw, scene))
    return _MEMO[key]


def _cached(ref: str, res: str) -> bool:
    from .errors import CameraError  # noqa: PLC0415
    from .looks.textures import local_texture_files  # noqa: PLC0415
    try:
        return bool(local_texture_files(ref, res))
    except (CameraError, FileNotFoundError, OSError):
        return False


def _texture(spec: dict, key: str) -> tuple[str | None, str]:
    from .looks.spec import resolve_spec  # noqa: PLC0415
    eff = resolve_spec(key, spec)
    return eff.get("texture"), str(eff.get("res", "2k"))


def _with_cached_textures(raw: dict, scene) -> dict:
    """``raw`` with each material whose texture the cache lacks given its item's own library texture (when cached)."""
    from alloy.scene.variants.looks import TEXTURE_FIELDS  # noqa: PLC0415
    out = {**raw, "materials": dict(raw["materials"])}
    own: dict[str, dict] = {}
    for mkey, spec in raw["materials"].items():
        item, _, key = mkey.partition(":")
        ref, res = _texture(spec, mkey)
        if not ref or _cached(ref, res):
            continue
        if item not in own:
            own[item] = _library_materials(scene, item)
        base = own[item].get(key)
        bref, bres = _texture(base, mkey) if base else (None, "")
        if base and bref and _cached(bref, bres):
            out["materials"][mkey] = {**{k: v for k, v in spec.items() if k not in TEXTURE_FIELDS},
                                      **{k: base[k] for k in TEXTURE_FIELDS if k in base}}
            log.info("camera: %s keeps its library texture %s (%s is not cached)", mkey, bref, ref)
    return out


def _library_materials(scene, item: str) -> dict:
    """The look materials ``alloy.looks`` gives scene item ``item`` (its card's object look, else its material's)."""
    try:
        from alloy.api import catalog  # noqa: PLC0415
        from alloy.cards.library import CardLibrary  # noqa: PLC0415
        from alloy.looks.resolve import material_look, object_look  # noqa: PLC0415
        spec = scene.items[item]
        lib = CardLibrary(roots=catalog.card_roots(getattr(scene, "roots", ())))
        card, material = spec.get("card_name") or spec.get("card"), spec.get("material")
        look = (object_look(card, lib) if card else None) or (material_look(material, lib) if material else None)
        return dict(look.materials) if look else {}
    except Exception as exc:  # noqa: BLE001 - no library look to fall back to: the texture is dropped
        log.info("camera: no library look for item %s (%s)", item, exc)
        return {}


def _drawable(raw: dict) -> dict:
    from alloy.scene.variants.sheet import drawable  # noqa: PLC0415

    layer, dropped = drawable(raw)
    if dropped:
        log.warning("camera: item looks without %d detail(s) the camera cannot draw here: %s%s (missing textures: "
                    "python -m alloy.looks --fetch-textures)", len(dropped), "; ".join(dropped[:4]),
                    " ..." if len(dropped) > 4 else "")
    return {"materials": layer["materials"], "assign": layer["assign"]}
