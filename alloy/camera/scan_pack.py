# SPDX-License-Identifier: Apache-2.0
"""Scanned objects' photo textures in the full-material scene (SCAN-TEXTURES part A for the full tracer).

:func:`pack_with_scans` is ``model_pack.pack_from_model`` plus, for an ``alloy.Sim``, every scanned item drawn by its
glb parts (``scan_textures``) instead of its flat ``<item>/visual`` display mesh: each part becomes a textured mesh
riding the item's body with a generic material (its image as base colour, by the mesh's UVs; ``texture_tint`` when the
glb's colour factor is not 1), as ``meshes.model_scene`` rides ``TexturedMesh`` objects. A look that names the object
(the caller's look, the item looks, or the card's own object look: ``card_looks.look_over_scan``) keeps its material
and the display mesh. The part meshes go before the per-world dynamic meshes (deformables, liquids), which stay last.

Example::

    pk, frame, binding = pack_with_scans(sim.engine.model, None, None, None, sim=sim)
"""
from __future__ import annotations

import logging

import numpy as np

from .scan_textures import ScanSource, scan_sources, with_scan_textures

log = logging.getLogger(__name__)


def linear_to_srgb(rgb) -> tuple:
    """sRGB encoding of a linear RGB colour (TexturedMesh colours are sRGB)."""
    c = np.clip(np.asarray(rgb, np.float64), 0.0, 1.0)
    out = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return tuple(float(v) for v in out)


def _kept(sim, sources: list[ScanSource], look, items) -> list[ScanSource]:
    """The sources no look claims (a look that names the object keeps its material)."""
    from .card_looks import look_over_scan
    out = []
    for src in sources:
        item = src.label.rsplit("/", 1)[0]
        card = (getattr(sim, "scene", None).items.get(item) or {}).get("card_name")
        if look_over_scan([src.label, item], card, look, items) is None:
            out.append(src)
    return out


def scan_objects(model, sim, look=None, items=None) -> tuple[list, set[str]]:
    """(TexturedMesh per scan part of world 0, the display labels they replace) for ``sim``'s scanned items."""
    from .from_newton import scene_from_model
    from .meshes import TexturedMesh
    sources = _kept(sim, scan_sources(sim), look, items)
    if not sources:
        return [], set()
    info: dict = {}
    sd = with_scan_textures(scene_from_model(model), sim, sources, info)
    body_labels = [str(b) for b in (getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])]
    objects, replaced = [], set()
    for i, name in enumerate(sd.inst_name):
        label = name.split("#", 1)[0]
        if sd.inst_group[i] > 0 or label not in {s.label for s in sources}:
            continue
        mesh, b = sd.meshes[sd.inst_mesh[i]], int(sd.inst_body[i])
        if mesh.name not in info:
            continue  # the display mesh itself (the glb was not used)
        replaced.add(label)
        color = tuple(float(c) for c in sd.inst_color[i])
        textured = mesh.texture >= 0
        uv = None if mesh.uv is None else np.stack([mesh.uv[:, 0], 1.0 - mesh.uv[:, 1]], axis=1)  # Blender: v up
        objects.append(TexturedMesh(
            name, mesh.points, mesh.tris, uv=uv, texture=sd.textures[mesh.texture] if textured else None,
            color=linear_to_srgb(color), roughness=info[mesh.name][0], metallic=info[mesh.name][1],
            pose=tuple(float(v) for v in sd.inst_local[i]),
            body=body_labels[b] if 0 <= b < len(body_labels) else "",
            texture_tint=textured and not np.allclose(color, 1.0)))
    return objects, replaced


def pack_with_scans(model, look, lighting, items, layout=None, liquid=None, sim=None):
    """``pack_from_model(model, look, lighting, items, deform=layout, liquid=liquid)`` with ``sim``'s scanned items
    drawn by their photo textures (module docstring). Returns (pack, frame, (body, local, present))."""
    from .meshes import _body_rows, _pack_meshes
    from .model_pack import pack_from_model
    pk, frame, binding = pack_from_model(model, look, lighting, items, deform=layout, liquid=liquid)
    st = pk.static
    st.setdefault("texture_arrays", {})
    st.setdefault("textures", {})
    if sim is not None and layout is not None:
        _soft_materials(st, frame, layout, sim, look, items)
    objects, replaced = scan_objects(model, sim, look, items) if sim is not None else ([], set())
    if not objects:
        return pk, frame, binding
    extra = _pack_meshes(st, objects)
    st["materials_used_by_meshes"] = sorted(set(st["materials_used_by_meshes"]) | {m.meta["materials"][0]
                                                                                    for m in extra})
    body, local, present = binding
    worlds = body.shape[1]
    present = present.copy()
    for k, m in enumerate(frame.meshes):
        if m.meta.get("name") in replaced:
            present[k] = False  # the flat display mesh is drawn by its scan parts instead
    n_dyn = (0 if layout is None else layout.worlds * len(layout.objects)) + \
        (0 if liquid is None else liquid.worlds * len(liquid.objects))
    at = len(frame.meshes) - n_dyn
    rows = _body_rows(model, objects, worlds)
    ident = np.tile(np.asarray([0, 0, 0, 0, 0, 0, 1], np.float32), (len(objects), worlds, 1))
    frame.meshes[at:at] = extra
    binding = (np.concatenate([body[:at], rows, body[at:]]).astype(np.int32),
               np.concatenate([local[:at], ident, local[at:]]).astype(np.float32),
               np.concatenate([present[:at], np.ones(rows.shape, bool), present[at:]]))
    log.debug("camera: %d scan parts replace %d display meshes", len(objects), len(replaced))
    return pk, frame, binding


def _soft_materials(st: dict, frame, layout, sim, look, items) -> None:
    """Scanned soft bodies (``scan_soft``: DeformObjects with ``uv`` / ``image``): their per-world meshes get the UVs
    (Blender convention, per corner) and a generic material with the photo, unless a look names the object."""
    from .card_looks import look_over_scan
    scene = getattr(sim, "scene", None)
    for o in layout.objects:
        if o.image is None:
            continue
        card = ((getattr(scene, "items", None) or {}).get(o.name) or {}).get("card_name")
        if look_over_scan([o.name], card, look, items) is not None:
            continue
        key = f"scan_soft:{o.name}"
        st["texture_arrays"][f"array:{key}"] = o.image
        st["textures"][key] = {"file": f"array:{key}", "colorspace": "sRGB"}
        st["materials"][key] = {"generic": {"base_color": [1.0, 1.0, 1.0], "roughness": 0.5, "metallic": 0.0,
                                            "texture": key, "transmission": 0.0, "ior": 1.45}}
        corner = np.stack([o.uv[:, 0], 1.0 - o.uv[:, 1]], axis=1)[o.tris].astype(np.float32)  # (T, 3, 2), v up
        for m in frame.meshes:
            if m.meta.get("dynamic") and m.meta.get("object_name") == o.name:
                m.uv = {"UVMap": corner}
                m.meta["materials"] = [key]
    st["materials_used_by_meshes"] = sorted({m.meta["materials"][0] for m in frame.meshes})
