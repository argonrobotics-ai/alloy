# SPDX-License-Identifier: Apache-2.0
"""Static set dressing (look.props): CC0 Poly Haven models or bundle files placed by pose, render only.

Each distinct (model, material) is imported once into an off-scene collection and placed by collection-instance
empties, so ten props of one model cost one import. A prop that cannot be fetched or imported is skipped with a
warning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import bpy
import numpy as np

import imports
import meshdata
import motion
from assets import AssetCache
from io_bundle import Bundle
from materials import MaterialLibrary

log = logging.getLogger(__name__)


def resolve(ref: str, bundle: Bundle, assets: AssetCache, res: str) -> Path | None:
    """Local file for 'polyhaven:<id>' or 'file:<bundle-relative path>' (None, logged, if unavailable)."""
    if ref.startswith("file:"):
        path = bundle.root / ref[len("file:"):]
        if path.is_file() and path.suffix.lower() in imports.IMPORTERS:
            return path
        log.warning("prop %s: no .glb/.gltf/.obj file at %s; skipping it", ref, path)
        return None
    return assets.model(ref, res)


def _override_material(coll: bpy.types.Collection, mat: bpy.types.Material) -> None:
    for ob in coll.all_objects:
        if ob.type == "MESH":
            ob.data.materials.clear()
            ob.data.materials.append(mat)


def build_props(look, bundle: Bundle, assets: AssetCache, mats: MaterialLibrary,
                coll: bpy.types.Collection) -> int:
    """Place every look.props entry; returns how many were placed."""
    sources: dict[tuple, bpy.types.Collection | None] = {}
    placed = 0
    for i, prop in enumerate(look.raw.get("props") or []):
        key = (prop["model"], prop.get("material"))
        if key not in sources:
            path = resolve(prop["model"], bundle, assets, prop.get("res", "1k"))
            sources[key] = imports.into_collection(path, f"prop_src_{len(sources)}") if path else None
            if sources[key] is not None and prop.get("material"):
                _override_material(sources[key], mats.get(prop["material"]))
        if sources[key] is None:
            continue
        empty = bpy.data.objects.new(f"prop_{i}", None)
        empty.instance_type = "COLLECTION"
        empty.instance_collection = sources[key]
        pose = np.asarray(prop.get("pose", [0, 0, 0, 0, 0, 0, 1]), float)
        empty.location = pose[:3]
        empty.rotation_mode = "QUATERNION"
        empty.rotation_quaternion = motion.xyzw_to_wxyz(motion.normalize_quat(pose[3:7]))
        scale = prop.get("scale", 1.0)
        empty.scale = (scale,) * 3 if isinstance(scale, (int, float)) else tuple(scale)
        coll.objects.link(empty)
        placed += 1
    log.info("props: %d placed", placed)
    return placed


def build_ground(look, bundle: Bundle, mats: MaterialLibrary, coll: bpy.types.Collection) -> bpy.types.Object | None:
    """look.ground {"z": -0.75, "material": "floor", "size_m": 4000, "center": [x, y]}: one render-only plane under
    every world. Per-world floors in a grid overlap and shadow each other (black patches), so hide them and use this.

    `center` defaults to the middle of the bundle's world offsets."""
    cfg = look.raw.get("ground")
    if not cfg:
        return None
    half = 0.5 * float(cfg.get("size_m", 4000.0))
    offs = np.array(list(bundle.offsets.values()) or [np.zeros(3)])
    cx, cy = cfg.get("center") or (0.5 * (offs[:, 0].min() + offs[:, 0].max()), 0.5 * (offs[:, 1].min() + offs[:, 1].max()))
    verts = np.array([[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]], np.float32)
    me = meshdata.make_mesh("ground", verts, np.arange(4, dtype=np.int32), np.array([4], np.int32), 0.0)
    me.materials.append(mats.get(cfg["material"]))
    ob = meshdata.new_object("ground", me, coll)
    ob.location = (float(cx), float(cy), float(cfg["z"]))
    log.info("ground: %.0f m plane at (%.2f, %.2f, %.4f), material %s", 2 * half, cx, cy, float(cfg["z"]),
             cfg["material"])
    return ob
