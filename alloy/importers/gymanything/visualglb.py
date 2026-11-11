# SPDX-License-Identifier: Apache-2.0
"""``visual.glb`` from Gym-Anything's render parts: one glTF mesh per part with the PBR values their render code sets
(base colour, roughness, metallic, generated label art as a base-colour texture). Shared importer format
(``alloy.importers.common.visual.export_glb``: glTF +Y up; the body frame is +Z up). Needs trimesh and Pillow.

Glass parts (``name`` contains "glass") carry alpha 0.25 so viewers and the thumbnail tool can tell them apart.

Example::

    path = write_visual_glb(spec, Path("gar"))        # gar/meshes/<card>/visual.glb, or None without parts
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from alloy.cards import geometry as geo
from alloy.importers.gymanything.spec import GarSpec, VisualPart

log = logging.getLogger("alloy.importers.gymanything")

GLASS_ALPHA = 0.25


def srgb_to_linear(c) -> list[float]:
    c = np.clip(np.asarray(c, float), 0, 1)
    return [float(x) for x in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)]


def read_obj_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(vertices, faces, per-vertex uv or None) of an exporter part OBJ (``f a/a b/b c/c`` when it has UVs)."""
    v, vt, f = [], [], []
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            v.append([float(x) for x in t[1:4]])
        elif t[0] == "vt":
            vt.append([float(x) for x in t[1:3]])
        elif t[0] == "f":
            f.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    uv = np.asarray(vt, float) if vt and len(vt) == len(v) else None
    return np.asarray(v, float), np.asarray(f, np.int64), uv


def _part_mesh(p: VisualPart):
    import trimesh  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    v, f, uv = read_obj_uv(p.mesh)
    glass = "glass" in p.name.lower()
    factor = [*srgb_to_linear(p.base_color), GLASS_ALPHA if glass else 1.0]
    mat = trimesh.visual.material.PBRMaterial(
        name=p.name or p.mesh.stem, baseColorFactor=factor, roughnessFactor=p.roughness, metallicFactor=p.metallic,
        baseColorTexture=Image.open(p.texture).convert("RGB") if p.texture and uv is not None else None,
        alphaMode="BLEND" if glass else None)
    vis = trimesh.visual.TextureVisuals(uv=uv if p.texture else None, material=mat)
    return trimesh.Trimesh(v, f, visual=vis, process=False)


def write_visual_glb(spec: GarSpec, out: Path) -> Path | None:
    """Write ``meshes/<card>/visual.glb`` from the spec's visual parts (None when it has none)."""
    if not spec.visual:
        return None
    import trimesh  # noqa: PLC0415

    from alloy.importers.common.visual import export_glb  # noqa: PLC0415

    scene = trimesh.Scene()
    for i, p in enumerate(spec.visual):
        scene.add_geometry(_part_mesh(p), node_name=f"part{i}_{p.name or p.mesh.stem}"[:60], geom_name=f"part{i}")
    path = out / "meshes" / spec.card_name / "visual.glb"
    path.parent.mkdir(parents=True, exist_ok=True)
    export_glb(scene, path)
    log.debug("%s: visual.glb with %d parts", spec.card_name, len(spec.visual))
    return path


def visual_check(spec: GarSpec, mesh_path: Path) -> dict:
    """Do the visual parts cover the card mesh? Bounding boxes within 5 % of the largest extent."""
    pts = np.concatenate([read_obj_uv(p.mesh)[0] for p in spec.visual])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    m = geo.read_mesh(mesh_path)
    mlo, mhi = m.vertices.min(axis=0), m.vertices.max(axis=0)
    tol = 0.05 * float(np.max(mhi - mlo))
    return {"parts": len(spec.visual), "bbox_ok": bool(np.all(np.abs(lo - mlo) <= tol) and np.all(np.abs(hi - mhi) <= tol))}
