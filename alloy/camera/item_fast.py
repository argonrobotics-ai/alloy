# SPDX-License-Identifier: Apache-2.0
"""Scene item looks (``alloy.vary``'s texture swaps and tints, ``item_looks``) drawn by the fast diffuse tracer.

The full-material tracer draws an item look as Blender would. The fast tracer, which a Gym environment's policy
cameras use (``camera_scene(sim, quality="fast")``), draws an approximation of the same look on each shape the look
layer assigns:

- an untextured material: its base colour (sRGB, made linear as the full tracer does) times its colour gain;
- a textured one: its colour map times the colour gain (the texture replaces the base colour, as in the full
  tracer), tiled every ``scale_m`` metres in the shape's frame by a planar projection per triangle along the axis
  its normal is closest to (the full tracer blends the three projections; on flat faces they agree). A colour map
  the camera cannot read here (not cached, an EXR, no Pillow) draws the flat tone of its texture family
  (``alloy.camera.item_looks.flat_tone``; the preset colour without a family), with one log line.

Not drawn: bump, roughness, metal, the ``tint_var`` noise and the other details of the full tracer. Lights stay the
fast tracer's own, so a policy image and a full render of the same pose differ in shading as well (the Gym laws in
``tests/envs/test_camera_looks.py`` state by how much).

Example::

    from alloy.camera.from_newton import scene_from_model
    from alloy.camera.item_looks import items_look
    sd = with_item_looks(scene_from_model(sim.engine.model), items_look(sim))
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from .scene import MeshGeom, SceneData

log = logging.getLogger(__name__)

_AXES = ((1, 2), (0, 2), (0, 1))  # (u, v) coordinates of the projection along x, y, z (Blender's triplanar order)


def with_item_looks(sd: SceneData, items: dict | None) -> SceneData:
    """``sd`` with each instance an item look assigns drawn in that look's colour or texture (module docstring);
    ``sd`` itself without item looks."""
    if not items or not items.get("assign"):
        return sd
    from .card_looks import assigned
    meshes, textures = list(sd.meshes), list(sd.textures)
    made: dict[tuple[int, str], int] = {}
    tex_index: dict[str, int] = {}
    mesh_of, color, textured = sd.inst_mesh.copy(), sd.inst_color.copy(), sd.inst_textured.copy()
    for i, name in enumerate(sd.inst_name):
        key = assigned(items, [name])
        if key is None:
            continue
        rgb, image, scale = _material(key, items["materials"][key])
        color[i], textured[i] = rgb, image is not None
        if image is None:
            continue
        if image[1] not in tex_index:
            tex_index[image[1]] = len(textures)
            textures.append(image[0])
        src = int(sd.inst_mesh[i])
        if (src, key) not in made:
            made[(src, key)] = len(meshes)
            meshes.append(projected(sd.meshes[src], scale, tex_index[image[1]], f"{sd.meshes[src].name}:{key}"))
        mesh_of[i] = made[(src, key)]
    return replace(sd, meshes=meshes, inst_mesh=mesh_of, inst_color=color, inst_textured=textured,
                   textures=textures)


def _material(key: str, spec: dict) -> tuple[np.ndarray, tuple[np.ndarray, str] | None, float]:
    """(linear RGB that multiplies the image, or the flat colour: the base colour, sRGB, made linear as the full
    tracer does; (sRGB image, key) or None; tile size in metres)."""
    from .item_looks import flat_tone
    from .looks.material import rgba
    from .looks.spec import resolve_spec
    eff = resolve_spec(key, spec)
    gain = float(eff.get("color_gain", 1.0))
    image = _color_map(str(eff["texture"]), str(eff.get("res", "2k")), key) if eff.get("texture") else None
    if image is not None:
        return np.full(3, gain, np.float32), image, float(eff.get("scale_m") or 1.0)
    tone = flat_tone({**spec, "texture": eff["texture"]}) if eff.get("texture") else None   # not the white placeholder
    base = np.asarray(rgba(tone or eff.get("base_color", (0.8, 0.8, 0.8)))[:3], np.float64)  # sRGB, as the full tracer
    return (base * gain).astype(np.float32), None, 1.0


def _color_map(ref: str, res: str, key: str) -> tuple[np.ndarray, str] | None:
    """(uint8 sRGB image, cache key) of texture ``ref``'s colour map from the local asset cache, or None."""
    from .errors import CameraError
    from .looks.textures import local_texture_files
    from .scan_textures import decode_image
    try:
        path = Path(local_texture_files(ref, res)["color"])
        if path.suffix.lower() == ".exr":
            raise CameraError(f"{path.name} is an EXR", "the fast tracer reads 8-bit colour maps")
        out = decode_image(path.read_bytes())
    except (CameraError, KeyError, OSError) as exc:
        log.info("camera: item look %s drawn in a flat colour on the fast tracer (%s)", key, exc)
        return None
    return out


def projected(mesh: MeshGeom, scale_m: float, texture: int, name: str) -> MeshGeom:
    """A copy of ``mesh`` with three vertices per triangle and UVs from the planar projection along the axis each
    triangle's normal is closest to, one texture tile every ``scale_m`` metres (v down, as ``MeshGeom.uv``)."""
    p = mesh.points[mesh.tris].astype(np.float64)                 # (T, 3 corners, 3)
    axis = np.argmax(np.abs(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])), axis=1)
    uv = np.empty(p.shape[:2] + (2,), np.float64)
    for a, (u, v) in enumerate(_AXES):
        sel = axis == a
        uv[sel, :, 0], uv[sel, :, 1] = p[sel, :, u], -p[sel, :, v]
    n = len(mesh.tris)
    return MeshGeom(name, p.reshape(-1, 3), np.arange(3 * n, dtype=np.int32).reshape(n, 3),
                    (uv / max(scale_m, 1e-6)).reshape(-1, 2), texture)
