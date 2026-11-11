# SPDX-License-Identifier: Apache-2.0
"""Scanned library objects drawn with their own photo texture (the ``visual.glb`` beside the card's physics mesh).

A scanned object card (YCB, Google Scanned Objects, Objaverse, ...: ``alloy fetch``) ships ``meshes/<card>/visual.glb``
next to ``physics.obj``: the scan's own mesh with UVs and a base-colour image. The scene draws the object by a
display mesh made from the physics mesh (``<item>/visual``, no UVs). For an ``alloy.Sim`` the camera swaps that display
mesh for the glb's meshes, UVs and images (one mesh per glb primitive, on the same body), for both tracers:

- the fast tracer reads ``MeshGeom.uv`` / ``MeshGeom.texture`` / ``SceneData.textures`` (:func:`with_scan_textures`);
- the full tracer gets the same parts as textured meshes riding the bodies (:func:`scan_objects`).

Frame: a glb is +Y up; the body frame is (x, -z, y) of it (``alloy.importers.common.visual.Z_UP_TO_GLTF``), with the
card's mesh scale. A glb whose box does not match the display mesh's (2 % + 2 mm per axis) is not used. Colour: the
image times the glb's ``baseColorFactor``, except trimesh's default (0.4, 0.4, 0.4) beside an image, which the scan
importers leave when the scan's MTL has no Kd and which stands for 1. Without Pillow (the ``import`` extra, which every
fetched scan needs) or a usable glb, the object keeps its flat colour, with one log line; a scene never fails here.

Example::

    sd = with_scan_textures(scene_from_model(sim.engine.model), sim)   # what camera_scene(sim) draws
    [m.name for m in sd.meshes if m.texture >= 0]                       # ["mug/visual:0", ...]
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .glb import GlbError, GlbPart, read_glb
from .scene import MeshGeom, SceneData, transform_points

log = logging.getLogger(__name__)

VISUAL = "visual.glb"
TEXTURE_MAX_ENV = "ALLOY_CAMERA_TEXTURE_MAX"
TEXTURE_MAX = 2048  # long side, texels; larger images are box-filtered down at load
TRIMESH_DEFAULT_FACTOR = (0.4, 0.4, 0.4)  # trimesh's diffuse for an MTL without Kd (stands for "no factor")
FIT_REL, FIT_ABS = 0.02, 0.002  # box match of a glb against the display mesh: 2 % of the extent + 2 mm per axis
_IMAGES: dict[str, np.ndarray] = {}  # decoded images by sha256 of their bytes (shared by every scene)
_IMAGES_MAX = 64
_SAID: set[str] = set()


@dataclass
class ScanPart:
    """One glb primitive in the display mesh's frame: points (V, 3), tris (T, 3), uv (V, 2) glTF convention or None,
    image uint8 (H, W, 3) sRGB (row 0 = top) or None, linear RGB colour (multiplies the image), roughness, metallic."""

    points: np.ndarray
    tris: np.ndarray
    uv: np.ndarray | None
    image: np.ndarray | None
    image_key: str | None
    color: tuple
    roughness: float
    metallic: float


def _say(key: str, msg: str, *args) -> None:
    if key not in _SAID:
        _SAID.add(key)
        log.warning(msg, *args)


def texture_max() -> int:
    """Long-side cap for scan images (``ALLOY_CAMERA_TEXTURE_MAX``, default 2,048)."""
    try:
        return max(16, int(os.environ.get(TEXTURE_MAX_ENV, TEXTURE_MAX)))
    except ValueError:
        return TEXTURE_MAX


def decode_image(data: bytes) -> tuple[np.ndarray, str] | None:
    """(uint8 (H, W, 3) sRGB image, sha256 key) of encoded image bytes, long side capped; None without Pillow."""
    key = hashlib.sha256(data).hexdigest()[:16]
    if key in _IMAGES:
        return _IMAGES[key], key
    try:
        from PIL import Image  # noqa: PLC0415 - the import extra; every fetched scan comes with it
    except ImportError:
        _say("pillow", "camera: scanned objects are drawn without their photo texture: Pillow is missing "
                       "(pip install pillow, or the import extra: pip install 'alloy-engine[import]')")
        return None
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        cap = texture_max()
        if max(im.size) > cap:
            s = cap / max(im.size)
            im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BOX)
        arr = np.ascontiguousarray(np.asarray(im, np.uint8))
    if len(_IMAGES) >= _IMAGES_MAX:
        _IMAGES.pop(next(iter(_IMAGES)))
    _IMAGES[key] = arr
    return arr, key


def part_color(part: GlbPart) -> tuple:
    """Linear RGB that multiplies a part's image (or is its colour without one)."""
    rgb = tuple(float(c) for c in part.factor[:3])
    if part.image is not None and np.allclose(rgb, TRIMESH_DEFAULT_FACTOR, atol=1e-6):
        return (1.0, 1.0, 1.0)
    return rgb


def to_body(points: np.ndarray, scale) -> np.ndarray:
    """glb (+Y up) points in the body frame: (x, -z, y), times the card's mesh scale."""
    p = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)
    return (p * np.asarray(scale, np.float32).reshape(1, 3)).astype(np.float32)


def fit_offset(points: np.ndarray, display: np.ndarray) -> np.ndarray | None:
    """Shift that centres ``points``' box on ``display``'s, or None when the box sizes differ (2 % + 2 mm per axis)."""
    lo, hi, dlo, dhi = points.min(0), points.max(0), display.min(0), display.max(0)
    size, dsize = hi - lo, dhi - dlo
    if (np.abs(size - dsize) > FIT_REL * np.maximum(size, dsize) + FIT_ABS).any():
        return None
    return (0.5 * (dlo + dhi) - 0.5 * (lo + hi)).astype(np.float32)


def scan_parts(glb: Path, display: MeshGeom, local: np.ndarray, scale=(1.0, 1.0, 1.0)) -> list[ScanPart] | None:
    """The glb's primitives in the frame of ``display`` (whose instance pose in the body frame is ``local``), or None
    (one log line says why) when the glb is not used. Example: ``scan_parts(path, sd.meshes[k], sd.inst_local[i])``."""
    try:
        raw = read_glb(glb)
    except (GlbError, OSError, KeyError, IndexError, ValueError) as exc:
        _say(f"read:{glb}", "camera: %s is not used (%s); the object keeps its flat colour", glb, exc)
        return None
    body = [to_body(p.points, scale) for p in raw]
    inv = _inverse(local)
    shape = [transform_points(inv, b) for b in body]
    shift = fit_offset(np.concatenate(shape), display.points)
    if shift is None:
        _say(f"fit:{glb}", "camera: %s does not match its object's mesh (box sizes differ); the object keeps its flat "
                           "colour", glb)
        return None
    parts = [_part(p, s + shift) for p, s in zip(raw, shape)]
    if any(part is None for part in parts):
        return None  # an image did not decode (Pillow missing): keep the flat colour rather than a white object
    return parts


def _part(p: GlbPart, points: np.ndarray) -> ScanPart | None:
    for note in p.notes:
        _say(f"note:{note}", "camera: scan texture: %s", note)
    if p.image is None or p.uv is None:
        return ScanPart(points.astype(np.float32), p.tris, None, None, None, part_color(p), p.roughness, p.metallic)
    img = decode_image(p.image.data)
    if img is None:
        return None
    return ScanPart(points.astype(np.float32), p.tris, p.uv, img[0], img[1], part_color(p), p.roughness, p.metallic)


def _inverse(xf: np.ndarray) -> np.ndarray:
    """Inverse of a (7,) pose (px, py, pz, qx, qy, qz, qw)."""
    p, q = np.asarray(xf[:3], np.float64), np.asarray(xf[3:7], np.float64)
    qi = np.array([-q[0], -q[1], -q[2], q[3]])
    u, w = qi[:3], qi[3]
    pi = -(p + 2.0 * np.cross(u, np.cross(u, p) + w * p))
    return np.concatenate([pi, qi]).astype(np.float32)


@dataclass(frozen=True)
class ScanSource:
    """A scene item drawn from a scan: its display instances' label, the glb and the card's mesh scale."""

    label: str
    glb: Path
    scale: tuple


def scan_sources(sim) -> list[ScanSource]:
    """The scanned items of an ``alloy.Sim`` (its scene's card items whose mesh has a ``visual.glb`` beside it);
    [] for any other source. Items with their own look keep it (they are not listed)."""
    from .item_looks import scene_of
    scene = scene_of(sim)
    if scene is None or getattr(sim, "engine", None) is None:
        return []
    cards = {name: spec for name, spec in scene.items.items()
             if isinstance(spec, dict) and spec.get("card_name") and not spec.get("look")}
    if not cards:
        return []
    try:
        from alloy.api import catalog  # noqa: PLC0415
        from alloy.cards.library import CardLibrary  # noqa: PLC0415
        lib = CardLibrary(roots=catalog.card_roots(getattr(scene, "roots", ())))
    except Exception as exc:  # noqa: BLE001 - no library: nothing to texture
        log.info("camera: no card library for scan textures (%s)", exc)
        return []
    found = [src for name, spec in cards.items() if (src := _source(lib, name, spec["card_name"])) is not None]
    from .card_looks import look_over_scan  # the card's own object look, when it names the part, wins (both tracers)
    return [s for s in found if look_over_scan([s.label, s.label.rsplit("/", 1)[0]],
                                               cards[s.label.rsplit("/", 1)[0]]["card_name"]) is None]


def _source(lib, item: str, card_name: str) -> ScanSource | None:
    try:
        shape = lib.object(card_name).shape
        path = lib.asset_path(shape.path) if getattr(shape, "path", None) else None
    except Exception as exc:  # noqa: BLE001 - a card the camera cannot read keeps its flat colour
        log.info("camera: item %s (card %s) has no scan texture (%s)", item, card_name, exc)
        return None
    glb = path.parent / VISUAL if path is not None else None
    if glb is None or not glb.is_file():
        return None
    return ScanSource(f"{item}/visual", glb, tuple(float(v) for v in getattr(shape, "scale", (1.0, 1.0, 1.0))))


def with_scan_textures(sd: SceneData, sim, sources: list[ScanSource] | None = None,
                       info: dict | None = None) -> SceneData:
    """``sd`` with every scanned item of ``sim`` (or only ``sources``) drawn by its glb (module docstring); ``sd``
    itself when there is none. ``info``, when given, receives {scan mesh name: (roughness, metallic)}.
    Example: ``with_scan_textures(scene_from_model(sim.engine.model), sim)``."""
    sources = scan_sources(sim) if sources is None else sources
    if not sources:
        return sd
    meshes, textures, keys = list(sd.meshes), list(sd.textures), {}
    rows = [list(r) for r in zip(sd.inst_mesh, sd.inst_group, sd.inst_body, sd.inst_local, sd.inst_color,
                                 sd.inst_name, sd.inst_textured)]
    for src in sources:
        hits = [i for i, n in enumerate(sd.inst_name) if n == src.label]
        if not hits:
            continue
        parts = scan_parts(src.glb, sd.meshes[sd.inst_mesh[hits[0]]], sd.inst_local[hits[0]], src.scale)
        if parts is None:
            continue
        first = len(meshes)
        for k, part in enumerate(parts):
            meshes.append(MeshGeom(f"{src.label}:{k}", part.points, part.tris, part.uv,
                                   _texture_index(part, textures, keys)))
            if info is not None:
                info[meshes[-1].name] = (part.roughness, part.metallic)
        for i in hits:
            rows[i][0], rows[i][4] = first, np.asarray(parts[0].color, np.float32)
            rows.extend([first + k, *rows[i][1:4], np.asarray(p.color, np.float32), f"{src.label}#{k}", True]
                        for k, p in enumerate(parts[1:], 1))
    m, g, b, x, c, nm, tx = zip(*rows)
    return replace(sd, meshes=meshes, inst_mesh=np.array(m), inst_group=np.array(g), inst_body=np.array(b),
                   inst_local=np.stack(x), inst_color=np.stack(c), inst_name=list(nm), textures=textures,
                   inst_textured=np.array(tx, bool))


def _texture_index(part: ScanPart, textures: list, keys: dict) -> int:
    if part.image is None:
        return -1
    if part.image_key not in keys:
        keys[part.image_key] = len(textures)
        textures.append(part.image)
    return keys[part.image_key]
