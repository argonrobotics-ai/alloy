# SPDX-License-Identifier: Apache-2.0
"""Load an object's render geometry as textured parts in its body frame (metres, +Z up).

Sources: ``visual.glb`` (glTF, +Y up; the shared importer format), an MJCF model (RoboCasa: visual geoms with
MuJoCo materials and 2-D textures), or a bare mesh (``physics.obj``) with a flat colour. Every part is returned
unwelded (one vertex per triangle corner), so per-corner UVs need no seams handling.

Example::

    parts = load_visual(Path("meshes/ycb_025_mug/visual.glb"))
    parts[0].texture.shape, parts[0].uv.shape          # (1024, 1024, 3), (3 * T, 2)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("thumbnails")

GLTF_TO_BODY = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)  # +Y up -> +Z up
DEFAULT_COLOR = (0.62, 0.62, 0.62)
MAX_TEXTURE_PX = 2048


class VisualError(ValueError):
    """The visual cannot be loaded; the message names the file and what is wrong."""


@dataclass
class Part:
    """One render part: per-corner geometry (points (3T, 3), tris (T, 3)), optional per-corner ``uv`` (3T, 2) in the
    OpenGL convention (origin bottom-left) with a ``texture`` (H, W, 3|4 uint8), sRGB base colour and PBR scalars."""

    name: str
    points: np.ndarray
    tris: np.ndarray
    uv: np.ndarray | None = None
    texture: np.ndarray | None = None
    base_color: tuple[float, float, float] = DEFAULT_COLOR
    roughness: float = 0.5
    metallic: float = 0.0
    glass: bool = False
    note: str = ""          # why the part looks simpler than its source (e.g. a texture it has no UVs for)


def _corners(v: np.ndarray, f: np.ndarray, uv: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pts = np.asarray(v, np.float32)[np.asarray(f, np.int64)].reshape(-1, 3)
    tris = np.arange(len(pts), dtype=np.int32).reshape(-1, 3)
    return pts, tris, (None if uv is None else np.asarray(uv, np.float32)[np.asarray(f)].reshape(-1, 2))


def _image(img) -> np.ndarray | None:
    if img is None:
        return None
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
    if max(img.size) > MAX_TEXTURE_PX:
        img = img.copy()
        img.thumbnail((MAX_TEXTURE_PX, MAX_TEXTURE_PX))
    return np.asarray(img, np.uint8)


def _srgb(linear) -> tuple[float, float, float]:
    c = np.clip(np.asarray(linear, float)[:3], 0, 1)
    return tuple(float(x) for x in np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055))


def _glb_material(g) -> dict:
    """Base colour (sRGB), texture, roughness, metallic of a trimesh geometry's PBR or simple material."""
    mat = getattr(g.visual, "material", None)
    out: dict = {}
    if mat is None:
        return out
    factor = getattr(mat, "baseColorFactor", None)
    if factor is not None:
        raw = np.asarray(factor)
        f = raw.astype(float) / (255.0 if np.issubdtype(raw.dtype, np.integer) or raw.max() > 1.0 else 1.0)
        out["base_color"] = _srgb(f)                                     # glTF factors are linear
        if len(f) > 3 and f[3] < 0.5:
            out["glass"] = True
    tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    out["texture"] = _image(tex)
    for key, attr in (("roughness", "roughnessFactor"), ("metallic", "metallicFactor")):
        if getattr(mat, attr, None) is not None:
            out[key] = float(getattr(mat, attr))
    return out


def load_glb(path: Path) -> list[Part]:
    """Every geometry of a glTF/GLB scene, scene transforms applied, moved to the +Z-up body frame."""
    import trimesh  # noqa: PLC0415 - the thumbnail env has it; Alloy's does not need it

    scene = trimesh.load(str(path), force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not hasattr(g, "faces") or len(g.faces) == 0:
            continue
        v = trimesh.transform_points(np.asarray(g.vertices), GLTF_TO_BODY @ tf)
        uv = getattr(g.visual, "uv", None)
        info = _glb_material(g)
        if info.get("texture") is None or uv is None:
            uv, info["texture"] = None, None
        pts, tris, cuv = _corners(v, g.faces, uv)
        parts.append(Part(f"{node}", pts, tris, cuv, **info))
    if not parts:
        raise VisualError(f"{path}: no triangle geometry")
    return parts


def load_obj(path: Path, color=DEFAULT_COLOR) -> list[Part]:
    """A bare mesh (body frame) as one flat-colour part (``alloy.cards.geometry.read_mesh``: OBJ or STL)."""
    from alloy.cards import geometry as geo  # noqa: PLC0415

    m = geo.read_mesh(Path(path))
    pts, tris, _ = _corners(m.vertices, m.faces, None)
    return [Part(Path(path).stem, pts, tris, None, None, tuple(color))]


def load_visual(path: Path, **kw) -> list[Part]:
    """Dispatch on the file type: ``.glb``/``.gltf`` -> ``load_glb``, ``.xml`` -> ``mjcf.load_mjcf``, else
    ``load_obj``."""
    suffix = Path(path).suffix.lower()
    if suffix in (".glb", ".gltf"):
        return load_glb(Path(path))
    if suffix == ".xml":
        from mjcf_visual import load_mjcf  # noqa: PLC0415 - sibling module (tools/thumbnails on sys.path)
        return load_mjcf(Path(path))
    return load_obj(Path(path), **kw)


def bounds(parts: list[Part]) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of every part's points."""
    pts = np.concatenate([p.points for p in parts])
    return pts.min(axis=0), pts.max(axis=0)
