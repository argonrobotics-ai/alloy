# SPDX-License-Identifier: Apache-2.0
"""Objaverse-specific mesh checks: real textures, one object (not a scene), no display base, sane proportions.

Sketchfab uploads often hold a showroom: several objects, a ground plane or a plinth, or untextured clay renders.
These checks run on the loaded GLB scene (Y-up glTF, arbitrary units) before the shared repair/hull pipeline and
reject such uploads. Needs the optional ``trimesh`` dependency (``pip install alloy[import]``).

Example:
    >>> import numpy as np
    >>> from alloy.importers.objaverse.quality import clusters_of_boxes
    >>> boxes = np.array([[[0, 0, 0], [1, 1, 1]], [[0.5, 0.5, 0.5], [2, 2, 2]], [[5, 5, 5], [6, 6, 6]]], float)
    >>> clusters_of_boxes(boxes, margin=0.0)
    [[0, 1], [2]]
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from alloy.importers.objaverse.errors import ImportRejected

log = logging.getLogger("alloy.importers.objaverse")

#: glTF is Y-up; Alloy is Z-up: rotate +90 deg about X.
Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class QualityLimits:
    """Thresholds for :func:`check_scene` (fractions are of total surface area unless stated)."""

    min_textured_area: float = 0.6       # area fraction with a varied image texture
    min_texture_std: float = 4.0         # grey-level std (0-255) below which a texture counts as flat colour
    significant_area: float = 0.05       # a component smaller than this is a detail, not a separate object
    max_aspect: float = 40.0             # longest / shortest bbox side (degenerate sheets and sticks)
    base_flatness: float = 0.03          # a bottom component thinner than this x its footprint is a plinth/plane
    base_footprint: float = 0.8          # ...and covering this fraction of the object footprint


@dataclass(frozen=True)
class QualityReport:
    """What :func:`check_scene` measured (JSON-able via ``as_row``)."""

    textured_area: float
    n_components: int
    n_clusters: int
    has_base: bool
    aspect: float
    faces: int
    extents_raw: tuple[float, float, float]

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def clusters_of_boxes(boxes: np.ndarray, margin: float) -> list[list[int]]:
    """Group axis-aligned boxes ``(n, 2, 3)`` that overlap (after growing each by ``margin``); union-find."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lo, hi = boxes[:, 0] - margin, boxes[:, 1] + margin
    for i in range(n):
        overlap = np.all((lo[i] <= hi) & (lo <= hi[i]), axis=1)
        for j in np.nonzero(overlap)[0]:
            parent[find(int(j))] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def texture_std(image: Any) -> float:
    """Grey-level standard deviation of a PIL image on a 64x64 thumbnail (0 for a flat colour); vertex-coloured
    meshes use the spread of their vertex colours instead."""
    grey = np.asarray(image.convert("L").resize((64, 64)), dtype=float)
    return float(grey.std())


def _geom_texture_std(geom: Any) -> float:
    vis = getattr(geom, "visual", None)
    if getattr(vis, "kind", None) == "vertex":                    # coloured scans: per-vertex colour variation
        cols = np.asarray(vis.vertex_colors, dtype=float)[:, :3]
        return float((cols @ np.array([0.299, 0.587, 0.114])).std()) if len(cols) else 0.0
    mat = getattr(vis, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None or getattr(vis, "uv", None) is None:
        return 0.0
    try:
        return texture_std(img)
    except (OSError, ValueError) as e:
        log.debug("texture read failed: %s", e)
        return 0.0


def textured_fraction(geoms: list[Any], min_std: float) -> float:
    """Area fraction of the geometries whose base-colour image texture is not a flat colour."""
    areas = np.array([g.area for g in geoms], dtype=float)
    if areas.sum() <= 0:
        return 0.0
    ok = np.array([_geom_texture_std(g) >= min_std for g in geoms])
    return float(areas[ok].sum() / areas.sum())


def _component_boxes(mesh: Any, min_area: float) -> tuple[np.ndarray, int]:
    parts = mesh.split(only_watertight=False)
    big = [p for p in parts if p.area >= min_area]
    if not big:
        big = [mesh]
    return np.array([p.bounds for p in big], dtype=float), len(parts)


def _has_base(boxes: np.ndarray, lim: QualityLimits) -> bool:
    """A flat component at the bottom that spans most of the footprint, with other components above it."""
    if len(boxes) < 2:
        return False
    lo, hi = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
    foot = max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1e-12)
    for b in boxes:
        size = b[1] - b[0]
        flat = size[2] <= lim.base_flatness * max(size[0], size[1])
        wide = size[0] * size[1] >= lim.base_footprint * foot
        bottom = b[0][2] <= lo[2] + 1e-6 * max(hi - lo)
        if flat and wide and bottom:
            return True
    return False


def measure(geoms: list[Any], lim: QualityLimits = QualityLimits()) -> QualityReport:
    """Measure Z-up world-space geometries (``trimesh.Trimesh`` list with visuals)."""
    import trimesh  # noqa: PLC0415 - optional dependency

    whole = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False) for g in geoms])
    whole.merge_vertices()
    boxes, n_parts = _component_boxes(whole, lim.significant_area * whole.area)
    ext = np.sort(whole.extents)[::-1]
    margin = 0.02 * float(ext[0])
    return QualityReport(textured_area=round(textured_fraction(geoms, lim.min_texture_std), 4),
                         n_components=n_parts, n_clusters=len(clusters_of_boxes(boxes, margin)),
                         has_base=_has_base(boxes, lim), aspect=float(ext[0] / max(ext[2], 1e-12)),
                         faces=int(len(whole.faces)), extents_raw=tuple(float(x) for x in ext))


def check_report(uid: str, rep: QualityReport, lim: QualityLimits = QualityLimits(), flat_ok: bool = False) -> None:
    """Raise :class:`ImportRejected` (stage ``quality``) when a report fails a limit; ``flat_ok`` for plates etc."""
    if rep.textured_area < lim.min_textured_area:
        raise ImportRejected(uid, "quality", f"only {rep.textured_area:.0%} of the surface has a real texture "
                                             f"(need {lim.min_textured_area:.0%})")
    if rep.n_clusters > 1:
        raise ImportRejected(uid, "quality", f"{rep.n_clusters} separate objects (a scene, not one object)")
    if rep.has_base:
        raise ImportRejected(uid, "quality", "sits on a display base or ground plane")
    if rep.aspect > lim.max_aspect and not flat_ok:
        raise ImportRejected(uid, "quality", f"aspect {rep.aspect:.0f} > {lim.max_aspect:.0f} (degenerate)")


def load_zup(glb_path: str) -> list[Any]:
    """World-space, Z-up geometries of a GLB (scene graph transforms applied, visuals kept)."""
    import trimesh  # noqa: PLC0415

    scene = trimesh.load(glb_path, force="scene", process=False)
    geoms = [g for g in scene.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not geoms:
        raise ImportRejected(str(glb_path), "quality", "no triangle geometry in the GLB")
    for g in geoms:
        g.apply_transform(Y_UP_TO_Z_UP)
    return geoms
