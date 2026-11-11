# SPDX-License-Identifier: Apache-2.0
"""Visual review of surveyed Objaverse objects: 3/4-view previews and numbered contact sheets.

Metadata and mesh checks miss wrong LVIS labels (a skyscraper filed under "scraper"), brand labels baked into
textures, props placed on boards, and fantasy items. A reviewer (a person or a vision model) looks at numbered contact
sheets and returns, per number, a JSON verdict (:data:`VERDICT_KEYS`); :func:`apply_verdicts` drops or relabels.

Rendering needs ``pyrender`` and an OpenGL backend (``PYOPENGL_PLATFORM=osmesa`` on headless machines).

Example:
    >>> from alloy.importers.objaverse.review import apply_verdicts
    >>> rows = [{"uid": "a", "category": "pot"}, {"uid": "b", "category": "mug"}]
    >>> v = {"a": {"keep": True, "category": "flowerpot"}, "b": {"keep": False, "why": "brand logo"}}
    >>> [(r["uid"], r["category"], r["review"]) for r in apply_verdicts(rows, v, {"flowerpot", "mug"})]
    [('a', 'flowerpot', 'ok (relabelled from pot)'), ('b', 'mug', 'rejected: brand logo')]
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

#: Keys a reviewer returns per object: keep (bool), why (str, when not kept), category (optional relabel).
VERDICT_KEYS = ("keep", "why", "category")

REVIEW_PROMPT = (
    "Each tile shows one 3D object (number top-left, its claimed category under it). For every tile return JSON "
    '{"<n>": {"keep": true|false, "why": "...", "category": "<better category or omit>"}}. Reject (keep=false) when: '
    "the object is not the claimed category and not another category in the list; it is several objects or a scene "
    "(items on a board/plate/stand count, a single product with its lid or cap does not); a real brand name or logo "
    "is readable; it is a cartoon, toy-like, fantasy or weapon item; it is broken, unrecognisable or mostly "
    "untextured. Keep realistic everyday items even if plain.")


def _camera_pose(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, float]:
    c, r = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2
    d = np.array([1.0, -1.2, 0.8])
    eye = c + d / np.linalg.norm(d) * r * 2.6
    z = (eye - c) / np.linalg.norm(eye - c)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, np.cross(z, x), z, eye
    return pose, r


def render_preview(geoms: list[Any], size: int = 256, bg: float = 0.62) -> Any:
    """A PIL image of Z-up ``trimesh`` geometries from a 3/4 view on a grey background."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import pyrender  # noqa: PLC0415 - optional, batch machines only
    import trimesh  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    scene = pyrender.Scene(bg_color=[bg, bg, bg, 1.0], ambient_light=[0.35] * 3)
    for g in geoms:
        try:
            scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))
        except (ValueError, TypeError, AttributeError):
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(g.vertices, g.faces), smooth=False))
    allv = np.vstack([g.vertices for g in geoms])
    pose, _ = _camera_pose(allv.min(0), allv.max(0))
    scene.add(pyrender.PerspectiveCamera(yfov=np.radians(40)), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    r = pyrender.OffscreenRenderer(size, size)
    try:
        rgba, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        r.delete()
    rgba = np.asarray(rgba)
    if rgba.dtype != np.uint8:                      # some backends return float colours in [0, 1]
        rgba = (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(rgba[..., :4]).convert("RGBA")
    back = Image.new("RGBA", img.size, tuple(int(bg * 255) for _ in range(3)) + (255,))
    return Image.alpha_composite(back, img).convert("RGB")


def contact_sheets(items: list[tuple[str, str, Path]], out: Path, cols: int = 6, rows: int = 6,
                   tile: int = 200) -> list[dict[str, Any]]:
    """Numbered sheets of ``(uid, label, preview path)`` items; returns [{sheet, numbers: {n: uid}}].

    Example:
        >>> contact_sheets([], Path("/tmp/none"))
        []
    """
    if not items:
        return []
    from PIL import Image, ImageDraw  # noqa: PLC0415

    per = cols * rows
    index = []
    for s in range(0, len(items), per):
        sheet = Image.new("RGB", (cols * tile, rows * tile), "white")
        draw = ImageDraw.Draw(sheet)
        numbers = {}
        for k, (uid, label, path) in enumerate(items[s:s + per]):
            n = s + k
            x, y = (k % cols) * tile, (k // cols) * tile
            sheet.paste(Image.open(path).convert("RGB").resize((tile, tile)), (x, y))
            draw.rectangle([x, y, x + tile - 1, y + 24], fill="white")
            draw.text((x + 3, y + 1), f"{n}", fill=(200, 0, 0))
            draw.text((x + 3, y + 12), label[:30], fill=(0, 0, 160))
            numbers[n] = uid
        out.mkdir(parents=True, exist_ok=True)
        name = out / f"sheet_{s // per:03d}.jpg"
        sheet.save(name, quality=85)
        index.append({"sheet": str(name), "numbers": numbers})
    return index


def apply_verdicts(rows: Iterable[Mapping[str, Any]], verdicts: Mapping[str, Mapping[str, Any]],
                   categories: set[str]) -> list[dict[str, Any]]:
    """Rows with a ``review`` field; kept rows may be relabelled to another known category. Unreviewed rows are
    marked ``unreviewed`` (not dropped); a relabel to an unknown category is ignored and noted."""
    out = []
    for row in rows:
        r = dict(row)
        v = verdicts.get(r["uid"])
        if v is None:
            r["review"] = "unreviewed"
        elif not v.get("keep", False):
            r["review"] = f"rejected: {v.get('why', 'no reason given')}"
        else:
            new = v.get("category")
            if new and new != r["category"] and new in categories:
                r["review"] = f"ok (relabelled from {r['category']})"
                r["category"] = new
            else:
                r["review"] = "ok" if not new or new == r["category"] else f"ok (unknown relabel {new!r} ignored)"
        out.append(r)
    return out
