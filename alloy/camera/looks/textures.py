# SPDX-License-Identifier: Apache-2.0
"""Texture files of a look material, without Blender and without network access.

``AssetCache`` (``alloy.render.blender.assets``) downloads Poly Haven / ambientCG sets on first use; here
``LocalAssets`` only reads what is already on disk, with the same layout and choice of maps and resolution.

Example::

    files = LocalAssets("~/.cache/alloy-render").texture("polyhaven:plywood", "2k")
    # {"color": Path(".../plywood_diff_2k.jpg"), "roughness": ..., "height": ...}
    image_entry("color", files["color"])   # ('plywood_diff_2k.jpg', {'file': '.../plywood_diff_2k.jpg', 'colorspace': 'sRGB'})
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from alloy.render.blender.assets import AssetCache, default_cache

from ..errors import CameraError

log = logging.getLogger(__name__)


class LocalAssets(AssetCache):
    """An ``AssetCache`` that never downloads: a missing file raises instead of being fetched."""

    def _ph_files(self, ident: str) -> dict:
        cached = self.root / "polyhaven" / ident / "files.json"
        if not cached.exists():
            raise FileNotFoundError(cached)
        return json.loads(cached.read_text())

    def _ph_download(self, ident: str, url: str) -> Path:
        path = self.root / "polyhaven" / ident / url.rsplit("/", 1)[-1]
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _acg_texture(self, ident: str, res: str) -> dict:
        if not (self.root / "ambientcg" / f"{ident}_{res.upper()}" / ".done").exists():
            raise FileNotFoundError(self.root / "ambientcg" / f"{ident}_{res.upper()}")
        return super()._acg_texture(ident, res)


def local_texture_files(ref: str, res: str, root: str | Path | None = None) -> dict[str, Path]:
    """Role -> image path of texture ``ref`` from the local asset cache (``alloy.render.blender.assets.default_cache``:
    ``$ALLOY_RENDER_ASSETS``, else ``$ALLOY_CACHE/render``, else ~/.cache/alloy-render).

    Raises CameraError, naming the download call, when the set is not cached.
    """
    files = LocalAssets(root or default_cache()).texture(ref, res)
    if not files:
        raise CameraError(f"texture {ref!r} ({res}) is not in the local asset cache",
                          "download it once with alloy.render.blender.assets.AssetCache().texture(ref, res), "
                          "or pass textures={ref: {'color': path, ...}} to material_tree")
    return files


def colorspace_of(role: str, path: Path) -> str:
    """Colour space Blender gives the image: sRGB for colour maps (scene-linear for EXR), Non-Color otherwise."""
    if role != "color":
        return "Non-Color"
    return "Linear Rec.709" if Path(path).suffix.lower() == ".exr" else "sRGB"


def image_entry(role: str, path: str | Path) -> tuple[str, dict]:
    """(texture key, {file, colorspace}) of an image file; the key is the file name, as Blender names the image."""
    path = Path(path)
    return path.name, {"file": str(path), "colorspace": colorspace_of(role, path)}
