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

from alloy.render.blender.assets import RES_ORDER, AssetCache, default_cache

from ..errors import CameraError

log = logging.getLogger(__name__)
_NOTICED: set[tuple[str, str]] = set()  # (set, asked resolution) already reported, so each notice prints once


def _nearest_first(res: str) -> list[str]:
    """The other resolutions of ``RES_ORDER``, nearest to ``res`` first and the larger on a tie (2k: 4k, 1k, 8k);
    largest first for a resolution outside ``RES_ORDER``."""
    alts = [r for r in RES_ORDER if r != res]
    if res not in RES_ORDER:
        return alts[::-1]
    i = RES_ORDER.index
    return sorted(alts, key=lambda r: (abs(i(r) - i(res)), -i(r)))


class LocalAssets(AssetCache):
    """An ``AssetCache`` that never downloads: a missing file raises instead of being fetched. A set that is on disk
    only at other resolutions (the public texture library holds 1k sets, looks ask for 2k by default) is served at
    the nearest resolution there (the larger one on a tie), with a one-line notice."""

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

    def _ph_texture(self, ident: str, res: str) -> dict:
        return self._any_res(self._ph_exact, f"polyhaven:{ident}", ident, res)

    def _ph_exact(self, ident: str, res: str) -> dict:
        if res not in self._ph_files(ident).get("Diffuse", {}):  # else _pick_res serves another res as this one
            raise FileNotFoundError(f"polyhaven:{ident} lists no {res} maps")
        return super()._ph_texture(ident, res)

    def _acg_texture(self, ident: str, res: str) -> dict:
        return self._any_res(self._acg_exact, f"ambientcg:{ident}", ident, res)

    def _acg_exact(self, ident: str, res: str) -> dict:
        if not (self.root / "ambientcg" / f"{ident}_{res.upper()}" / ".done").exists():
            raise FileNotFoundError(self.root / "ambientcg" / f"{ident}_{res.upper()}")
        return super()._acg_texture(ident, res)

    @staticmethod
    def _any_res(get, ref: str, ident: str, res: str) -> dict:
        """``get(ident, res)``, else the set at the nearest other resolution on disk, the larger one on a tie
        (noticed once), else the first FileNotFoundError. Nearest, not largest: an 8k set fetched once must not serve
        every 2k ask at 16x the pixels."""
        try:
            return get(ident, res)
        except FileNotFoundError as missing:
            for alt in _nearest_first(res):
                try:
                    files = get(ident, alt)
                except FileNotFoundError:
                    continue
                if (ref, res) not in _NOTICED:
                    _NOTICED.add((ref, res))
                    log.warning("texture %s: no %s maps in the local asset cache; using the %s maps there", ref, res,
                                alt)
                return files
            raise missing


def local_texture_files(ref: str, res: str, root: str | Path | None = None) -> dict[str, Path]:
    """Role -> image path of texture ``ref`` from the local asset cache (``alloy.render.blender.assets.default_cache``:
    ``$ALLOY_RENDER_ASSETS``, else ``$ALLOY_CACHE/render``, else ~/.cache/alloy-render).

    A set cached only at other sizes is served at the nearest one (``LocalAssets``). Raises CameraError, naming the
    public fetch command, when no size of the set is cached.
    """
    files = LocalAssets(root or default_cache()).texture(ref, res)
    if not files:
        raise CameraError(f"texture {ref!r} ({res}) is not in the local asset cache at any size",
                          fetch_fix(ref))
    return files


def fetch_fix(ref: str) -> str:
    """The fix for texture ``ref`` with no size cached: ``--family <f>`` for a texture-pool set, the full fetch for a
    set that presets or library looks name (the full fetch takes those too), else what that fetch can bring."""
    cmd = "python -m alloy.looks --fetch-textures"
    try:
        from alloy.looks import pool  # noqa: PLC0415 - only on the error path
        family = next((e.family for e in pool.entries() if e.texture == ref), None)
        named = family is None and ref in pool.look_textures()
    except Exception:  # noqa: BLE001 - an unreadable library must not hide the missing-texture error
        family, named = None, True
    if family:
        return f"fetch it with {cmd} --family {family}, or render with quality='fast'"
    if named:
        return f"fetch it with {cmd}, or render with quality='fast'"
    return (f"{cmd} takes only the texture pool's sets and those presets and library looks name: use one of those "
            "(python -m alloy.looks --pool lists the pool), or render with quality='fast'")


def colorspace_of(role: str, path: Path) -> str:
    """Colour space Blender gives the image: sRGB for colour maps (scene-linear for EXR), Non-Color otherwise."""
    if role != "color":
        return "Non-Color"
    return "Linear Rec.709" if Path(path).suffix.lower() == ".exr" else "sRGB"


def image_entry(role: str, path: str | Path) -> tuple[str, dict]:
    """(texture key, {file, colorspace}) of an image file; the key is the file name, as Blender names the image."""
    path = Path(path)
    return path.name, {"file": str(path), "colorspace": colorspace_of(role, path)}
