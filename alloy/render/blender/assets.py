# SPDX-License-Identifier: Apache-2.0
"""CC0 asset cache: Poly Haven HDRIs and texture sets, ambientCG texture sets (stdlib only, no bpy).

Every failure is logged and returns None, so a farm render falls back to a procedural material or a studio world
instead of crashing on a texture.
"""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

USER_AGENT = "alloy-render/1.0"
TIMEOUT_S = 60
PH_FILES = "https://api.polyhaven.com/files/{id}"
ACG_ZIP = "https://ambientcg.com/get?file={id}_{res}-JPG.zip"
# Poly Haven map key -> our role, with preferred formats.
PH_MAPS = {"color": ("Diffuse", ("jpg", "png")), "roughness": ("Rough", ("jpg", "png")),
           "height": ("Displacement", ("png", "jpg"))}
ACG_MAPS = {"color": "_Color.", "roughness": "_Roughness.", "height": "_Displacement."}
RES_ORDER = ("1k", "2k", "4k", "8k")


def default_cache() -> Path:
    """The render asset cache (both renderers and ``python -m alloy.looks --fetch-textures`` use it):
    ``$ALLOY_RENDER_ASSETS`` if set, else ``$ALLOY_CACHE/render`` (Alloy's cache, which ``alloy fetch`` uses too),
    else ``~/.cache/alloy-render``."""
    for var, sub in (("ALLOY_RENDER_ASSETS", ""), ("ALLOY_CACHE", "render")):
        raw = os.environ.get(var, "").strip()
        if raw:
            return Path(raw).expanduser() / sub
    return Path.home() / ".cache" / "alloy-render"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _pick_res(available: dict, want: str) -> str | None:
    if want in available:
        return want
    order = [r for r in RES_ORDER if r in available]
    lower = [r for r in order if RES_ORDER.index(r) < RES_ORDER.index(want)] if want in RES_ORDER else order
    return (lower or order or [None])[-1]


def split_ref(ref: str) -> tuple[str, str]:
    """'polyhaven:id' -> ('polyhaven', 'id'); 'file:rel' or a bare path -> ('file', path)."""
    if ref.startswith("file:"):
        return "file", ref[len("file:"):]
    if ":" in ref and not ref.startswith("/"):
        src, ident = ref.split(":", 1)
        return src.lower(), ident
    return "file", ref


class AssetCache:
    """Downloads CC0 assets on first use into `root`, then serves them from disk."""

    def __init__(self, root: str | Path | None = None, bundle_root: str | Path | None = None):
        self.root = Path(root) if root else default_cache()
        self.bundle_root = Path(bundle_root) if bundle_root else None
        self._ph_index: dict[str, dict] = {}

    def _ph_files(self, ident: str) -> dict | None:
        if ident not in self._ph_index:
            cached = self.root / "polyhaven" / ident / "files.json"
            if cached.exists():
                self._ph_index[ident] = json.loads(cached.read_text())
            else:
                data = _fetch(PH_FILES.format(id=ident))
                _write_atomic(cached, data)
                self._ph_index[ident] = json.loads(data)
        return self._ph_index[ident]

    def _ph_download(self, ident: str, url: str) -> Path:
        path = self.root / "polyhaven" / ident / url.rsplit("/", 1)[-1]
        if not path.exists():
            log.info("downloading %s", url)
            _write_atomic(path, _fetch(url))
        return path

    def hdri(self, ref: str, res: str = "4k") -> Path | None:
        """Local .hdr/.exr path of an HDRI reference, or None (logged) if it cannot be had."""
        src, ident = split_ref(ref)
        try:
            if src == "file":
                return Path(ident) if Path(ident).exists() else None
            if src != "polyhaven":
                raise ValueError(f"HDRIs come from polyhaven:<id>, got {ref!r}")
            files = self._ph_files(ident)["hdri"]
            r = _pick_res(files, res)
            fmt = "hdr" if "hdr" in files[r] else "exr"
            return self._ph_download(ident, files[r][fmt]["url"])
        except Exception as exc:  # noqa: BLE001 - never crash a render for an asset
            log.warning("HDRI %s unavailable (%s); using the studio fallback world", ref, exc)
            return None

    def model(self, ref: str, res: str = "1k") -> Path | None:
        """Local .gltf of a Poly Haven model (with its .bin and textures beside it), or None (logged)."""
        src, ident = split_ref(ref)
        try:
            if src != "polyhaven":
                raise ValueError(f"models come from polyhaven:<id>, got {ref!r}")
            files = self._ph_files(ident)["gltf"]
            entry = files[_pick_res(files, res)]["gltf"]
            folder = self.root / "polyhaven" / ident / f"gltf_{res}"
            main = folder / entry["url"].rsplit("/", 1)[-1]
            for rel, inc in entry.get("include", {}).items():
                if not (folder / rel).exists():
                    _write_atomic(folder / rel, _fetch(inc["url"]))
            if not main.exists():
                log.info("downloading model %s", entry["url"])
                _write_atomic(main, _fetch(entry["url"]))
            return main
        except Exception as exc:  # noqa: BLE001 - a missing prop is skipped, never fatal
            log.warning("model %s unavailable (%s); skipping that prop", ref, exc)
            return None

    def texture(self, ref: str, res: str = "2k") -> dict[str, Path] | None:
        """Role -> local image path ({color, roughness, height}) of a texture set, or None (logged)."""
        src, ident = split_ref(ref)
        try:
            if src == "file" and ref.startswith("file:") and self.bundle_root is not None:
                path = self.bundle_root / ref[len("file:"):]
                if not path.is_file():
                    raise FileNotFoundError(path)
                return {"color": path}
            if src == "polyhaven":
                return self._ph_texture(ident, res)
            if src == "ambientcg":
                return self._acg_texture(ident, res)
            raise ValueError(f"textures come from polyhaven:<id> or ambientcg:<id>, got {ref!r}")
        except Exception as exc:  # noqa: BLE001 - never crash a render for an asset
            log.warning("texture %s unavailable (%s); using the procedural material", ref, exc)
            return None

    def _ph_texture(self, ident: str, res: str) -> dict[str, Path]:
        files = self._ph_files(ident)
        out: dict[str, Path] = {}
        for role, (key, fmts) in PH_MAPS.items():
            if key not in files:
                continue
            r = _pick_res(files[key], res)
            fmt = next((f for f in fmts if f in files[key][r]), None)
            if fmt:
                out[role] = self._ph_download(ident, files[key][r][fmt]["url"])
        if "color" not in out:
            raise ValueError(f"polyhaven:{ident} has no Diffuse map")
        return out

    def _acg_texture(self, ident: str, res: str) -> dict[str, Path]:
        res_u = res.upper()
        folder = self.root / "ambientcg" / f"{ident}_{res_u}"
        if not (folder / ".done").exists():
            log.info("downloading ambientcg %s %s", ident, res_u)
            data = _fetch(ACG_ZIP.format(id=ident, res=res_u))
            folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    if any(tag in name for tag in ACG_MAPS.values()) and "/" not in name:
                        (folder / name).write_bytes(zf.read(name))
            (folder / ".done").write_text("ok")
        out = {}
        for role, tag in ACG_MAPS.items():
            hit = sorted(p for p in folder.iterdir() if tag in p.name)
            if hit:
                out[role] = hit[0]
        if "color" not in out:
            raise ValueError(f"ambientcg:{ident} has no Color map")
        return out
