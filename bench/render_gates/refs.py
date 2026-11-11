# SPDX-License-Identifier: Apache-2.0
"""Cycles reference sets for the render gates: one tar per scene, size and kind (PNG or EXR), hashed in a manifest.

The tars live in a bucket or folder named by ``$ALLOY_RENDER_GATES_URI`` (or ``--uri``); git keeps only
``refs_manifest.json`` (file name, sha256, bytes, image count). ``fetch`` downloads, verifies and unpacks them into a
gate root (``<scene>/<frame>/<engine>/...``). Scene packs (kind "pack": the scene description each engine renders,
one per scene, all gate frames) unpack to ``<packs>/<scene>/``. Kind "passes": for every control image rendered without
denoising, ``<res>_<spp>spp_off_passes.exr`` holds Cycles' Combined (the noisy colour), Denoising Albedo and
Denoising Normal (camera space) of one render, for denoiser work; use its own Combined layer with its albedo and
normal (the ``_off.exr`` beside it is the same seed and spp, equal to float noise).

Example:
    >>> m = load_manifest()
    >>> m["schema"]
    'alloy.render_gates.refs/1'
    >>> select(m, scenes=["c3"], sizes=["128x128"], kind="png")  # doctest: +SKIP
    [{'file': 'c3_128x128_png.tar', ...}]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
from collections.abc import Iterable
from pathlib import Path

from .layout import REF_ENGINE, is_control, parse_name

log = logging.getLogger(__name__)
MANIFEST_PATH = Path(__file__).with_name("refs_manifest.json")
URI_ENV = "ALLOY_RENDER_GATES_URI"
KINDS = {"png": (".png", ".json"), "exr": (".exr",), "passes": (".exr",), "pack": ()}
PASSES = "_passes"  # <res>_<spp>spp_off_passes.exr: multilayer Combined + Denoising Albedo + Denoising Normal (camera space)


class RefsError(RuntimeError):
    """A reference set is missing, corrupt or not configured; the message says what to do."""


def sha256(path: Path) -> str:
    """Hex sha256 of a file.

    Example:
        >>> len(sha256(Path(__file__)))
        64
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path | None = None) -> dict:
    """The reference manifest (an empty one if the file does not exist yet).

    Example:
        >>> isinstance(load_manifest()["sets"], list)
        True
    """
    p = path or MANIFEST_PATH
    if not p.exists():
        return {"schema": "alloy.render_gates.refs/1", "sets": []}
    return json.loads(p.read_text())


def _members(root: Path, scene: str, res: str, kind: str) -> list[Path]:
    exts = KINDS[kind]
    out = []
    for p in sorted((root / scene).glob("*/*/*")):
        if p.suffix not in exts or not (p.parent.name == REF_ENGINE or is_control(p.parent.name)):
            continue  # only Cycles folders go into a reference set, never an engine under test
        is_passes = p.stem.endswith(PASSES)
        if is_passes != (kind == "passes"):
            continue
        parsed = parse_name(p.stem.removesuffix(PASSES) + ".png")
        if (parsed and parsed[0] == res) or (p.name == "view.json"):
            out.append(p)
    return out


def pack_set(root: Path, out_dir: Path, scene: str, res: str, kind: str = "png") -> dict | None:
    """Write ``<scene>_<res>_<kind>.tar`` of one reference set and return its manifest entry (None if empty).

    Example:
        >>> pack_set(Path("refs"), Path("tars"), "c3", "128x128")  # doctest: +SKIP
        {'scene': 'c3', 'res': '128x128', 'kind': 'png', 'file': 'c3_128x128_png.tar', 'images': 205, ...}
    """
    files = _members(root, scene, res, kind)
    if not any(f.name != "view.json" for f in files):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    tar = out_dir / f"{scene}_{res}_{kind}.tar"
    with tarfile.open(tar, "w") as tf:
        for f in files:
            tf.add(f, arcname=f.relative_to(root).as_posix())
    return {"scene": scene, "res": res, "kind": kind, "file": tar.name, "sha256": sha256(tar),
            "bytes": tar.stat().st_size, "images": sum(f.name != "view.json" for f in files)}


def pack_scene(src: Path, out_dir: Path, scene: str) -> dict:
    """Write ``<scene>_pack.tar`` of a scene pack folder (members under ``<scene>/``) and return its manifest entry
    (``kind`` "pack", ``res`` "*"). Fetch it with ``kind="pack"`` into a packs folder.

    Example:
        >>> pack_scene(Path("packs/c3"), Path("tars"), "c3")  # doctest: +SKIP
        {'scene': 'c3', 'res': '*', 'kind': 'pack', 'file': 'c3_pack.tar', ...}
    """
    if not src.is_dir():
        raise RefsError(f"pack folder {src} does not exist")
    out_dir.mkdir(parents=True, exist_ok=True)
    tar = out_dir / f"{scene}_pack.tar"
    files = sorted(f for f in src.rglob("*") if f.is_file())
    with tarfile.open(tar, "w") as tf:
        for f in files:
            tf.add(f, arcname=f"{scene}/{f.relative_to(src).as_posix()}")
    return {"scene": scene, "res": "*", "kind": "pack", "file": tar.name, "sha256": sha256(tar),
            "bytes": tar.stat().st_size, "files": len(files)}


def select(manifest: dict, scenes: Iterable[str] | None = None, sizes: Iterable[str] | None = None,
           kind: str = "png") -> list[dict]:
    """Manifest entries for the given scenes, sizes and kind (all when None).

    Example:
        >>> select({"sets": [{"scene": "c3", "res": "128x128", "kind": "png"}]}, ["c3"])
        [{'scene': 'c3', 'res': '128x128', 'kind': 'png'}]
    """
    sc, sz = set(scenes or []), set(sizes or [])
    return [e for e in manifest["sets"] if (not sc or e["scene"] in sc) and (not sz or e["res"] in sz or e["res"] == "*")
            and e["kind"] == kind]


def _copy(uri: str, name: str, dest: Path) -> None:
    src = f"{uri.rstrip('/')}/{name}"
    if uri.startswith("gs://"):
        cmd = ["gcloud", "storage", "cp", src, str(dest)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RefsError(f"could not copy {name} from the reference bucket ({r.stderr.strip()[-300:]}); "
                            f"check {URI_ENV} and your gcloud login")
    else:
        if not Path(src).exists():
            raise RefsError(f"{src} does not exist; set {URI_ENV} (or --uri) to the folder holding the tars")
        shutil.copyfile(src, dest)


def fetch(entries: list[dict], dest: Path, uri: str | None = None, cache: Path | None = None) -> list[str]:
    """Download (or reuse from ``cache``), verify and unpack reference tars into ``dest``. Returns the tar names.

    Example:
        >>> fetch(select(load_manifest(), ["c3"]), Path("gate_root"))  # doctest: +SKIP
        ['c3_128x128_png.tar', 'c3_640x480_png.tar', 'c3_1280x720_png.tar']
    """
    uri = uri or os.environ.get(URI_ENV)
    if not uri:
        raise RefsError(f"no reference location: set {URI_ENV} to the bucket or folder of the tars, or pass --uri")
    cache = cache or dest / ".refs_cache"
    cache.mkdir(parents=True, exist_ok=True)
    done = []
    for e in entries:
        tar = cache / e["file"]
        if not (tar.exists() and sha256(tar) == e["sha256"]):
            _copy(uri, e["file"], tar)
        got = sha256(tar)
        if got != e["sha256"]:
            raise RefsError(f"{e['file']}: sha256 {got[:12]} != manifest {e['sha256'][:12]}; delete it and fetch again")
        with tarfile.open(tar) as tf:
            tf.extractall(dest, filter="data")
        done.append(e["file"])
        log.info("unpacked %s (%d images)", e["file"], e.get("images", 0))
    return done
