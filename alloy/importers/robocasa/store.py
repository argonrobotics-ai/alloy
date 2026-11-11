# SPDX-License-Identifier: Apache-2.0
"""Binary store for imported assets: git keeps cards, model XML and the manifest; meshes and images live elsewhere.

``BINARY_SUFFIXES`` files (meshes, textures, thumbnails) are not committed; ``manifest.json`` lists each with its
SHA-256. ``fetch`` downloads every listed file that is missing or whose hash differs, from a store given as a
``gs://`` prefix (``gcloud storage``), an ``http(s)://`` prefix or a local folder. ``upload_command`` is the
``gcloud storage rsync`` line that publishes a finished import root.

Example::

    >>> from alloy.importers.robocasa.store import is_binary
    >>> is_binary("mjcf/glass_cup/robocasa_glass_cup_003/hulls/hull_00.msh"), is_binary("objects/a/b.yaml")
    (True, False)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path

from alloy.importers.robocasa.pipeline import sha256

log = logging.getLogger("alloy.importers.robocasa")

BINARY_SUFFIXES = (".msh", ".png", ".jpg", ".jpeg", ".obj")


def is_binary(rel: str) -> bool:
    """True for files kept out of git (meshes, textures, thumbnails)."""
    return rel.lower().endswith(BINARY_SUFFIXES)


def wanted(root: Path) -> dict[str, str]:
    """rel path -> sha256 of every binary in ``root/manifest.json`` that is missing or differs locally."""
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no manifest; run the importer's finalize step or pick the import root")
    manifest = json.loads(path.read_text())
    out = {}
    for entry in manifest["assets"].values():
        for rel, meta in (entry.get("files") or {}).items():
            p = root / rel
            if is_binary(rel) and (not p.is_file() or sha256(p) != meta["sha256"]):
                out[rel] = meta["sha256"]
    return out


def _copy_one(store: str, rel: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = f"{store.rstrip('/')}/{rel}"
    if store.startswith("gs://"):
        subprocess.run(["gcloud", "storage", "cp", "--quiet", src, str(dest)], check=True, capture_output=True)
    elif store.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=120) as r, open(dest, "wb") as fh:
            shutil.copyfileobj(r, fh)
    else:
        shutil.copyfile(Path(store) / rel, dest)


def fetch(root: Path, store: str) -> dict:
    """Download missing or changed binaries; returns counts and any files whose hash still differs."""
    todo = wanted(root)
    bad = []
    for rel, digest in sorted(todo.items()):
        try:
            _copy_one(store, rel, root / rel)
        except (OSError, subprocess.CalledProcessError) as exc:
            log.error("fetch %s failed: %s", rel, exc)
            bad.append(rel)
            continue
        if sha256(root / rel) != digest:
            bad.append(rel)
    return {"wanted": len(todo), "fetched": len(todo) - len(bad), "bad": bad}


def upload_command(root: Path, store: str) -> list[str]:
    """The command that copies an import root's binaries to ``store`` (gs:// only)."""
    pattern = r".*\.(msh|png|jpe?g|obj|json)$"
    return ["gcloud", "storage", "rsync", "--recursive", "--exclude", rf"^(?!{pattern[2:]})", str(root), store]


__all__ = ["BINARY_SUFFIXES", "fetch", "is_binary", "upload_command", "wanted"]
