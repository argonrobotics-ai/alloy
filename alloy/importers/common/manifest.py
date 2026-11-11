# SPDX-License-Identifier: Apache-2.0
"""Import manifests: which binary files each imported card needs, their sha256, source and licence; fetch them.

Layout of an import library root (``alloy-assets/imports/<lib>/``)::

    manifest.json                 # this module; in git
    objects/<src>/<name>.yaml     # cards; in git
    materials/<src>/<name>_eff.yaml
    queue.jsonl                   # ESTIMATE values for the sourcing tool; in git
    ATTRIBUTION.md                # licence notices; in git
    meshes/<name>/physics.obj ... # binaries; NOT in git, fetched from the asset store by hash

The asset store is a directory, ``gs://`` prefix or ``https://`` prefix that holds ``<lib>/meshes/<name>/<file>``.
It is given per call or by ``$ALLOY_ASSET_STORE``; no store location is written in code or cards.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from alloy.importers.common.errors import ImporterError
from alloy.importers.common.objio import sha256_path

log = logging.getLogger("alloy.importers")

SCHEMA = "alloy.import_manifest/1"
STORE_ENV = "ALLOY_ASSET_STORE"
MESH_DIR = "meshes"


def empty(lib: str) -> dict:
    """A new manifest for library ``lib``.

    Example::

        m = empty("ycb_gso")
    """
    return {"schema": SCHEMA, "lib": lib, "objects": {}}


def load(path: Path, lib: str) -> dict:
    """The manifest at ``path``, or an empty one if the file does not exist.

    Example::

        m = load(Path("imports/ycb_gso/manifest.json"), "ycb_gso")
    """
    p = Path(path)
    if not p.is_file():
        return empty(lib)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ImporterError(str(p), "manifest", f"schema {data.get('schema')!r}, expected {SCHEMA!r}")
    return data


def save(path: Path, manifest: dict) -> None:
    """Write sorted, indented JSON (stable diffs).

    Example::

        save(Path("manifest.json"), m)
    """
    manifest["objects"] = dict(sorted(manifest["objects"].items()))
    Path(path).write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def file_entry(path: Path) -> dict:
    """``{"sha256", "bytes"}`` of one file.

    Example::

        file_entry(Path("meshes/ycb_025_mug/physics.obj"))
    """
    return {"sha256": sha256_path(path), "bytes": Path(path).stat().st_size}


def store_root(store: str | None) -> str:
    """The asset store to use: ``store`` or ``$ALLOY_ASSET_STORE``; raises with the fix if neither is set.

    Example::

        store_root("/data/alloy-store")   # "/data/alloy-store"
    """
    s = store or os.environ.get(STORE_ENV, "")
    if not s:
        raise ImporterError("<manifest>", "fetch", f"no asset store: pass --store or set ${STORE_ENV} "
                                                   "(a directory, gs:// or https:// prefix)")
    return s.rstrip("/")


def _copy_one(src: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.startswith("gs://"):
        subprocess.run(["gcloud", "storage", "cp", "-q", src, str(dest)], check=True, capture_output=True)
    elif src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=120) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    else:
        shutil.copyfile(src, dest)


def missing_files(root: Path, manifest: dict, names: list[str] | None = None) -> list[tuple[str, str]]:
    """``(name, file)`` pairs whose local copy is absent or has the wrong sha256.

    Example::

        missing_files(Path("imports/ycb_gso"), m, ["ycb_025_mug"])
    """
    out = []
    for name in names or sorted(manifest["objects"]):
        entry = manifest["objects"].get(name)
        if entry is None:
            raise ImporterError(name, "fetch", "not in the manifest; check the name with the 'list' command")
        for fname, meta in entry["files"].items():
            p = Path(root) / MESH_DIR / name / fname
            if not p.is_file() or sha256_path(p) != meta["sha256"]:
                out.append((name, fname))
    return out


def fetch(root: Path, manifest: dict, names: list[str] | None = None, store: str | None = None) -> dict:
    """Copy missing or stale binaries from the store and verify their sha256; returns a JSON summary.

    Example::

        fetch(Path("imports/ycb_gso"), m, ["ycb_025_mug"], store="/data/alloy-store")
    """
    base = f"{store_root(store)}/{manifest['lib']}/{MESH_DIR}"
    todo = missing_files(root, manifest, names)
    for name, fname in todo:
        dest = Path(root) / MESH_DIR / name / fname
        _copy_one(f"{base}/{name}/{fname}", dest)
        want = manifest["objects"][name]["files"][fname]["sha256"]
        if sha256_path(dest) != want:
            dest.unlink()
            raise ImporterError(name, "fetch", f"{fname} from the store has the wrong sha256; the store is stale")
    log.info("fetched %d files for %d objects", len(todo), len({n for n, _ in todo}))
    return {"fetched": len(todo), "objects": sorted({n for n, _ in todo})}
