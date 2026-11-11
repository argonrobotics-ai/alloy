# SPDX-License-Identifier: Apache-2.0
"""Fetch one imported object's files (meshes, textures) on demand, so the builder can use any object in the library.

Git holds the cards; the binaries sit in an asset store given by ``$ALLOY_ASSET_STORE`` (a ``gs://`` or ``https://``
prefix, or a folder). An import root ``<alloy-assets>/imports/<source>/`` lists every object's files with their
sha256 in ``manifest.json``, in either importer layout:

- ``{"objects": {name: {"files": {"physics.obj": {"sha256": ...}}}}}``: files under ``meshes/<name>/``;
- ``{"assets": {name: {"files": {"mjcf/<...>/hull_00.msh": {"sha256": ...}}}}}``: paths relative to the root.

A file is copied from ``<store>/<source>/<path in the root>`` when it is missing or its sha256 differs, and checked
after the copy.

Where fetched files go (``default_dest``): into the import root itself when it is a writable checkout (alloy-assets);
else, for cards packaged in the installed ``alloy`` (read-only, wiped by an upgrade), into the cache
``cache_root(<source>)`` = ``$ALLOY_CACHE/imports/<source>`` or ``${XDG_CACHE_HOME:-~/.cache}/alloy/imports/<source>``,
at the same relative paths the cards name. A cached object is complete there: files that sit in the read-only root
(an MJCF ``model.xml``) are copied next to the fetched ones, so relative mesh paths inside it still resolve. Card
libraries see the cache by listing ``cache_roots()`` as library roots before the import roots (they hold no cards).

Example::

    >>> fetch_object(Path("alloy-assets/imports/ycb_gso"), "ycb_025_mug", "gs://my-bucket")     # doctest: +SKIP
    4
    >>> cache_root("ycb_gso").parts[-2:]
    ('imports', 'ycb_gso')
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from alloy.compose.spec import ComposeError

log = logging.getLogger("alloy.compose")

STORE_ENV = "ALLOY_ASSET_STORE"
CACHE_ENV = "ALLOY_CACHE"
MESH_DIR = "meshes"
IMPORTS = "imports"


def cache_base() -> Path:
    """``$ALLOY_CACHE``, else ``${XDG_CACHE_HOME:-~/.cache}/alloy``. Example: ``cache_base() / "imports"``."""
    raw = os.environ.get(CACHE_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "alloy"


def cache_root(source: str) -> Path:
    """Where fetched files of import library ``source`` go when its root is read-only. Example: see the module."""
    return cache_base() / IMPORTS / source


def cache_roots() -> list[Path]:
    """The cache root of every import library that has one on disk (sorted); add them as library roots before the
    import roots. Example: ``CardLibrary(roots=[*cache_roots(), *import_roots])``."""
    base = cache_base() / IMPORTS
    return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def default_dest(root: Path) -> Path:
    """``root`` when it is a writable checkout, else ``cache_root(root.name)`` (a root inside the installed package
    is never written). Example: ``default_dest(Path("site-packages/alloy/_data/imports/ycb_gso"))`` is the cache."""
    import alloy  # noqa: PLC0415

    r = Path(root).resolve()
    if r.is_relative_to(Path(alloy.__file__).resolve().parent) or not os.access(r, os.W_OK):
        return cache_root(r.name)
    return r


def store_from_env() -> str | None:
    """``$ALLOY_ASSET_STORE`` without a trailing slash, or None. Example: ``store_from_env()``."""
    s = os.environ.get(STORE_ENV, "").strip()
    return s.rstrip("/") or None


def files_of(root: Path, name: str) -> dict[str, str]:
    """Path under ``root`` -> sha256 of every file the manifest lists for ``name`` ({} if it lists none).

    Example: ``files_of(Path("imports/ycb_gso"), "ycb_025_mug")``."""
    path = Path(root) / "manifest.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("objects"), dict):
        entry = data["objects"].get(name) or {}
        return {f"{MESH_DIR}/{name}/{f}": m["sha256"] for f, m in (entry.get("files") or {}).items()}
    entry = (data.get("assets") or {}).get(name) or {}
    return {rel: m["sha256"] for rel, m in (entry.get("files") or {}).items()}


def missing(root: Path, name: str, dest: Path | None = None) -> list[str]:
    """Listed files of ``name`` that are absent from ``dest`` (default ``default_dest(root)``) or differ from the
    manifest's sha256. Example: ``missing(root, "ycb_025_mug") == []`` once fetched."""
    dest = Path(dest) if dest is not None else default_dest(root)
    return [rel for rel, digest in files_of(root, name).items() if not _valid(dest / rel, digest)]


def _valid(path: Path, digest: str) -> bool:
    from alloy.importers.common.objio import sha256_path  # noqa: PLC0415

    return path.is_file() and sha256_path(path) == digest


def fetch_object(root: Path, name: str, store: str | None = None, dest: Path | None = None) -> int:
    """Put ``name``'s files into ``dest`` (default ``default_dest(root)``): copied from ``root`` when they are valid
    there, else from ``store`` (default ``$ALLOY_ASSET_STORE``). Returns how many files were written. Raises
    ``ComposeError`` with the fix when there is no store or a copy fails."""
    import shutil  # noqa: PLC0415

    root = Path(root)
    dest = Path(dest) if dest is not None else default_dest(root)
    want = files_of(root, name)
    todo = missing(root, name, dest)
    local = [rel for rel in todo if dest.resolve() != root.resolve() and _valid(root / rel, want[rel])]
    for rel in local:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, dest / rel)
    todo = [rel for rel in todo if rel not in local]
    if not todo:
        return len(local)
    store = (store or store_from_env() or "").rstrip("/")
    if not store:
        raise ComposeError(f"{name}: {len(todo)} file(s) are not on disk and no asset store is set",
                           f"set ${STORE_ENV} to the asset store (gs://, https:// or a folder), or pick "
                           "another object", field="object")
    try:
        _copy(f"{store}/{root.resolve().name}", dest, todo)
    except Exception as exc:  # noqa: BLE001 - any copy failure becomes one error with the fix
        raise ComposeError(f"{name}: fetching {len(todo)} file(s) failed ({type(exc).__name__}: {exc})",
                           f"check ${STORE_ENV} ({store}) and your access to it", field="object") from exc
    for rel in todo:
        if not _valid(dest / rel, want[rel]):
            (dest / rel).unlink(missing_ok=True)
            raise ComposeError(f"{name}: {rel} from the store has the wrong sha256",
                               "the store is older than the manifest; update the store or the import root",
                               field="object")
    log.info("compose: fetched %d file(s) for %s into %s", len(todo), name, dest)
    return len(todo) + len(local)


def _copy(base: str, dest: Path, rels: list[str]) -> None:
    """Copy ``<base>/<rel>`` to ``dest/<rel>``: one ``gcloud storage cp`` per folder for gs:// stores (a call costs
    seconds), else file by file."""
    import subprocess  # noqa: PLC0415

    from alloy.importers.common.manifest import _copy_one  # noqa: PLC0415 - the importers' gs/https/folder copy

    if not base.startswith("gs://"):
        for rel in rels:
            _copy_one(f"{base}/{rel}", dest / rel)
        return
    folders: dict[Path, list[str]] = {}
    for rel in rels:
        folders.setdefault((dest / rel).parent, []).append(f"{base}/{rel}")
    for folder, srcs in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", "-q", *srcs, f"{folder}/"], check=True, capture_output=True)


__all__ = ["CACHE_ENV", "STORE_ENV", "cache_base", "cache_root", "cache_roots", "default_dest", "fetch_object",
           "files_of", "missing", "store_from_env"]
