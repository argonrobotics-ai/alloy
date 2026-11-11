# SPDX-License-Identifier: Apache-2.0
"""Layout of the object library dataset (``alloy-library`` on the Hugging Face Hub), shared by the builder and fetch.

::

    README.md                        dataset card: what it is, licences, how to use it with Alloy, citation
    index.jsonl                      one row per object (below)
    ATTRIBUTION.md                   every object's credit line, by library
    LICENSES/<SPDX id>.txt           the full text of every licence used
    libraries/<library>/<path>       files a library's objects share (manifest.json, shared material cards, looks)
    objects/<library>/<name>/<file>  one folder per object: card.yaml, material.yaml, meshes, hull sidecars,
                                     textures, thumb.png

An index row::

    {"name": "ycb_025_mug", "library": "ycb_gso", "category": "ycb", "title": "025_mug", "licence": "CC-BY-4.0",
     "attribution": "YCB Object and Model Set ...", "source": "https://...", "showcase_ok": true,
     "mass_basis": "sourced", "estimate": false, "sources": {"density": "fit", "mu_static": "model"},
     "flags": "(only when an importer flagged it: the reason)",
     "size": 1234567, "sha256": "<tree digest>",
     "files": [{"file": "physics.obj", "path": "meshes/ycb_025_mug/physics.obj", "sha256": "...", "bytes": 123}]}

``file`` is the name inside the object's folder. ``path`` is where Alloy puts it under the library's cache root
(``$ALLOY_CACHE/imports/<library>/``): the import-root layout that the card's relative paths expect. The row's
``sha256`` is the object's tree digest: the sha256 of its sorted ``"<sha256>  <file>"`` lines. Library files are
listed the same way in ``libraries/<library>/files.json``.

Example::

    >>> object_dir("ycb_gso", "ycb_025_mug")
    'objects/ycb_gso/ycb_025_mug'
    >>> dataset_file("meshes/ycb_025_mug/physics.obj", home="meshes/ycb_025_mug")
    'physics.obj'
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

INDEX = "index.jsonl"
LIBRARY_FILES = "files.json"
ATTRIBUTION = "ATTRIBUTION.md"
LICENSES = "LICENSES"
THUMB = "thumb.png"
CARD = "card.yaml"
MATERIAL = "material.yaml"
SCHEMA = "alloy.library_index/1"


def object_dir(library: str, name: str) -> str:
    """The object's folder in the dataset. Example: ``object_dir("objaverse", "objv_mug_1a2b3c4d")``."""
    return f"objects/{library}/{name}"


def library_dir(library: str) -> str:
    """The folder of a library's shared files. Example: ``library_dir("robocasa") == "libraries/robocasa"``."""
    return f"libraries/{library}"


def dataset_file(rel: str, home: str, card: str | None = None, material: str | None = None,
                 thumb: str | None = None) -> str:
    """Name inside the object's folder of the import-root file ``rel``: ``card.yaml``, ``material.yaml`` (the
    object's own material card) and ``thumb.png``; files under the object's home folder (``meshes/<name>``,
    ``mjcf/<category>/<name>``) keep their path below it; anything else keeps its file name. Example: see the module
    docstring."""
    named = {card: CARD, material: MATERIAL, thumb: THUMB}
    if rel in named and rel is not None:
        return named[rel]
    home = home.rstrip("/") + "/"
    return rel[len(home):] if rel.startswith(home) else Path(rel).name


def tree_digest(files: Iterable[dict]) -> str:
    """sha256 of the sorted ``"<sha256>  <file>"`` lines of an object's files. Example: ``tree_digest(row["files"])``."""
    lines = sorted(f"{f['sha256']}  {f['file']}" for f in files)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file, read in 1 MiB blocks. Example: ``sha256_file(Path("thumb.png"))``."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
