# SPDX-License-Identifier: Apache-2.0
"""Which cards an import library has, where each one's render source is, and where its thumbnail goes.

Two layouts are read from the library's ``manifest.json`` (``alloy.import_manifest/1``):

- the shared importer layout (YCB/GSO, Objaverse, Gym-Anything): ``objects[name].files`` with ``meshes/<name>/
  visual.glb`` (else ``physics.obj``), thumbnail ``meshes/<name>/thumb.png``;
- RoboCasa: ``assets[name]`` with ``model`` (an MJCF file) and a ``thumbs/<category>/<name>.png`` file.

Example::

    jobs = library_jobs(Path("alloy-assets/imports/ycb_gso"))
    jobs[0].name, jobs[0].source.name, jobs[0].thumb.name       # ('gso_...', 'visual.glb', 'thumb.png')
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

THUMB = "thumb.png"
VISUAL_ORDER = ("visual.glb", "physics.obj")


@dataclass(frozen=True)
class Job:
    """One thumbnail to render: library, card name, render source file, thumbnail path (all absolute)."""

    lib: str
    name: str
    source: Path | None
    thumb: Path
    kind: str = "object"

    def as_dict(self) -> dict:
        return {"lib": self.lib, "name": self.name, "source": str(self.source) if self.source else None,
                "thumb": str(self.thumb), "kind": self.kind}


def _shared(root: Path, lib: str, objects: dict) -> list[Job]:
    jobs = []
    for name, e in sorted(objects.items()):
        d = root / "meshes" / name
        files = e.get("files", {})
        src = next((d / f for f in VISUAL_ORDER if f in files or (d / f).is_file()), None)
        jobs.append(Job(lib, name, src, d / THUMB, e.get("kind", "object")))
    return jobs


def _robocasa(root: Path, lib: str, assets: dict) -> list[Job]:
    jobs = []
    for name, e in sorted(assets.items()):
        thumb = next((f for f in e.get("files", {}) if f.startswith("thumbs/")), f"thumbs/{name}.png")
        model = e.get("model")
        jobs.append(Job(lib, name, root / model if model else None, root / thumb, e.get("kind", "object")))
    return jobs


def library_jobs(root: Path, lib: str | None = None) -> list[Job]:
    """Every card of the import library at ``root`` (sorted by name)."""
    root = Path(root)
    man = json.loads((root / "manifest.json").read_text())
    lib = lib or man.get("lib") or root.name
    if "objects" in man:
        return _shared(root, lib, man["objects"])
    if "assets" in man:
        return _robocasa(root, lib, man["assets"])
    raise ValueError(f"{root}/manifest.json: no 'objects' or 'assets' table (not an import library)")
