# SPDX-License-Identifier: Apache-2.0
"""Small MJCF edits for environments: drop bodies a task does not need (and everything that refers to them).

A library robot file often ships with props (a table, a mug, boxes). A reaching task does not need them, and every
convex piece costs contact work in every world. ``without_bodies`` writes a trimmed copy into a cache folder, with
mesh paths made absolute so the copy loads from anywhere. Example::

    path = without_bodies(default_root() / "robots/emb_e2_panda/scene.xml", ["box", "cylinder", "mug"])
    scene.add_robot("panda", path)
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from alloy.envs.errors import EnvError


def cache_dir() -> Path:
    """Where trimmed robot files go: ``$ALLOY_CACHE`` or ``~/.cache/alloy``, sub-folder ``envs``."""
    root = Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy"))
    return root / "envs"


def without_bodies(path: str | Path, bodies: Sequence[str]) -> Path:
    """A copy of the MJCF ``path`` without the named top-level bodies, their meshes, contact pairs and keyframes.

    Example: ``without_bodies("scene.xml", ["mug"])`` -> a cached file path."""
    src = Path(path).resolve()
    if not src.is_file():
        raise EnvError(f"robot file {src.name!r} not found", "pass the path of an existing MJCF file", field="path")
    text = src.read_text(encoding="utf-8")
    # the copy holds absolute paths into src's folder, so two checkouts of one file must not share it
    key = hashlib.sha256((str(src.parent) + "|" + text + "|" + ",".join(sorted(bodies))).encode()).hexdigest()[:16]
    out = cache_dir() / f"{src.stem}-{key}.xml"
    if out.is_file():
        return out
    root = ET.fromstring(text)
    _drop_bodies(root, set(bodies))
    _absolute_files(root, src.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    ET.ElementTree(root).write(tmp, encoding="unicode")
    tmp.replace(out)
    return out


def _drop_bodies(root: ET.Element, names: set[str]) -> None:
    world = root.find("worldbody")
    missing = set(names)
    for body in list(world) if world is not None else []:
        if body.tag == "body" and body.get("name") in names:
            world.remove(body)
            missing.discard(body.get("name"))
    if missing:
        raise EnvError(f"no top-level bodies {sorted(missing)} in the robot file", "name bodies directly under "
                       "<worldbody>", field="bodies")
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)                       # qpos sizes no longer match
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used:
                asset.remove(mesh)
    kept = {b.get("name") for b in root.iter("body")}
    for contact in root.findall("contact"):
        for item in list(contact):
            refs = [item.get(k) for k in ("body1", "body2") if item.get(k)]
            if any(r not in kept for r in refs):
                contact.remove(item)


def _absolute_files(root: ET.Element, base: Path) -> None:
    meshdir = root.find("compiler")
    sub = meshdir.get("meshdir") if meshdir is not None else None
    for el in root.iter():
        f = el.get("file")
        if f and not Path(f).is_absolute():
            folder = base / sub if sub and el.tag == "mesh" else base
            el.set("file", str((folder / f).resolve()))


__all__ = ["cache_dir", "without_bodies"]
