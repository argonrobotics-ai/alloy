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


TRIM_VERSION = "2"             # what a trimmed copy holds: change it with any change to the trim's output


def cache_dir() -> Path:
    """Where trimmed robot files go: ``$ALLOY_CACHE`` or ``~/.cache/alloy``, sub-folder ``envs``."""
    root = Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy"))
    return root / "envs"


# attributes that name another element, by what they name (MJCF reference: equality, actuator, sensor, contact, tendon)
_REFS = {"bodies": ("body", "body1", "body2"), "joints": ("joint", "joint1", "joint2", "jointinparent"),
         "geoms": ("geom", "geom1", "geom2"), "sites": ("site", "site1", "site2"),
         "tendons": ("tendon", "tendon1", "tendon2")}
_NAMED = {"body": "bodies", "joint": "joints", "freejoint": "joints", "geom": "geoms", "site": "sites"}
_OBJTYPE = {"body": "bodies", "xbody": "bodies", "joint": "joints", "geom": "geoms", "site": "sites",
            "tendon": "tendons"}


def without_bodies(path: str | Path, bodies: Sequence[str]) -> Path:
    """A copy of the MJCF ``path`` without the named top-level bodies, their meshes and keyframes, and every
    equality, actuator, sensor, contact or tendon row that names something inside them. A name can also be a bare
    geom directly under ``<worldbody>`` (the H1-2 file's static obstacle box is one, where the G1 and Go2 files wrap
    theirs in a body).

    Example: ``without_bodies("scene.xml", ["mug"])`` -> a cached file path."""
    src = Path(path).resolve()
    if not src.is_file():
        raise EnvError(f"robot file {src.name!r} not found", "pass the path of an existing MJCF file", field="path")
    text = src.read_text(encoding="utf-8")
    # the copy holds absolute paths into src's folder, so two checkouts of one file must not share it; a copy an
    # older trimmer wrote must not be reused either
    raw = "|".join([TRIM_VERSION, str(src.parent), text, ",".join(sorted(bodies))])
    key = hashlib.sha256(raw.encode()).hexdigest()[:16]
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
    missing, gone = set(names), {k: set() for k in _REFS}       # what the dropped bodies held, by kind
    for part in list(world) if world is not None else []:
        if part.tag in ("body", "geom") and part.get("name") in names:   # a body, or a bare geom fixed to the world
            for el in part.iter():
                if el.get("name") and el.tag in _NAMED:
                    gone[_NAMED[el.tag]].add(el.get("name"))
            world.remove(part)
            missing.discard(part.get("name"))
    if missing:
        raise EnvError(f"no top-level bodies or geoms {sorted(missing)} in the robot file", "name bodies or geoms "
                       "directly under <worldbody>", field="bodies")
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)                       # qpos sizes no longer match
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used:
                asset.remove(mesh)
    gone["tendons"] |= _drop_dangling(root, ("tendon",), gone)    # first: actuators and sensors can name a tendon
    _drop_dangling(root, ("equality", "actuator", "sensor", "contact"), gone)


def _drop_dangling(root: ET.Element, sections: tuple[str, ...], gone: dict[str, set[str]]) -> set[str]:
    """Remove each row of ``sections`` that (or any element inside it) names a body, joint, geom, site or tendon in
    ``gone`` (an equality on a dropped finger would make the build fail); returns the removed rows' names. Only what
    was dropped counts: the world body ("world") and names from included files stay named."""
    def dangling(row: ET.Element) -> bool:
        for el in row.iter():
            if any(el.get(a) in gone[cat] for cat, attrs in _REFS.items() for a in attrs):
                return True
            kind = _OBJTYPE.get(el.get("objtype", ""))
            if kind and el.get("objname") in gone[kind]:
                return True
        return False

    removed: set[str] = set()
    for section in (s for tag in sections for s in root.findall(tag)):
        for row in [r for r in section if dangling(r)]:
            section.remove(row)
            removed |= {row.get("name")} - {None}
    return removed


def _absolute_files(root: ET.Element, base: Path) -> None:
    """Every relative ``file`` made absolute the way MuJoCo resolves it: meshes, height fields and skins under
    ``meshdir``, textures under ``texturedir``, both falling back to ``assetdir``, anything else beside the file."""
    compiler = root.find("compiler")
    get = compiler.get if compiler is not None else (lambda _key: None)
    asset = get("assetdir")
    folders = {"mesh": get("meshdir") or asset, "hfield": get("meshdir") or asset, "skin": get("meshdir") or asset,
               "texture": get("texturedir") or asset}
    for el in root.iter():
        f = el.get("file")
        if f and not Path(f).is_absolute():
            sub = folders.get(el.tag)
            folder = base / sub if sub else base
            el.set("file", str((folder / f).resolve()))


__all__ = ["cache_dir", "without_bodies"]
