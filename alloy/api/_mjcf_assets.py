# SPDX-License-Identifier: Apache-2.0
"""Asset paths of an MJCF string, resolved by MuJoCo's rule for XML text when ``Scene.add_robot(mjcf=...)`` runs.

MuJoCo resolves the files an XML string names from the working directory: ``<mesh>``, ``<hfield>`` and ``<skin>``
files under the compiler's ``meshdir``, ``<texture>`` files under its ``texturedir`` (both default to ``assetdir``,
and are themselves relative to the working directory; Alloy's importer also finds a texture under ``meshdir``),
``<include>`` files from the working directory, and with ``strippath`` only each file's base name. The build writes
an ``mjcf=`` robot to ``robots/<name>.xml`` in its own card folder, where the importer resolved relative paths from
that folder: a mesh beside the user's script was looked for in the card folder and failed with the mesh loader's
bare "string is not a file". ``absolute_assets`` writes every such path absolute, and the compiler's directories too
(included files resolve their own relative assets under them, as MuJoCo does), so the text means the same wherever
it is written or saved. A file that is not there raises ``SceneError`` naming the robot, the element and the path it
resolved to.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from alloy.api.errors import SceneError

MESH_DIR_TAGS = ("mesh", "hfield", "skin")
TEXTURE_FILES = ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")
_DIRS = ("meshdir", "texturedir")


def absolute_assets(name: str, text: str, cwd: Path) -> str:
    """``text`` (an MJCF string) with every asset and include path absolute, resolved from ``cwd`` by MuJoCo's rule;
    unchanged when it names no file. Raises ``SceneError`` for a file that is not there.

    Example: ``absolute_assets("arm", '<mujoco><asset><mesh file="a.stl"/></asset></mujoco>', Path.cwd())``."""
    root = ElementTree.fromstring(text)
    named = [e for e in root.iter() if e.tag in (*MESH_DIR_TAGS, "texture", "include") and _files(e)]
    if not named:
        return text
    comp = root.find("compiler")
    if comp is None:
        comp = ElementTree.Element("compiler")
        root.insert(0, comp)
    dirs = _asset_dirs(comp, cwd)
    strip = comp.get("strippath", "false").lower() in ("true", "1")
    for elem in named:
        bases = [cwd] if elem.tag == "include" else [dirs["meshdir"]]
        if elem.tag == "texture":               # the importer also reads a texture under meshdir: kept as a fallback
            bases.insert(0, dirs["texturedir"])
        for attr, value in _files(elem):
            paths = [_resolve(b, value, strip and elem.tag != "include") for b in bases]
            path = next((q for q in paths if q.is_file()), None)
            if path is None:
                raise SceneError(f"robot {name!r}: {_label(elem)} names {paths[0]}, which is not a file",
                                 _MISSING_FIX, field="mjcf")
            elem.set(attr, str(path))
    for key in _DIRS:
        comp.set(key, str(dirs[key]))
    comp.attrib.pop("assetdir", None)
    return ElementTree.tostring(root, encoding="unicode")


_MISSING_FIX = ("put the file there, or name it by its absolute path: relative paths in an MJCF string are read from "
                "the working directory when add_robot runs (MuJoCo's rule), under the compiler's meshdir / texturedir; "
                "or pass the robot's file, add_robot(name, path), whose relative paths are read from its folder")


def _files(elem: ElementTree.Element) -> list[tuple[str, str]]:
    attrs = TEXTURE_FILES if elem.tag == "texture" else ("file",)
    return [(a, elem.get(a)) for a in attrs if elem.get(a)]


def _asset_dirs(comp: ElementTree.Element, cwd: Path) -> dict[str, Path]:
    """The compiler's mesh and texture directories, absolute (``assetdir`` is the default of both)."""
    asset = comp.get("assetdir")
    out = {}
    for key in _DIRS:
        rel = comp.get(key, asset)
        out[key] = cwd if rel is None else (cwd / Path(rel).expanduser()).resolve()
    return out


def _resolve(base: Path, value: str, strip: bool) -> Path:
    path = Path(value.replace("\\", "/")).expanduser()
    if strip:
        path = Path(path.name)
    return path if path.is_absolute() else (base / path).resolve()


def _label(elem: ElementTree.Element) -> str:
    shown = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items() if k == "name" or k in TEXTURE_FILES)
    return f"<{elem.tag} {shown}>" if shown else f"<{elem.tag}>"


__all__ = ["absolute_assets"]
