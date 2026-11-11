# SPDX-License-Identifier: Apache-2.0
"""Files an MJCF model names (meshes, textures, includes), so a dataset object and a fetch carry everything a model
loads.

A model can use another object's files: RoboCasa's composed cabinets (``mjcf/cabinets_composed/<name>/model.xml``)
take their door and handle meshes from a door-panel object and a handle object of the same library::

    <mesh name="vis0" file="../../../mjcf/cabinets/robocasa_cabinet_door_panel_001/visuals/CabinetDoorPanel001.msh"/>

``alloy.assets.hub.fetch`` follows these names to the objects that ship them.

Example::

    >>> named_files(cache_root("robocasa"), ["mjcf/cabinets_composed/robocasa_cabinet_drawers_001_001/model.xml"])
    ['mjcf/cabinets/robocasa_cabinet_door_panel_001/visuals/CabinetDoorPanel001.msh', ...]   # doctest: +SKIP
"""
from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger("alloy.assets.hub")

#: the ``<compiler>`` folder each asset element's ``file`` is read from (``assetdir`` stands in for both)
MJCF_DIR = {"mesh": "meshdir", "skin": "meshdir", "texture": "texturedir", "hfield": "texturedir"}


def mjcf_refs(path: Path) -> tuple[list[Path], list[Path]]:
    """The files one MJCF file names: (assets, includes), joined to its folder and its ``<compiler>`` folders.
    Raises ``xml.etree.ElementTree.ParseError`` on a file that is not XML. Example: ``mjcf_refs(Path("model.xml"))``."""
    tree = ET.parse(path)
    comp = tree.find(".//compiler")
    asset_dir = comp.get("assetdir", "") if comp is not None else ""
    dirs = {k: (comp.get(k, asset_dir) if comp is not None else "") for k in ("meshdir", "texturedir")}
    assets, includes = [], []
    for el in tree.iter():
        f = el.get("file")
        if not f:
            continue
        if el.tag == "include":
            includes.append(path.parent / f)
        else:
            assets.append(path.parent / dirs.get(MJCF_DIR.get(el.tag, ""), "") / f)
    return assets, includes


def _under(base: Path, path: Path) -> str:
    """``path`` relative to ``base`` with ``/`` separators (``../`` kept when it is outside ``base``)."""
    return os.path.relpath(os.path.normpath(path), base).replace(os.sep, "/")


def named_files(base: Path, rels: Iterable[str]) -> list[str]:
    """Every file the MJCF files among ``rels`` (paths under ``base``) name, followed through the includes that are
    on disk, as sorted paths under ``base``. MJCF files not on disk are skipped, and so (logged) are files that are
    not XML. Example: see the module docstring."""
    out: set[str] = set()
    todo, seen = [base / r for r in rels if r.endswith(".xml")], set()
    while todo:
        xml = todo.pop()
        if xml in seen or not xml.is_file():
            continue
        seen.add(xml)
        try:
            assets, includes = mjcf_refs(xml)
        except ET.ParseError as exc:
            log.warning("%s is not readable XML (%s): the files it names are not followed", _under(base, xml), exc)
            continue
        out.update(_under(base, p) for p in assets + includes)
        todo += includes
    return sorted(out)


def absent_files(base: Path, rels: Iterable[str]) -> list[str]:
    """The files ``named_files`` finds that are not on disk under ``base``. Example:
    ``absent_files(root, ["mjcf/cabinets_composed/<name>/model.xml"]) == []`` once its door panel is fetched."""
    return [r for r in named_files(base, rels) if not (base / r).is_file()]
