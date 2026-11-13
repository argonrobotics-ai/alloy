# SPDX-License-Identifier: Apache-2.0
"""Read an MJCF file with its ``<include>`` files written in, as MuJoCo's parser does.

MuJoCo replaces each ``<include file="..."/>`` by the children of the included file's root element, so a scene file
that includes its robot (MuJoCo Menagerie's ``scene.xml`` files) is one document. Sections that then appear twice
(``<worldbody>``) are merged into the first, in document order, which is how MuJoCo reads them.

Example::

    from alloy.io.mjcf_include import read_mjcf
    root = read_mjcf("unitree_g1/scene.xml")          # the <mujoco> element, no <include> left in it
    print([b.get("name") for b in root.find("worldbody") if b.tag == "body"])
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

MAX_DEPTH = 16


class IncludeError(ValueError):
    """An ``<include>`` could not be resolved; the message names the file."""


def _find(name: str, here: Path, main: Path) -> Path:
    """MuJoCo looks beside the main model file; a path relative to the including file is accepted too."""
    for base in (main, here):
        cand = (base / name).resolve()
        if cand.is_file():
            return cand
    raise IncludeError(f"<include file={name!r}> not found beside {main}")


def _expand(el: ET.Element, here: Path, main: Path, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise IncludeError(f"<include> files nest deeper than {MAX_DEPTH}: a file includes itself")
    for child in list(el):
        if child.tag != "include":
            _expand(child, here, main, depth)
            continue
        path = _find(child.get("file", ""), here, main)
        sub = ET.parse(path).getroot()
        _expand(sub, path.parent, main, depth + 1)
        at = list(el).index(child)
        el.remove(child)
        for k, new in enumerate(list(sub)):
            el.insert(at + k, new)


def _merge_worldbodies(root: ET.Element) -> None:
    worlds = root.findall("worldbody")
    for extra in worlds[1:]:
        worlds[0].extend(list(extra))
        root.remove(extra)


def read_mjcf(path: str | Path) -> ET.Element:
    """The ``<mujoco>`` element of ``path`` with every ``<include>`` written in and one ``<worldbody>``."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    _expand(root, path.parent, path.parent, 0)
    _merge_worldbodies(root)
    return root
