# SPDX-License-Identifier: Apache-2.0
"""Robots the command line and tool server can name without a file path: the library's robots (``alloy list
robots``), so a recipe's ``{"add": "robot", "name": "arm", "path": "emb_e2_panda"}`` (or the library-relative
``"robots/emb_e2_panda/scene.xml"``) finds the file. No engine import.

A robot is a folder under ``<root>/robots/`` holding one robot file (``scene.xml`` first); in the shipped library,
only folders whose ``source.json`` names an embodiment (the others hold task scenes). ``--library`` folders come
first. Example::

    from alloy.cli.robots import resolve, robots
    [r["name"] for r in robots()][:2]           # ['emb_e1_fr3', 'emb_e1_fr3_route']
    resolve("emb_e2_panda")                     # .../library/robots/emb_e2_panda/scene.xml
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger("alloy.cli")

ROBOT_SUFFIXES = (".xml", ".urdf", ".usd", ".usda", ".usdc")
_PREFERRED = ("scene.xml", "robot.xml", "robot.urdf")


def robots(roots: Sequence[str | Path] = ()) -> list[dict[str, Any]]:
    """Every robot folder in ``roots`` then the shipped library (the first root with a name wins): name, title (from
    its ``source.json``, else "") and the robot file. Example: ``robots(["my_library"])[0]["path"]``."""
    return [{"name": name, "title": meta.get("title", ""), "path": str(file)}
            for name, (file, meta) in sorted(_folders(roots).items())]


def resolve(path: str, roots: Sequence[str | Path] = ()) -> Path | None:
    """The robot file meant by ``path``: a robot name (``alloy list robots``), else a path relative to a library root
    (``robots/emb_e2_panda/scene.xml``); None when it is neither (an existing file needs no lookup).

    Example: ``resolve("emb_e2_panda").name == "scene.xml"``."""
    hit = _folders(roots).get(path)
    if hit is not None:
        return hit[0]
    for root in _roots(roots):
        candidate = root / path
        if candidate.is_file():
            return candidate
    return None


def _roots(roots: Sequence[str | Path]) -> list[Path]:
    from alloy.cards.library import default_root  # noqa: PLC0415

    return [*(Path(r).expanduser() for r in roots), Path(default_root())]


def _folders(roots: Sequence[str | Path]) -> dict[str, tuple[Path, dict[str, Any]]]:
    *extra, shipped = _roots(roots)
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for root in [*extra, shipped]:
        base = root / "robots"
        if not base.is_dir():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir() and p.name not in found):
            file, meta = robot_file(folder), _source(folder)
            if file is None or (root == shipped and "embodiment" not in meta):
                continue
            found[folder.name] = (file, meta)
    return found


def robot_file(folder: Path) -> Path | None:
    """The robot file of a folder: ``scene.xml`` / ``robot.xml`` / ``robot.urdf``, else its only MJCF, URDF or USD
    file; None if there is none or several. Example: ``robot_file(Path("library/robots/emb_e2_panda"))``."""
    for name in _PREFERRED:
        if (folder / name).is_file():
            return folder / name
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in ROBOT_SUFFIXES]
    return files[0] if len(files) == 1 else None


def _source(folder: Path) -> dict[str, Any]:
    path = folder / "source.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("robots: %s is not readable JSON (%s); listed without a title", path, exc)
        return {}
    return data if isinstance(data, dict) else {}
