# SPDX-License-Identifier: Apache-2.0
"""Cut a converted MJCF down to a subset of its top-level bodies (e.g. the arms and bench without the task objects).

Works on the writer's own explicit MJCF (:mod:`alloy.tasks.mjcf_tree`): removes the dropped top-level ``<body>``
subtrees and every element that names something inside them (contact pairs and excludes, equalities, actuators),
then drops mesh assets no remaining geom uses. CLI:
``python -m alloy.tasks.mjcf_subset <scene.xml> <out.xml> <model name> --drop body1 body2 ...``.
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from alloy.core.errors import TraceError

log = logging.getLogger("alloy.tasks")


def _names_inside(bodies: list[ET.Element]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"body": set(), "joint": set(), "geom": set(), "site": set()}
    for b in bodies:
        for el in b.iter():
            if el.tag in out and el.get("name"):
                out[el.tag].add(el.get("name", ""))
    return out


def _refs(el: ET.Element, gone: dict[str, set[str]]) -> bool:
    """True if ``el`` names a removed body, joint, geom or site through any reference attribute."""
    keys = {"body": ("body", "body1", "body2"), "joint": ("joint", "joint1", "joint2", "jointinparent"),
            "geom": ("geom1", "geom2"), "site": ("site", "site1", "site2", "refsite")}  # fmt: skip
    return any(el.get(attr) in gone[kind] for kind, attrs in keys.items() for attr in attrs)


def _prune_section(root: ET.Element, tag: str, gone: dict[str, set[str]]) -> int:
    removed = 0
    for section in root.findall(tag):
        for el in list(section):
            if _refs(el, gone):
                section.remove(el)
                removed += 1
    return removed


def _prune_meshes(root: ET.Element) -> int:
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    removed = 0
    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used:
                asset.remove(mesh)
                removed += 1
    return removed


def subset(xml_in: Path, xml_out: Path, model_name: str, drop: set[str]) -> Path:
    """Write ``xml_out`` = ``xml_in`` without the top-level bodies in ``drop`` and everything that refers to them.

    ``xml_out`` must sit at the same depth as ``xml_in`` (mesh files are referenced relative to the file)."""
    tree = ET.parse(xml_in)
    root = tree.getroot()
    world = root.find("worldbody")
    dropped = [b for b in world.findall("body") if b.get("name") in drop] if world is not None else []
    missing = drop - {b.get("name") for b in dropped}
    if missing:
        raise TraceError(f"{xml_in}: no top-level bodies {sorted(missing)}")
    gone = _names_inside(dropped)
    for b in dropped:
        world.remove(b)
    counts = {tag: _prune_section(root, tag, gone) for tag in ("contact", "equality", "actuator")}
    counts["mesh"] = _prune_meshes(root)
    root.set("model", model_name)
    ET.indent(tree, space=" ")
    xml_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(xml_out, encoding="unicode", xml_declaration=False)
    log.info("subset %s -> %s: dropped %d bodies, removed %s", xml_in, xml_out, len(dropped), counts)
    return xml_out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("xml_in", type=Path)
    p.add_argument("xml_out", type=Path)
    p.add_argument("model_name")
    p.add_argument("--drop", nargs="+", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    subset(args.xml_in, args.xml_out, args.model_name, set(args.drop))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
