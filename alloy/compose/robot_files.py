# SPDX-License-Identifier: Apache-2.0
"""Robot files for composed scenes: the robot alone, out of an MJCF that also holds a floor, a table or loose
objects (Alloy's library embodiments are whole task scenes: arm, table, floor and a few objects).

``robot_only`` keeps the worldbody bodies whose tree has a hinge or slide joint (the robot) and drops the rest:
world geoms (floor planes, tables), bodies without joints (scenery) and free bodies (loose objects), with the
contact pairs, equalities, actuators, sensors and keyframes that name them. Mesh files and licences are copied
with it, so a saved scene folder loads on its own.

Example::

    >>> text = '<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
    >>> text += '<body name="arm"><joint name="j" type="hinge"/><geom size=".1"/></body></worldbody></mujoco>'
    >>> new, dropped = robot_only(text)
    >>> "floor" in new, dropped
    (False, ['floor'])
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from alloy.compose.spec import RobotSpec

_FILE = re.compile(r"\bfile=\"([^\"]+)\"")
_REFS = ("body", "body1", "body2", "geom", "geom1", "geom2", "joint", "joint1", "joint2", "site", "objname", "tendon")


def _is_robot(body: ET.Element) -> bool:
    return any(j.get("type", "hinge") in ("hinge", "slide") for j in body.iter("joint"))


def robot_only(text: str) -> tuple[str, list[str]]:
    """The MJCF ``text`` with only its robot bodies left in the worldbody, and the names it dropped."""
    root = ET.fromstring(text)
    world = root.find("worldbody")
    if world is None:
        return text, []
    gone: set[str] = set()
    dropped: list[str] = []
    for child in list(world):
        if child.tag in ("geom", "site") or (child.tag == "body" and not _is_robot(child)):
            dropped.append(child.get("name") or child.tag)
            gone |= {e.get("name") for e in child.iter() if e.get("name")}
            world.remove(child)
    if not dropped:
        return text, []
    for section in ("contact", "equality", "actuator", "sensor", "tendon"):
        sec = root.find(section)
        for el in list(sec) if sec is not None else ():
            if any(el.get(a) in gone for a in _REFS) or any(e.get(a) in gone for e in el.iter() for a in _REFS):
                sec.remove(el)
    for key in root.findall("keyframe"):
        root.remove(key)
    return ET.tostring(root, encoding="unicode"), dropped


def robot_copy(spec: RobotSpec, library, out: dict[str, bytes]) -> tuple[RobotSpec, str]:
    """``spec`` pointing at a robot-only copy (written into ``out``: relative path -> bytes) when its MJCF holds
    more than the robot; else ``spec`` unchanged. Returns the spec and a note for the result ("" when unchanged).

    Example: ``spec, note = robot_copy(RobotSpec("robots/emb_e2_panda/scene.xml"), lib, files)``."""
    if spec.kind != "mjcf":
        return spec, ""
    src = library.asset_path(spec.path)
    if src is None:
        return spec, ""
    src = Path(src)
    text = src.read_text(encoding="utf-8")
    new, dropped = robot_only(text)
    if not dropped:
        return spec, ""
    folder = f"robots/{src.parent.name}_robot"
    out[f"{folder}/{src.name}"] = new.encode("utf-8")
    for rel in sorted(set(_FILE.findall(new))):
        f = (src.parent / rel).resolve()
        if f.is_file():
            out[_normal(f"{folder}/{rel}")] = f.read_bytes()
    for lic in [*sorted(src.parent.glob("LICENSE*")), *sorted(src.parent.glob("source.json"))]:
        out[f"{folder}/{lic.name}"] = lic.read_bytes()
    note = (f"{spec.name}: {spec.path} also holds {', '.join(dropped)}; using the robot alone ({folder}) so it "
            "stands on the scene's table")
    return replace(spec, path=f"{folder}/{src.name}"), note


def _normal(rel: str) -> str:
    parts: list[str] = []
    for p in rel.split("/"):
        if p == "..":
            parts.pop()
        elif p not in ("", "."):
            parts.append(p)
    return "/".join(parts)


__all__ = ["robot_copy", "robot_only"]
