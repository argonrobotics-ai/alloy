# SPDX-License-Identifier: Apache-2.0
"""Fixed-base articulated MJCF objects (SCR-DC4-2): a `motion: static` card whose MJCF file has non-free joints.

The emitter imports such a file with a fixed root (`floating=False`), and Newton keeps its nested slide / hinge / ball
joints. Only the MuJoCo entry steps them (SolverMuJoCo converts the articulation), so the router gives these objects the
MuJoCo entry. Their route stays STATIC: a fixed base in the shared pipeline. The joints' types are read from the file
as MuJoCo resolves them: a joint's `type`, else its `class` default, else the nearest enclosing body's `childclass`
default, else "main"'s, else hinge. `<freejoint>` and `type="free"` don't count, since the root is fixed. Joints in
`<include>`d files are not read.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping

from alloy.cards.types import MJCFSource, ResolvedObject
from alloy.core.errors import RouteError

MAIN = "main"


def _default_types(node: ET.Element, inherited: str | None, out: dict[str, str | None]) -> None:
    """class name -> joint type from a <default> tree (nested classes inherit; None = unset)."""
    name = node.get("class", MAIN)
    joint = node.find("joint")
    own = joint.get("type") if joint is not None else None
    out[name] = own or inherited
    for child in node.findall("default"):
        _default_types(child, out[name], out)


def _body_joints(body: ET.Element, cls: str, defaults: Mapping[str, str | None], out: list[str]) -> None:
    cls = body.get("childclass", cls)
    for j in body.findall("joint"):
        kind = j.get("type") or defaults.get(j.get("class", cls)) or defaults.get(MAIN) or "hinge"
        if kind != "free":
            out.append(kind)
    for child in body.findall("body"):
        _body_joints(child, cls, defaults, out)


def mjcf_joint_types(path: str) -> tuple[str, ...]:
    """Types of the non-free joints in an MJCF file's <worldbody>, in file order."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RouteError(f"{path}: cannot read the MJCF to find its joints: {exc}") from exc
    defaults: dict[str, str | None] = {}
    for d in root.findall("default"):
        _default_types(d, None, defaults)
    out: list[str] = []
    for wb in root.findall("worldbody"):
        for body in wb.findall("body"):
            _body_joints(body, MAIN, defaults, out)
    return tuple(out)


def articulated_joints(obj: ResolvedObject) -> tuple[str, ...]:
    """The non-free joint types of a static MJCF object (empty for any other object)."""
    if obj.card.motion != "static" or not isinstance(obj.card.shape, MJCFSource):
        return ()
    return mjcf_joint_types(obj.card.shape.path)


__all__ = ["articulated_joints", "mjcf_joint_types"]
