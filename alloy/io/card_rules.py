# SPDX-License-Identifier: Apache-2.0
"""Rules the card writer follows for what a card cannot hold: which groups stay a robot file as written, the pair
cards of a file's explicit contact pairs, and which bodies are kinematic.

An object card holds one material, one motion and no constraint, so a group stays a file (a ``RobotRef``, loaded as
the author wrote it) when it has

- a driven joint (a servo or a motor, also on the joint that holds its base),
- joints and a base fixed to the world (a cabinet on a wall, a gantry on its rail): an object card would float it,
- joints and more than one surface material (the card's one material would replace them all),
- a weld or connect to another body or the world, or a gravity-compensated body,
- a joint the writers have no form for (several axes on one body): it is never written as a free body.

Example::

    from alloy.io.card_rules import file_reason
    from alloy.io.groups import split_groups
    for g in split_groups(io):
        print(g.name, file_reason(io, g) or "becomes a card")
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from alloy.cards import types as ct
from alloy.core.quantity import Q
from alloy.io.from_model import clean_name
from alloy.io.groups import Group
from alloy.io.model import IOJoint, IOModel


UNWRITABLE = "a joint the card writer cannot express"
FIXED_BASE = "joints on a base fixed to the world"


def group_joints(g: Group) -> list[IOJoint]:
    """Every joint of the group, the one that holds its base to the world included."""
    return ([g.base_joint] if g.base_joint is not None else []) + list(g.joints)


def is_driven(j: IOJoint) -> bool:
    return j.actuated or j.stiffness > 0 or j.motor_gear is not None


def surface_materials(g: Group) -> set[str]:
    """Names of the surface materials on the group's colliding shapes, over all its bodies."""
    return {s.material for b in g.bodies for s in b.shapes if s.collide and s.material}


def file_reason(io: IOModel, g: Group) -> str:
    """Why the group must stay a robot file (module docstring), or "" when a card holds it."""
    if g.kind == "robot" or any(is_driven(j) for j in group_joints(g)):
        return "driven joints"
    names = {b.name for b in g.bodies}
    if names & set(io.unwritable):
        return UNWRITABLE
    if any(e.body1 in names or e.body2 in names for e in io.equalities
           if not (e.kind == "weld" and e.body2 is None and not g.articulated)):   # a lone world weld: a static card
        return "a weld or connect constraint"
    if any(b.gravcomp for b in g.bodies):
        return "gravity compensation"
    if g.articulated and g.fixed_base:
        return FIXED_BASE
    if g.articulated and len(surface_materials(g)) > 1:
        return "joints and several surface materials"
    return ""


def mocap_bodies(src: Path, fmt: str) -> set[str]:
    """Cleaned names of the MJCF's ``mocap="true"`` bodies (moved by the user, not by forces: kinematic cards)."""
    if fmt != "mjcf":
        return set()
    from alloy.io.mjcf_include import read_mjcf  # noqa: PLC0415

    try:
        root = read_mjcf(src)
    except (ValueError, ET.ParseError, OSError):
        return set()
    return {clean_name(b.get("name"), "") for b in root.iter("body") if b.get("mocap") == "true" and b.get("name")}


def pair_cards(io: IOModel, material_of_shape: dict[str, str], src: str) -> tuple[list[ct.PairCard], int]:
    """Pair cards for the file's explicit contact pairs between two carded objects, and how many pairs were left
    out (a shape of a robot file: the pair stays in that file)."""
    out: dict[tuple[str, str], ct.PairCard] = {}
    skipped = 0
    for p in io.pairs:
        a, b = material_of_shape.get(p.shape1), material_of_shape.get(p.shape2)
        if a is None or b is None or a == b:
            skipped += 1
            continue
        note = f"imported from {src}: the file's contact pair {p.shape1} / {p.shape2}"
        mu = Q(float(p.mu), "model", note=note)
        e = Q(float(p.restitution), "model", note=note + " (from its damping ratio)") if p.restitution else None
        card = ct.PairCard(a, b, mu, mu, e)
        out.setdefault(card.key, card)
    return list(out.values()), skipped
