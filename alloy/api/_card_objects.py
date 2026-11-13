# SPDX-License-Identifier: Apache-2.0
"""Library objects in an ``alloy.Scene`` (``Scene.add_object``): the item a placement records, and what the build
needs to give one object of a card values of its own.

``add_object`` takes the shared keywords of every ``add_*``: ``static``, ``kinematic``, ``rigid``, ``material``,
``mass`` or ``density``, ``friction`` and ``restitution``. They change this object only: the build writes a copy of
its card under the object's name, with a derived material ``<base>_for_<object>`` when a value changes, as it does for
``add_box`` objects (``alloy.api._materialize``). ``Scene.set_friction`` and a randomized static friction below the
card's dynamic friction take the same derived material, so they work on library objects and on the objects of a
loaded scene as they do on primitives.

A card whose parts are of several materials (``part_materials``) takes ``static``, ``kinematic``, ``rigid`` and
``material`` (its main material), but a value or a pair friction would change only its main material, so those raise
a ``SceneError`` naming the parts.

Example::

    scene.add_object("cell", "aa_cell", pos=(0, 0, 0.03), mass=0.024, friction=0.9)
    scene.add_object("sponge", "sponge_kitchen", pos=(0, 0, 0.05), rigid=True)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from alloy.api import _args as A
from alloy.api import _worlds, catalog
from alloy.api._items import _overrides
from alloy.api.errors import SceneError

ANCHORS = ("auto", "center", "base", "origin")
#: card primitive (sizes: half extents, [radius], [radius, half height]) -> its volume in m^3
PRIMITIVE_VOLUME = {
    "box": lambda s: 8.0 * s[0] * s[1] * s[2],
    "ellipsoid": lambda s: 4.0 / 3.0 * math.pi * s[0] * s[1] * s[2],
    "sphere": lambda s: 4.0 / 3.0 * math.pi * s[0] ** 3,
    "cylinder": lambda s: 2.0 * math.pi * s[0] ** 2 * s[1],
    "capsule": lambda s: 2.0 * math.pi * s[0] ** 2 * s[1] + 4.0 / 3.0 * math.pi * s[0] ** 3,
    "cone": lambda s: 2.0 / 3.0 * math.pi * s[0] ** 2 * s[1],
}


def card_item(scene: Any, name: str, card: str, pos: Any, orientation: Sequence[Any], anchor: str,
              shared: Mapping[str, Any]) -> dict[str, Any]:
    """The ``Scene.items`` entry of library object ``card`` placed as ``name``; ``orientation`` is (quat_xyzw,
    quat_wxyz, rpy) and ``shared`` the shared ``add_*`` keywords (static, kinematic, rigid, material, mass, density,
    friction, restitution; None or False: the card's own).

    Example: ``card_item(scene, "cell", "aa_cell", (0, 0, 0.03), (None, None, None), "auto", {"friction": 0.9})``."""
    A.check_name(name, list(scene.items))
    if not catalog.can_place(card, scene.roots):
        raise catalog.missing_object(card, scene.roots)
    static, kinematic, rigid = shared.get("static"), bool(shared.get("kinematic")), shared.get("rigid")
    if static and kinematic:
        raise SceneError("an object cannot be both static and kinematic", "pass one of static=True (never moves) or "
                         "kinematic=True (moved by sim.drive / set_pose)", field="kinematic")
    if static and _worlds.per_world(pos):
        _worlds.refuse_static(name)
    material = shared.get("material")
    if material is not None:
        scene._check_material(material)
    p0, per_world = A.positions(pos, scene.worlds)
    item: dict[str, Any] = {"kind": "card", "card_name": card, "static": static, "per_world_pos": per_world,
                            "pose": p0 + A.orientation(*orientation), "randomize": [],
                            "anchor": A.one_of(anchor, ANCHORS, "anchor")}
    extra = card_extra(static, kinematic, rigid)
    ov = _overrides(shared.get("mass"), shared.get("density"), shared.get("friction"), shared.get("restitution"),
                    bool(static))
    item.update({k: v for k, v in (("material", material), ("overrides", ov), ("card_extra", extra)) if v})
    return item


def card_extra(static: bool | None, kinematic: bool, rigid: bool | None) -> dict[str, Any]:
    """The object card fields ``kinematic`` and ``rigid`` set, as ``Scene.add_box`` writes them.

    Example: ``card_extra(None, False, True)`` -> ``{"route": {"route": "rigid", "reason": ...}}``."""
    out: dict[str, Any] = {}
    if kinematic:
        out["motion"] = "kinematic"
    if rigid is not None and not static:
        if rigid not in (True, False):
            raise SceneError(f"rigid must be True, False or None, got {rigid!r}", "pass rigid=True (always a rigid "
                             "body) or rigid=False (a soft body)", field="rigid")
        out["route"] = {"route": "rigid" if rigid else "soft_vbd", "reason": f"set through alloy.Scene (rigid={rigid})"}
    return out


def needs_material(name: str, item: Mapping[str, Any], paired: set[str]) -> bool:
    """Whether a library object may need a material of its own: ``material=`` or a value keyword, a pair friction
    (``paired``: the objects ``Scene.set_friction`` names) or a randomized static friction (its dynamic friction may
    need a cap). Example: ``needs_material("cell", scene.items["cell"], {"cell"})`` -> True."""
    lowers = any(r["field"] == "material.surface.mu_static" for r in item.get("randomize") or [])
    return bool(name in paired or lowers or item.get("overrides") or item.get("material"))


def changes(name: str, item: Mapping[str, Any], paired: set[str]) -> bool:
    """Whether a library object needs a card of its own beyond ``static``: a material of its own
    (``needs_material``), or ``kinematic=`` / ``rigid=``. Example: ``changes("cell", scene.items["cell"], set())``."""
    return needs_material(name, item, paired) or bool(item.get("card_extra"))


def refuse_parts(name: str, data: Mapping[str, Any], asked: Sequence[str]) -> None:
    """A SceneError when a card of several materials (``part_materials``) is asked for values that would change only
    its main material. Example: ``refuse_parts("driver", card_data, ["friction"])``."""
    parts = dict(data.get("part_materials") or {})
    if not parts or not asked:
        return
    raise SceneError(f"{name!r}: card {data.get('name')!r} is made of several materials (main {data.get('material')!r}"
                     f", parts {parts}); {', '.join(asked)} would change only its main material",
                     "copy the card into a folder of your own (alloy.Scene(roots=[folder])) and set each part's "
                     "material there, or place the part that matters with add_mesh(..., friction=...)",
                     field=asked[0])


def value_asks(name: str, ov: Mapping[str, Any], paired: set[str]) -> list[str]:
    """The keywords and calls that set the object's own material values (``refuse_parts`` names them).

    Example: ``value_asks("cell", {"mass": 0.02}, {"cell"})`` -> ``["mass=", "set_friction"]``."""
    return [*(f"{k}=" for k in ov), *(["set_friction"] if name in paired else [])]


def volume(name: str, card: str, roots: Sequence[str]) -> float:
    """The volume (m^3) of library object ``card``, which ``mass=`` turns into a density; a SceneError when its card
    does not fix one (an MJCF or USD source, an open mesh). Example: ``volume("cell", "aa_cell", [])``."""
    from alloy.api import _anchor  # noqa: PLC0415
    from alloy.cards import geometry as geo  # noqa: PLC0415
    from alloy.cards import types as ct  # noqa: PLC0415

    lib = _anchor.card_library(roots, [card])
    obj = lib.peek_object(card)
    if isinstance(obj.shape, ct.Primitive) and obj.shape.kind in PRIMITIVE_VOLUME:
        v = PRIMITIVE_VOLUME[obj.shape.kind]([float(s) for s in obj.shape.size])
    else:
        mesh = lib.mesh_for(obj)
        v = geo.signed_volume(mesh) if mesh is not None and len(mesh.faces) else 0.0
    if not v > 0.0:
        raise SceneError(f"{name!r}: card {card!r} ({type(obj.shape).__name__}) has no closed volume to turn mass= "
                         "into a density", "pass density= (kg/m^3) instead of mass=", field="mass")
    return v


def refuse_shared(name: str, card: str, items: Mapping[str, Mapping[str, Any]]) -> None:
    """A SceneError when an object of a loaded scene needs a card of its own but shares its card with other objects
    there (the engine names them ``card``, ``card#1``, ...): a changed card would change them all.

    Example: ``refuse_shared("cube", "cube", scene.items)``."""
    others = sorted(n for n, it in items.items()
                    if n != name and str((it.get("raw_placement") or {}).get("object")) == card)
    if not others and name == card:
        return
    raise SceneError(f"{name!r} shares object card {card!r} with {others or [card]} in this loaded scene, so "
                     "set_friction or its friction range would change them all",
                     f"give it a card of its own: copy objects/{card}.yaml to objects/<new name>.yaml (name: <new "
                     "name>) in the scene's card folder and place <new name> in the scene card", field="name")
