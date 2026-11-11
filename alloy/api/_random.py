# SPDX-License-Identifier: Apache-2.0
"""``Scene.randomize`` fields: the short names, and the check that names a wrong field when it is given.

A field is a short name (``FIELDS``) or a card path to a number on the object's card or its material, such as
``"material.surface.mu_dynamic"`` (``alloy.cards.paths.kind_for_path`` checks the path against the card types, so no
card is needed yet). A wrong name used to fail only at build, as ``CardError: unknown card path``, without the valid
names (agent trial r2, R2-D2); now ``randomize`` raises a ``SceneError`` that lists them and says what to use instead
of a quantity that is not a card field (mass comes from density).

Only the paths the engine can write per world are taken (``DRAWN``, as ``alloy.scene.randomize``; ``ROUTING`` paths
are sized at the worst end of their range). Any other card number (``material.elastic.poisson``) failed the build
with "not supported", and ``material.surface.restitution`` with a ReadbackError (AUDIT-STACK AGENTS-1 / API-2 /
WORLDS-1 / WORLDS-2): rigid contacts step in MuJoCo, which keeps one restitution for every world, so ``randomize``
refuses both and says so. ``material.surface.contact_stiffness`` is refused for the same reason (WP2, 2026-11-10: it
was offered, and a plain ``add_box`` then failed the build with "not a numeric leaf": its card has none).
"""

from __future__ import annotations

from typing import Any

from alloy.api.errors import SceneError

#: short names accepted by Scene.randomize -> card paths
FIELDS = {"x": "pose.0", "y": "pose.1", "z": "pose.2", "density": "material.density",
          "friction": "material.surface.mu_static", "youngs": "material.elastic.youngs"}
#: names agents reach for that are not card fields -> what to randomize instead
INSTEAD = {"mass": "density (mass = density x volume)", "weight": "density (mass = density x volume)",
           "pos": "x, y and z", "position": "x, y and z", "mu": "friction", "friction_static": "friction",
           "youngs_modulus": "youngs", "stiffness": "youngs", "e": "youngs"}
EXAMPLE_PATH = "material.surface.mu_dynamic"
#: card paths the engine writes per world (alloy.scene.randomize); ROUTING: first segments sized at their worst end
DRAWN = frozenset({"material.density", "material.elastic.youngs", "material.surface.mu_static",
                   "material.surface.mu_dynamic", "pose.0", "pose.1", "pose.2"})
ROUTING = ("loads", "features", "budget")
#: card numbers that exist but cannot differ between worlds -> why, and what to do
NOT_PER_WORLD = {"material.surface.restitution": "rigid contacts keep one restitution for every world; set it for "
                                                 "all worlds with restitution= on the object",
                 "material.surface.contact_stiffness": "rigid contacts keep one contact stiffness for every world, "
                                                       "and a shape's material card has none to draw"}


def card_path(name: str, field: str) -> str:
    """The card path ``Scene.randomize(name, field, ...)`` draws, or a SceneError listing the valid fields.

    Example: ``card_path("cube", "density")`` -> ``"material.density"``."""
    from alloy.cards.paths import PathError, kind_for_path  # noqa: PLC0415 - the card types, on first use

    path = FIELDS.get(field, field)
    try:
        kind_for_path(path)
    except (PathError, ValueError, TypeError, IndexError) as exc:
        instead = INSTEAD.get(str(field).lower())
        hint = f"; randomize {instead} instead" if instead else ""
        raise SceneError(f"{name!r}: cannot randomize {field!r}: not a field of the object's card{hint}",
                         f"use one of {sorted(FIELDS)} or a card path to a number, such as {EXAMPLE_PATH!r}",
                         field="field") from exc
    if path not in DRAWN and path.split(".")[0] not in ROUTING:
        why = NOT_PER_WORLD.get(path, "the engine cannot draw this card number per world")
        paths = sorted(DRAWN - set(FIELDS.values()))
        raise SceneError(f"{name!r}: cannot randomize {field!r}: {why}",
                         f"randomize one of {sorted(FIELDS)}, or a card path among {paths}", field="field")
    return path


def add_row(scene: Any, name: str, field: str, lo: float, hi: float, when: str = "build") -> None:
    """``Scene.randomize``: check the row and record it on the scene (an object's item, or ``scene.robot_random``
    for a robot's physical fields and gravity).

    Example: ``add_row(scene, "cube", "x", -0.005, 0.005)``."""
    from alloy.api import _robot_random, _robots, _rods, _worlds  # noqa: PLC0415

    if not hi >= lo:
        raise SceneError(f"randomize range [{lo}, {hi}] is empty", "pass lo <= hi", field="lo")
    if _robot_random.is_robot_row(scene, name, field):
        return _robot_random.add(scene, name, field, lo, hi, when)
    if when != "build":
        raise SceneError(f"{name!r}: when={when!r} is for robot fields and gravity; an object's {field!r} is drawn "
                         "at build", "drop when= (the value is drawn once per world, at build)", field="when")
    item = scene._item(name)
    if item["kind"] == "robot":
        return _robots.add_random(item, name, field, lo, hi)
    if field in _worlds.AXES and (item.get("static") or item["kind"] == "ground"):
        _worlds.refuse_static(name)
    if field in _worlds.AXES and item["kind"] == "rod":
        _rods.refuse_per_world(name)
    if field in _worlds.AXES and item["kind"] in _worlds.PARTICLE_KINDS:
        return _worlds.add_start_random(item, name, field, lo, hi)
    if "randomize" not in item:
        raise SceneError(f"{name!r} cannot be randomized ({item['kind']})", "randomize objects and robot bases",
                         field="name")
    row = {"field": card_path(name, field), "lo": float(lo), "hi": float(hi)}
    raw = item.get("raw_placement")
    (raw.setdefault("randomize", []) if raw is not None else item["randomize"]).append(row)
