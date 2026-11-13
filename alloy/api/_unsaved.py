# SPDX-License-Identifier: Apache-2.0
"""What ``scene.save()``'s cards cannot keep, said when it saves: one warning per kind, naming the objects and the
fix (``scene.to_json()`` / ``Scene.from_dict`` keep everything).

The card schema (``alloy.cards.types``) has no per-world start rows, no fixture that makes several parts one body, no
control period apart from the engine's own step, no anchor of a placed object, no heat, no solver settings (cloth and
liquid accuracy, rigid step) and none of the robot physics the API adds (yielding joints, robot randomization, timed
pushes), and its decoder refuses keys it does not know, so none of these can be written into the cards. Without a
word, a scene loaded from them would start every world at world 0's start, load a kinematic fixture as separate parts
that ``sim.drive(name)`` cannot find (and driving its floor leaves the walls behind), step a precise-fit scene's fine
step per ``sim.step()``, read an anchored object's ``.pos`` at its body origin, run cold, step cloth at the default
accuracy and let bent joints spring back. Rod clamps, robot sysid and relaxed contacts are said by
``_contact_relax.warn_unsaved``; pours are kept as card emitters (``_pour.save_as_emitters``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from alloy.api import _say

KEEP = "Keep them with scene.to_json() / Scene.from_dict"


def warn(scene: Any) -> None:
    """Every warning ``scene.save()`` gives for what its cards do not keep (anchors: ``anchored``, after the cards
    are written). Example: ``warn(scene)``."""
    from alloy.api import _contact_relax  # noqa: PLC0415

    _contact_relax.warn_unsaved(scene)            # rod clamps, robot sysid, relaxed contacts
    per_world_starts(scene.items)
    kinematic_fixtures(scene.items)
    precise_period(scene)
    settings(scene)
    physics(scene)


def _owner(name: str, item: Mapping[str, Any]) -> str:
    """A fixture part's fixture name, else the object's own name."""
    return str(item.get("fixture") or name)


def per_world_starts(items: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Warn about (and return) what starts per world in a way the cards cannot hold: ``pos=`` rows, a robot's
    randomized base, a cloth's or liquid's randomized start. An object's randomized x / y / z is a card field (a
    placement's ``randomize``) and is kept. Example: ``per_world_starts(scene.items)`` -> ``["cube"]``."""
    lost = sorted({_owner(n, it) for n, it in items.items()
                   if it.get("per_world_pos") is not None or it.get("start_random")
                   or (it.get("kind") == "robot" and it.get("randomize"))})
    if lost:
        _say.warn("scene.save(): the cards do not keep the per-world starts (pos= rows, randomized robot bases, cloth "
                  "and liquid starts) of %s; a scene loaded from them starts every world at world 0's position. %s.",
                  ", ".join(lost), KEEP)
    return lost


def kinematic_fixtures(items: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Warn about (and return) the kinematic fixtures: the cards hold each part as its own kinematic object, so a
    scene loaded from them has no body named after the fixture. Example: ``kinematic_fixtures(scene.items)`` ->
    ``["bin"]``."""
    from alloy.api.fixtures import kinematic_groups  # noqa: PLC0415

    groups = kinematic_groups(items)
    for name, parts in groups.items():
        _say.warn("scene.save(): the cards keep the kinematic fixture %s as %d separate kinematic objects (%s), not "
                  "one body: a scene loaded from them has no %r to sim.drive, and driving %r moves that part alone. "
                  "Keep it with scene.to_json() / Scene.from_dict.", name, len(parts), ", ".join(parts), name,
                  parts[0])
    return sorted(groups)


def precise_period(scene: Any) -> int:
    """Warn when a precise fixture (``add_block_with_hole``) makes the cards' control period the fine rigid step:
    a scene loaded from them steps that much less time per ``sim.step()`` (its physics and ``sim.run(seconds)`` are
    the same). Returns the engine periods per ``scene.dt`` (1: nothing said). Example: ``precise_period(scene)`` ->
    ``64`` at dt 0.01."""
    from alloy.api.fixtures import precise_substeps  # noqa: PLC0415

    n = precise_substeps(scene)
    if n > 1:
        names = sorted({_owner(k, it) for k, it in scene.items.items() if it.get("precise")})
        _say.warn("scene.save(): the cards keep the precise-fit step of %s as their control period, so a scene loaded "
                  "from them has dt %.6g s, not %.6g s: one sim.step() covers %dx less time (its physics and "
                  "sim.run(seconds) are the same). Keep dt with scene.to_json() / Scene.from_dict.",
                  ", ".join(names), scene.dt / n, scene.dt, n)
    return n


def settings(scene: Any) -> list[str]:
    """Warn about (and return) the scene's solver settings the cards cannot hold: ``cloth_accuracy``,
    ``liquid_accuracy`` and ``rigid_step`` away from their defaults (Scene JSON keeps them). Example:
    ``settings(alloy.Scene(cloth_accuracy="fast"))`` -> ``["cloth_accuracy='fast'"]``."""
    from alloy.api import _rigid_step  # noqa: PLC0415

    lost = [f"{key}={value!r}" for key, value in (("cloth_accuracy", scene.cloth_accuracy),
                                                  ("liquid_accuracy", scene.liquid_accuracy)) if value != "default"]
    if not _rigid_step.is_default(scene.rigid_step):
        lost.append(f"rigid_step={scene.rigid_step!r}")
    if lost:
        _say.warn("scene.save(): the cards do not keep %s; a scene loaded from them steps at the defaults. Keep "
                  "them with scene.to_json() / Scene.from_dict, or set them again on the scene Scene.load returns.",
                  ", ".join(lost))
    return lost


def heated(scene: Any) -> list[str]:
    """What heats or is heated in ``scene`` that the cards drop: start temperatures (a pour's is said with the pour),
    heaters, and ``scene.set_heat`` (as "set_heat"). Example: ``heated(scene)`` -> ``["hotplate", "water"]``."""
    thermal = getattr(scene, "thermal", None) or {}
    pours = {n for n, it in scene.items.items() if it.get("pour")}
    names = [n for n in thermal.get("objects") or {} if n not in pours] + list(thermal.get("heaters") or {})
    return sorted(set(names)) + (["set_heat"] if thermal.get("config") else [])


# kind -> (the names it applies to, what a scene loaded from the cards does without it)
PHYSICS: dict[str, tuple[Callable[[Any], list[str]], str]] = {
    "heat": (heated, 'it has no heat: no "<name>.temperature" in its state, sim.heat() raises, and a heater is a '
             "plain static box"),
    "set_joint_yield": (lambda s: sorted(getattr(s, "joint_yield", None) or {}), "those joints spring back "
                        'elastically, with no "<joint>.rest" or "<joint>.plastic_strain"'),
    "robot randomization": (lambda s: [f"{r['target']}.{r['field']}" for r in getattr(s, "robot_random", None) or []],
                            "every world starts with the robot file's values"),
    "add_push": (lambda s: [str(p["target"]) for p in getattr(s, "pushes", None) or []], "no body is pushed"),
}


def physics(scene: Any) -> dict[str, list[str]]:
    """Warn about (and return) the physics the API adds that the cards cannot hold, one warning per kind (heat,
    yielding joints, robot randomization, timed pushes). Example: ``physics(scene)`` ->
    ``{"set_joint_yield": ["clip/hinge"]}``."""
    lost = {kind: names_of(scene) for kind, (names_of, _) in PHYSICS.items()}
    lost = {kind: names for kind, names in lost.items() if names}
    for kind, names in lost.items():
        _say.warn("scene.save(): the cards do not keep %s (%s); in a scene loaded from them %s. %s, or set them "
                  "again on the scene Scene.load returns.", kind, ", ".join(names), PHYSICS[kind][1], KEEP)
    return lost


def alloy_only(scene: Any) -> list[tuple[str, str]]:
    """(item, reason) rows for the physics the API adds that no card or export format keeps (``PHYSICS``):
    ``alloy.io.export_scene`` lists them as unmapped. Example: ``alloy_only(scene)`` ->
    ``[("set_joint_yield clip/hinge", "Alloy-only: no export format carries it; ...")]``."""
    reason = "Alloy-only: no export format carries it; keep the scene with scene.to_json() / Scene.from_dict"
    return [(f"{kind} {name}", reason) for kind, (names_of, _) in PHYSICS.items() for name in names_of(scene)]


def anchored(resolved: Any) -> list[str]:
    """Warn about (and return) the objects ``Scene.add_object`` placed by an anchor off their card's origin, given
    ``_anchor.resolve``'s output: the cards place them by their body origin, so in a scene loaded from them
    ``"<name>.pos"`` reads that origin, ``"<name>.body_pos"`` is gone, and ``set_pose`` / ``randomize("z", ...)``
    at the old values move the origin there. Example: ``anchored(_anchor.resolve(scene))`` -> ``["can"]``."""
    moved = {n: -np.asarray(it["anchor_shift"], np.float64) for n, it in resolved.items.items()
             if it.get("anchor_shift") is not None}
    if moved:
        where = ", ".join(f"{n} (origin {(np.round(d * 1e3, 1) + 0.0).tolist()} mm from its anchor)"
                          for n, d in moved.items())
        _say.warn('scene.save(): the cards place %s by the body origin, not by the anchor add_object put at pos; in a '
                  'scene loaded from them "<name>.pos" reads that origin, there is no "<name>.body_pos", and set_pose '
                  "or randomize take origin values (the anchor's values put the object off by that offset: into what "
                  "it stands on when its origin is above the anchor). Keep anchors with scene.to_json() / "
                  "Scene.from_dict.", where)
    return sorted(moved)
