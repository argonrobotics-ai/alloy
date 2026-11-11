# SPDX-License-Identifier: Apache-2.0
"""Route pairs the engine refuses in one scene, found from the plan before any build.

``alloy.coupling.unsupported`` lists the solver pairs Alloy cannot step together yet (cloth or a soft body beside
grains or a paste, cloth beside a film sheet, ...), and ``alloy.build`` refuses them with the pair and the nearest
supported alternative. The scene builder plans each scene (routes only, no build: about a second) and takes out one
side of a refused pair, so a sentence such as "a bin of rice and a towel" gives a scene that builds, with the part
it left out reported missing and why, instead of one whose first build raises.

Cables are left out of the plan: their routes (chain, rod_vbd) are in no refused pair, and routing one runs the
engine's cable self-test (minutes on a cold machine).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from alloy.cards import types as ct

log = logging.getLogger("alloy.compose")


def refused(card: ct.SceneCard, library: Any, skip: frozenset[str] = frozenset()) -> tuple[Exception, dict] | None:
    """(the engine's ``UnsupportedPairError``, object name -> route) when ``card`` holds a route pair the engine
    refuses in one scene, else None. ``skip``: objects (cables) left out of the plan. A plan that fails for another
    reason gives None: the load check reports it.

    Example: ``refused(card, lib)[0].pair`` is ``(cloth_vbd, grain_mpm)`` for a towel beside a bin of rice."""
    from alloy.coupling.unsupported import UnsupportedPairError, check_built  # noqa: PLC0415
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    if skip:
        card = dataclasses.replace(card, objects=tuple(p for p in card.objects if p.object not in skip),
                                   attachments=())
    try:
        plan = SceneAssembler(library).plan(card)
    except Exception as exc:  # noqa: BLE001 - not this check's finding; the load check names it
        log.debug("compose: pair check skipped, the plan failed: %s: %s", type(exc).__name__, exc)
        return None
    try:
        check_built(plan)
    except UnsupportedPairError as exc:
        return exc, {name: str(d.route) for name, d in plan.report.decisions.items()}
    return None


def pick(pair: tuple[Any, Any], routes: dict[str, str], extras: set[str]) -> list[str]:
    """The objects to take out for a refused ``pair``: every object on the side with fills or poured parts
    (``extras``; the second side if both have some), else every object on the second side.

    Example: ``pick(("cloth_vbd", "grain_mpm"), {"towel": "cloth_vbd", "rice": "grain_mpm"}, {"rice"}) == ["rice"]``."""
    sides = [[n for n, r in routes.items() if r == str(p)] for p in pair]
    filled = [s for s in sides if any(n in extras for n in s)]
    return filled[-1] if filled else sides[1]


def reason(pair: tuple[Any, Any], routes: dict[str, str], gone: list[str], exc: Exception,
           label: dict[str, str] | None = None) -> str:
    """Why ``gone`` was left out, with what to ask for instead, then the engine's message; ``label`` names objects
    by the sentence's words (object name -> "rice").

    Example: ``reason(pair, routes, ["scene_rice_1"], exc, {"scene_rice_1": "rice", "scene_towel_1": "towel"})``
    starts ``"the rice cannot share a scene with the towel yet"``."""
    label = label or {}
    kept = [n for n, r in routes.items() if r in {str(p) for p in pair} and n not in gone]
    return (f"{_words(gone, label)} cannot share a scene with {_words(kept, label)} yet: ask for one of them per "
            f"scene. Engine: {exc}")


def _words(names: list[str], label: dict[str, str]) -> str:
    """``"the rice and the sand"`` for those objects, each named once. Example: ``_words(["a"], {"a": "rice"})``."""
    return " and ".join(dict.fromkeys(f"the {label.get(n, n)}" for n in names))


__all__ = ["pick", "reason", "refused"]
