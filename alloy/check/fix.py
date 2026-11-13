# SPDX-License-Identifier: Apache-2.0
"""Apply the mechanical fixes of a ``CheckReport`` to the scene card: new placement poses (overlaps, settled poses)
and replacement object names (unknown objects with a close library match).

Only findings with a ``suggestion`` are applied; the rest need a person or an agent to choose. Check again after
applying: one overlap fixed can reveal the next. ``card`` is what ``alloy.check`` took. A ``SceneCard`` (or a library
scene name) comes back as a fixed ``SceneCard``; a scene ``.yaml`` file (what ``alloy.Scene.save`` returns, as a
``Path`` or ``str``) comes back as the path of a fixed copy beside it, ``<name>_fixed.yaml``, where the scene's own
object and material cards still resolve (the file given is never changed).

Example::

    >>> report = check_load(card)                                    # doctest: +SKIP
    >>> card2, applied = apply_fixes(card, report)
    >>> applied
    ['objects.1.pose: lift cube_beech_30 by 3.5 mm (pose z 0.012 -> 0.0155 m)']
    >>> check_load(card2).ok
    True
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

from alloy.api.errors import SceneError
from alloy.cards import types as ct
from alloy.check.report import CheckReport
from alloy.check.source import SceneInputError, load_source

log = logging.getLogger("alloy.check")

_PLACEMENT = re.compile(r"^(?:[^.]+\.)?objects\.(?P<i>\d+)\.(?P<what>pose|object)$")


def apply_fixes(card: Any, report: CheckReport, levels: tuple[str, ...] = ("error", "warn")
                ) -> tuple[Any, list[str]]:
    """``card`` with every suggested pose / object-name edit of findings at ``levels`` applied, and a list of what
    changed (one line each); a ``.yaml`` file comes back as the path of its fixed copy (module docstring). A placement
    gets at most one edit (the first, i.e. the most severe). A ``card`` or ``report`` of another kind raises
    ``alloy.SceneError`` (``SceneInputError``) with the fix.

    Example::

        >>> path = scene.save("checked_scene")              # doctest: +SKIP
        >>> fixed, applied = apply_fixes(path, alloy.check(path, run=False))
        >>> fixed.name, applied[0]
        ('checked_scene_fixed.yaml', 'objects.1.pose: lift mug by 4 mm')
    """
    path = _scene_file(card)
    fixed, applied = _apply(scene_card(card), report, levels)
    if path is None:
        return fixed, applied
    from alloy.cards import dump_card  # noqa: PLC0415

    out = path.with_name(f"{path.stem}_fixed{path.suffix}")
    out.write_text(dump_card(fixed), encoding="utf-8")
    log.info("check fixes: wrote %s", out)
    return (str(out) if isinstance(card, str) else out), applied


def _scene_file(card: Any) -> Path | None:
    """The existing scene file ``card`` names, else None (a SceneCard, a library scene name, YAML text)."""
    if not isinstance(card, (str, Path)) or (isinstance(card, str) and "\n" in card):
        return None
    path = Path(card).expanduser()
    return path if path.is_file() else None


def _apply(card: ct.SceneCard, report: CheckReport, levels: tuple[str, ...]) -> tuple[ct.SceneCard, list[str]]:
    if not isinstance(report, CheckReport):
        raise SceneError(f"apply_fixes(report=...) got a {type(report).__name__}",
                         "pass the CheckReport that alloy.check (or check_load) returned for this card",
                         field="report")
    objects = list(card.objects)
    done: set[int] = set()
    applied: list[str] = []
    for f in sorted(report.findings, key=lambda f: f.level != "error"):
        s = f.suggestion
        if f.level not in levels or not s:
            continue
        m = _PLACEMENT.match(str(s.get("field", "")))
        if m is None or int(m.group("i")) >= len(objects) or int(m.group("i")) in done:
            continue
        i = int(m.group("i"))
        pl = objects[i]
        if m.group("what") == "pose" and "pose" in s:
            objects[i] = dataclasses.replace(pl, pose=tuple(float(v) for v in s["pose"]))
        elif m.group("what") == "object" and "replace_with" in s:
            objects[i] = dataclasses.replace(pl, object=str(s["replace_with"]))
        else:
            continue
        done.add(i)
        applied.append(f"objects.{i}.{m.group('what')}: {f.fix}")
    log.info("check fixes: %d applied", len(applied))
    return dataclasses.replace(card, objects=tuple(objects)), applied


def scene_card(card: Any) -> ct.SceneCard:
    """``card`` read into a ``SceneCard``: as is, or from a scene ``.yaml`` path, YAML text or a library scene name,
    as ``alloy.check`` reads it; ``SceneInputError`` for anything else.

    Example::

        >>> scene_card("demo_peg_d12").worlds  # doctest: +SKIP
        16
    """
    if isinstance(card, ct.SceneCard):
        return card
    if isinstance(card, (str, Path)):
        return load_source(card).card
    raise SceneInputError(f"apply_fixes(card=...) got a {type(card).__name__}",
                          "pass the scene card that was checked: an alloy.cards.types.SceneCard, its .yaml path (what "
                          "alloy.Scene.save returns) or a library scene name; apply_fixes returns the fixed SceneCard",
                          field="card")


__all__ = ["apply_fixes", "scene_card"]
