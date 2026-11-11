# SPDX-License-Identifier: Apache-2.0
"""Apply the mechanical fixes of a ``CheckReport`` to the scene card: new placement poses (overlaps, settled poses)
and replacement object names (unknown objects with a close library match).

Only findings with a ``suggestion`` are applied; the rest need a person or an agent to choose. Check again after
applying: one overlap fixed can reveal the next.

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

from alloy.cards import types as ct
from alloy.check.report import CheckReport

log = logging.getLogger("alloy.check")

_PLACEMENT = re.compile(r"^(?:[^.]+\.)?objects\.(?P<i>\d+)\.(?P<what>pose|object)$")


def apply_fixes(card: ct.SceneCard, report: CheckReport, levels: tuple[str, ...] = ("error", "warn")
                ) -> tuple[ct.SceneCard, list[str]]:
    """``card`` with every suggested pose / object-name edit of findings at ``levels`` applied, and a list of what
    changed (one line each). A placement gets at most one edit (the first, i.e. the most severe).

    Example::

        >>> fixed, applied = apply_fixes(src.card, report)  # doctest: +SKIP
        >>> applied[0]
        'objects.1.pose: lift mug by 4 mm'
    """
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


__all__ = ["apply_fixes"]
