# SPDX-License-Identifier: Apache-2.0
"""Look rules for renderers: one WP20 look fragment that gives every card of a scene its resolved look.

Renderers match a look's ``assign`` regexes against shape and body labels. Alloy labels a card's body with the card
name and its shapes ``<card>/<part>`` (copies ``<card>#k``). ``fallback_look(cards)`` returns
``{"materials": {...}, "assign": [...]}`` with one rule per card (and one per part rule of an object look) in the
same format, so a renderer consults it **after** the scene's own look: authored looks still win, and a shape no rule
matched gets its card's look instead of grey.

Example::

    >>> from alloy.looks.fallback import fallback_look
    >>> fb = fallback_look(["screwdriver_pz2"])                         # doctest: +SKIP
    >>> fb["assign"][0]                                                 # doctest: +SKIP
    {'match': '^(?=screwdriver_pz2(?:[/#]|$)).*(?:grip)', 'material': 'screwdriver_pz2.grip'}
    >>> fb["materials"]["screwdriver_pz2.grip"]["preset"]              # doctest: +SKIP
    'plastic_gloss'
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from alloy.cards import CardLibrary
from alloy.cards import types as ct
from alloy.looks.resolve import resolve


def card_name(label: str) -> str:
    """The card a body or shape label belongs to: ``"mug/hull_3"`` -> ``"mug"``, ``"puck#1"`` -> ``"puck"``."""
    return re.split(r"[/#]", label, maxsplit=1)[0]


def _card(item: str | ct.ObjectCard, lib: CardLibrary) -> ct.ObjectCard | None:
    if isinstance(item, ct.ObjectCard):
        return item
    name = card_name(item)
    return lib.peek_object(name) if name and lib.has_object(name) else None


def fallback_look(cards: Iterable[str | ct.ObjectCard], lib: CardLibrary | None = None) -> dict:
    """WP20 look fragment for ``cards`` (card names, body/shape labels, or ObjectCards; unknown labels skipped).

    Part rules of an object look match the part inside that card's labels only; part regexes are not anchored.
    """
    lib = lib or CardLibrary()
    materials: dict[str, dict] = {}
    assign: list[dict] = []
    seen: set[str] = set()
    for item in cards:
        card = _card(item, lib)
        if card is None or card.name in seen:
            continue
        seen.add(card.name)
        look = resolve(card, lib)
        head = f"^(?={re.escape(card.name)}(?:[/#]|$))"
        for rule in look.assign:
            key = f"{card.name}.{rule['material']}"
            materials[key] = dict(look.materials[rule["material"]])
            assign.append({"match": f"{head}.*(?:{rule['match']})", "material": key})
        key = f"{card.name}.{look.default}"
        materials[key] = dict(look.materials[look.default])
        assign.append({"match": head, "material": key})
    return {"materials": materials, "assign": assign}


__all__ = ["card_name", "fallback_look"]
