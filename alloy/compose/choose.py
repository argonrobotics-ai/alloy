# SPDX-License-Identifier: Apache-2.0
"""Which object card answers each item of a spec: search the catalog, keep what can be placed, pick one.

The base scene takes the best match; a variation (``swap=True``) draws among the matches that score as well as the
best (any mug for "a mug"), so variations differ in objects as well as in poses. "small" / "large" pick the
smallest / largest of those. Items with no placeable match are reported, not fatal, unless nothing is left.

Example::

    >>> pick = choose(spec, catalog, rng=None)            # doctest: +SKIP
    >>> pick.cards                                         # doctest: +SKIP
    {'mug': 'mug_stoneware', 'cube': 'cube_beech_30'}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from alloy.compose import words as W
from alloy.compose.spec import ComposeError, ItemSpec, SceneSpec

log = logging.getLogger("alloy.compose")

TIER = 1.0                 # matches within this score of the best are equally good
LOOK = 25                  # matches examined per item (by score) for a placeable one


@dataclass
class Choice:
    """The card for each item query (None: nothing placeable) and notes for the result."""

    cards: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def choose(spec: SceneSpec, catalog, rng: np.random.Generator | None = None, *, swap: bool = False) -> Choice:
    """A card per item (see the module docstring); ``ComposeError`` when no item has a placeable match."""
    out = Choice()
    soft = soft_queries(spec)
    for item in spec.items:
        if item.query not in out.cards and item.query not in soft:
            out.cards[item.query] = choose_one(item, catalog, rng if swap else None, out.notes)
    if not any(out.cards.values()) and not soft:
        raise ComposeError(f"no placeable object for any of {[i.query for i in spec.items]}",
                           "use other nouns: alloy.compose.search('<noun>') lists what exists and what can be placed",
                           field="items")
    return out


def soft_queries(spec: SceneSpec) -> set[str]:
    """Queries the soft builder makes (liquids, grains, cloth, cables) or holds (a pour's vessel), not chosen here.

    Example: ``soft_queries(parse("pour water from a bottle into a cup")) == {"water", "bottle"}``."""
    out = {i.query for i in spec.items if i.soft is not None}
    out |= {i.soft.source for i in spec.items if i.soft is not None and i.soft.source}
    return out


def _no_match_hint() -> str:
    """What to do when no object matches: the ``import`` extra on a base install (the Hub holds most objects), else
    another word (GEO P4)."""
    from alloy.api.catalog import _hub_unreachable  # noqa: PLC0415

    if _hub_unreachable():
        return 'the object library on the Hugging Face Hub needs the import extra: pip install "alloy-engine[import]"'
    return "try another word (alloy.compose.search lists names)"


def choose_one(item: ItemSpec, catalog, rng: np.random.Generator | None, notes: list[str]) -> str | None:
    """Draw from the best-scoring tier, reading (and fetching) only the card drawn; an unplaceable draw is dropped
    and the draw repeated, down the ranking."""
    if item.object is not None:
        info = catalog.info(item.object)
        if not info.placeable:
            notes.append(f"{item.query}: {item.object} cannot be placed ({info.reason})")
        return item.object if info.placeable else None
    matches = catalog.search(item.query, item.words, k=LOOK)
    if not matches:
        notes.append(f"{item.query}: no object matches; {_no_match_hint()}")
        return None
    skipped: list[tuple[str, str]] = []
    floor = matches[0].score - TIER                  # never fall back to a weaker meaning ("apple" -> a game box)
    matches = [m for m in matches if m.score > floor]
    while matches:
        tier = _sized(item, [m.name for m in matches], catalog, rng)
        name = tier[0] if rng is None else tier[int(rng.integers(len(tier)))]
        info = catalog.info(name)
        if info.placeable and not (info.sheet and not _named(item, name)):
            return name
        skipped.append((name, info.reason or "a flat sheet: ask for it as cloth ('a sheet of kraft paper')"))
        matches = [m for m in matches if m.name != name]
    notes.append(f"{item.query}: the best matches cannot be placed: "
                 + "; ".join(f"{n} ({why})" for n, why in skipped[:3]))
    return None


def _named(item: ItemSpec, name: str) -> bool:
    """True when the item's words include a word only ``W.CARD_WORDS`` gives the card ("a bag insert")."""
    return bool({item.query, *item.words} & set(W.CARD_WORDS.get(name, ())))


def _sized(item: ItemSpec, tier: list[str], catalog, rng: np.random.Generator | None) -> list[str]:
    """For "small" / "large": the tier sorted by volume, cut to its smaller / larger half (the extreme without rng);
    unplaceable members sort last."""
    size = next((W.SIZE_WORDS[w] for w in item.words if w in W.SIZE_WORDS), None)
    if size is None or len(tier) < 2:
        return tier
    infos = {n: catalog.info(n) for n in tier}
    ok = [n for n in tier if infos[n].placeable] or tier
    ok = sorted(ok, key=lambda n: float(np.prod(infos[n].size or (1.0, 1.0, 1.0))), reverse=size == "large")
    return ok[:max(1, len(ok) // 2)] if rng is not None else ok[:1]


__all__ = ["Choice", "choose", "choose_one", "soft_queries"]
