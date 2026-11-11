# SPDX-License-Identifier: Apache-2.0
"""The soft parts of one composed scene: generated cards, the placements that hang off other objects (a liquid in
its cup, a cable plugged into its target), attachments, emitters, result rows and estimates.

A placement that depends on other objects is computed from the layout when the scene card is made, so when the
final check moves a cup, its water moves with it; when the cup is dropped, the water is reported missing.

Example::

    >>> soft = SoftScene()
    >>> soft.placements({}) , soft.missing
    ([], [])
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alloy.cards import types as ct
from alloy.compose.soft.cards import SoftCard

log = logging.getLogger("alloy.compose")


@dataclass
class Extra:
    """A soft placement outside the layout: ``card``, the layout keys it ``depends`` on, ``pose(placed)`` (placed:
    layout key -> Placed) and its result ``row``."""

    card: SoftCard
    depends: tuple[str, ...]
    pose: Callable[[dict[str, Any]], tuple[float, ...]]
    row: dict[str, Any]


@dataclass
class LazyEmitter:
    """An emitter whose pose follows the container it pours into (``depends``)."""

    depends: tuple[str, ...]
    make: Callable[[dict[str, Any]], ct.EmitterSpec]
    material: str = ""


@dataclass
class Weld:
    """A cable end (``cable`` card name and link index) welded to a layout object (``key``)."""

    cable: str
    link: int
    key: str


@dataclass
class SoftScene:
    """Everything soft in one scene (see the module docstring). ``settle_s``: how long the scene takes to come to
    rest (drapes, piles) or to finish moving (pours); ``expect_rest``: False when it moves by design (a pour)."""

    cards: dict[str, SoftCard] = field(default_factory=dict)
    extras: list[Extra] = field(default_factory=list)
    welds: list[Weld] = field(default_factory=list)
    emitters: list[LazyEmitter] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    estimated: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    watches: list[Any] = field(default_factory=list)          # alloy.compose.soft.probe.Watch per soft item
    settle_s: float = 0.0
    expect_rest: bool = True
    hold_still: bool = False                                  # place the scene's objects as static copies

    def still(self, card: str, library: Any, work: Path) -> str:
        """The name of a static copy of object ``card`` (made once): a tilting pour holds the scene's objects still,
        since beside free rigid bodies Alloy's liquid does not yet feel a moving (kinematic) vessel."""
        from alloy.compose.soft.cards import static_copy  # noqa: PLC0415

        name = f"{card}_still"
        if name not in self.cards:
            self.add_card(static_copy(name, library.object(card)), work)
            library.rescan()                                      # the checks read the new card from the work root
        return name

    def add_card(self, card: SoftCard, work: Path) -> None:
        """Keep ``card`` and write it (and its files) into the work card root."""
        self.cards[card.name] = card
        (work / "objects").mkdir(parents=True, exist_ok=True)
        (work / "objects" / f"{card.name}.yaml").write_text(card.text, encoding="utf-8")
        for rel, data in card.files.items():
            p = work / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def estimate(self, note: str) -> None:
        if note not in self.estimated:
            self.estimated.append(note)

    def placements(self, placed: dict[str, Any]) -> list[ct.Placement]:
        """Placements of the extras whose objects are still placed (``placed``: layout key -> Placed)."""
        out = []
        for e in self.extras:
            if all(k in placed for k in e.depends):
                out.append(ct.Placement(e.card.name, tuple(float(v) for v in e.pose(placed))))
        return out

    def dropped(self, placed: dict[str, Any]) -> list[str]:
        """Extras lost because an object they hang off was dropped by the final repairs."""
        return [f"{e.row.get('item', e.card.name)} ({e.card.name}: its {', '.join(k for k in e.depends if k not in
                placed)} was dropped)" for e in self.extras if not all(k in placed for k in e.depends)]

    def attachments_for(self, placed: dict[str, Any], names: dict[str, str]) -> tuple[ct.Attachment, ...]:
        """Welds whose cable and object are both in the scene (``names``: engine name -> layout key)."""
        engine = {k: n for n, k in names.items()}
        present = {e.card.name for e in self.extras if all(k in placed for k in e.depends)}
        return tuple(ct.Attachment(f"{w.cable}:{w.link}", engine[w.key], "weld") for w in self.welds
                     if w.cable in present and w.key in engine)

    def emitters_for(self, placed: dict[str, Any]) -> tuple[ct.EmitterSpec, ...]:
        return tuple(e.make(placed) for e in self.emitters if all(k in placed for k in e.depends))

    def rows_for(self, placed: dict[str, Any]) -> list[dict[str, Any]]:
        """Result rows of the soft items still in the scene."""
        keep = {e.card.name for e in self.extras if all(k in placed for k in e.depends)} | set(placed) | \
            {p.card for p in placed.values()}
        rows = [r for r in self.rows if r.get("name") in keep or r.get("card") in keep or r.get("into") in placed]
        return [{k: (v(placed) if callable(v) else v) for k, v in r.items()} for r in rows]

    def to_dict(self) -> dict[str, Any]:
        return {"settle_s": round(self.settle_s, 3), "moves_by_design": not self.expect_rest,
                "estimated": list(self.estimated)}


__all__ = ["Extra", "LazyEmitter", "SoftScene", "Weld"]
