# SPDX-License-Identifier: Apache-2.0
"""Load and dump cards: YAML files and strings to card dataclasses and back (``load(dump(c)) == c`` exactly)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from alloy.cards import types as ct
from alloy.cards.codec import Ctx, decode, encode
from alloy.cards.units_yaml import yaml_dump, yaml_load
from alloy.core.errors import CardError

log = logging.getLogger("alloy.cards")

Card = ct.MaterialCard | ct.PairCard | ct.ObjectCard | ct.SceneCard
T = TypeVar("T", ct.MaterialCard, ct.PairCard, ct.ObjectCard, ct.SceneCard)

SCHEMAS: dict[type, str] = {
    ct.MaterialCard: "alloy.material/1", ct.PairCard: "alloy.pair/1",
    ct.ObjectCard: "alloy.object/1", ct.SceneCard: "alloy.scene/1",
}
CARD_TYPES: dict[str, type] = {schema: cls for cls, schema in SCHEMAS.items()}


def _card_label(data: Any, fallback: str) -> str:
    if isinstance(data, Mapping):
        name = data.get("name")
        if isinstance(name, str) and name:
            return name
        a, b = data.get("a"), data.get("b")
        if isinstance(a, str) and isinstance(b, str):
            return "__".join(sorted((a, b)))
    return fallback


def from_data(cls: type[T], data: Any, card: str) -> T:
    """Decode already-parsed YAML data into a card of type ``cls``; the schema key must match if present."""
    label = _card_label(data, card)
    if not isinstance(data, Mapping):
        raise CardError(label, "<card>", f"a card file must hold a mapping, got {type(data).__name__}")
    schema = data.get("schema", SCHEMAS[cls])
    if schema != SCHEMAS[cls]:
        raise CardError(label, "schema", f"expected {SCHEMAS[cls]!r}, got {schema!r}")
    return decode(cls, data, "", Ctx(label))


def loads_card(text: str, cls: type[T], card: str = "<string>") -> T:
    """Parse one card of type ``cls`` from YAML text (structure and units only; see ``alloy.cards.validate``)."""
    return from_data(cls, yaml_load(text, card), card)


def load_card(path: Path, cls: type[T]) -> T:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CardError(path.stem, "<file>", f"cannot read {path}: {exc}") from exc
    card = loads_card(text, cls, path.stem)
    log.debug("loaded %s %s from %s", cls.__name__, path.stem, path)
    return card


def load_any(path: Path) -> Card:
    """Load a card file whose type is given by its ``schema`` key."""
    path = Path(path)
    data = yaml_load(path.read_text(encoding="utf-8"), path.stem)
    schema = data.get("schema") if isinstance(data, Mapping) else None
    if schema not in CARD_TYPES:
        raise CardError(path.stem, "schema", f"unknown or missing schema {schema!r} (one of {sorted(CARD_TYPES)})")
    return from_data(CARD_TYPES[schema], data, path.stem)


def load_material(path: Path) -> ct.MaterialCard:
    """Load a material card. Value rules are applied by ``validate_material`` (``CardLibrary`` runs it)."""
    return load_card(path, ct.MaterialCard)


def load_pair(path: Path) -> ct.PairCard:
    return load_card(path, ct.PairCard)


def load_object(path: Path) -> ct.ObjectCard:
    return load_card(path, ct.ObjectCard)


def load_scene(path: Path) -> ct.SceneCard:
    return load_card(path, ct.SceneCard)


def to_data(card: Card) -> dict[str, Any]:
    """Plain data for a card, schema first, defaults omitted."""
    if type(card) not in SCHEMAS:
        raise TypeError(f"dump_card: not a card: {type(card).__name__}")
    body = encode(card)
    return {"schema": card.schema, **{k: v for k, v in body.items() if k != "schema"}}


def dump_card(card: Card, header: str = "") -> str:
    """Canonical YAML for a card; ``loads_card(dump_card(c), type(c)) == c`` and the dump is idempotent."""
    return yaml_dump(to_data(card), header=header)
