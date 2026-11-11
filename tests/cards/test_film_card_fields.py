# SPDX-License-Identifier: Apache-2.0
"""Film-only values on material cards (engine item 8 step 7; coordinator ruling on SCR-SOFT-1-1; WP1 reviews).

An optional ``film`` block (dead-fold yield angle, contact radius, air scale) overrides ``FILM_DEFAULTS`` field by
field in the film emitter. A card without it decodes, validates, hashes and emits exactly as before: the block is
left out of digests while it is None (``OMIT_IF_NONE``, WP1 ruling 2026-11-06).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from alloy.cards import types as ct
from alloy.cards.loader import dump_card, load_material, loads_card
from alloy.cards.validate import errors, validate_material
from alloy.core.ids import digest_of
from alloy.solvers.film.emit import film_extras
from alloy.solvers.film.materials import FILM_DEFAULTS, FilmExtras

LIB = Path(__file__).resolve().parents[2] / "library" / "materials"
HDPE_DIGEST = "13b0bffe749bb4425df926f40e75cf0683babe46ffa95c3c9a0d1023367fd9aa"   # main e57b85a9b, before the block


def _card(name: str, film: str = "") -> ct.MaterialCard:
    """The library HDPE card renamed, with an optional ``film:`` YAML block appended."""
    text = (LIB / "hdpe.yaml").read_text().replace("name: hdpe\n", f"name: {name}\n", 1)
    return loads_card(text + film, ct.MaterialCard, name)


def _block(**values: float) -> str:
    rows = "".join(f"  {k}:\n    value: {v!r}\n    source: model\n    note: test value\n" for k, v in values.items())
    return "film:\n" + rows


def test_a_card_without_the_block_keeps_its_hash() -> None:
    card = load_material(LIB / "hdpe.yaml")
    assert card.film is None
    assert digest_of(card) == HDPE_DIGEST


def test_the_block_round_trips_and_validates() -> None:
    card = _card("film_test", _block(dead_fold_yield=math.radians(73.0), contact_radius=4.0e-4, air_scale=0.5))
    assert card.film is not None and card.film.contact_radius.value == pytest.approx(4.0e-4)
    assert errors(validate_material(card)) == []
    assert loads_card(dump_card(card), ct.MaterialCard, "film_test") == card
    assert digest_of(card) != digest_of(_card("film_test"))


@pytest.mark.parametrize("field, value", [("dead_fold_yield", math.pi), ("dead_fold_yield", -0.1),
                                          ("contact_radius", 0.0), ("air_scale", -1.0)])
def test_out_of_range_values_are_errors(field: str, value: float) -> None:
    card = _card("film_test", _block(**{field: value}))
    assert [i.field for i in errors(validate_material(card))] == [f"film.{field}"]


def test_card_fields_override_the_table_field_by_field() -> None:
    assert film_extras(_card("film_hdpe")) == FILM_DEFAULTS["film_hdpe"]          # no block: the table, unchanged
    got = film_extras(_card("film_hdpe", _block(air_scale=0.5)))
    assert got.air == 0.5 and got.yield_angle == FILM_DEFAULTS["film_hdpe"].yield_angle
    unlisted = film_extras(_card("film_new", _block(dead_fold_yield=1.0, contact_radius=3.0e-4)))
    assert unlisted.yield_angle == 1.0 and unlisted.contact_radius == 3.0e-4
    assert unlisted.air == FilmExtras().air and unlisted.membrane_scale == FilmExtras().membrane_scale
