# SPDX-License-Identifier: Apache-2.0
"""Liquids and their containers in a sentence. An amount of a material the builder does not have ("200 ml of
detergent", "pour coolant ...") is an error that names the word and lists the liquids and grains it has (before: a
pour said "a pour names no liquid" and a fill made forty "detergent" objects). A container that is filled or poured
into is a card with a cavity ("a tube with 5 ml of water": the sample tube, not the solid tube rack)."""

from __future__ import annotations

import pytest

from alloy.cards.library import CardLibrary, default_root
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog
from alloy.compose.choose import fill_targets
from alloy.compose.errors import ComposeError
from alloy.compose.parse import parse


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


@pytest.mark.parametrize(("sentence", "word"), [("a cup with 200 ml of detergent", "detergent"),
                                                ("a cup with 2 l of reagent", "reagent"),
                                                ("pour 100 ml of detergent from a bottle into a bin", "detergent"),
                                                ("pour coolant into a mug", "coolant")])
def test_an_unknown_liquid_is_named_with_the_liquids_there_are(sentence, word):
    with pytest.raises(ComposeError) as e:
        parse(sentence)
    assert e.value.message.startswith(f"'{word}' is not a liquid")
    assert "water" in e.value.fix and "honey" in e.value.fix and "oil" in e.value.fix


def test_a_size_is_not_an_amount():
    spec = parse("a 200 ml cup on a table")
    assert [(i.query, i.count) for i in spec.items] == [("cup", 1)]


def test_a_filled_container_is_one_that_can_hold_it(cat):
    assert fill_targets(parse("a tube with 5 ml of water")) == {"tube"}
    r = compose("a tube with 5 ml of water", check="none", catalog=cat)
    assert [o["card"] for o in r.scenes[0].objects if o["item"] == "tube"] == ["sample_tube_13x75"]
    assert not any("no cavity" in m for m in r.scenes[0].missing), r.scenes[0].missing
    plain = compose("a tube on a table", check="none", catalog=cat)                    # nothing poured: any tube
    assert len(plain.scenes[0].objects) == 1
