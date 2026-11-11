# SPDX-License-Identifier: Apache-2.0
"""OSS-19 sentence parser and spec: counts, plurals, materials, supports, relations, clutter, JSON round trip, and
errors that say what to change. Pure Python (no engine)."""

from __future__ import annotations

import pytest

from alloy.compose import words as W
from alloy.compose.parse import parse
from alloy.compose.spec import ComposeError, ItemSpec, SceneSpec


def _items(spec: SceneSpec) -> list[tuple]:
    return [(i.query, i.count, i.words, i.relation, i.target) for i in spec.items]


def test_counts_materials_and_support():
    spec = parse("a mug, a spoon and three wooden cubes on a plywood table")
    assert _items(spec) == [("mug", 1, (), None, None), ("spoon", 1, (), None, None),
                            ("cube", 3, ("wooden",), None, None)]
    assert spec.support == "table" and spec.clutter == "normal" and spec.arrangement == "scatter"
    assert any("plywood not used" in n for n in spec.notes)


def test_container_relation_puts_the_target_first():
    spec = parse("a cluttered table with a mug, two spoons and three nuts in a tray")
    assert _items(spec) == [("mug", 1, (), None, None), ("spoon", 2, (), None, None), ("tray", 1, (), None, None),
                            ("nut", 3, (), "in", "tray")]
    assert spec.clutter == "cluttered"


def test_pronoun_target_floor_worlds_and_digits():
    spec = parse("a tray with 2 nuts on it on the floor in 64 worlds")
    assert _items(spec) == [("tray", 1, (), None, None), ("nut", 2, (), "on", "tray")]
    assert spec.support == "floor" and spec.worlds == 64


def test_plural_without_number_and_rows():
    spec = parse("pencils in a row")
    assert _items(spec) == [("pencil", 3, (), None, None)] and spec.arrangement == "row"
    assert any("plural with no number" in n for n in spec.notes)


def test_tidy_desk_keeps_unknown_words_as_hints():
    spec = parse("a tidy desk with a small screwdriver and a red mug")
    assert spec.clutter == "tidy" and spec.support == "table"
    assert _items(spec) == [("screwdriver", 1, ("small",), None, None), ("mug", 1, ("red",), None, None)]


def test_nothing_to_place_says_what_to_write():
    with pytest.raises(ComposeError) as e:
        parse("a table")
    assert "no object" in e.value.message and "e.g." in e.value.fix
    with pytest.raises(ComposeError):
        parse("   ")


def test_spec_round_trip_and_strings_as_items():
    spec = SceneSpec.from_dict({"items": ["mug", {"query": "cube", "count": 3, "words": ["wooden"]}],
                                "clutter": "tidy", "worlds": 8})
    assert spec.items[0] == ItemSpec("mug") and spec.items[1].words == ("wooden",)
    again = SceneSpec.from_dict(spec.to_dict())
    assert again == spec and again.to_dict()["schema"] == "alloy.compose_spec/1"


@pytest.mark.parametrize("data, field", [
    ({"items": []}, "items"),
    ({"items": ["mug"], "clutter": "chaos"}, "clutter"),
    ({"items": [{"query": "mug", "count": 0}]}, "items.count"),
    ({"items": [{"query": "nut", "relation": "in", "target": "tray"}]}, "items.target"),
    ({"items": ["mug"], "colour": "red"}, "colour"),
    ({"items": [{"query": "mug", "relation": "under", "target": "mug"}]}, "items.relation"),
])
def test_bad_specs_name_the_field_and_the_fix(data, field):
    with pytest.raises(ComposeError) as e:
        SceneSpec.from_dict(data)
    assert e.value.field == field and e.value.fix
    assert e.value.to_dict()["error"] == "ComposeError"


def test_words_helpers():
    assert [W.singular(w) for w in ("cubes", "knives", "boxes", "glass", "batteries", "bus")] == \
        ["cube", "knife", "box", "glass", "battery", "bus"]
    assert W.tokens("gso_Threshold_Porcelain_CoffeeMug") == ["gso", "threshold", "porcelain", "coffee", "mug"]
    assert "mug" in W.expand("cups")


def test_randomized_worlds_turn_on_jitter():
    spec = parse("three cubes on a table in 16 randomized worlds")
    assert spec.jitter and spec.worlds == 16 and spec.items[0].count == 3
    assert not parse("three cubes on a table").jitter
