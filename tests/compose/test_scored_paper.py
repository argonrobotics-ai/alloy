# SPDX-License-Identifier: Apache-2.0
"""Scored paper ("scored paper", "creased paper"): the library sheet whose crease is in its mesh is placed as it is,
at its own size, and the saved folder builds it as a film sheet. Asking for another size or for it folded in half is
reported with what to ask instead. Before, every such sentence raised a raw AttributeError."""

from __future__ import annotations

import pytest

import alloy.check
from alloy.cards.library import CardLibrary, default_root
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog
from alloy.compose.load import library
from alloy.compose.soft.cards import copied_from


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


def _sheet(r) -> dict:
    return next(x for x in r.scenes[0].soft["items"] if x["kind"] == "cloth")


@pytest.mark.parametrize("sentence", ["scored paper on a table", "creased paper on a table",
                                      "a wooden cube and scored paper on a table"])
def test_scored_paper_is_placed_as_its_library_sheet(cat, sentence):
    r = compose(sentence, check="none", catalog=cat)
    assert r.ok, r.summary()
    row = _sheet(r)
    assert row["template"] == "paper_kraft_scored_sheet" and row["size_m"] == [0.3, 0.3]
    assert copied_from(r.generated[row["card"]]) == "paper_kraft_scored_sheet"


def test_the_saved_scored_sheet_loads_as_a_film(cat, tmp_path):
    r = compose("scored paper on a table", check="none", catalog=cat)
    r.save(tmp_path)
    rep = alloy.check.check_load(str(tmp_path / "scenes" / "scene.yaml"), device="cpu",
                                 library=library([str(tmp_path)]))
    assert rep.ok, rep.summary()


@pytest.mark.parametrize(("sentence", "says"), [("a sheet of scored paper folded in half on a table", "fold it"),
                                                ("a large sheet of scored paper on a table", "its own size")])
def test_a_size_or_a_fold_is_reported_with_what_to_ask(cat, sentence, says):
    r = compose(sentence, check="none", catalog=cat)
    assert not r.ok and says in r.scenes[0].missing[0] and "kraft paper" in r.scenes[0].missing[0]


def test_the_scored_sheet_pins_as_a_cloth_card(cat):
    spec = {"items": [{"query": "sheet", "object": "paper_kraft_scored_sheet", "soft": {"kind": "cloth"}}],
            "support": "table"}
    r = compose(spec, check="none", catalog=cat)
    assert r.ok and _sheet(r)["template"] == "paper_kraft_scored_sheet"
