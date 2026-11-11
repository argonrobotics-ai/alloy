# SPDX-License-Identifier: Apache-2.0
"""The starter library: counts, round trips, canonical files, zero validation errors, sourced leaves, required objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from alloy.cards import types as ct
from alloy.cards.library import KINDS, CardLibrary, default_root
from alloy.cards.loader import dump_card, load_card, loads_card
from alloy.cards.paths import iter_q_leaves
from alloy.cards.validate import errors, validate_material, validate_pair, validate_scene

LIB = CardLibrary(assets=None)
FILES = [(kind, p) for kind in KINDS for p in sorted((default_root() / kind).glob("*.yaml"))]


def test_counts() -> None:
    assert len(LIB.names("materials")) >= 30
    assert len(LIB.names("pairs")) >= 20
    assert len(LIB.names("objects")) >= 40


@pytest.mark.parametrize("kind, path", FILES, ids=lambda x: x.stem if isinstance(x, Path) else x)
def test_round_trip_every_library_card(kind: str, path: Path) -> None:
    card = load_card(path, KINDS[kind])
    text = dump_card(card)
    back = loads_card(text, KINDS[kind])
    assert back == card
    assert dump_card(back) == text
    if kind != "scenes":                                   # generated cards are stored in canonical form
        assert path.read_text(encoding="utf-8") == text


def test_no_validation_errors_anywhere() -> None:
    errs = []
    for _, m in LIB.iter_cards("materials"):
        errs += errors(validate_material(m))
    for _, p in LIB.iter_cards("pairs"):
        errs += errors(validate_pair(p, LIB))
    for name in LIB.names("objects"):
        errs += errors(LIB.object_issues(name))
    for name, s in LIB.iter_cards("scenes"):
        errs += errors(validate_scene(s, LIB, name=name))
    assert errs == []


def test_every_leaf_sourced_or_default() -> None:
    bad = []
    for kind in ("materials", "pairs", "objects"):
        for name, card in LIB.iter_cards(kind):
            for path, q in iter_q_leaves(card):
                if q.source != "default" and not q.note.strip():
                    bad.append((kind, name, path))
    assert bad == []


def _features(name: str) -> tuple[ct.PrecisionFeature, ...]:
    return LIB.peek_object(name).precision_features


def test_required_objects_from_the_spec() -> None:
    names = set(LIB.names("objects"))
    for must in ("peg_d12", "block_bore_d12_2", "nut_m10", "bolt_m10x60", "usb_a_plug", "usb_a_receptacle", "usb_cable_1m",
                 "towel_cotton", "sponge_kitchen", "cup_pp_disposable", "rice_cup_fill", "water_cup", "dough_ball"):
        assert must in names, must
    bore = _features("block_bore_d12_2")[0]
    assert bore.clearance is not None and bore.clearance.value == pytest.approx(2e-4)
    assert _features("nut_m10")[0].analytic_id == "iso_thread:M10x1.5:6H"
    assert _features("bolt_m10x60")[0].analytic_id == "iso_thread:M10x1.5:6g"
    assert LIB.material(LIB.peek_object("water_cup").material).fluid is not None
    assert LIB.material(LIB.peek_object("dough_ball").material).plastic.yield_stress is not None


def test_pair_files_are_named_by_sorted_materials() -> None:
    for name, p in LIB.iter_cards("pairs"):
        assert name == "__".join(sorted((p.a, p.b)))
        assert LIB.has_material(p.a) and LIB.has_material(p.b)
