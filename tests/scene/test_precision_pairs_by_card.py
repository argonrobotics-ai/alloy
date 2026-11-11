# SPDX-License-Identifier: Apache-2.0
"""PHYSA-4: a precision pair forms whatever the scene calls its objects, and the build says what it made.

A card names its precision partner by card name ("block_bore_d12_2:bore"); the engine looked the partner up among the
scene's object names only. ``add_object("peg", "peg_d12")`` beside ``add_object("blk", "block_bore_d12_2")`` (the
names in ``Scene.add_object``'s own example) made no pair and said nothing: the block then collided as its convex
hull, and a peg centred over the 12.2 mm bore rested on top of it with health OK (audit PHYSA-4). Now:
- the API writes a renamed card's partners by the names the scene gives the partner card's objects;
- the engine also finds a partner by card name (a scene card's second copy is "peg_d12#1");
- each pair is in both objects' route reasons, and two objects whose partners are both absent get a warning.
The insertion laws are in tests/contact/test_peg_demo_inserts.py.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace as NS

import alloy
from alloy.api import _materialize as M
from alloy.scene import precision_pairs as PP

PEG, BLOCK = "peg_d12", "block_bore_d12_2"


def _obj(card: str, partner: str, feature: str) -> NS:
    feat = NS(name=feature, clearance=NS(value=2.0e-4))
    return NS(card=NS(name=card, precision_partners=(partner,), precision_features=(feat,)))


def _pairs(objs: dict[str, NS]) -> set[frozenset[str]]:
    return {frozenset((a, b)) for a, b, _, _ in PP.pair_features(objs)}


def test_a_partner_is_found_by_object_name_then_by_card_name():
    block, peg = _obj(BLOCK, f"{PEG}:shaft", "bore"), _obj(PEG, f"{BLOCK}:bore", "shaft")
    objs = {BLOCK: block, PEG: peg, f"{PEG}#1": peg}                      # a scene card with two pegs
    assert PP.partner_names(objs, PEG, f"{BLOCK}:bore") == [BLOCK]
    assert PP.partner_names(objs, BLOCK, f"{PEG}:shaft") == [PEG, f"{PEG}#1"]
    assert _pairs(objs) == {frozenset((BLOCK, PEG)), frozenset((BLOCK, f"{PEG}#1"))}
    both = {**objs, f"{BLOCK}#1": block}                                  # two of each: every peg with every block
    assert len(_pairs(both)) == 4 and frozenset((f"{BLOCK}#1", f"{PEG}#1")) in _pairs(both)
    renamed = {"a": block, "b": peg}                                      # other object names, the cards as shipped
    assert PP.partner_names(renamed, "b", f"{BLOCK}:bore") == ["a"]
    (a, b, fa, fb), = PP.pair_features(renamed)
    assert (a, b, fa.name, fb.name) == ("a", "b", "bore", "shaft")
    assert PP.pair_features({"b": peg}) == []                             # a peg alone: no pair, no error


def test_the_api_writes_a_renamed_cards_partners_by_object_name():
    items = {"peg": {"kind": "card", "card_name": PEG}, "blk": {"kind": "card", "card_name": BLOCK},
             "box": {"kind": "box"}}
    names = M.names_by_card(items)
    assert names == {PEG: ["peg"], BLOCK: ["blk"]}
    assert M._partners([f"{BLOCK}:bore"], names) == ["blk:bore"]
    assert M._partners(["nut_m10", "nut_m6:thread"], names) == ["nut_m10", "nut_m6:thread"]      # not in the scene
    two = M.names_by_card({**items, "peg2": {"kind": "card", "card_name": PEG}})
    assert M._partners([f"{PEG}:shaft"], two) == ["peg:shaft", "peg2:shaft"]
    assert M.names_by_card({PEG: {"kind": "card", "card_name": PEG}}) == {}       # nothing renamed: cards as shipped


def test_routes_name_the_pair_for_the_docstring_example():
    """``Scene.add_object``'s example is ``scene.add_object("peg", "peg_d12", pos=(0, 0, 0.04))``."""
    assert 'scene.add_object("peg", "peg_d12", pos=(0, 0, 0.04))' in alloy.Scene.add_object.__doc__
    scene = alloy.Scene(dt=0.02)
    scene.add_object("blk", BLOCK, pos=(0, 0, 0.0045))
    scene.add_object("peg", "peg_d12", pos=(0, 0, 0.04))
    routes = alloy.estimate(scene, worlds=1, seconds=0.1, device="cpu")["routes"]
    assert "precision pair with blk: shaft / bore, clearance 0.2 mm" in routes["peg"]["reasons"]
    assert "precision pair with peg: bore / shaft, clearance 0.2 mm" in routes["blk"]["reasons"]
    lone = alloy.Scene(dt=0.02)
    lone.add_ground()
    lone.add_object("peg", "peg_d12", pos=(0, 0, 0.04))
    reasons = alloy.estimate(lone, worlds=1, seconds=0.1, device="cpu")["routes"]["peg"]["reasons"]
    assert not any("precision pair" in r for r in reasons)


def test_two_objects_whose_partners_are_absent_get_a_warning(caplog):
    """Hand-written cards whose partner names match nothing in the scene: the pair that was meant cannot form."""
    report = NS(decisions={})
    mine = [NS(name="my_peg", obj=_obj("my_peg", f"{BLOCK}:bore", "shaft")),
            NS(name="my_block", obj=_obj("my_block", f"{PEG}:shaft", "bore"))]
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        assert PP.routed_features(report, mine) == []
    assert "no precision pair is made for my_block -> peg_d12:shaft, my_peg -> block_bore_d12_2:bore" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        assert PP.routed_features(report, mine[:1]) == []                 # one object alone: nothing to say
    assert caplog.text == ""
