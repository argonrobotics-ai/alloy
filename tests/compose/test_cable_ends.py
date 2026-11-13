# SPDX-License-Identifier: Apache-2.0
"""Cables between objects: the two ends go to two different objects ("from a mug to a mug" takes both mugs), a
cable with both ends on the only one is reported missing with what to ask for (before: 0/0 gave NaN points, a card
error or a false "no free path"), a second cable between the same two objects runs beside the first instead of on top
of it (before: the same curve twice, both deleted by the overlap check), and a bare plural ("cables", "towels")
places 3, as "cubes" does."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from alloy.cards import types as ct
from alloy.cards.library import CardLibrary, default_root
from alloy.cards.loader import loads_card
from alloy.compose import parse
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


@pytest.fixture
def chain(monkeypatch):
    """Route the ropes as a chain: a cold rod self-test takes minutes on a CPU and is not what is tested here."""
    from alloy.compose.soft import cards as K

    real = K.route_of
    monkeypatch.setattr(K, "route_of", lambda lib, name: ("chain", {"links": 14.0}, "") if "rope" in name
                        else real(lib, name))


def _ropes(r):
    return [row for row in (r.scenes[0].soft or {}).get("items", []) if row["kind"] == "cable"]


def _laid(r, row):
    """The rope's points as laid (world): its card's polyline from the placed start."""
    pose = next(o["pose"] for o in r.scenes[0].objects if o["name"] == row["name"])
    pts = np.asarray(loads_card(r.generated[row["card"]], ct.ObjectCard).shape.points)
    return pts + np.asarray(pose[:3])


def _compose(sentence, cat):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="invalid value", category=RuntimeWarning)       # no 0/0
        return compose(sentence, check="none", device="cpu", catalog=cat)


def test_a_cable_from_a_mug_to_a_mug_takes_both_mugs(cat, chain):
    r = _compose("two mugs and a 0.35 m cotton rope from a mug to a mug on a table", cat)
    (rope,) = _ropes(r)
    assert r.ok, r.summary()
    assert len(set(rope["attached"])) == 2
    assert np.isfinite(_laid(r, rope)).all()


def test_both_ends_on_the_one_mug_is_reported_with_what_to_ask(cat, chain):
    r = _compose("a mug and a 0.35 m cotton rope from the mug to the mug on a table", cat)
    assert not r.ok and not _ropes(r)
    (miss,) = [m for m in r.scenes[0].missing if m.startswith("rope")]
    assert "both its ends are on the one mug" in miss and "ask for a second mug" in miss
    assert not any("nan" in text for text in r.generated.values())


def test_two_ropes_between_the_same_objects_lie_apart(cat, chain):
    r = _compose("two 0.35 m cotton ropes from a mug to a jar on a table", cat)
    a, b = _ropes(r)
    assert r.ok, r.summary()
    assert a["attached"] == b["attached"]
    pa, pb = _laid(r, a), _laid(r, b)
    gap = min(math.dist(p, q) for p in pa for q in pb)
    assert gap > 1e-3 * (a["diameter_mm"] + b["diameter_mm"]) / 2 + 1e-3        # clear of each other, not overlapping


def test_bare_soft_plurals_place_three():
    for sentence, noun in (("cables on a table", "cable"), ("towels on a table", "towel"),
                           ("cubes on a table", "cube")):
        (item,) = parse(sentence).items
        assert (item.query, item.count) == (noun, 3)
    assert parse("a few towels on a table").items[0].count == 3
    assert parse("a towel on a table").items[0].count == 1
