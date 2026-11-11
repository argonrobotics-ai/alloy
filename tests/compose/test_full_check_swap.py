# SPDX-License-Identifier: Apache-2.0
"""An object that fails the full check in two places gives its place to another object of the same item (audit
EXAMPLES-6 / EXAMPLES-9: on a GPU a scanned mug kept sliding wherever it stood, was dropped, and the example's own
sentence came back NOT OK with the mug missing while a shipped mug that rests was one search result away)."""

from __future__ import annotations

import pytest

import alloy.check
from alloy.cards.library import CardLibrary, default_root
from alloy.check.report import CheckReport, Finding
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog


@pytest.fixture
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))          # two spoons match "a spoon"


def _sliding(monkeypatch, bad: set[str]) -> list[list[str]]:
    """Replace the full check by one that says every object in ``bad`` still slides; returns the scenes it saw."""
    seen: list[list[str]] = []
    real = alloy.check.check

    def check(card, *args, **kwargs) -> CheckReport:
        if kwargs.get("run") is False:                                           # a placement's load check
            return real(card, *args, **kwargs)
        names = [pl.object for pl in card.objects]
        seen.append(names)
        slides = [n for n in names if n in bad]
        return CheckReport("scene", [Finding("rest_drift", "error", f"{n}: still slides 13.9 mm over the last 0.96 s",
                                             "lower its friction", objects=(n,)) for n in slides])

    monkeypatch.setattr(alloy.check, "check", check)
    return seen


def test_an_object_that_fails_twice_is_swapped_for_another_match(cat, monkeypatch):
    first = compose("a spoon and a mug on a table", check="none", catalog=cat).scenes[0].objects
    spoon = next(o["card"] for o in first if o["item"] == "spoon")
    other = ({"spoon_steel", "spoon_wood_oxo"} - {spoon}).pop()
    seen = _sliding(monkeypatch, {spoon})
    r = compose("a spoon and a mug on a table", check="full", device="cpu", catalog=cat)
    assert r.ok, r.summary()
    s = r.scenes[0]
    assert {o["card"] for o in s.objects} == {other, "mug_stoneware"} and not s.missing
    assert [spoon in names for names in seen] == [True, True, False]             # moved once, then swapped, then passes
    note = next(n for n in s.log if "failed the full check twice" in n)
    assert note.startswith(f"spoon: {spoon} failed") and note.endswith(f"using {other} instead") and "slides" in note
    assert not cat.info(spoon).placeable and cat.info(spoon).reason.startswith("does not rest: ")


def test_with_no_other_match_the_object_is_dropped_and_reported(cat, monkeypatch):
    _sliding(monkeypatch, {"spoon_steel", "spoon_wood_oxo", "mug_stoneware"})
    r = compose("a mug on a table", check="full", device="cpu", catalog=cat)
    assert not r.ok
    s = r.scenes[0]
    assert s.missing == ["mug (mug_stoneware)"]
    assert any(n.startswith("mug_stoneware: dropped (failed the full check twice") for n in s.log), s.log


def test_an_object_named_by_its_card_is_not_swapped(cat, monkeypatch):
    _sliding(monkeypatch, {"spoon_steel"})
    r = compose({"items": [{"query": "spoon", "object": "spoon_steel"}], "support": "table"}, check="full",
                device="cpu", catalog=cat)
    assert not r.ok and not any("instead" in n for n in r.scenes[0].log)
    assert cat.info("spoon_steel").placeable                                # an asked-for card stays on offer


def test_a_contact_overflow_the_check_reads_fails_the_scene_and_the_summary_says_why(cat, monkeypatch):
    """Audit EXAMPLES-8: a scene whose contacts were dropped must not come back OK. The engine's health raises
    CONTACT_OVERFLOW, ``alloy.check`` reports it as a health error naming no object, and the builder, with nothing
    to move, returns the scene NOT OK with the finding in its summary and report."""
    real = alloy.check.check
    lost = Finding("health", "error", "64 world(s) INVALID: the contact or constraint capacity overflowed, so "
                   "contacts were dropped", "simplify collision shapes or reduce clutter per world")

    def check(card, *args, **kwargs) -> CheckReport:
        return real(card, *args, **kwargs) if kwargs.get("run") is False else CheckReport("scene", [lost])

    monkeypatch.setattr(alloy.check, "check", check)
    r = compose("a spoon and a mug on a table", check="full", device="cpu", catalog=cat)
    s = r.scenes[0]
    assert not r.ok and not s.missing and len(s.objects) == 2                    # nothing dropped: it is the scene
    assert s.report["errors"] == 1 and "contacts were dropped" in s.report["findings"][0]["message"]
    assert "NOT OK" in r.summary() and "contacts were dropped" in r.summary(), r.summary()
