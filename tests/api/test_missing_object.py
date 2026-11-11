# SPDX-License-Identifier: Apache-2.0
"""A library object that is not on disk (ease test, 2026-11-09): on a fresh install without the ``import`` extra the
error says first that the object library is on the Hugging Face Hub and how to fetch from it, then the close names.

The README's ``ycb_025_mug`` used to fail with "no object card ... use one of ['pcb_80x50', ...]" and no word of
``alloy fetch``. Fresh installs are faked: ``hub_mode`` on, ``huggingface_hub`` not importable, no Hub names listed,
and a name not on disk (a checkout with the imported libraries has the mug).
"""

from __future__ import annotations

import importlib.util

import pytest

import alloy
from alloy.api import catalog
from alloy.api.errors import SceneError


@pytest.fixture
def fresh_install(monkeypatch):
    """No imported library or asset store, and no huggingface_hub (so the Hub's list cannot be read)."""
    import alloy.assets.hub.lazy as lazy

    real = importlib.util.find_spec
    monkeypatch.setattr(lazy, "hub_mode", lambda *a, **k: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "huggingface_hub" else real(n, *a))
    monkeypatch.setattr(catalog, "hub_names", lambda: [])


def _error(scene: alloy.Scene, card: str) -> SceneError:
    with pytest.raises(SceneError) as info:
        scene.add_object("thing", card)
    return info.value


def test_add_object_names_the_hub_and_the_fetch_first(fresh_install) -> None:
    e = _error(alloy.Scene(), "ycb_999_teapot")
    assert e.message.startswith("object card 'ycb_999_teapot' is not on disk") and "Hugging Face Hub" in e.message
    assert e.fix.index("alloy-engine[import]") < e.fix.index("alloy fetch ycb_999_teapot") < e.fix.index("use one of")
    assert "install.sh" not in e.fix and e.field == "card"     # pip only, as docs/guide/fetch.md quotes it


def test_a_build_from_cards_gives_the_same_error(fresh_install) -> None:
    with pytest.raises(SceneError) as info:
        catalog.object_card("ycb_999_teapot")
    assert "alloy fetch ycb_999_teapot" in info.value.fix and "is not on disk" in info.value.message


def test_without_a_fresh_install_the_close_names_stay(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "_hub_unreachable", lambda: False)
    e = _error(alloy.Scene(), "not_a_card_anywhere")
    assert e.message == "no object card 'not_a_card_anywhere'" and e.fix.startswith("use one of [")


def test_hub_unreachable_needs_both_a_fresh_install_and_no_hub(monkeypatch, fresh_install) -> None:
    import alloy.assets.hub.lazy as lazy

    assert catalog._hub_unreachable()
    monkeypatch.setattr(lazy, "hub_mode", lambda *a, **k: False)
    assert not catalog._hub_unreachable()
