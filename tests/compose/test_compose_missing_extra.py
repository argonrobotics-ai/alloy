# SPDX-License-Identifier: Apache-2.0
"""GEO P4 (2026-11-11): on a base install, `alloy compose "a mug and a bowl on a table"` left the bowl out yet answered
top-level ok true, and its note blamed the word. A missing object now fails the call (top-level ok false) and the fix
names the import extra when the object library on the Hub is out of reach."""

from __future__ import annotations

import pytest

import alloy.api.catalog as catalog
from alloy.cli.registry import respond_op
from alloy.compose import choose, ops


@pytest.mark.parametrize("hub_missing", [True, False])
def test_a_missing_object_fails_the_call_with_the_fix(monkeypatch, hub_missing: bool) -> None:
    monkeypatch.setattr(catalog, "_hub_unreachable", lambda: hub_missing)
    reply = respond_op(None, "compose", {"sentence": "a cube and a zqxjvw on the floor", "check": "none"})
    assert reply["ok"] is False and reply["op"] == "compose"
    err = reply["error"]
    assert "zqxjvw" in err["message"] and err["field"] == "sentence"
    if hub_missing:
        assert 'pip install "alloy-engine[import]"' in err["fix"] and "Hugging Face" in err["fix"]
    else:
        assert "another word" in err["fix"] and "import" not in err["fix"]


def test_a_complete_scene_still_answers_ok(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "_hub_unreachable", lambda: True)
    reply = respond_op(None, "compose", {"sentence": "a cube on the floor", "check": "none"})
    assert reply["ok"] is True and reply["result"]["ok"] is True


def test_the_no_match_note_names_the_extra_on_a_base_install(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "_hub_unreachable", lambda: True)
    assert 'pip install "alloy-engine[import]"' in choose._no_match_hint()
    monkeypatch.setattr(catalog, "_hub_unreachable", lambda: False)
    assert choose._no_match_hint().startswith("try another word") and ops.missing_fix().startswith("use another word")
