# SPDX-License-Identifier: Apache-2.0
"""``alloy.Scene.set_inertia_floor``: the scene option that turns the inertia floor off (``alloy.contact.inertia_floor``).

Unset means on, and an unset option leaves the scene JSON, the scene card's digest and every object's card_hash as
they were (the field is omitted from digests while None). False is written to the scene card, survives
``to_dict`` / ``from_dict`` and ``save`` / ``Scene.load``, and the build then raises nobody."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import alloy
from alloy.api.errors import SceneError
from alloy.cards import types as ct
from alloy.core.ids import canonical_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contact"))
from test_inertia_floor import _parts_scene  # noqa: E402


def test_the_option_round_trips_and_unset_leaves_the_json_unchanged():
    s = _parts_scene()
    before = s.to_json()
    s.set_inertia_floor(True)
    assert s.to_json() == before and "inertia_floor" not in before
    s.set_inertia_floor(False)
    assert json.loads(s.to_json())["extra"]["inertia_floor"] is False
    assert alloy.Scene.from_dict(s.to_dict()).to_json() == s.to_json()
    s.set_inertia_floor(True)
    assert s.to_json() == before
    with pytest.raises(SceneError):
        s.set_inertia_floor("off")


def test_unset_leaves_the_scene_digest_and_card_hashes_unchanged():
    card = ct.SceneCard(robots=(), objects=())
    assert b"inertia_floor" not in canonical_json(card)
    assert b"inertia_floor" in canonical_json(ct.SceneCard(robots=(), objects=(), inertia_floor=False))
    on, off = alloy.build(_parts_scene(), device="cpu"), alloy.build(_parts_scene(floor=False), device="cpu")
    assert {n: h.card_hash for n, h in on.built.handles.items()} == {n: h.card_hash for n, h in off.built.handles.items()}
    assert set(on.built.report.inertia_floor) == {"pen"} and off.built.report.inertia_floor == {}


def test_a_saved_scene_card_keeps_the_option(tmp_path):
    s = _parts_scene(floor=False)
    loaded = alloy.Scene.load(s.save(tmp_path / "off"))
    assert loaded.extra.get("inertia_floor") is False
    assert alloy.build(loaded, device="cpu").built.report.inertia_floor == {}
    default = alloy.Scene.load(_parts_scene().save(tmp_path / "on"))
    assert "inertia_floor" not in default.extra
    assert set(alloy.build(default, device="cpu").built.report.inertia_floor) == {"pen"}
