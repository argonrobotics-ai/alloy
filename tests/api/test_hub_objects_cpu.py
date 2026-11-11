# SPDX-License-Identifier: Apache-2.0
"""Objects from the Hugging Face object library in the Scene API on a fresh install (SCR OSS-11-9, CPU).

A fresh install has no alloy-assets checkout, no packaged imports and no asset store; `alloy fetch <name>` puts
objects in the fetch cache. `Scene.add_object` used to reject them ("no object card": the API's catalog did not list
the fetch caches the engine's default roots read since OSS-11-8). Now a fetched object places and builds like a
library object, and an object that is only in the Hub's index is accepted and fetched by the build. The Hub is the
fake one of tests/assets/test_hub.py (a staged toy library, no network).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
import test_hub as toy  # noqa: E402  (a staged toy library)
from test_hub import hub  # noqa: E402, F401  (fixture: a fake Hub serving `dataset`)

import alloy  # noqa: E402
from alloy.api import catalog  # noqa: E402
from alloy.assets.hub import build, card, fetch, source  # noqa: E402

SHA = hashlib.sha256(toy.OBJ.encode()).hexdigest()
SURFACE = """  mu_dynamic: {{value: 0.4, source: model, note: "typical"}}
  restitution: {{value: 0.2, source: model, note: "typical"}}
"""                                                             # str.format template, as test_hub's MATERIAL


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """test_hub's toy library with cards that build, complete as real cards are (mesh sha256, both frictions)."""
    monkeypatch.setattr(toy, "CARD", toy.CARD.replace("collision: convex\n", f"collision: convex\n    sha256: {SHA}\n"))
    monkeypatch.setattr(toy, "MATERIAL", toy.MATERIAL + SURFACE)
    root, store = toy._library(tmp_path, ["cup_a", "brand_box"])
    out, rows = tmp_path / "ds", tmp_path / "rows" / "toylib.jsonl"
    assert not build.stage("toylib", root, store, out, rows)["failed"]
    (tmp_path / "texts").mkdir()
    (tmp_path / "texts" / "CC-BY-4.0.txt").write_text("Attribution 4.0 International")
    card.write_top(rows.parent, tmp_path / "texts", out)
    return out


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A fresh install: no alloy-assets checkout, no packaged imports, no asset store, the default Hub repo."""
    from alloy.cards import roots

    monkeypatch.setenv("ALLOY_ASSETS", "")
    monkeypatch.setattr(roots, "package_data", lambda: tmp_path / "_data")
    monkeypatch.setattr(catalog, "_package_data", lambda: tmp_path / "_data")
    monkeypatch.delenv("ALLOY_ASSET_STORE", raising=False)
    monkeypatch.delenv(source.REPO_ENV, raising=False)
    catalog.refresh()
    yield
    catalog.refresh()


def cup_scene(card: str = "cup_a") -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_object("cup", card, pos=(0.0, 0.0, 0.05))         # renamed: the API copies the fetched card
    return scene


def rests_on_the_ground(scene: alloy.Scene) -> None:
    sim = alloy.build(scene, device="cpu")
    z = sim.run(0.5, record=["cup"]).world(0)["cup.pos"][-1, 2]
    assert abs(z) < 2e-3 and sim.check()["ok"]                  # the toy mesh's origin is a corner on its base
    sim.close()


@pytest.mark.compile_heavy
def test_fetched_object_places_and_builds_like_a_library_object(fresh, hub, monkeypatch):
    from alloy.cards import roots

    fetch.fetch(names=["cup_a"])                                 # `alloy fetch cup_a`
    monkeypatch.setenv(source.REPO_ENV, "")                      # the Hub off: only the cache serves it
    assert catalog.fetched_roots() == roots.fetched_roots(roots.import_roots())     # the fast mirror
    path = catalog.card_path("objects", "cup_a")
    assert path is not None and path.is_relative_to(catalog.cache_base()) and "cup_a" in catalog.objects()
    rests_on_the_ground(cup_scene())


@pytest.mark.compile_heavy
def test_hub_object_is_accepted_then_fetched_by_the_build(fresh, hub):
    scene = cup_scene()                                          # only in the Hub's index: accepted
    assert not any(c.endswith("physics.obj") for c in hub)       # nothing fetched yet
    with pytest.raises(alloy.SceneError, match="no object card") as err:
        scene.add_object("other", "cup_q")
    assert "cup_a" in err.value.fix                              # close names include the Hub's
    rests_on_the_ground(scene)
    assert any(c.endswith("cup_a/physics.obj") for c in hub)


def test_an_object_fetched_after_a_lookup_is_found(fresh, hub, monkeypatch):
    fetch.fetch(names=["brand_box"])
    monkeypatch.setenv(source.REPO_ENV, "")
    assert "brand_box" in catalog.objects() and "cup_a" not in catalog.objects()   # names read and kept
    with pytest.raises(alloy.SceneError, match="no object card"):
        alloy.Scene().add_object("cup", "cup_a")                 # Hub off and not fetched: refused
    fetch.fetch(names=["cup_a"])                                 # into the library folder read above
    alloy.Scene().add_object("cup", "cup_a")
