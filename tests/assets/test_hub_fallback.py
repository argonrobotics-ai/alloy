# SPDX-License-Identifier: Apache-2.0
"""The Hub fallback (SCR OSS-11-6) on CPU: with no imported library and no store, the API's build step, compose's
catalog and check fetch an object from a fake Hub on first use. Offline, the fix says `alloy fetch <name>`. On a
machine with imported libraries nothing changes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hub import dataset, hub  # noqa: E402, F401  (fixtures: a staged toy library, a fake Hub serving it)

from alloy.assets.hub import fetch, lazy, source  # noqa: E402


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """A fresh install: no alloy-assets checkout, no packaged imports, no asset store, the default repo."""
    from alloy.cards import roots

    monkeypatch.setenv("ALLOY_ASSETS", "")
    monkeypatch.setattr(roots, "package_data", lambda: tmp_path / "_data")
    monkeypatch.delenv("ALLOY_ASSET_STORE", raising=False)
    monkeypatch.delenv(source.REPO_ENV, raising=False)


def test_offline_is_reported_as_offline_not_as_a_missing_file():
    from huggingface_hub import errors

    exc = source.explain(errors.LocalEntryNotFoundError("not in the local cache"), "o/r", "objects/x/card.yaml")
    assert "cannot reach" in exc.message and "check the network" in exc.fix


def _offline(monkeypatch):
    def down(path, local_dir, repo=None):
        raise source.explain(ConnectionError("no route to host"), repo or source.repo_id(), path)

    monkeypatch.setattr(source, "download", down)
    monkeypatch.setattr(fetch, "download", down)


def _lib(tmp_path):
    from alloy.cards.library import CardLibrary, default_root

    return CardLibrary(roots=[default_root()], assets=None)


def test_hub_mode_only_without_store_or_imports(fresh, tmp_path, monkeypatch):
    assert lazy.hub_mode()
    assert not lazy.hub_mode(store="gs://bucket")
    monkeypatch.setenv(source.REPO_ENV, "")                   # the Hub turned off
    assert not lazy.hub_mode()
    monkeypatch.delenv(source.REPO_ENV)
    (tmp_path / "a" / "imports" / "ycb_gso" / "objects").mkdir(parents=True)
    monkeypatch.setenv("ALLOY_ASSETS", str(tmp_path / "a"))   # a checkout with imported libraries
    assert not lazy.hub_mode()


def test_ensure_fetches_once_then_uses_the_cache(fresh, hub, tmp_path):
    lib = _lib(tmp_path)
    assert not lib.has_object("cup_a")
    root = lazy.ensure(lib, "cup_a")
    assert lib.has_object("cup_a") and lib.asset_path("meshes/cup_a/physics.obj").is_file()
    assert root in lib.roots and any(c.endswith("cup_a/physics.obj") for c in hub)
    hub.clear()
    lib2 = _lib(tmp_path)
    lazy.ensure(lib2, "cup_a")
    assert lib2.has_object("cup_a") and hub == []            # complete in the cache: no Hub call


def test_api_build_fetches_a_missing_card_and_says_alloy_fetch_offline(fresh, hub, tmp_path, monkeypatch):
    from alloy.api import _build
    from alloy.api.errors import SceneError

    lib = _lib(tmp_path)
    _build.fetch_imported(lib, ["cup_a"])
    assert lib.has_object("cup_a")
    _offline(monkeypatch)
    with pytest.raises(SceneError) as e:
        _build.fetch_imported(_lib(tmp_path), ["brand_box"])
    assert "cannot reach" in e.value.message and "alloy fetch brand_box" in e.value.fix


def test_check_reports_offline_with_alloy_fetch_and_unknown_names(fresh, hub, tmp_path, monkeypatch):
    from alloy.check.fetch import fetch_imported

    lib = _lib(tmp_path)
    (f,) = fetch_imported(lib, ["cup_b"])                      # not in the library: the close names
    assert f.level == "error" and "cup_a" in f.fix
    assert fetch_imported(lib, ["cup_a"]) == [] and lib.has_object("cup_a")
    _offline(monkeypatch)
    (f,) = fetch_imported(_lib(tmp_path), ["brand_box"])
    assert "check the network" in f.fix and "run `alloy fetch brand_box`" in f.fix


def test_compose_catalog_lists_hub_objects_and_fetches_on_first_use(fresh, hub, monkeypatch):
    from alloy.compose.catalog import Catalog

    cat = Catalog()
    assert cat.hub and cat.entries["cup_a"].source == "toylib" and "cup_a" in [m.name for m in cat.search("cup_a")]
    assert not any("cup_a/" in c for c in hub)                 # listed from the index alone
    cat.info("cup_a")
    assert cat.library.has_object("cup_a") and any(c.endswith("cup_a/physics.obj") for c in hub)
    off = Catalog(fetch=False)                                 # search: fetched objects only, no index
    assert "cup_a" in off.entries and "brand_box" not in off.entries
    hub.clear()
    assert "brand_box" in Catalog().entries and hub == []      # an index checked within the hour: no Hub call
    monkeypatch.setenv(source.REPO_ENV, "")                   # Hub off: objects fetched earlier are still there
    off = Catalog()
    assert not off.hub and off.entries["cup_a"].source == "toylib" and off.info("cup_a").name == "cup_a"
    assert "brand_box" not in off.entries and off.library.has_object("cup_a")


def test_a_machine_with_imported_libraries_never_asks_the_hub(tmp_path, monkeypatch, hub):
    from alloy.compose.catalog import Catalog

    (tmp_path / "a" / "imports" / "toylib" / "objects").mkdir(parents=True)
    monkeypatch.setenv("ALLOY_ASSETS", str(tmp_path / "a"))
    monkeypatch.delenv("ALLOY_ASSET_STORE", raising=False)
    cat = Catalog()
    assert not cat.hub and "cup_a" not in cat.entries and hub == []
