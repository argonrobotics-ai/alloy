# SPDX-License-Identifier: Apache-2.0
"""On a fresh install (no imported library, no asset store) ``alloy.compose.search`` lists the object library on the
Hub from its index: each object not fetched yet is in the results, not placeable, with the ``alloy fetch`` that
gets it, and search downloads no object. Before, search built its catalog without the Hub's index and found none of
the Hub's objects ("apple", "pan", "basket" gave nothing while ``alloy fetch --list`` listed about 5,400)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
from test_hub import dataset, hub  # noqa: E402, F401  (fixtures: a staged toy library, a fake Hub serving it)
from test_hub_fallback import fresh  # noqa: E402, F401  (fixture: a fresh install)

import alloy.compose  # noqa: E402


def test_search_lists_hub_objects_that_are_not_fetched_yet(fresh, hub):
    rows = alloy.compose.search("brand box", k=20)
    row = next(r for r in rows if r["name"] == "brand_box")
    assert row["source"] == "toylib" and not row["placeable"]
    assert "not fetched yet" in row["reason"] and "alloy fetch brand_box" in row["reason"]
    assert not any("brand_box/" in path for path in hub)                 # the index only: no object downloaded


def test_search_without_measuring_still_lists_them(fresh, hub):
    assert "cup_a" in [r["name"] for r in alloy.compose.search("cup a", k=20, placeable=False)]
