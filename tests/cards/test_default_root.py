# SPDX-License-Identifier: Apache-2.0
"""Where the shipped card library is found (SCR-OSS2-2, coordinator 2026-11-08): ``default_root()`` is
``$ALLOY_LIBRARY``, else the checkout's ``library/``, else the package's ``alloy/_data/library``; every reader goes
through it; and when there is no library (the first Mac test: a plain ``pip install .`` from a clone shipped none),
the error names where it looked and the fix instead of "use one of []"."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from alloy.cards import library as L
from alloy.core.errors import CardError

REPO = Path(__file__).resolve().parents[2]
# a library path built by hand instead of default_root(): <repo>/library from __file__, or REPO / "library"
BYPASS = re.compile(r"""(parents\[\d\]|parent\.parent|\bREPO|\bROOT)\s*/\s*["']library["']""")


def test_alloy_library_overrides_the_shipped_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(L.LIBRARY_ENV, str(tmp_path))
    assert L.default_root() == tmp_path.resolve()
    monkeypatch.setenv(L.LIBRARY_ENV, "  ")
    checkout, packaged = L._shipped_roots()                          # empty: the shipped library, checkout or package
    assert L.default_root() == (checkout if checkout.is_dir() else packaged)
    monkeypatch.setenv(L.LIBRARY_ENV, str(tmp_path / "nowhere"))
    with pytest.raises(CardError, match=L.LIBRARY_ENV):
        L.default_root()


def test_without_a_checkout_the_root_is_the_package_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(L.LIBRARY_ENV, raising=False)
    packaged = tmp_path / "_data" / "library"
    monkeypatch.setattr(L, "_shipped_roots", lambda: (tmp_path / "no_checkout" / "library", packaged))
    assert L.default_root() == packaged
    missing = L.library_missing()
    assert missing is not None and str(tmp_path / "no_checkout" / "library") in missing and str(packaged) in missing


def test_an_empty_library_says_where_it_looked_and_the_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The card layer and the API both name the searched folder, say the library was not found, and give the fix."""
    import alloy
    from alloy.api.errors import SceneError

    assert L.library_missing() is None                              # this checkout has its library
    monkeypatch.setenv(L.LIBRARY_ENV, str(tmp_path))
    missing = L.library_missing()
    assert missing == f"the card library was not found (no material cards in {tmp_path.resolve()})"
    with pytest.raises(CardError) as card_err:
        L.CardLibrary(assets=None).material("plywood_marine")
    assert "card library was not found" in str(card_err.value) and "pip install alloy-engine" in str(card_err.value)
    scene = alloy.Scene(dt=0.01)                                    # AGENTS.md's quick start, with no library
    with pytest.raises(SceneError) as api_err:                      # its ground's default material
        scene.add_ground()
    assert str(tmp_path.resolve()) in api_err.value.message and "card library was not found" in api_err.value.message
    assert api_err.value.fix == L.LIBRARY_FIX and "use one of" not in str(api_err.value)


def test_every_library_reader_goes_through_default_root() -> None:
    """No engine module builds ``<repo>/library`` itself: an installed package has no checkout above it."""
    allowed = {REPO / "alloy" / "cards" / "library.py"}             # _shipped_roots: the one place it is spelled
    bypass = [f"{p.relative_to(REPO)}:{n}" for p in sorted((REPO / "alloy").rglob("*.py"))
              if p not in allowed and "_vendor" not in p.parts
              for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if BYPASS.search(line)]
    assert bypass == []
