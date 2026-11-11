# SPDX-License-Identifier: Apache-2.0
"""Objects fetched from the Hub on first use: the fallback that the API, compose and check use when this machine has
no imported libraries and no asset store.

Hub mode is on when all of these hold:
- no asset store is set (``$ALLOY_ASSET_STORE`` or the caller's own);
- no imported library is on disk (an alloy-assets checkout, or a wheel's ``alloy/_data/imports``);
- ``$ALLOY_ASSETS_REPO`` is not set to the empty string, which turns the Hub off.

Machines with imported libraries keep their asset store, and their tests never reach the network.
An object is fetched with its library's shared files into ``cache_root(<library>)``, laid out like an import root,
and the card library that asked is rescanned so it finds the card.

Example::

    if hub_mode():
        ensure(lib, "ycb_025_mug")       # fetched (or found in the cache), now ``lib.has_object("ycb_025_mug")``
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from alloy.assets.hub.source import REPO_ENV, HubError

log = logging.getLogger("alloy.assets.hub")

ROW_KEYS = ("name", "library", "category", "title")
#: the catalog lists objects from an index checked within this many seconds without asking the Hub again (a compose
#: run builds several catalogs, and the Hub rate-limits busy tokens for minutes)
INDEX_FRESH_S = 3600.0


def hub_mode(store: str | None = None) -> bool:
    """True when objects come from the Hub: no asset store, no imported library on this machine, and the Hub
    not turned off (``ALLOY_ASSETS_REPO=""``). Example: ``hub_mode()`` is True on a fresh ``pip install``."""
    from alloy.cards.roots import import_roots  # noqa: PLC0415 - the card layer imports compose, not the reverse
    from alloy.compose.fetch import store_from_env  # noqa: PLC0415

    raw = os.environ.get(REPO_ENV)
    if store or store_from_env() or (raw is not None and not raw.strip()):
        return False
    return not import_roots()


def catalog_rows(repo: str | None = None) -> list[dict[str, Any]]:
    """Name, library, category and title of every object in the Hub's index (re-checked when older than
    ``INDEX_FRESH_S``); [] (logged with the fix) when it cannot be read. Example: ``{r["library"] for r in
    catalog_rows()}``."""
    from alloy.assets.hub.source import load_index  # noqa: PLC0415

    try:
        rows = load_index(repo=repo, max_age_s=INDEX_FRESH_S)
    except HubError as exc:
        log.warning("object library on the Hub not listed: %s (fix: %s)", exc.message, exc.fix)
        return []
    return [{k: str(r.get(k) or "") for k in ROW_KEYS} for r in rows]


def library_roots(rows: list[dict[str, Any]]) -> list[Path]:
    """The cache root of every library in ``rows``, plus every cache root already on disk (sorted, no repeats)."""
    from alloy.compose.fetch import cache_root, cache_roots  # noqa: PLC0415

    return sorted({*(cache_root(r["library"]) for r in rows), *cache_roots()})


def _complete_root(name: str) -> Path | None:
    """The cache root that already holds object ``name`` with every file its manifest lists and every file its MJCF
    model names (a composed cabinet fetched without its door panel is not complete), or None."""
    from alloy.assets.hub.mjcf import absent_files  # noqa: PLC0415
    from alloy.compose.fetch import cache_roots, files_of, missing  # noqa: PLC0415

    for root in cache_roots():
        objects = root / "objects"
        if objects.is_dir() and any(objects.rglob(f"{name}.yaml")) and not missing(root, name, root) \
                and not absent_files(root, files_of(root, name)):
            return root
    return None


def ensure(lib: Any, name: str, repo: str | None = None) -> Path:
    """Make object card ``name`` visible to card library ``lib``. It is taken from the cache when it is complete
    there, else fetched from the Hub; then its cache root is added to ``lib.roots`` and ``lib`` is rescanned.
    Returns that cache root. Raises ``HubError`` (no access, offline with nothing cached, or an unknown name)."""
    from alloy.assets.hub.fetch import fetch  # noqa: PLC0415

    root = _complete_root(name)
    if root is None:
        log.info("fetching object %s from the object library on the Hub", name)
        root = Path(fetch(names=[name], repo=repo)["roots"][0])
    root = root.resolve()
    if root not in tuple(lib.roots):
        lib.roots = (*tuple(lib.roots), root)
    lib.rescan()
    return root


def fetch_fix(name: str, exc: HubError | None = None) -> str:
    """The fix for object ``name`` whose files are not on this machine: ``alloy fetch <name>``, after the Hub
    error's own fix (an unknown name gets the close names instead). Example: ``fetch_fix("ycb_025_mug")``."""
    hint = f"run `alloy fetch {name}` (the object library on the Hugging Face Hub)"
    if exc is None:
        return hint
    if exc.field == "names":
        return exc.fix
    return f"{exc.fix}; then {hint}"


__all__ = ["catalog_rows", "ensure", "fetch_fix", "hub_mode", "library_roots"]
