# SPDX-License-Identifier: Apache-2.0
"""Imported objects (``<alloy-assets>/imports/<source>/``) keep their meshes in an asset store, not in git. Before a
check assembles a scene, the files of every imported object it places are fetched from ``$ALLOY_ASSET_STORE``
(sha256-checked, ``alloy.compose.fetch``); without a store, each object whose files are missing is one card error
that says how to get them. On a machine with no imported library and no store, an object card the library lacks is
fetched from the object library on the Hub (``alloy.assets.hub.lazy``); when that fails (offline, no access), the
error's fix says what to do, then ``alloy fetch <name>``.

Example::

    >>> found = fetch_imported(src.lib, ["ycb_025_mug"])          # doctest: +SKIP
    >>> found[0].fix                                              # no store set, files missing
    'run `alloy fetch ycb_025_mug` (the object library on the Hugging Face Hub), set $ALLOY_ASSET_STORE ...'
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from alloy.check.report import Finding
from alloy.check.source import IMPORTS_DIR

log = logging.getLogger("alloy.check")


def fetch_imported(lib: Any, names: Iterable[str]) -> list[Finding]:
    """Fetch the missing files of each imported object in ``names``; a card error per object still missing."""
    out: list[Finding] = []
    for name in dict.fromkeys(names):
        if not lib.has_object(name):
            out += _from_hub(lib, name)
            continue
        root = import_root_of(lib, name)
        if root is None:
            continue
        try:
            from alloy.compose.fetch import STORE_ENV, fetch_object, missing  # noqa: PLC0415
        except ImportError:        # no importer tooling in this install: card validation reports the missing file
            return out
        if not missing(root, name):
            continue
        if os.environ.get(STORE_ENV):
            log.info("check: fetching %s from the asset store (imported library %s)", name, root.name)
            try:
                fetch_object(root, name, None)
                continue
            except Exception as exc:  # noqa: BLE001 - a failed fetch is a finding with the store's message
                why = f"fetching from ${STORE_ENV} failed: {type(exc).__name__}: {exc}"
        else:
            why = "its mesh files are not on this machine"
        from alloy.assets.hub.lazy import fetch_fix  # noqa: PLC0415

        out.append(Finding("cards", "error", f"{name} (imported library {root.name}): {why}",
                           f"{fetch_fix(name)}, set ${STORE_ENV} to the asset store (gs://..., https://... or a "
                           f"folder) and check again, or run python -m alloy.importers.{root.name} fetch",
                           objects=(name,), field=name))
    return out


def _from_hub(lib: Any, name: str) -> list[Finding]:
    """[] when ``name`` is now in ``lib`` (fetched from the Hub or found in the cache) or the Hub is not in use here;
    else one card error whose fix says what to do, then ``alloy fetch <name>``."""
    from alloy.assets.hub.lazy import ensure, fetch_fix, hub_mode  # noqa: PLC0415
    from alloy.assets.hub.source import HubError  # noqa: PLC0415

    if not hub_mode():
        return []
    try:
        ensure(lib, name)
        return []
    except HubError as exc:
        return [Finding("cards", "error", f"{name}: not on this machine, and the Hub fetch failed: {exc.message}",
                        fetch_fix(name, exc), objects=(name,), field=name)]


def import_root_of(lib: Any, name: str) -> Path | None:
    """The import library root (``.../imports/<source>``) that holds object card ``name``, or None."""
    if not lib.has_object(name):
        return None
    path = Path(lib.path_of("objects", name)).resolve()
    for parent in path.parents:
        if parent.parent.name == IMPORTS_DIR:
            return parent
    return None


__all__ = ["fetch_imported", "import_root_of"]
