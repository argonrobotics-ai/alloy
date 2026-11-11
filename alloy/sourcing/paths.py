# SPDX-License-Identifier: Apache-2.0
"""Where sourcing reads cards and writes records, inside the alloy-assets checkout.

- ``<assets>/imports/<library>/``: importer output (object and material cards), read only here.
- ``<assets>/sources/imports/``: what this package writes: ``cache/`` (per category / material),
  ``<library>/<object>.yaml`` (per-object records) and ``coverage.json`` / ``COVERAGE.md``.

The assets checkout is ``$ALLOY_ASSETS`` or the sibling ``../alloy-assets`` (``alloy.cards.roots.assets_root``).

Example:
    >>> from pathlib import Path
    >>> sources_root(Path("/data/alloy-assets")).as_posix()
    '/data/alloy-assets/sources/imports'
"""

from __future__ import annotations

from pathlib import Path

from alloy.cards.roots import assets_root
from alloy.sourcing.fields import SourcingError


def assets(path: Path | None = None) -> Path:
    """The alloy-assets checkout to use: ``path`` if given, else ``assets_root()``; raises if there is none."""
    if path is not None:
        return Path(path)
    root = assets_root()
    if root is None:
        raise SourcingError("no alloy-assets checkout found; set ALLOY_ASSETS=/path/to/alloy-assets or pass --assets")
    return root


def sources_root(path: Path | None = None) -> Path:
    """``<assets>/sources/imports``: cache, per-object records and coverage reports."""
    return assets(path) / "sources" / "imports"


def imports_root(path: Path | None = None) -> Path:
    """``<assets>/imports``: one folder per importer library (``ycb_gso``, ``robocasa``, ...)."""
    return assets(path) / "imports"
