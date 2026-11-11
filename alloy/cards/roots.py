# SPDX-License-Identifier: Apache-2.0
"""Where cards live: the engine's ``library/`` roots and the separate alloy-assets checkout (SCR-DC-1).

The assets root comes from ``$ALLOY_ASSETS`` (an empty value turns it off; a value that is not a directory is an
error), else the sibling checkout ``<repo>/../alloy-assets`` when it exists, so every worktree
``../alloy-wpN`` finds ``../alloy-assets``; else there is none.

Card files are found recursively (``objects/<area>/<name>.yaml``; folders starting with ``.`` are skipped) and the
file name is the card name. Library roots overlay each other (the first root that has a name wins, as before). The
assets root shares one name space with them: a name defined in two files of the assets root, or in the assets root
and a library root, is a clash, and looking it up raises ``CardError`` naming both files (no silent shadowing).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from alloy.core.errors import CardError

log = logging.getLogger("alloy.cards")

ASSETS_ENV = "ALLOY_ASSETS"
ASSETS_DIRNAME = "alloy-assets"
IMPORTS_DIR = "imports"  # <alloy-assets>/imports/<source>/: imported object libraries (SCR-OSS3-4)
KIND_DIRS: tuple[str, ...] = ("materials", "pairs", "objects", "scenes")


def repo_root() -> Path:
    """The checkout (or worktree) that holds the ``alloy`` package."""
    return Path(__file__).resolve().parents[2]


def assets_root() -> Path | None:
    """The alloy-assets checkout: ``$ALLOY_ASSETS``, else ``<repo>/../alloy-assets`` if it exists, else None."""
    raw = os.environ.get(ASSETS_ENV)
    if raw is not None:
        if not raw.strip():
            return None
        path = Path(raw).expanduser()
        if not path.is_dir():
            raise CardError("<library>", ASSETS_ENV, f"{raw!r} is not a directory (unset it or set it empty for none)")
        return path.resolve()
    sibling = repo_root().parent / ASSETS_DIRNAME
    return sibling.resolve() if sibling.is_dir() else None


def package_data() -> Path:
    """``alloy/_data``: what an installed wheel carries in place of the checkouts (OSS-2's layout)."""
    return Path(__file__).resolve().parents[1] / "_data"


def import_roots(assets: Path | None = None) -> list[Path]:
    """The imported card libraries that hold object cards, sorted: ``<assets>/imports/<source>/`` (``assets``, else
    ``assets_root()``), else what an installed wheel carries in ``alloy/_data/imports/<source>/``; empty if neither.
    Example: ``[.../alloy-assets/imports/gymanything, .../imports/robocasa, .../imports/ycb_gso]``."""
    base = assets if assets is not None else assets_root()
    parents = ([Path(base) / IMPORTS_DIR] if base is not None else []) + [package_data() / IMPORTS_DIR]
    for parent in parents:
        found = sorted(p for p in parent.iterdir() if (p / "objects").is_dir()) if parent.is_dir() else []
        if found:
            return found
    return []


def fetch_caches(imports: Sequence[Path]) -> list[Path]:
    """The user cache of each imported library (``alloy.compose.fetch.cache_root``: meshes fetched on first use,
    where an installed library cannot be written), one per import root whether or not it exists yet."""
    from alloy.compose.fetch import cache_root  # noqa: PLC0415 - OSS-19's fetch cache; compose imports cards

    return [cache_root(r.name) for r in imports]


def fetched_roots(imports: Sequence[Path]) -> list[Path]:
    """Every fetch cache a default library reads (SCR-OSS11-8): each import root's (``fetch_caches``), then every
    cache root on disk (``alloy.compose.fetch.cache_roots``), duplicates removed. A library fetched with no import
    root (a clean wheel install, ``alloy fetch`` from the Hugging Face library) is a cache root only."""
    from alloy.compose.fetch import cache_roots  # noqa: PLC0415 - compose imports cards

    return list(dict.fromkeys([*fetch_caches(imports), *cache_roots()]))


def default_roots() -> list[Path]:
    """The library roots every default card library uses (SCR-OSS3-4): the engine's ``library/``, then the fetch
    caches (``fetched_roots``), then the imported libraries. The first root that has a name wins: a library/ card is
    never shadowed, and a fetched copy (a RoboCasa MJCF, say) wins over the import root's."""
    from alloy.cards.library import default_root  # noqa: PLC0415 - library imports this module

    imports = import_roots()
    return [default_root(), *fetched_roots(imports), *imports]


def scan_kind(root: Path, kind: str) -> dict[str, list[Path]]:
    """Card files of one kind under ``root/kind`` (recursive, sorted): name -> every file with that name."""
    base = root / kind
    found: dict[str, list[Path]] = {}
    for path in sorted(base.rglob("*.yaml")):
        if any(part.startswith(".") for part in path.relative_to(base).parts):
            continue
        found.setdefault(path.stem, []).append(path)
    return found


@dataclass
class CardIndex:
    """name -> file per kind, plus the names that clash (defined by more than one file)."""

    paths: dict[str, dict[str, Path]] = field(default_factory=lambda: {k: {} for k in KIND_DIRS})
    clashes: dict[str, dict[str, tuple[Path, ...]]] = field(default_factory=lambda: {k: {} for k in KIND_DIRS})

    def names(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(set(self.paths[kind]) | set(self.clashes[kind])))

    def has(self, kind: str, name: str) -> bool:
        return name in self.paths[kind] or name in self.clashes[kind]

    def lookup(self, kind: str, name: str, where: str) -> Path:
        """The one file that defines ``name``; ``CardError`` (field ``name``) if none or several do."""
        files = self.clashes[kind].get(name)
        if files:
            listed = ", ".join(map(str, files))
            raise CardError(name, "name", f"{kind[:-1]} {name!r} is defined in {len(files)} files: {listed} "
                                          "(library/ and alloy-assets share one name space; rename one)")
        try:
            return self.paths[kind][name]
        except KeyError:
            raise CardError(name, "name", f"no {kind[:-1]} {name!r} in {where}") from None

    def _clash(self, kind: str, name: str, files: Sequence[Path]) -> None:
        self.paths[kind].pop(name, None)
        self.clashes[kind][name] = tuple(files)
        log.warning("library: %s %r is defined in %d files: %s", kind[:-1], name, len(files), ", ".join(map(str, files)))

    def add_overlay(self, root: Path) -> None:
        """A library root: names an earlier root already has are shadowed (logged); duplicates inside it clash."""
        for kind in KIND_DIRS:
            for name, files in scan_kind(root, kind).items():
                if self.has(kind, name):
                    log.info("library: %s/%s in %s is shadowed by an earlier root", kind, name, root)
                elif len(files) > 1:
                    self._clash(kind, name, files)
                else:
                    self.paths[kind][name] = files[0]

    def add_assets(self, root: Path) -> None:
        """The assets root: any name it shares with a library root, or defines twice, clashes."""
        for kind in KIND_DIRS:
            for name, files in scan_kind(root, kind).items():
                earlier = self.clashes[kind].get(name) or ((self.paths[kind][name],) if name in self.paths[kind] else ())
                every = [*earlier, *files]
                if len(every) > 1:
                    self._clash(kind, name, every)
                else:
                    self.paths[kind][name] = files[0]


def build_index(roots: Sequence[Path], assets: Path | None) -> CardIndex:
    index = CardIndex()
    for root in roots:
        index.add_overlay(root)
    if assets is not None:
        index.add_assets(assets)
    return index
