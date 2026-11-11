# SPDX-License-Identifier: Apache-2.0
"""The card library: named materials, pairs, objects and scenes under one or more roots.

Layout per root: ``materials/<name>.yaml``, ``pairs/<a>__<b>.yaml`` (names sorted), ``objects/<name>.yaml``,
``scenes/<name>.yaml``, found recursively (``objects/<area>/<name>.yaml``); assets (meshes, robots) by path relative
to a root. Library roots overlay each other (the first root that has a name wins, later ones are shadowed, logged).
The alloy-assets checkout (``alloy.cards.roots.assets_root``: ``$ALLOY_ASSETS`` or the sibling ``../alloy-assets``) is
one more root in the same name space: a name it shares with a library root raises ``CardError`` naming both files.
Cards are loaded lazily, checked against their file name, validated and cached. ``put`` writes a validated card
(in place for a card that lives in the assets root, else to the first root) and appends a row to ``history.jsonl``.
"""

from __future__ import annotations

import datetime as _dt
import enum
import json
import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alloy.cards import geometry as geo
from alloy.cards import types as ct
from alloy.cards.identity import resolve_object
from alloy.cards.loader import dump_card, load_card
from alloy.cards.roots import CardIndex, assets_root, build_index, fetched_roots, import_roots
from alloy.cards.validate import log_issues, raise_on_error, validate_material, validate_pair
from alloy.cards.validate_object import validate_object
from alloy.cards.validate_scene import validate_scene
from alloy.core.errors import CardError
from alloy.core.ids import digest_of

log = logging.getLogger("alloy.cards")

KINDS: dict[str, type] = {"materials": ct.MaterialCard, "pairs": ct.PairCard, "objects": ct.ObjectCard,
                          "scenes": ct.SceneCard}
RIGID_DT = 1.25e-3                       # s, rigid target on the D4 grid (1/600 s snapped)


class _Auto(enum.Enum):
    AUTO = "auto"


AUTO = _Auto.AUTO                         # assets: the alloy-assets checkout when ``roots`` is the default


LIBRARY_ENV = "ALLOY_LIBRARY"          # a card library folder to use instead of the shipped one (empty: unset)


def _shipped_roots() -> tuple[Path, Path]:
    """``(<repo>/library, alloy/_data/library)``: a source checkout's library and the package's installed copy."""
    here = Path(__file__).resolve()
    return here.parents[2] / "library", here.parents[1] / "_data" / "library"


def default_root() -> Path:
    """The shipped card library (SCR-OSS2-2): ``$ALLOY_LIBRARY`` when set (it must be a folder), else ``<repo>/library``
    in a source checkout, else the copy installed with the package (``alloy/_data/library``). Every reader of the
    library's materials, objects, scenes or robots goes through here. Example: ``(default_root() / "materials")``."""
    raw = os.environ.get(LIBRARY_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_dir():
            raise CardError("<library>", LIBRARY_ENV, f"{raw!r} is not a directory (unset it for the shipped library)")
        return path.resolve()
    repo, packaged = _shipped_roots()
    return repo if repo.is_dir() else packaged


LIBRARY_FIX = f"install Alloy with pip (pip install alloy-engine), or set {LIBRARY_ENV} to a card library folder"


def library_missing() -> str | None:
    """None when the default card library holds material cards; else where it was searched, for errors that would
    otherwise list no names (a plain ``pip install .`` from a clone shipped no library: the first Mac test). The fix
    to give with it is ``LIBRARY_FIX``."""
    root = default_root()
    if next((root / "materials").rglob("*.yaml"), None) is not None:
        return None
    searched = [str(root)] if os.environ.get(LIBRARY_ENV, "").strip() else [str(r) for r in _shipped_roots()]
    return f"the card library was not found (no material cards in {' or '.join(searched)})"


def pair_key(a: str, b: str) -> str:
    return "__".join(sorted((a, b)))


class CardLibrary:
    """Lazy, validated access to the cards under ``roots`` and ``assets``.

    ``roots``: default ``alloy.cards.roots.default_roots()``: the repo's ``library/``, the fetch caches (each imported
    library's, and every fetched library on disk; SCR-OSS11-8), then the imported libraries
    (``<assets>/imports/<source>/``, or a wheel's ``alloy/_data/imports/``; SCR-OSS3-4). ``imports=False`` keeps the default to ``library/`` alone (sweeps over the engine's own cards); an
    explicit ``assets=None`` leaves the imports out too. Explicit roots are used as given.
    ``assets``: a directory, None, or ``AUTO`` (default): ``assets_root()`` when ``roots`` is the default, None when
    explicit roots are given (tests on a copied library stay independent of the checkout; pass it to add it)."""

    def __init__(self, roots: Sequence[Path] | None = None, assets: Path | str | None | _Auto = AUTO,
                 imports: bool = True) -> None:
        opted_out = assets is None  # an explicit assets=None: no checkout and no imported libraries
        if assets is AUTO:
            assets = assets_root() if not roots else None
        self.assets: Path | None = Path(assets).resolve() if assets is not None else None
        with_imports = imports and not opted_out and not roots
        found = import_roots(self.assets) if with_imports else []
        default = [default_root(), *(fetched_roots(found) if with_imports else []), *found]
        self.roots: tuple[Path, ...] = tuple(Path(r).resolve() for r in (roots or default))
        self._index = CardIndex()
        self._cache: dict[tuple[str, str], Any] = {}
        self._issues: dict[tuple[str, float], list[ct.CardIssue]] = {}
        self._meshes: dict[Path, geo.MeshData] = {}
        self._mjcf: dict[tuple[Path, str | None], geo.MeshData] = {}
        self.rescan()

    # ------------------------------------------------------------------------------------------- index

    def rescan(self) -> None:
        """Re-read the directory listing and drop every cached card."""
        self._index = build_index(self.roots, self.assets)
        self._cache.clear()
        self._issues.clear()
        self._meshes.clear()
        self._mjcf.clear()

    @property
    def search_roots(self) -> tuple[Path, ...]:
        """Library roots, then the assets root (if any)."""
        return self.roots + ((self.assets,) if self.assets is not None else ())

    def names(self, kind: str) -> tuple[str, ...]:
        """Every name of a kind, clashing names included (looking one up raises)."""
        return self._index.names(kind)

    def path_of(self, kind: str, name: str) -> Path:
        """The file that defines a card; ``CardError`` (field ``name``) if none does or two do. When the default card
        library is missing, the error says so and gives the fix (``library_missing``)."""
        try:
            return self._index.lookup(kind, name, ", ".join(map(str, self.search_roots)))
        except CardError as e:
            missing = library_missing() if default_root().resolve() in self.roots else None
            if missing is None or self._index.has(kind, name):
                raise
            raise CardError(name, "name", f"{e.message}: {missing}; fix: {LIBRARY_FIX}") from None

    def in_assets(self, kind: str, name: str) -> bool:
        """True if the card comes from the assets root."""
        path = self._index.paths[kind].get(name)
        return path is not None and self.assets is not None and path.is_relative_to(self.assets)

    def has_material(self, name: str) -> bool:
        return self._index.has("materials", name)

    def has_object(self, name: str) -> bool:
        return self._index.has("objects", name)

    def has_pair(self, a: str, b: str) -> bool:
        return self._index.has("pairs", pair_key(a, b))

    # ------------------------------------------------------------------------------------------- cards

    def _load(self, kind: str, name: str) -> Any:
        key = (kind, name)
        if key not in self._cache:
            card = load_card(self.path_of(kind, name), KINDS[kind])
            stem = pair_key(card.a, card.b) if kind == "pairs" else getattr(card, "name", name)
            if kind != "scenes" and stem != name:
                raise CardError(name, "name", f"file {self.path_of(kind, name).name} holds {stem!r}")
            self._cache[key] = card
        return self._cache[key]

    def material(self, name: str) -> ct.MaterialCard:
        """A validated material card; errors raise ``CardError``."""
        card = self._load("materials", name)
        if ("materials:valid", name) not in self._cache:
            issues = validate_material(card)
            log_issues(issues)
            raise_on_error(issues, f"materials/{name}")
            self._cache[("materials:valid", name)] = True
        return card

    def pair(self, a: str, b: str) -> ct.PairCard | None:
        """The sourced pair card for two materials (unordered), or None if the library has none."""
        key = pair_key(a, b)
        if not self._index.has("pairs", key):
            return None
        card = self._load("pairs", key)
        if ("pairs:valid", key) not in self._cache:
            raise_on_error(validate_pair(card, self), f"pairs/{key}")
            self._cache[("pairs:valid", key)] = True
        return card

    def peek_object(self, name: str) -> ct.ObjectCard:
        """An object card parsed and name-checked but not validated (for cross-reference checks)."""
        return self._load("objects", name)

    def object_issues(self, name: str, rigid_dt: float = RIGID_DT) -> list[ct.CardIssue]:
        key = (name, rigid_dt)
        if key not in self._issues:
            self._issues[key] = validate_object(self.peek_object(name), self, rigid_dt)
            log_issues(self._issues[key])
        return self._issues[key]

    def object(self, name: str, rigid_dt: float = RIGID_DT) -> ct.ObjectCard:
        """A validated object card; errors raise ``CardError``."""
        raise_on_error(self.object_issues(name, rigid_dt), f"objects/{name}")
        return self.peek_object(name)

    def peek_scene(self, name: str) -> ct.SceneCard:
        """A scene card parsed but not validated."""
        return self._load("scenes", name)

    def scene(self, name: str, rigid_dt: float = RIGID_DT) -> ct.SceneCard:
        """A validated scene card (its objects included)."""
        card = self._load("scenes", name)
        raise_on_error(validate_scene(card, self, rigid_dt, name=name), f"scenes/{name}")
        return card

    def iter_cards(self, kind: str) -> Iterator[tuple[str, Any]]:
        """(name, parsed card) for every card of a kind, in name order (not validated)."""
        for name in self.names(kind):
            yield name, self._load(kind, name)

    def resolve(self, card: ct.ObjectCard, placement: ct.Placement | None = None,
                rigid_dt: float = RIGID_DT) -> ct.ResolvedObject:
        """Validate and resolve an object card (see ``alloy.cards.identity``); raises ``CardError``."""
        return resolve_object(card, self, placement, rigid_dt)

    # ------------------------------------------------------------------------------------------ assets

    def asset_path(self, rel: str) -> Path | None:
        """An asset path: absolute as given, else the first library root or the assets root that has it; None if
        nowhere. A relative path that exists under both a library root and the assets root raises ``CardError``."""
        p = Path(rel)
        if p.is_absolute():
            return p if p.exists() else None
        in_lib = next((root / p for root in self.roots if (root / p).exists()), None)
        in_assets = self.assets / p if self.assets is not None and (self.assets / p).exists() else None
        if in_lib is not None and in_assets is not None:
            raise CardError(rel, "path", f"{rel!r} exists in both {in_lib} and {in_assets} (rename one)")
        return in_assets if in_assets is not None else in_lib

    def mesh(self, rel: str) -> geo.MeshData:
        """A mesh asset (OBJ/STL), cached by resolved path."""
        path = self.asset_path(rel)
        if path is None:
            raise FileNotFoundError(rel)
        if path not in self._meshes:
            self._meshes[path] = geo.read_mesh(path)
        return self._meshes[path]

    def mesh_for(self, card: ct.ObjectCard) -> geo.MeshData | None:
        """The (scaled) mesh behind a mesh/SDF/tet-backed object, or None for other sources."""
        s = card.shape
        if isinstance(s, ct.MeshSource):
            return self.mesh(s.path).scaled(s.scale)
        if isinstance(s, ct.MJCFSource):
            return self.mjcf_cloud(s)
        rel = s.path if isinstance(s, ct.SDFSource) else (s.from_mesh if isinstance(s, ct.TetSource) else None)
        if rel and Path(rel).suffix.lower() in (".obj", ".stl") and self.asset_path(rel):
            return self.mesh(rel)
        return None

    def mjcf_cloud(self, s: ct.MJCFSource) -> geo.MeshData | None:
        """Collision-shape points of an MJCF source (faces empty: extents only), or None if the file is missing.
        Raises ``KeyError`` for an unknown ``body`` and whatever Newton's MJCF importer raises for a bad file."""
        path = self.asset_path(s.path)
        if path is None:
            return None
        key = (path, s.body)
        if key not in self._mjcf:
            from alloy.cards.mjcf import mjcf_points  # noqa: PLC0415 - imports Newton

            self._mjcf[key] = geo.MeshData(mjcf_points(path, s.body), np.zeros((0, 3), dtype=np.int64))
        return self._mjcf[key]

    # --------------------------------------------------------------------------------------------- put

    def _target(self, card: ct.MaterialCard | ct.PairCard | ct.ObjectCard) -> tuple[str, str]:
        if isinstance(card, ct.MaterialCard):
            return "materials", card.name
        if isinstance(card, ct.PairCard):
            return "pairs", pair_key(card.a, card.b)
        if isinstance(card, ct.ObjectCard):
            return "objects", card.name
        raise TypeError(f"put: not a material, pair or object card: {type(card).__name__}")

    def _issues_for(self, card: ct.MaterialCard | ct.PairCard | ct.ObjectCard) -> list[ct.CardIssue]:
        if isinstance(card, ct.MaterialCard):
            return validate_material(card)
        if isinstance(card, ct.PairCard):
            return validate_pair(card, self)
        return validate_object(card, self, RIGID_DT)

    def _put_target(self, kind: str, stem: str) -> tuple[Path, Path]:
        """(root, file) a put writes: the card's own file if it lives in the assets root, else the first root."""
        if self._index.has(kind, stem) and self.in_assets(kind, stem):
            return self.assets, self.path_of(kind, stem)  # type: ignore[return-value]
        if self._index.clashes[kind].get(stem):
            self.path_of(kind, stem)                      # raises: the name clashes
        return self.roots[0], self.roots[0] / kind / f"{stem}.yaml"

    def put(self, card: ct.MaterialCard | ct.PairCard | ct.ObjectCard, *, reason: str) -> Path:
        """Validate ``card`` and write it (see ``_put_target``), replacing any card of that name; returns the path.
        The history row goes to ``<root>/history.jsonl``, or next to the card for a card in the assets root."""
        kind, stem = self._target(card)
        if not reason.strip():
            raise CardError(stem, "<put>", "put needs a reason (it is recorded in the file and history.jsonl)")
        raise_on_error(self._issues_for(card), f"put {kind}/{stem}")
        root, path = self._put_target(kind, stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(dump_card(card, header=f"written by CardLibrary.put {stamp}: {reason}"), encoding="utf-8")
        os.replace(tmp, path)
        row = {"time": stamp, "kind": kind, "name": stem, "path": str(path.relative_to(root)), "reason": reason,
               "digest": digest_of(card)}
        history = path.parent / "history.jsonl" if root == self.assets else root / "history.jsonl"
        with open(history, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        log.info("library.put %s/%s (%s): %s", kind, stem, row["digest"][:12], reason)
        self.rescan()
        return path
