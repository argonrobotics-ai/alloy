# SPDX-License-Identifier: Apache-2.0
"""The objects the scene builder can use: every object card in Alloy's ``library/``, the alloy-assets checkout and each
imported library under ``<alloy-assets>/imports/<source>/`` (YCB, Google Scanned Objects, RoboCasa, Objaverse, ...).

The index is cheap (file names plus the import manifests' titles and categories), so a library of thousands of
objects is searched in milliseconds; cards are read, routed and measured only for the matches actually used. A new
import directory is picked up on the next ``Catalog()`` with no code change.

On a machine with no imported library and no asset store (a fresh ``pip install``), the index also lists every
object of the object library on the Hugging Face Hub (its ``index.jsonl``), and an object is fetched when it is first
used (``alloy.assets.hub.lazy``); objects fetched earlier are found in the fetch cache.

Example::

    >>> cat = Catalog()
    >>> [m.name for m in cat.search("cube", k=2)]                     # doctest: +SKIP
    ['cube_beech_30', ...]
    >>> cat.info("mug_stoneware").placeable                            # doctest: +SKIP
    True
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alloy.cards.roots import import_roots  # one definition (SCR-OSS3-4), re-exported here
from alloy.compose import words as W
from alloy.compose.body import Body, mesh_com, shell_com
from alloy.compose.spec import ComposeError

log = logging.getLogger("alloy.compose")

#: rigid bodies, fixtures and soft solids (fruit, foam): things that keep their shape when set down
PLACEABLE_ROUTES = frozenset({"rigid", "rigid_compliant", "static", "soft_vbd"})
#: thin-shell routes placed as objects when the card is a made shape (a mesh: an open gift bag, a carrier bag); flat
#: sheet cards on these routes are the soft builder's cloth templates ("a sheet of kraft paper")
SHELL_ROUTES = frozenset({"cloth_film"})
IMPORTS_DIR = "imports"


@dataclass(frozen=True)
class Entry:
    """One object card in the index: name, where it comes from, and the words it answers to."""

    name: str
    source: str                         # "library", "assets" or the import library ("ycb_gso", ...)
    path: str
    title: str
    words: frozenset[str]
    root: str = ""                      # the import root holding the card ("" for library and alloy-assets)


@dataclass(frozen=True)
class Info:
    """What the builder knows about one object after reading its card: ``placeable`` or ``reason`` why not."""

    name: str
    placeable: bool
    reason: str = ""
    route: str = ""
    material: str = ""
    motion: str = ""
    size: tuple[float, float, float] | None = None      # rest-frame extents, m
    sheet: bool = False                                  # a flat shell sheet: placed only when named (CARD_WORDS)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "placeable", "reason", "route", "material", "motion")}
        d["size_m"] = [round(s, 4) for s in self.size] if self.size else None
        return d


@dataclass(frozen=True)
class Match:
    """A search hit: ``score`` (higher is better; 10 = the noun itself, 6 = a synonym) and the entry."""

    name: str
    score: float
    entry: Entry

    def to_dict(self) -> dict:
        return {"name": self.name, "score": round(self.score, 2), "source": self.entry.source,
                "title": self.entry.title}


class Catalog:
    """Search and measure objects across the library roots.

    ``roots``: extra card roots searched first (your own cards); ``imports``: also use the import libraries;
    ``library``: an existing ``CardLibrary`` to use as is (its roots define what is searched); ``store``: where to
    fetch an imported object's missing files (default ``$ALLOY_ASSET_STORE``; ``fetch=False`` never fetches). With no
    imported library and no store, the Hub's objects are listed too and fetched on first use (``fetch=False``: only
    those already in the fetch cache, unless ``hub_index``: then every object of the Hub's index is listed and the
    ones not fetched yet say so, as ``alloy.compose.search`` does).
    Example: ``Catalog(roots=["my_cards"]).search("mug")``."""

    def __init__(self, roots: Sequence[str | Path] = (), *, imports: bool = True, library=None,
                 store: str | None = None, fetch: bool = True, hub_index: bool = False) -> None:
        from alloy.cards.library import CardLibrary, default_root  # noqa: PLC0415
        from alloy.cards.roots import assets_root  # noqa: PLC0415

        if library is None:
            extra = [Path(r).expanduser().resolve() for r in roots]
            self.import_roots = import_roots() if imports else []
            caches = self._cache_setup(imports, fetch or hub_index, store)
            library = CardLibrary(roots=[*extra, default_root(), *caches, *self.import_roots], assets=assets_root())
        else:
            self.import_roots = [r for r in library.roots if r.parent.name == IMPORTS_DIR and (r / "objects").is_dir()]
            self.hub, self.hub_rows, self.fetched_libs = False, [], []
        self.library = library
        self.store, self.fetch = store, fetch
        self.entries = self._index()
        self._info: dict[str, Info] = {}
        self._bodies: dict[str, Body] = {}
        log.info("compose catalog: %d objects from %d roots", len(self.entries), len(self.library.search_roots))

    def _cache_setup(self, imports: bool, listed: bool, store: str | None) -> list[Path]:
        """The fetch caches the library reads (``alloy.cards.roots.fetched_roots``), plus, in Hub mode (no imported
        library, no store), the Hub's index rows (when ``listed``) and their libraries' caches. Sets ``fetched_libs``:
        the caches of libraries with no import root, whose cards count as that library's (``_source``)."""
        from alloy.assets.hub.lazy import catalog_rows, hub_mode, library_roots  # noqa: PLC0415
        from alloy.cards.roots import fetched_roots  # noqa: PLC0415

        self.hub = bool(imports) and not self.import_roots and hub_mode(store)
        self.hub_rows = catalog_rows() if self.hub and listed else []
        caches = [*(fetched_roots(self.import_roots) if imports else []),
                  *(library_roots(self.hub_rows) if self.hub else [])]
        caches = list(dict.fromkeys(caches))
        own = {r.name for r in self.import_roots}
        self.fetched_libs = [r for r in caches if r.name not in own]
        return caches

    # --- index ---------------------------------------------------------------------------------------------------

    def _index(self) -> dict[str, Entry]:
        meta = self._manifests()
        out: dict[str, Entry] = {}
        for name in self.library.names("objects"):
            try:
                path = self.library.path_of("objects", name)
            except Exception as exc:  # noqa: BLE001 - a clashing name is skipped and logged, not fatal
                log.warning("compose catalog: skipping %s: %s", name, exc)
                continue
            title, category = meta.get(name, ("", ""))
            words = frozenset(W.tokens(name) + W.tokens(title) + W.tokens(category) + list(W.CARD_WORDS.get(name, ())))
            source, root = self._source(path)
            out[name] = Entry(name, source, str(path), title or category, words, root)
        for r in self.hub_rows:                     # not fetched yet: no card file, fetched on first use
            if r["name"] not in out:
                out[r["name"]] = _hub_entry(r)
        return out

    def _manifests(self) -> dict[str, tuple[str, str]]:
        meta: dict[str, tuple[str, str]] = {}
        for root in [*self.import_roots, *self.fetched_libs]:
            path = root / "manifest.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("compose catalog: %s unreadable (%s); its titles are not searched", path, exc)
                continue
            rows = data.get("objects") or data.get("assets") or {}       # importers name the table either way
            for name, row in rows.items():
                if isinstance(row, dict):
                    title = row.get("title") or str(row.get("asset") or "").replace("/", " ")
                    meta[name] = (str(title), str(row.get("category") or ""))
        return meta

    def _source(self, path: Path) -> tuple[str, str]:
        """(source name, import root or "") of a card file."""
        p = Path(path).resolve()
        for root in [*self.import_roots, *self.fetched_libs]:
            if p.is_relative_to(root.resolve()):
                return root.name, str(root)
        assets = self.library.assets
        if assets is not None and p.is_relative_to(Path(assets).resolve()):
            return "assets", ""
        return "library", ""

    # --- search --------------------------------------------------------------------------------------------------

    def search(self, query: str, hints: Sequence[str] = (), k: int = 20) -> list[Match]:
        """Objects matching the noun ``query`` (an exact card name returns that card alone), best first.

        ``hints`` (e.g. "wooden", "small") re-rank matches by material and name words.
        Example: ``[m.name for m in Catalog().search("ball", ["steel"])]``."""
        if query in self.entries:
            return [Match(query, 100.0, self.entries[query])]
        words = W.tokens(query)
        if not words:
            return []
        noun, hints = words[-1], [*words[:-1], *hints]          # "fruit bowl": bowl, hinted by fruit
        syn = W.expand(noun) - {noun}
        hits = []
        for e in self.entries.values():
            base = 10.0 if noun in e.words else (6.0 if syn & e.words else 0.0)
            if base:
                head = 1.0 if noun in _head(e.name, e.source) else 0.0      # a mug, not a mug tree
                hits.append(Match(e.name, base + head + self._hint_score(e, hints), e))
        hits.sort(key=lambda m: (-m.score, len(m.name), m.name))
        return hits[:k]

    def _hint_score(self, e: Entry, hints: Sequence[str]) -> float:
        score = 0.0
        for h in hints:
            if h in W.MATERIAL_WORDS:
                toks = set(e.words) | set(self._material_of(e.name).split("_"))
                score += 3.0 if any(_family(s, t) for s in W.MATERIAL_WORDS[h] for t in toks) else 0.0
            elif h not in W.SIZE_WORDS and W.singular(h) in e.words:
                score += 2.0
        return score

    def _material_of(self, name: str) -> str:
        try:
            return self.library.peek_object(name).material
        except Exception:  # noqa: BLE001 - an unreadable card scores no material hint; info() reports it
            return ""

    # --- per-object facts ----------------------------------------------------------------------------------------

    def info(self, name: str) -> Info:
        """Read, route and measure one object (cached): placeable, or why not, plus its route and size."""
        if name not in self._info:
            self._info[name] = self._analyze(name)
        return self._info[name]

    def mark_broken(self, name: str, reason: str, what: str = "does not build") -> None:
        """Record that ``name`` failed in a scene (``what``: "does not build", "does not rest"): it is not offered
        again in this catalog. Example: ``cat.mark_broken("ycb_025_mug", "slides 14 mm/s", "does not rest")``."""
        self._info[name] = Info(name, False, f"{what}: {reason}")
        self._bodies.pop(name, None)

    def body(self, name: str) -> Body:
        """The rest-frame geometry of a placeable object (``info(name).placeable`` must be True)."""
        if name not in self._bodies:
            info = self.info(name)
            if not info.placeable:
                raise ComposeError(f"{name} cannot be placed: {info.reason}", "pick another object (search lists "
                                   "placeable ones first)", field="object")
        return self._bodies[name]

    def _analyze(self, name: str) -> Info:
        from alloy.cards import types as ct  # noqa: PLC0415

        if name not in self.entries:
            raise ComposeError(f"no object card named {name!r}", f"use a name from search(); close: "
                               f"{[m.name for m in self.search(name, k=3)]}", field="object")
        entry = self.entries[name]
        why = self._files(entry) if entry.root else ""
        if why:
            return Info(name, False, why)
        try:
            card = self.library.object(name)
        except Exception as exc:  # noqa: BLE001 - an invalid card is not placeable; the reason says why
            hint = "" if self.entries[name].source in ("library", "assets") else \
                " (an imported object: are its files fetched? python -m alloy.importers.<source> fetch)"
            return Info(name, False, f"card does not load: {type(exc).__name__}: {str(exc)[:200]}{hint}")
        shape = card.shape
        if not isinstance(shape, (ct.Primitive, ct.MeshSource, ct.MJCFSource)):
            return Info(name, False, f"{type(shape).__name__} shapes are not placed by the scene builder yet "
                                     "(primitives, meshes and MJCF bodies are)", material=card.material,
                        motion=card.motion)
        mesh = _mesh(self.library, card, shape)
        if isinstance(mesh, str):
            return Info(name, False, mesh, material=card.material, motion=card.motion)
        route = self._route(name)
        if route not in PLACEABLE_ROUTES | SHELL_ROUTES:
            return Info(name, False, f"route {route}: the scene builder places rigid, static and thin-shell objects "
                        "(an open bag)", route=route, material=card.material, motion=card.motion)
        v, f = mesh
        shell = route in SHELL_ROUTES
        com = np.zeros(3) if isinstance(shape, ct.Primitive) else shell_com(v, f) if shell else mesh_com(v, f)
        self._bodies[name] = Body.from_mesh(name, v, f, com)
        return Info(name, True, route=route, material=card.material, motion=card.motion,
                    size=self._bodies[name].size, sheet=shell and isinstance(shape, ct.Primitive))

    def _files(self, entry: Entry) -> str:
        """"" when an imported object's files are on disk (fetched now when needed), else why they are not."""
        from alloy.assets.hub.lazy import fetch_fix  # noqa: PLC0415
        from alloy.compose.fetch import STORE_ENV, fetch_object, missing  # noqa: PLC0415

        root, name = Path(entry.root), entry.name
        if self.hub:
            return self._from_hub(entry)
        try:
            if self.fetch:
                fetch_object(root, name, self.store)
            elif missing(root, name):
                return f"its files are not fetched yet ({fetch_fix(name)}, or set ${STORE_ENV} and compose fetches " \
                       "them)"
        except ComposeError as exc:
            return f"{exc.message}; {fetch_fix(name)}, or {exc.fix}"
        return ""

    def _from_hub(self, entry: Entry) -> str:
        """Hub mode: "" when the object is in the fetch cache (complete) or was fetched now, else why not."""
        from alloy.assets.hub.lazy import ensure, fetch_fix  # noqa: PLC0415
        from alloy.assets.hub.source import HubError  # noqa: PLC0415
        from alloy.compose.fetch import missing  # noqa: PLC0415

        root, name = Path(entry.root), entry.name
        if self.library.has_object(name) and not missing(root, name, root):
            return ""
        if not self.fetch:
            return f"its files are not fetched yet ({fetch_fix(name)})"
        try:
            ensure(self.library, name)
        except HubError as exc:
            return f"{exc.message}; {fetch_fix(name, exc)}"
        return ""

    def _route(self, name: str) -> str:
        from alloy.cards import types as ct  # noqa: PLC0415
        from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

        scene = ct.SceneCard(robots=(), objects=(ct.Placement(name, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),))
        try:
            return str(SceneAssembler(self.library).plan(scene).report.decisions[name].route)
        except Exception as exc:  # noqa: BLE001 - routing failure makes the object unplaceable, with the reason
            return f"unroutable ({type(exc).__name__}: {exc})"


def _hub_entry(row: dict[str, str]) -> Entry:
    """A Hub index row as an entry: words from its name, title and category; root = its library's fetch cache."""
    from alloy.compose.fetch import cache_root  # noqa: PLC0415

    name = row["name"]
    words = frozenset(W.tokens(name) + W.tokens(row["title"]) + W.tokens(row["category"]) +
                      list(W.CARD_WORDS.get(name, ())))
    return Entry(name, row["library"], "", row["title"] or row["category"], words, str(cache_root(row["library"])))


def _mesh(lib, card, shape) -> tuple[np.ndarray, np.ndarray] | str:
    """(vertices, faces) in the body frame, or the reason there is none."""
    from alloy.cards import types as ct  # noqa: PLC0415
    from alloy.compose import geom  # noqa: PLC0415

    if isinstance(shape, ct.Primitive):
        return geom.primitive_mesh(shape.kind, shape.size)
    path = lib.asset_path(shape.path)
    fetch = "fetch it first (imported libraries: python -m alloy.importers.<source> fetch)"
    if path is None:
        return f"{shape.path} is not on disk; {fetch}"
    if isinstance(shape, ct.MJCFSource):
        from alloy.compose.mjcf import mjcf_mesh  # noqa: PLC0415

        try:
            return mjcf_mesh(path, shape.body)
        except Exception as exc:  # noqa: BLE001 - a missing collision mesh or a bad file: not placeable, with why
            return f"MJCF {shape.path} does not load ({type(exc).__name__}: {str(exc)[:160]}); {fetch}"
    md = lib.mesh_for(card)
    return np.asarray(md.vertices, dtype=np.float64), np.asarray(md.faces, dtype=np.int64)


def _family(stem: str, token: str) -> bool:
    """A material-family stem names a word: exactly, or as its start for stems of 4+ letters ("alumin" ->
    "aluminium"); short stems ("pe", "pp") match whole words only, so "paper" is not a plastic."""
    return token == stem or (len(stem) >= 4 and token.startswith(stem))


#: words in card names that are not the object's noun: units, sources, materials
_NOT_NOUNS = frozenset({"mm", "cm", "ml", "oz", "kg", "al", "in", "ycb", "gso", "robocasa", "objaverse", "job",
                        "sc", "dc", "untasked", "fill", "disposable", "stub", "show", "rack",
                        *W.MATERIAL_WORDS, *(m for ms in W.MATERIAL_WORDS.values() for m in ms)})


_COLOURS = frozenset({"red", "green", "blue", "yellow", "orange", "purple", "pink", "black", "white", "grey", "gray",
                      "brown", "silver", "gold", "beige", "tan", "navy", "teal", "clear", "multicolor", "multicolour"})


def _head(name: str, source: str = "") -> set[str]:
    """The object's noun in a card name: the last word ("ycb_056_tennis_ball" -> ball, "robocasa_mug_tree_001" ->
    tree) unless it is a trailing colour ("gso_android_figure_orange" -> figure; "ycb_017_orange" stays orange), or
    also the first one for Alloy's own noun-first names ("spoon_wood_oxo" -> spoon)."""
    t = [w for w in W.tokens(name) if w not in _NOT_NOUNS and len(w) > 1]
    while len(t) > 1 and t[-1] in _COLOURS:
        t = t[:-1]
    if not t:
        return set()
    return {t[-1], t[0]} if source == "library" else {t[-1]}


__all__ = ["Catalog", "Entry", "IMPORTS_DIR", "Info", "Match", "PLACEABLE_ROUTES", "SHELL_ROUTES", "import_roots"]
