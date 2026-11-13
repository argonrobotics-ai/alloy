# SPDX-License-Identifier: Apache-2.0
"""What the card library holds: materials, objects and scenes by name (fast: reads YAML, imports no engine).

The library is the shipped ``library/`` plus every imported library (``<alloy-assets>/imports/<source>/``: thousands
of scanned and catalogued objects; each card records where its mass and friction come from, and for most that is an
estimate from category and mesh volume, its note saying ESTIMATE) and alloy-assets itself. An imported object's
meshes are fetched from ``$ALLOY_ASSET_STORE`` the first time a scene that uses it is built. On a fresh install
(no imported library, no store) objects come from the object library on the Hugging Face Hub: ``alloy fetch <name>``
puts them in the fetch cache (``$ALLOY_CACHE``), where they work like library objects, and a build fetches any it
still lacks.

Example::

    import alloy
    print([m["name"] for m in alloy.materials()][:5])     # ['aa_cell_effective', 'abs', ...]
    print(alloy.scenes())                                   # ['demo_cloth_drop', 'demo_nut_on_bolt', ...]
"""

from __future__ import annotations

import difflib
import functools
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

KINDS = ("materials", "pairs", "objects", "scenes")


def _package_data() -> Path:
    """``alloy/_data``: the library and the import cards an installed wheel carries."""
    return Path(__file__).resolve().parents[1] / "_data"


def library_root() -> Path:
    """The shipped card library, ``alloy.cards.library.default_root()`` (``$ALLOY_LIBRARY``, else ``<repo>/library``
    in a checkout, else the wheel's ``alloy/_data/library``). Example: ``(library_root() / "materials").is_dir()``."""
    from alloy.cards.library import default_root  # noqa: PLC0415 - the card layer, only when a root is needed

    return default_root()


IMPORTS_DIR = "imports"


def assets_dir() -> Path | None:
    """The alloy-assets checkout, by the rules of ``alloy.cards.roots.assets_root``: ``$ALLOY_ASSETS`` (empty: none),
    else ``<repo>/../alloy-assets`` if it exists. Example: ``assets_dir()``."""
    raw = os.environ.get("ALLOY_ASSETS")
    if raw is not None:
        path = Path(raw).expanduser()
        return path.resolve() if raw.strip() and path.is_dir() else None
    sibling = Path(__file__).resolve().parents[2].parent / "alloy-assets"
    return sibling.resolve() if sibling.is_dir() else None


def import_roots() -> list[Path]:
    """A fast mirror of ``alloy.cards.roots.import_roots`` (no engine import; tests keep the two equal). Imported
    libraries that hold object cards (YCB + Google Scanned Objects, RoboCasa, Objaverse, Gym-Anything,
    ...), sorted: ``<alloy-assets>/imports/<source>/`` in a development checkout, else the cards an installed wheel
    carries in ``alloy/_data/imports/<source>/`` (meshes come from ``$ALLOY_ASSET_STORE`` on first use).

    Example: ``[p.name for p in import_roots()]``."""
    base = assets_dir()
    for parent in ([base / IMPORTS_DIR] if base is not None else []) + [_package_data() / IMPORTS_DIR]:
        if parent.is_dir():
            found = sorted(p for p in parent.iterdir() if (p / "objects").is_dir())
            if found:
                return found
    return []


def card_roots(extra: Sequence[str | Path] = ()) -> list[Path]:
    """Library roots for building a scene: ``extra``, then the engine's default roots (``alloy.cards.roots.
    default_roots``: the shipped library, each imported library's fetch cache, the imported libraries).

    Example: ``CardLibrary(roots=card_roots([workdir]), assets=assets_root())``."""
    from alloy.cards.roots import default_roots  # noqa: PLC0415 - the card layer (imports the engine's cards)

    return [*(Path(r) for r in extra), *default_roots()]


def cache_base() -> Path:
    """Where fetched objects go: ``$ALLOY_CACHE``, else ``${XDG_CACHE_HOME:-~/.cache}/alloy`` (a fast mirror of
    ``alloy.compose.fetch.cache_base``). Example: ``cache_base() / "imports"``."""
    raw = os.environ.get("ALLOY_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "alloy"


def fetched_roots(imports: Sequence[Path] | None = None) -> list[Path]:
    """Every fetch cache, as ``alloy.cards.roots.fetched_roots`` lists it (tests keep the two equal): each imported
    library's cache, then every library fetched into the cache (``alloy fetch`` from the object library on the Hugging
    Face Hub, with no import root on a fresh install), duplicates removed. Example: ``fetched_roots()``."""
    base = cache_base() / IMPORTS_DIR
    on_disk = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    mine = import_roots() if imports is None else imports
    return list(dict.fromkeys([*(base / r.name for r in mine), *on_disk]))


def search_roots(extra: Sequence[str | Path] = ()) -> list[Path]:
    """Library roots in lookup order: ``extra``, then the engine's default roots (the shipped library, the fetch
    caches, the imported libraries), then alloy-assets. Example: ``search_roots(["my_cards"])``."""
    assets, imports = assets_dir(), import_roots()
    return ([Path(r) for r in extra] + [library_root(), *fetched_roots(imports), *imports]
            + ([assets] if assets else []))


def refresh() -> None:
    """Forget the card names read so far (a fetch added cards to a root already read). Example: ``refresh()``."""
    _names.cache_clear()


def hub_names() -> list[str]:
    """On a fresh install (no imported library, no asset store; ``alloy.assets.hub.lazy.hub_mode``): every object in
    the Hub's object library, which a build fetches on first use; else []. Example: ``"ycb_025_mug" in hub_names()``."""
    from alloy.assets.hub.lazy import catalog_rows, hub_mode  # noqa: PLC0415 - only when a name is not on disk

    return sorted(r["name"] for r in catalog_rows()) if hub_mode() else []


def object_card(name: str, extra: Sequence[str | Path] = ()) -> Path:
    """The card file of object ``name``: on disk, else fetched from the Hub on a fresh install (``alloy fetch``).

    Example: ``object_card("ycb_025_mug")``."""
    path = card_path("objects", name, extra)
    if path is None:
        refresh()
        path = card_path("objects", name, extra)
    if path is not None:
        return path
    from alloy.api.errors import SceneError  # noqa: PLC0415

    if name not in hub_names():
        raise missing_object(name, extra)
    from alloy.assets.hub.fetch import fetch  # noqa: PLC0415
    from alloy.assets.hub.lazy import fetch_fix  # noqa: PLC0415
    from alloy.assets.hub.source import HubError  # noqa: PLC0415

    try:
        fetch(names=[name])
    except HubError as exc:
        raise SceneError(f"object card {name!r}: {exc.message}", fetch_fix(name, exc), field="card") from exc
    refresh()
    path = card_path("objects", name, extra)
    if path is None:
        raise SceneError(f"object card {name!r} was fetched but is not in the cache {cache_base()}",
                         f"run `alloy fetch {name}` and check $ALLOY_CACHE", field="card")
    return path


def missing_object(name: str, extra: Sequence[str | Path] = ()) -> Exception:
    """The ``SceneError`` for object card ``name`` that is not on disk. On a fresh install without the ``import``
    extra, the Hub's object library can be neither listed nor fetched, so the error says that first, with the fetch
    command, and the close names after it. Example: ``raise missing_object("ycb_025_mug")``."""
    from alloy.api.errors import SceneError  # noqa: PLC0415

    close = (f"use one of {near('objects', name, extra)}, or pass roots=[<folder>] to alloy.Scene if the card is in "
             "your own folder")
    if not _hub_unreachable():
        return SceneError(f"no object card {name!r}", close, field="card")
    return SceneError(f"object card {name!r} is not on disk. Library objects that do not ship with Alloy (about "
                      "5,400: YCB, Google Scanned Objects, RoboCasa, ...) are in the object library on the Hugging "
                      "Face Hub, and fetching one needs the import extra (huggingface_hub is not installed)",
                      f"pip install 'alloy-engine[import]', then run `alloy fetch "
                      f"{name}`; a build then also fetches the library objects you name. If you meant a card on "
                      f"disk: {close}", field="card")


def _hub_unreachable() -> bool:
    """A fresh install (objects come from the Hub) without huggingface_hub, the ``import`` extra."""
    import importlib.util  # noqa: PLC0415

    from alloy.assets.hub.lazy import hub_mode  # noqa: PLC0415

    return hub_mode() and importlib.util.find_spec("huggingface_hub") is None


def import_root_of(path: Path) -> Path | None:
    """The imported library a card file belongs to, or None. Example: ``import_root_of(card_path("objects", n))``."""
    for root in import_roots():
        if Path(path).resolve().is_relative_to(root.resolve()):
            return root
    return None


@functools.lru_cache(maxsize=64)
def _names(root: str, kind: str) -> dict[str, str]:
    base = Path(root) / kind
    if not base.is_dir():
        return {}
    return {p.stem: str(p) for p in sorted(base.rglob("*.yaml"))
            if not any(part.startswith(".") for part in p.relative_to(base).parts)}


def names(kind: str, extra: Sequence[str | Path] = ()) -> list[str]:
    """Card names of one kind ("materials", "pairs", "objects", "scenes"). Example: ``names("scenes")``."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    out: set[str] = set()
    for root in search_roots(extra):
        out.update(_names(str(root), kind))
    return sorted(out)


def card_path(kind: str, name: str, extra: Sequence[str | Path] = ()) -> Path | None:
    """The file defining a card (first root wins), or None. Example: ``card_path("materials", "abs")``."""
    for root in search_roots(extra):
        hit = _names(str(root), kind).get(name)
        if hit is not None:
            return Path(hit)
    return None


def has_card(kind: str, name: str, extra: Sequence[str | Path] = ()) -> bool:
    """True if a card of that kind and name exists on disk (an object is looked up again on a miss: a fetch since
    the last lookup may have added it). Example: ``has_card("materials", "water")``."""
    if card_path(kind, name, extra) is not None:
        return True
    if kind != "objects":
        return False
    refresh()
    return card_path(kind, name, extra) is not None


def can_place(name: str, extra: Sequence[str | Path] = ()) -> bool:
    """True if ``Scene.add_object`` can place object card ``name``: on disk, or in the Hub's object library on a
    fresh install (fetched at build). Example: ``can_place("ycb_025_mug")``."""
    return has_card("objects", name, extra) or name in hub_names()


def near(kind: str, name: str, extra: Sequence[str | Path] = (), n: int = 6) -> list[str]:
    """Names close to ``name`` (for "did you mean" fixes; objects include the Hub's on a fresh install).

    Example: ``near("materials", "steel")``."""
    _no_library(kind, name)
    every = names(kind, extra)
    if kind == "objects":
        every = sorted({*every, *hub_names()})
    close = difflib.get_close_matches(name, every, n=n, cutoff=0.3)
    starts = [x for x in every if name and (x.startswith(name) or name in x)]
    return list(dict.fromkeys(starts + close))[:n] or every[:n]


def _no_library(kind: str, name: str) -> None:
    """Raise ``SceneError`` when the default card library is missing, so the error names where it was searched and
    the fix instead of "use one of []" (SCR-OSS2-2: a plain ``pip install .`` from a clone, the first Mac test)."""
    from alloy.api.errors import SceneError  # noqa: PLC0415
    from alloy.cards.library import LIBRARY_FIX, library_missing  # noqa: PLC0415

    missing = library_missing()
    if missing is not None:
        raise SceneError(f"no {kind[:-1]} {name!r}: {missing}", LIBRARY_FIX,
                         field="material" if kind == "materials" else "card")


def _value(node: Any) -> float | None:
    if isinstance(node, dict):
        node = node.get("value")
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def _material_row(name: str, path: str) -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    surface = data.get("surface") or {}
    kind = "fluid" if data.get("fluid") else "granular" if (data.get("plastic") or {}).get("friction_angle") \
        else "elastic" if data.get("elastic") else "rigid"
    return {"name": name, "kind": kind, "density": _value(data.get("density")),
            "friction": _value(surface.get("mu_static")), "restitution": _value(surface.get("restitution"))}


def materials(extra: Sequence[str | Path] = (), *, imports: bool = False) -> list[dict[str, Any]]:
    """Every shipped material with its kind (rigid / elastic / fluid / granular), density (kg/m^3), friction and
    restitution; ``imports=True`` adds the imported libraries' per-object materials (about 1,600).

    Example: ``[m["name"] for m in alloy.materials() if m["kind"] == "fluid"]`` -> ``['honey', 'olive_oil', 'water']``."""
    roots = [*(Path(r) for r in extra), library_root()]
    found: dict[str, str] = {}
    for root in (search_roots(extra) if imports else roots):
        for name, path in _names(str(root), "materials").items():
            found.setdefault(name, path)
    return [_material_row(n, found[n]) for n in sorted(found)]


def scenes(extra: Sequence[str | Path] = ()) -> list[str]:
    """Scene names in the library (``alloy.load(name)`` opens one). Example: ``alloy.scenes()``."""
    return names("scenes", extra)


def objects(extra: Sequence[str | Path] = ()) -> list[str]:
    """Object card names on disk (``Scene.add_object(name, card)`` places one): the 118 that ship with Alloy, plus
    any you fetched or imported. ``alloy fetch --list`` shows the about 5,400 in the object library on the Hugging
    Face Hub (with the ``import`` extra), which ``add_object`` also takes by name. Example: ``alloy.objects()[:3]``."""
    return names("objects", extra)
