# SPDX-License-Identifier: Apache-2.0
"""Use a saved composed scene: as an editable ``alloy.Scene``, or check it, with the imported libraries visible.

Scenes built from imported objects (``<alloy-assets>/imports/<source>/``) need those card roots; these helpers add
every import root, so a saved folder loads and checks with no extra arguments. ``Scene.load`` turns a card's pour
emitters into ``Scene.add_pour`` streams named ``<material>_pour_<k>``, so ``alloy.build`` pours them.

Example::

    >>> scene = load_scene("my_scenes/scenes/scene.yaml")        # doctest: +SKIP
    >>> sim = alloy.api.build(scene, worlds=64)                       # doctest: +SKIP
    >>> alloy.check("my_scenes/scenes/scene.yaml", library=library(["my_scenes"])).ok    # doctest: +SKIP
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from alloy.compose.catalog import import_roots


def library(roots: Sequence[str | Path] = ()):
    """A ``CardLibrary`` over ``roots`` (searched first), Alloy's library, the fetch cache of each import library,
    every import root and alloy-assets.

    Example: ``alloy.check(card, library=library(["my_scenes"]))``."""
    from alloy.cards.library import CardLibrary, default_root  # noqa: PLC0415
    from alloy.cards.roots import assets_root  # noqa: PLC0415
    from alloy.compose.fetch import cache_root  # noqa: PLC0415

    extra = [Path(r).expanduser().resolve() for r in roots]
    imports = import_roots()
    return CardLibrary(roots=[*extra, default_root(), *(cache_root(r.name) for r in imports), *imports],
                       assets=assets_root())


def load_scene(path: str | Path, *, worlds: int | None = None):
    """An editable ``alloy.Scene`` of a saved scene card, with the import roots added to ``scene.roots``.

    Example: ``alloy.api.build(load_scene("out/scenes/scene.yaml"), worlds=16)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415

    from alloy.compose.fetch import cache_root  # noqa: PLC0415

    fetch_sources(Path(path).resolve().parent.parent)
    scene = Scene.load(str(path), worlds=worlds)
    imports = import_roots()
    extra = [str(cache_root(r.name)) for r in imports] + [str(r) for r in imports]
    scene.roots = [*scene.roots, *(r for r in extra if r not in scene.roots)]
    return scene


def fetch_sources(folder: str | Path) -> list[str]:
    """Make sure the library objects that the generated copies in a saved ``folder`` were made from are on this
    machine, and return their names. The builder saves a rigid copy of a soft object (``<card>_rigid``) and a held
    copy for a tilting pour (``<card>_still``) under ``objects/``; a copy keeps the object's mesh path and material,
    which a build on another machine fetches for an object the scene names, not for a copy of it.

    Example: ``fetch_sources("my_scenes") == ["ycb_013_apple"]`` for "a bowl and three apples on a table"."""
    from alloy.compose.soft.cards import copied_from  # noqa: PLC0415

    cards = sorted((Path(folder) / "objects").glob("*.yaml"))
    names = list(dict.fromkeys(n for n in (copied_from(p.read_text(encoding="utf-8")) for p in cards) if n))
    if names:
        from alloy.api._build import fetch_imported  # noqa: PLC0415

        fetch_imported(library([folder]), names)
    return names


__all__ = ["fetch_sources", "library", "load_scene"]
