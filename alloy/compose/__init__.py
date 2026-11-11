# SPDX-License-Identifier: Apache-2.0
"""Scene builder: a sentence in, a physics-checked scene and N variations out (scene cards).

``compose("a mug, a spoon and three wooden cubes on a table", variations=4)`` picks objects from the card library
(Alloy's ``library/``, alloy-assets and every imported library under ``<alloy-assets>/imports/<source>/``),
places them with support, clearance and clutter rules, runs ``alloy.check`` after every placement and a full check
at the end, and fixes what the checks flag. Nothing heavy is imported until a function is called.

Example::

    >>> import alloy.compose
    >>> spec = alloy.compose.parse("three nuts in a tray on a cluttered table")
    >>> [(i.query, i.count, i.relation) for i in spec.items], spec.clutter
    ([('tray', 1, None), ('nut', 3, 'in')], 'cluttered')
    >>> result = alloy.compose.compose(spec, variations=2)                 # doctest: +SKIP
    >>> result.save("my_scenes")                                           # doctest: +SKIP

Command line (JSON on stdout): ``python -m alloy.compose "a mug and two cubes on a table" --variations 2 --out dir``.
"""

from __future__ import annotations

from typing import Any

from alloy.compose.parse import parse
from alloy.compose.spec import ComposeError, ItemSpec, SceneSpec

__all__ = ["ComposeError", "ItemSpec", "SceneSpec", "compose", "library", "load_scene", "parse", "search", "tilt"]


def compose(request: Any, **kwargs: Any) -> Any:
    """Build a checked scene and variations; see ``alloy.compose.build.compose`` for the arguments.

    Example: ``alloy.compose.compose("a mug on a table", variations=2, device="cpu").ok``."""
    from alloy.compose.build import compose as _compose  # noqa: PLC0415 - keeps `import alloy.compose` light

    return _compose(request, **kwargs)


def search(query: str, hints: tuple[str, ...] = (), k: int = 10, *, roots: tuple = (), placeable: bool = True
           ) -> list[dict[str, Any]]:
    """Library objects matching ``query`` as JSON rows (name, score, source, title, placeable, reason, size).

    ``placeable=True`` reads and measures each hit (slower, but says whether the builder can use it); imported
    objects whose files are not on disk are reported as such (search never downloads).
    Example: ``alloy.compose.search("mug", k=3)``."""
    from alloy.compose.catalog import Catalog  # noqa: PLC0415

    cat = Catalog(roots, fetch=False)
    rows = []
    for m in cat.search(query, hints, k=k):
        row = m.to_dict()
        if placeable:
            row.update({k2: v for k2, v in cat.info(m.name).to_dict().items() if k2 != "name"})
        rows.append(row)
    return rows


def library(roots: tuple = ()) -> Any:
    """A card library that also sees every imported library; see ``alloy.compose.load.library``."""
    from alloy.compose.load import library as _library  # noqa: PLC0415

    return _library(roots)


def load_scene(path: Any, **kwargs: Any) -> Any:
    """An editable ``alloy.Scene`` of a saved composed scene; see ``alloy.compose.load.load_scene``."""
    from alloy.compose.load import load_scene as _load  # noqa: PLC0415

    return _load(path, **kwargs)


def tilt(sim: Any, plan: dict[str, Any]) -> Any:
    """Start a tilting pour on a built scene: ``plan`` is the soft row with "vessel" (from a sentence such as "pour
    200 ml of water from a jar tilted at 40 deg/s into a mug"); see ``alloy.compose.soft.tilt``.

    Example: ``alloy.compose.tilt(sim, next(r for r in result.scenes[0].soft["items"] if "vessel" in r))``."""
    from alloy.compose.soft.tilt import tilt as _tilt  # noqa: PLC0415

    return _tilt(sim, plan)
