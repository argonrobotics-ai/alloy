# SPDX-License-Identifier: Apache-2.0
"""Turn what the caller passed (a YAML path, a library scene name, YAML or JSON text, or a ``SceneCard``) into a
card and the ``CardLibrary`` that resolves its objects.

A scene file inside ``<root>/scenes/`` adds ``<root>`` as a library root before the default ones, so an importer's
own folder of cards (``<root>/objects``, ``<root>/materials``) is found without configuration.

Example::

    >>> src = load_source("demo_peg_d12")          # a library scene name
    >>> src.label, len(src.card.objects)
    ('demo_peg_d12', 3)
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alloy.api.errors import SceneError
from alloy.cards import types as ct
from alloy.core.errors import CardError

log = logging.getLogger("alloy.check")

SceneLike = Any   # str | Path | SceneCard (str: a path, a library scene name, or YAML text)
IMPORTS_DIR = "imports"   # <alloy-assets>/imports/<source>/ holds each imported library (alloy.importers)


class SceneInputError(SceneError):
    """The scene argument could not be read: ``message`` says what was passed, ``fix`` what to pass instead. An
    ``alloy.SceneError`` (so also an ``alloy.AlloyAPIError`` and a ``ValueError``).

    Example::

        >>> try:
        ...     load_source(3)
        ... except SceneInputError as exc:
        ...     print(str(exc)[:27])
        check(scene=...) got a int
    """


@dataclass(frozen=True)
class Source:
    """A scene card ready to check: the card, the library that resolves it, a label and the file (if any).

    Example::

        >>> src = load_source("demo_peg_d12")
        >>> src.label, src.path.name
        ('demo_peg_d12', 'demo_peg_d12.yaml')
    """

    card: ct.SceneCard
    lib: Any            # CardLibrary
    label: str
    path: Path | None = None


def load_source(scene: SceneLike, library: Any = None) -> Source:
    """Read ``scene`` into a ``Source``; raises ``CardError`` for a malformed card, ``SceneInputError`` when the
    argument is neither a file, a library scene, YAML text nor a ``SceneCard``.

    Example::

        >>> load_source("library/scenes/demo_peg_d12.yaml").card.worlds
        16
    """
    if isinstance(scene, ct.SceneCard):
        return Source(scene, _library(library, None), "scene")
    if isinstance(scene, str) and (scene.lstrip().startswith("{") or "\n" in scene):
        from alloy.cards.loader import loads_card  # noqa: PLC0415 - YAML or JSON text (JSON is YAML)

        return Source(loads_card(scene, ct.SceneCard, "scene"), _library(library, None), "scene")
    if isinstance(scene, (str, Path)):
        return _from_path_or_name(str(scene), library)
    raise SceneInputError(f"check(scene=...) got a {type(scene).__name__}",
                          "pass a scene YAML path, a library scene name, YAML text or an alloy.cards.types.SceneCard",
                          field="scene")


def _from_path_or_name(text: str, library: Any) -> Source:
    path = Path(text).expanduser()
    if path.is_file():
        from alloy.cards.loader import load_scene  # noqa: PLC0415

        lib = _library(library, _card_root(path))
        return Source(load_scene(path), lib, path.stem, path.resolve())
    lib = _library(library, None)
    if path.suffix in (".yaml", ".yml") or "/" in text:
        raise SceneInputError(f"scene file {text!r} does not exist",
                              f"pass an existing .yaml path or a library scene name such as {_some_names(lib)}",
                              field="scene")
    if text not in lib.names("scenes"):
        close = difflib.get_close_matches(text, lib.names("scenes"), n=3)
        hint = f"did you mean {close}?" if close else f"library scenes include {_some_names(lib)}"
        raise SceneInputError(f"no scene file or library scene named {text!r}", hint, field="scene")
    return Source(lib.peek_scene(text), lib, text, lib.path_of("scenes", text))


def _card_root(path: Path) -> Path | None:
    """``<root>`` for a file at ``<root>/scenes/.../<name>.yaml`` (an importer's own card folder)."""
    for parent in path.resolve().parents:
        if parent.name == "scenes":
            return parent.parent
    return None


def default_roots() -> list[Path]:
    """The shipped library, then every imported library (``<alloy-assets>/imports/<source>/``, objects from YCB, GSO,
    RoboCasa, ...): ``alloy.cards.roots.default_roots`` once the card package has it, else the same list here.

    Example::

        >>> [p.name for p in default_roots()][:2]                     # doctest: +SKIP
        ['library', 'gymanything']
    """
    from alloy.cards import roots as R  # noqa: PLC0415
    from alloy.cards.library import default_root  # noqa: PLC0415

    shared = getattr(R, "default_roots", None)
    if callable(shared):
        return [Path(p) for p in shared()]
    base = R.assets_root()
    imports = base / IMPORTS_DIR if base is not None else None
    found = sorted(p for p in imports.iterdir() if (p / "objects").is_dir()) if imports and imports.is_dir() else []
    return [default_root(), *found]


def _library(library: Any, extra_root: Path | None) -> Any:
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.cards.roots import assets_root  # noqa: PLC0415

    if library is not None and not isinstance(library, (str, Path, list, tuple)):
        return library
    roots = [Path(r) for r in ([library] if isinstance(library, (str, Path)) else library or [])]
    defaults = default_roots()
    if extra_root is not None and all(extra_root.resolve() != d.resolve() for d in defaults):
        roots.append(extra_root)
    log.debug("check: card library roots %s", [str(r) for r in [*roots, *defaults]])
    return CardLibrary([*roots, *defaults], assets=assets_root())


def _some_names(lib: Any, n: int = 3) -> list[str]:
    return [s for s in lib.names("scenes") if s.startswith("demo_")][:n] or list(lib.names("scenes"))[:n]


def card_error_fields(exc: CardError) -> tuple[str, str, str]:
    """(card, field, message) of a ``CardError`` (its attributes, else its text).

    Example::

        >>> from alloy.core.errors import CardError
        >>> card_error_fields(CardError("mug", "mass", "must be > 0"))
        ('mug', 'mass', 'must be > 0')
    """
    card = str(getattr(exc, "card", "") or "scene")
    fld = str(getattr(exc, "field", "") or "")
    msg = str(getattr(exc, "message", "") or exc)
    return card, fld, msg


__all__ = ["SceneInputError", "Source", "card_error_fields", "default_roots", "load_source"]
