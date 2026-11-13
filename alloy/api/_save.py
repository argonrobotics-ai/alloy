# SPDX-License-Identifier: Apache-2.0
"""``scene.save(directory)``: the scene's cards in a folder that loads on its own (``alloy.load`` of its
``scenes/scene.yaml``, also after the folder is moved or copied).

``_materialize`` writes the cards the scene makes itself (its shapes, per-object materials, renamed library objects).
A card the scene takes from one of its own card roots (``Scene(roots=[...])``: an ``alloy.create`` or
``alloy.io.import_file`` folder, or a folder saved before) is in none of the roots a saved folder loads with (that
folder, then the installed library), so ``save`` copies what the saved cards name from those roots, at the same paths
relative to the folder: object cards, their material cards (and the materials those take pair cards from), the pair
cards between the scene's materials, their look files, and the files the object cards name (meshes), each with the
files beside it that share its stem (decomposed hulls ``<stem>.hulls.obj``) and a ``visual.glb`` (a scan's textured
look); a robot file named relative to an own root (``robots/arm.xml`` of a scene loaded from a saved folder) is
copied too, with its whole folder when it has one of its own (``robots/<robot>/``). Cards of the shipped library,
the imported libraries and the Hub are not copied: every install has them (the Hub's on first use). The cards are
written to a fresh folder first and then copied over ``directory``, so a file left there by an earlier save never
stands in for one of these.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("alloy.api")

ASSET_KEYS = ("path", "from_mesh")      # shape source fields that name a file (mesh, sdf, tet, usd, mjcf)
LOOK_BESIDE = "visual.glb"              # a scan's textured look, read from beside its physics mesh


def save(scene: Any, directory: str | Path) -> Path:
    """Write ``scene``'s cards to ``directory`` (``Scene.save``) and return its scene file; warns for what the cards
    cannot keep. Example: ``save(scene, "my_scene")`` -> ``Path("my_scene/scenes/scene.yaml")``."""
    from alloy.api import _materialize, _pour, _unsaved  # noqa: PLC0415 - imports the card layer

    _unsaved.warn(scene)                            # what the cards do not keep: one warning per kind
    dest = Path(directory)
    with tempfile.TemporaryDirectory(prefix="alloy-save-") as tmp:
        mat = _materialize.materialize(scene, Path(tmp))
        _unsaved.anchored(mat.scene)                # placed by an anchor: the cards hold the body origin
        copied = copy_own_cards(scene.roots, mat.root)
        scene_file = _pour.save_as_emitters(scene, mat)  # a pour: a card emitter
        shutil.copytree(mat.root, dest, dirs_exist_ok=True)
    if copied:
        log.info("scene.save(): copied %d file(s) from the scene's own card roots into %s", len(copied), dest)
    return dest / scene_file.relative_to(mat.root)


def own_roots(roots: Sequence[str | Path]) -> list[Path]:
    """The scene's own card roots: ``roots`` that exist, less the ones every install has (the shipped library, the
    imported libraries, their fetch caches, alloy-assets). Example: ``own_roots(["my_objects"])`` ->
    ``[Path("my_objects")]``."""
    from alloy.api import catalog  # noqa: PLC0415

    shared = {p.resolve() for p in catalog.search_roots()}
    paths = (Path(r).expanduser() for r in roots)
    return [p for p in paths if p.is_dir() and p.resolve() not in shared]


def copy_own_cards(roots: Sequence[str | Path], root: Path) -> list[str]:
    """Copy into the card folder ``root`` what its scene card names from the scene's own card roots among ``roots``
    (module docstring); returns the paths copied, relative to ``root``.

    Example: ``copy_own_cards(["my_objects"], Path(tmp))`` -> ``["objects/puck.yaml", "materials/puck_eff.yaml",
    "meshes/puck/physics.obj", ...]``."""
    from alloy.api._materialize import SCENE_NAME, _load_yaml  # noqa: PLC0415

    walk = _Walk([Path(r).expanduser() for r in roots], own_roots(roots), root)
    if not walk.own:
        return []
    card = _load_yaml(root / "scenes" / f"{SCENE_NAME}.yaml")
    for placement in card.get("objects") or []:
        walk.object(str(placement["object"]))
    for pair in card.get("pairs") or []:
        walk.material(pair.get("a"))
        walk.material(pair.get("b"))
    for emitter in card.get("emitters") or []:
        walk.material(emitter.get("material"))
    for robot in card.get("robots") or []:
        walk.asset(str(robot["path"]), folder=True)        # a robot of a scene loaded from a saved folder
    walk.pair_cards()
    return walk.copied


@dataclass
class _Walk:
    """The cards and files a saved folder needs, found where a build of the scene finds them: in the folder (what
    this save wrote), else in the first of the scene's roots that has them; copied when that root is its own."""

    roots: list[Path]
    own: list[Path]
    root: Path
    copied: list[str] = field(default_factory=list)
    used: set[str] = field(default_factory=set)              # every material name the saved cards use
    _seen: set[tuple[str, str]] = field(default_factory=set)
    _index: dict[tuple[Path, str], dict[str, Path]] = field(default_factory=dict)

    def object(self, name: str) -> None:
        """An object card, then its materials, files and look."""
        data = self._card("objects", name)
        if data is None:
            return
        self.material(data.get("material"))
        for part in (data.get("part_materials") or {}).values():
            self.material(part)
        for rel in _asset_paths(data.get("shape")):
            self.asset(rel)
        self._look("objects", name)

    def material(self, name: Any) -> None:
        """A material card, the material it takes pair cards from, and its look."""
        if not name:
            return
        self.used.add(str(name))
        data = self._card("materials", str(name))
        if data is not None:
            self.material(data.get("pairs_from"))
            self._look("materials", str(name))

    def asset(self, rel: str, folder: bool = False) -> None:
        """A file a card names, relative to a card root, with the files beside it that belong to it; ``folder``: a
        robot file in a folder of its own (``robots/<robot>/``) brings that whole folder (its meshes, includes)."""
        if Path(rel).is_absolute() or (self.root / rel).exists() or not self._first(("asset", rel)):
            return
        src, base = self._winner(lambda r: r / rel if (r / rel).is_file() else None)
        if src is None:
            return
        if folder and len(Path(rel).parts) > 2:
            files = [p for p in src.parent.rglob("*") if p.is_file()]
        else:
            files = [p for p in {src, *src.parent.glob(f"{src.stem}.*"), src.parent / LOOK_BESIDE} if p.is_file()]
        for path in sorted(files):
            self._copy(path, base)

    def pair_cards(self) -> None:
        """The own roots' pair cards between two materials the saved cards use (the first root that has one wins)."""
        from alloy.api._materialize import _load_yaml  # noqa: PLC0415

        for r in self.own:
            for stem, path in self._names(r, "pairs").items():
                data = _load_yaml(path)
                both_used = {str(data.get("a")), str(data.get("b"))} <= self.used
                if both_used and self._first(("pairs", stem)) and not (self.root / "pairs" / f"{stem}.yaml").exists():
                    self._copy(path, r)

    def _card(self, kind: str, name: str) -> dict[str, Any] | None:
        """The card's data (copied into the folder first when an own root has it), or None (the library's)."""
        from alloy.api._materialize import _load_yaml  # noqa: PLC0415

        if not self._first((kind, name)):
            return None
        written = self.root / kind / f"{name}.yaml"
        if written.is_file():
            return _load_yaml(written)
        src, base = self._winner(lambda r: self._names(r, kind).get(name))
        if src is None:
            return None
        self._copy(src, base)
        return _load_yaml(src)

    def _look(self, kind: str, name: str) -> None:
        """The look file ``looks/<kind>/**/<name>.json`` of an own root (a look the folder has already wins)."""
        if _look_file(self.root, kind, name) is None:
            src, base = self._winner(lambda r: _look_file(r, kind, name))
            if src is not None:
                self._copy(src, base)

    def _winner(self, find: Callable[[Path], Path | None]) -> tuple[Path | None, Path]:
        """(file, its root) from the first of the scene's roots where ``find(root)`` hits, when that root is one of
        its own; (None, root) when it is another root or no root hits (the library's file: not copied)."""
        own = {p.resolve() for p in self.own}
        for r in self.roots:
            hit = find(r) if r.is_dir() else None
            if hit is not None:
                return (hit, r) if r.resolve() in own else (None, r)
        return None, self.root

    def _names(self, r: Path, kind: str) -> dict[str, Path]:
        """{stem: file} of root ``r``'s ``<kind>/**/*.yaml`` cards, read once per save (hidden folders skipped)."""
        if (r, kind) not in self._index:
            base = r / kind
            files = sorted(base.rglob("*.yaml")) if base.is_dir() else []
            index: dict[str, Path] = {}
            for p in files:
                if not any(part.startswith(".") for part in p.relative_to(base).parts):
                    index.setdefault(p.stem, p)
            self._index[(r, kind)] = index
        return self._index[(r, kind)]

    def _first(self, key: tuple[str, str]) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def _copy(self, src: Path, base: Path) -> None:
        """Copy ``src`` (a file under root ``base``) to the same path under the saved folder."""
        rel = src.relative_to(base)
        if ".." in rel.parts:                                 # a "../" path: outside its root, left where it is
            log.warning("scene.save(): %s is outside its card root %s and is not copied; the saved folder still "
                        "names it there", src, base)
            return
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        self.copied.append(rel.as_posix())


def _look_file(r: Path, kind: str, name: str) -> Path | None:
    """Root ``r``'s look file ``looks/<kind>/**/<name>.json``, or None."""
    d = r / "looks" / kind
    return next(iter(sorted(d.rglob(f"{name}.json"))), None) if d.is_dir() else None


def _asset_paths(shape: Any) -> list[str]:
    """The file paths an object card's shape names (``mesh.path``, ``sdf.path``, ``tet.path`` / ``tet.from_mesh``,
    ``usd.path``, ``mjcf.path``)."""
    if not isinstance(shape, dict):
        return []
    return [str(src[k]) for src in shape.values() if isinstance(src, dict)
            for k in ASSET_KEYS if isinstance(src.get(k), str) and src.get(k)]
