# SPDX-License-Identifier: Apache-2.0
"""Public entry points: export scene and object cards to MJCF, URDF and USD.

Every call returns a result with the files written, what each simulator loaded, and the list of values the format
could not carry (``unmapped``). ``to_json()`` gives the same as plain data for agents and the command line.

Example::

    from alloy.io import export_scene, export_object
    res = export_scene("demo_peg_d12", "out/peg")            # a library scene name, a SceneCard or a .yaml path
    print(res.files["mjcf"], [u.item for u in res.unmapped])
    export_object("cube_beech_30", "out/cube", formats=("urdf",))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from alloy.cards import types as ct
from alloy.cards.library import CardLibrary
from alloy.io.model import IOModel, Unmapped
from alloy.io.quiet import quiet_engine

log = logging.getLogger("alloy.io")
FORMATS: tuple[str, ...] = ("mjcf", "urdf", "usd")


@dataclass
class ExportResult:
    """Files written per format (URDF: its ``index.json``), MuJoCo's load counts, and what did not map."""

    name: str
    files: dict[str, str] = field(default_factory=dict)
    counts: dict[str, dict] = field(default_factory=dict)
    unmapped: list[Unmapped] = field(default_factory=list)

    def to_json(self) -> dict:
        """Plain data: ``{"name", "files", "counts", "unmapped": [{"format", "item", "reason"}]}``."""
        return {"name": self.name, "files": dict(self.files), "counts": dict(self.counts),
                "unmapped": [u.as_dict() for u in self.unmapped]}


def _check_formats(formats) -> tuple[str, ...]:
    fmts = (formats,) if isinstance(formats, str) else tuple(formats)
    bad = [f for f in fmts if f not in FORMATS]
    if bad or not fmts:
        raise ValueError(f"unknown export format(s) {bad or fmts}; use one or more of {', '.join(FORMATS)}")
    return fmts


def _library(library: CardLibrary | str | Path | None) -> CardLibrary:
    """The default library, a library folder searched before the default one, or any library object as given."""
    if library is None:
        return CardLibrary()
    if not isinstance(library, (str, Path)):
        return library
    from alloy.cards.library import default_root  # noqa: PLC0415

    return CardLibrary([Path(library), default_root()])


def _with_cards(lib: CardLibrary, cards: list, folder: Path) -> CardLibrary:
    """``lib`` plus in-memory cards, written as YAML into ``folder`` (searched first)."""
    from alloy.cards.loader import dump_card  # noqa: PLC0415

    kinds = {ct.ObjectCard: "objects", ct.MaterialCard: "materials"}
    for c in cards:
        d = folder / kinds[type(c)]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{c.name}.yaml").write_text(dump_card(c))
    return CardLibrary([folder, *lib.roots])


def card_root(path: Path) -> Path:
    """The card folder of a scene file: the parent of its ``scenes/`` folder (``alloy.api.load``'s rule)."""
    return path.parent.parent if path.parent.name == "scenes" else path.parent


def _with_root(lib: CardLibrary, root: Path) -> CardLibrary:
    """``lib`` with ``root`` searched first (unchanged if it is already a root or ``lib`` has no roots)."""
    roots = list(getattr(lib, "roots", None) or [])
    if not roots or any(r.resolve() == root.resolve() for r in roots):
        return lib
    return CardLibrary([root, *roots])


def _scene_card(scene: ct.SceneCard | str | Path, lib: CardLibrary) -> tuple[ct.SceneCard, str, CardLibrary]:
    """The scene card, its default name and the library to build it with (a ``.yaml`` file's own card folder is
    searched first, so scenes saved by ``alloy.api.Scene.save`` find their objects)."""
    if isinstance(scene, ct.SceneCard):
        return scene, "scene", lib
    if not isinstance(scene, (str, Path)):
        raise TypeError(f"export_scene takes a library scene name, a scene .yaml path or a SceneCard, not "
                        f"{type(scene).__name__}; for an alloy.Scene pass scene.save(<folder>)")
    p = Path(scene)
    if p.suffix in (".yaml", ".yml"):
        from alloy.cards.loader import load_scene  # noqa: PLC0415

        if not p.exists():
            raise FileNotFoundError(f"scene card {p} not found; pass a library scene name or an existing .yaml")
        return load_scene(p), p.stem, _with_root(lib, card_root(p))
    names = lib.names("scenes")
    if str(scene) not in names:
        raise KeyError(f"no library scene {scene!r}; pass a SceneCard, a .yaml path or one of {sorted(names)[:12]}...")
    return lib.scene(str(scene)), str(scene), lib


def _write(io: IOModel, built, out: Path, fmts: tuple[str, ...], res: ExportResult) -> None:
    from alloy.io.mjcf_export import write_mjcf  # noqa: PLC0415
    from alloy.io.urdf_write import write_urdf  # noqa: PLC0415
    from alloy.io.usd_write import write_usd  # noqa: PLC0415

    stem = out / io.name
    if "mjcf" in fmts:
        res.counts["mjcf"] = write_mjcf(built, io, stem.with_suffix(".xml"))
        res.files["mjcf"] = str(stem.with_suffix(".xml"))
    if "urdf" in fmts:
        index = write_urdf(io, out / "urdf")
        res.files["urdf"] = str(out / "urdf" / "index.json")
        res.counts["urdf"] = {"files": len(index["groups"])}
    if "usd" in fmts:
        res.files["usd"] = str(write_usd(io, stem.with_suffix(".usda")))


def export_scene(scene: ct.SceneCard | str | Path, out_dir: str | Path, formats=FORMATS, *,
                 library: CardLibrary | str | Path | None = None, name: str | None = None) -> ExportResult:
    """Export a scene (library name, ``.yaml`` path, ``SceneCard`` or an ``alloy.Scene``) to ``out_dir``.

    Writes ``<name>.xml`` (MJCF, checked by loading it in MuJoCo), ``urdf/<object>.urdf`` + ``urdf/index.json``
    and ``<name>.usda``. Only world 0 at nominal values is exported; deformables, liquids and Alloy-only contact
    features are listed in ``unmapped``. Raises ``alloy.io.mjcf_export.ExportError`` with the fix to try.
    """
    from alloy.io.build import scene_model  # noqa: PLC0415 - heavy import

    fmts = _check_formats(formats)
    if callable(getattr(scene, "save", None)) and not isinstance(scene, (str, Path, ct.SceneCard)):
        import tempfile  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:       # an alloy.Scene: export from its saved cards
            return export_scene(scene.save(Path(tmp) / "cards"), out_dir, fmts, library=library, name=name)
    card, default_name, lib = _scene_card(scene, _library(library))
    with quiet_engine():
        io, built = scene_model(card, lib, name or default_name)
        res = ExportResult(io.name)
        _write(io, built, Path(out_dir), fmts, res)
    res.unmapped = [u for u in io.unmapped if u.format in ("all", "model", *fmts)]
    log.info("exported %s to %s: %s", io.name, out_dir, res.files)
    return res


def export_object(obj: ct.ObjectCard | str, out_dir: str | Path, formats=FORMATS, *,
                  library: CardLibrary | str | Path | None = None,
                  materials: tuple[ct.MaterialCard, ...] = ()) -> ExportResult:
    """Export one object card at the origin, as :func:`export_scene` does. ``obj`` is a library name or an
    ``ObjectCard``; a card (and ``materials``) not in the library are used for this call only."""
    import tempfile  # noqa: PLC0415

    from alloy.io.build import object_scene  # noqa: PLC0415

    lib = _library(library)
    card = obj if isinstance(obj, ct.ObjectCard) else lib.object(str(obj))
    if not isinstance(obj, ct.ObjectCard) and not materials:
        return export_scene(object_scene(card), out_dir, formats, library=lib, name=card.name)
    with tempfile.TemporaryDirectory() as tmp:
        lib2 = _with_cards(lib, [card, *materials], Path(tmp))
        return export_scene(object_scene(card), out_dir, formats, library=lib2, name=card.name)
