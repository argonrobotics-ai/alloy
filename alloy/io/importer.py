# SPDX-License-Identifier: Apache-2.0
"""Read MJCF, URDF and USD files into an ``IOModel`` and write them out as Alloy cards.

MJCF and USD are read by Newton's own importers (the same ones Alloy uses for robots), URDF by
``alloy.io.urdf_read`` (no extra dependencies). Reading USD needs ``usd-core`` and ``newton-usd-schemas`` (the
``usd`` extra); the error names the command.

Example::

    from alloy.io import import_file
    res = import_file("robot_scene.xml", "my_library")       # cards in my_library/{objects,materials,scenes}
    print(res.objects, res.robots, res.scene)

Then ``CardLibrary(["my_library", default_root()]).scene(res.name)`` assembles it in Alloy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from alloy.io.errors import ImportFileError, usd_fix
from alloy.io.from_model import read_model
from alloy.io.model import IOModel
from alloy.io.quiet import quiet_engine
from alloy.io.to_cards import ImportResult, write_cards

log = logging.getLogger("alloy.io")
SUFFIXES = {".xml": "mjcf", ".mjcf": "mjcf", ".urdf": "urdf", ".usd": "usd", ".usda": "usd", ".usdc": "usd",
            ".usdz": "usd"}


def file_format(path: str | Path) -> str:
    """"mjcf", "urdf" or "usd" from the suffix (``.xml`` files are MJCF unless their root is ``<robot>``)."""
    p = Path(path)
    fmt = SUFFIXES.get(p.suffix.lower())
    if fmt is None:
        raise ImportFileError(f"{p.name}: unknown suffix {p.suffix!r}; use one of {sorted(SUFFIXES)}")
    if fmt == "mjcf" and p.suffix.lower() == ".xml" and "<robot" in p.read_text(errors="replace")[:4000] \
            and "<mujoco" not in p.read_text(errors="replace")[:4000]:
        return "urdf"
    return fmt


def _newton_model(path: Path, fmt: str, floating: bool | None) -> IOModel:
    from alloy._vendor import newton  # noqa: PLC0415 - heavy import

    b = newton.ModelBuilder()
    if fmt == "mjcf":
        b.add_mjcf(str(path), floating=floating)
    else:
        try:
            import newton_usd_schemas  # noqa: F401, PLC0415
            import pxr  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportFileError(f"reading USD needs the packages usd-core and newton-usd-schemas ({exc.name} is "
                                  "not installed)", usd_fix()) from exc
        b.add_usd(str(path), floating=floating)
    return read_model(b.finalize("cpu"), path.stem)


def read_file(path: str | Path, *, floating: bool | None = None, package_dirs=()) -> IOModel:
    """``IOModel`` of an MJCF, URDF or USD file. ``floating`` None keeps the file's own free/fixed roots (URDF: free
    when the file has no joints)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    fmt = file_format(p)
    if fmt == "urdf":
        from alloy.io.urdf_read import read_urdf  # noqa: PLC0415

        return read_urdf(p, floating=floating, package_dirs=package_dirs)
    try:
        return _newton_model(p, fmt, floating)
    except ImportFileError:
        raise
    except Exception as exc:  # noqa: BLE001 - Newton's message plus the fix to try
        raise ImportFileError(f"Newton could not read {p.name} as {fmt.upper()}: {exc}") from exc


def _declare_min_wall(lib, name: str, res: ImportResult) -> bool:
    """Files carry no ``features.min_wall``; for a thin part, declare the thinnest member the card check measures
    (as library cards do). Returns True if the card was rewritten."""
    import dataclasses  # noqa: PLC0415
    import importlib  # noqa: PLC0415

    from alloy.cards.loader import dump_card  # noqa: PLC0415
    from alloy.core.quantity import Q  # noqa: PLC0415

    vo = importlib.import_module("alloy.cards.validate_object")   # alloy.cards re-exports a function of that name
    card = lib.peek_object(name)
    t_min = vo._min_member(card, lib.mesh_for(card)) if hasattr(vo, "_min_member") else None
    if not t_min:
        return False
    q = Q(float(t_min), "model", note="thinnest member of the imported geometry, as measured by the card check")
    card = dataclasses.replace(card, features=dataclasses.replace(card.features, min_wall=q))
    path = Path(res.library) / "objects" / f"{name}.yaml"
    path.write_text(dump_card(card, header="Imported by alloy.io; min_wall measured from the geometry."))
    lib.rescan()
    return True


def _validate(res: ImportResult) -> None:
    """Load every written object card through a library holding the import folder; record its issues."""
    from alloy.cards.library import CardLibrary, default_root  # noqa: PLC0415

    lib = CardLibrary([Path(res.library), default_root()])
    for name in res.objects:
        issues = lib.object_issues(name)
        if any(i.level == "error" and i.field == "features.min_wall" for i in issues) and \
                _declare_min_wall(lib, name, res):
            issues = lib.object_issues(name)
        defaults = 0
        for issue in issues:
            if issue.level == "warn" and "still at source 'default'" in issue.message:
                defaults += 1
                continue
            res.issues.append({"card": name, "level": issue.level, "field": issue.field, "message": issue.message})
        if defaults:
            res.issues.append({"card": name, "level": "info", "field": "*", "message": f"{defaults} values are "
                               "library defaults (not in the file); the physics check lists them"})


def import_file(path: str | Path, out_dir: str | Path, *, name: str | None = None, floating: bool | None = None,
                package_dirs=()) -> ImportResult:
    """Import ``path`` (MJCF, URDF or USD) into card files under the library folder ``out_dir``.

    Writes ``objects/``, ``materials/``, ``assets/<name>/`` (meshes, generated MJCF) and ``scenes/<name>.yaml``,
    then validates each object card; problems are in ``issues`` and values the cards cannot hold in ``unmapped``.
    """
    p = Path(path)
    with quiet_engine():
        io = read_file(p, floating=floating, package_dirs=package_dirs)
        res = write_cards(io, out_dir, source=p, fmt=file_format(p), prefix=name)
        _validate(res)
    log.info("imported %s: %d objects, %d robots, %d issues", p.name, len(res.objects), len(res.robots),
             len(res.issues))
    return res
