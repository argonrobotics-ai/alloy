# SPDX-License-Identifier: Apache-2.0
"""Checks on an import library until Alloy's physics-check tool covers them: Alloy's card validators and a drop test per object.

``check_one`` loads the card through ``CardLibrary`` (library root + the import root), reports each validator error
and warning with its field, and runs ``common.droptest`` on the object's hulls with the card's mass, friction and
restitution. ``run_checks`` does that for many objects in a process pool and merges the results into
``checks.json`` next to the manifest.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path

from alloy.cards import types as ct
from alloy.cards.library import RIGID_DT, CardLibrary, default_root
from alloy.importers.common import manifest as mf
from alloy.importers.common.droptest import drop_test
from alloy.importers.common.objio import read_hulls_obj
from alloy.importers.common.process import HULLS, PHYSICS

log = logging.getLogger("alloy.importers")

CHECKS = "checks.json"


def library(root: Path) -> CardLibrary:
    """A card library that sees Alloy's starter library plus the import root (no alloy-assets checkout).

    Example::

        lib = library(Path("alloy-assets/imports/ycb_gso"))
        lib.object("ycb_025_mug")
    """
    return CardLibrary(roots=[default_root(), Path(root)], assets=None)


def _issues(lib: CardLibrary, name: str) -> dict:
    try:
        issues = lib.object_issues(name)
    except Exception as exc:  # noqa: BLE001 - a card that cannot load is reported, not raised
        return {"errors": [f"load: {type(exc).__name__}: {exc}"], "warnings": []}
    return {"errors": [f"{i.field}: {i.message}" for i in issues if i.level == "error"],
            "warnings": [f"{i.field}: {i.message}" for i in issues if i.level == "warn"]}


def _drop(lib: CardLibrary, root: Path, name: str, volume: float) -> dict:
    try:
        card = lib.object(name)
        mat = lib.material(card.material)
        _, hulls = read_hulls_obj(root / mf.MESH_DIR / name / HULLS)
        return drop_test(hulls, mat.density.value * volume, mat.surface.mu_dynamic.value,
                         mat.surface.restitution.value).to_json()
    except Exception as exc:  # noqa: BLE001 - every failure is reported per object
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def check_one(args: tuple[str, str, float]) -> tuple[str, dict]:
    """Card issues and drop test of one object (``(root, name, physics volume)``; picklable for a pool).

    Example::

        check_one(("imports/ycb_gso", "ycb_025_mug", 1.378e-4))[1]["drop"]["ok"]
    """
    root, name, volume = Path(args[0]), args[1], float(args[2])
    lib = library(root)
    return name, {"cards": _issues(lib, name), "drop": _drop(lib, root, name, volume)}


def check_all(root: Path, names: list[str], volumes: dict[str, float], workers: int = 2) -> dict[str, dict]:
    """``check_one`` over ``names`` in a process pool.

    Example::

        check_all(Path("imports/ycb_gso"), ["ycb_025_mug"], {"ycb_025_mug": 1.378e-4}, workers=1)
    """
    jobs = [(str(root), n, volumes[n]) for n in names]
    with mp.get_context("spawn").Pool(workers) as pool:
        return dict(pool.imap_unordered(check_one, jobs, chunksize=4))


def run_checks(root: Path, lib: str, names: list[str] | None = None, workers: int = 2) -> dict:
    """Validate and drop-test ``names`` (all objects of manifest ``lib`` by default); merge into ``checks.json``.

    Example::

        run_checks(Path("imports/ycb_gso"), "ycb_gso", workers=3)
    """
    root = Path(root)
    man = mf.load(root / "manifest.json", lib)
    names = names or sorted(man["objects"])
    path = root / CHECKS
    checks = json.loads(path.read_text()) if path.is_file() else {}
    volumes = {n: man["objects"][n]["physics"]["volume"] for n in names}
    checks.update(check_all(root, names, volumes, workers))
    path.write_text(json.dumps(dict(sorted(checks.items())), indent=1) + "\n", encoding="utf-8")
    bad_cards = sorted(n for n in names if checks[n]["cards"]["errors"])
    bad_drops = sorted(n for n in names if not checks[n]["drop"].get("ok"))
    unsettled = sorted(n for n in names if checks[n]["drop"].get("ok") and not checks[n]["drop"].get("settled"))
    return {"checked": len(names), "card_errors": bad_cards, "drop_failures": bad_drops, "not_settled": unsettled}


# ------------------------------------------------------------------------------------------------ wall probe


def _wall(path: str) -> tuple[float, float]:
    """``(value to declare, the validator probe's reading)`` of one ``physics.obj`` (``walls.min_wall``)."""
    from alloy.cards import geometry as geo  # noqa: PLC0415
    from alloy.importers.common import walls  # noqa: PLC0415

    mesh = geo.read_mesh(Path(path))
    probe = float(geo.wall_thickness(mesh))
    return walls.min_wall(mesh, probe, THIN), probe


THIN = 2.0 * ct.LoadEnvelope().contact_speed.value * RIGID_DT   # m: the validator's min_wall threshold


def _cached_ok(entry: dict, sha: str) -> bool:
    """A cached reading is reused when the mesh is the same and it is plausible or was already re-measured."""
    from alloy.importers.common import walls  # noqa: PLC0415

    return (entry.get("physics_sha256") == sha and "min_wall" in entry
            and (entry["min_wall"] >= walls.PLAUSIBLE or "min_wall_probe" in entry))


def wall_records(items: list[tuple[str, str, Path]], cached: dict, workers: int) -> dict[str, tuple[float, float]]:
    """``name -> (min_wall to declare, the validator probe's reading)`` (m) of each ``physics.obj``, reusing
    ``cached[name]`` (``{"physics_sha256", "min_wall", "min_wall_probe"?}``, e.g. the manifest's ``physics`` entries)
    when ``_cached_ok``. The probe (``alloy.cards.geometry.wall_thickness``, the one the validator uses) costs about a
    second per mesh; a reading under ``walls.PLAUSIBLE`` is re-measured by ``walls.wall``. A card whose probe reads
    below ``THIN`` must declare ``features.min_wall``.

    Example::

        walls = wall_records([("ycb_025_mug", sha, Path("out/ycb_025_mug"))], {}, workers=3)
    """
    out = {n: (cached[n]["min_wall"], cached[n].get("min_wall_probe", cached[n]["min_wall"]))
           for n, sha, _ in items if n in cached and _cached_ok(cached[n], sha)}
    todo = [(n, str(Path(d) / PHYSICS)) for n, _, d in items if n not in out]
    if todo:
        with mp.get_context("spawn").Pool(workers) as pool:
            out.update(zip([n for n, _ in todo], pool.map(_wall, [p for _, p in todo], chunksize=8)))
    return out


def wall_thicknesses(items: list[tuple[str, str, Path]], cached: dict, workers: int) -> dict[str, float]:
    """``name -> min_wall to declare`` (m): ``wall_records`` without the probe readings.

    Example::

        walls = wall_thicknesses([("ycb_025_mug", sha, Path("out/ycb_025_mug"))], {}, workers=3)
    """
    return {n: v for n, (v, _) in wall_records(items, cached, workers).items()}
