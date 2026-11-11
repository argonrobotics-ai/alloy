# SPDX-License-Identifier: Apache-2.0
"""Final stage: pick the best objects and write the import root with the shared writers (``alloy.importers.common``).

Inputs: candidates from ``select`` (attribution metadata), objects kept by the review, the shared mesh pipeline's
``result.json`` per object and a folder holding each object's ``physics.obj`` + ``physics.hulls.obj``, centred by
:mod:`alloy.importers.objaverse.recentre`. The pick goes
round-robin over categories by rank (best first) until ``target`` objects, so every category gets its best objects
before any category gets many.

Layout written under ``root`` (``alloy-assets/imports/objaverse``; see :mod:`alloy.importers.common.manifest`)::

    objects/objaverse/<name>.yaml   materials/objaverse/<name>_eff.yaml   manifest.json   queue.jsonl
    ATTRIBUTION.md   .gitignore   meshes/<name>/physics.obj, physics.hulls.obj (not in git)

Example:
    >>> from alloy.importers.objaverse.assemble import round_robin
    >>> round_robin({"a": ["a1", "a2", "a3"], "b": ["b1"]}, 3)
    ['a1', 'b1', 'a2']
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alloy.cards import CardLibrary, dump_card
from alloy.cards import types as ct
from alloy.importers.common import cards as cw
from alloy.importers.common import checks as ch
from alloy.importers.common import manifest as mf
from alloy.importers.common import queue as qu
from alloy.importers.objaverse.cards import HULLS, LIBRARY, PHYSICS, Processed, card_spec
from alloy.importers.objaverse.categories import card_name, load_categories
from alloy.importers.objaverse.errors import ImporterConfigError, ImportRejected
from alloy.importers.objaverse.licence import CHANGES, DATASET_CREDIT
from alloy.importers.objaverse.materials import food_materials
from alloy.importers.objaverse.recentre import is_centred

log = logging.getLogger("alloy.importers.objaverse")

REPAIR_BONUS = {"as_is": 0.5, "repair": 0.25}     # prefer meshes that needed less repair
FILES = (PHYSICS, HULLS, "visual.glb", "thumb.png")
Wall = float | tuple[float, float] | None         # min_wall to declare [m], or (that, the validator probe's reading)


def round_robin(ranked: Mapping[str, Sequence[str]], target: int) -> list[str]:
    """Take rank 0 of every category, then rank 1, ... until ``target`` items (categories in sorted order)."""
    out: list[str] = []
    depth = max((len(v) for v in ranked.values()), default=0)
    for r in range(depth):
        for cat in sorted(ranked):
            if r < len(ranked[cat]) and len(out) < target:
                out.append(ranked[cat][r])
    return out


def pick(cands: Sequence[Mapping[str, Any]], kept: Sequence[Mapping[str, Any]], results: Mapping[str, Mapping],
         target: int, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Objects to import: reviewed-kept, mesh stage ok, not in ``exclude`` (card names that failed the checks),
    ranked per category (meta score + repair bonus), capped."""
    cats = load_categories()
    by_uid = {c["uid"]: c for c in cands}
    ranked: dict[str, list[tuple[float, str]]] = {}
    rows: dict[str, dict[str, Any]] = {}
    for k in kept:
        cat = k["category"]
        name = card_name(cat, k["uid"])                   # relabelled objects are re-processed under the new name
        res = results.get(name)
        if name in exclude or res is None or not res.get("ok", True) or not res.get("watertight"):
            continue
        score = by_uid[k["uid"]]["score"] + REPAIR_BONUS.get(res.get("repair", {}).get("method"), 0.0)
        ranked.setdefault(cat, []).append((-score, k["uid"]))
        rows[k["uid"]] = {"cand": by_uid[k["uid"]], "category": cat, "result": res}
    order = {c: [u for _, u in sorted(v)][: cats[c].cap] for c, v in ranked.items()}
    return [rows[u] for u in round_robin(order, target)]


def wall_probe(dirs: Mapping[str, Path], shas: Mapping[str, str], workers: int = 3,
               cached: Mapping[str, Mapping] | None = None) -> dict[str, tuple[float, float]]:
    """name -> ``(min_wall to declare, the validator probe's reading)`` [m] of each ``physics.obj``
    (``common.checks.wall_records``: a probe reading on a remesh defect is re-measured); ``cached`` (a manifest's
    ``physics`` entries) skips meshes whose sha256 is unchanged."""
    return ch.wall_records([(n, shas[n], dirs[n]) for n in sorted(dirs)], dict(cached or {}), workers)


def _wall_pair(wall: Wall) -> tuple[float | None, float | None]:
    return wall if isinstance(wall, tuple) else (wall, wall)


def _meta(c: Mapping[str, Any]) -> dict[str, Any]:
    return {"uid": c["uid"], "name": c["name"], "user_display": c["author"], "viewerUrl": c["url"],
            "license": c["spdx"]}


def _entry(row: Mapping[str, Any], spec: cw.CardSpec, wall: Wall) -> dict[str, Any]:
    c, res = row["cand"], row["result"]
    sizes = res.get("sizes", {})
    wall, probe = _wall_pair(wall)
    physics = {"volume": res["volume"], "n_hulls": res["n_hulls"], "repair": res["repair"]["method"],
               "mass_kg": round(spec.mass.value * res["volume"], 6), "mass_kind": spec.mass.kind,
               "min_wall": wall, "physics_sha256": res["files"][PHYSICS]}
    if probe != wall:                                   # re-measured (common.walls): keep the probe's reading too
        physics["min_wall_probe"] = probe
    return {"library": LIBRARY, "title": c["name"], "source": c["url"], "licence": c["spdx"], "author": c["author"],
            "uid": c["uid"], "category": row["category"], "card": f"objects/{LIBRARY}/{spec.name}.yaml",
            "material": f"materials/{LIBRARY}/{spec.name}{cw.EFF_SUFFIX}.yaml",
            "files": {f: {"sha256": res["files"][f], "bytes": sizes.get(f)} for f in FILES if f in res["files"]},
            "physics": physics}


def write_object(root: Path, row: Mapping[str, Any], mesh_dir: Path, base: ct.MaterialCard,
                 wall: Wall) -> tuple[dict[str, Any], qu.QueueRow]:
    """Both cards of one object, its local physics files, its manifest entry and its sourcing row."""
    cat = load_categories()[row["category"]]
    value, probe = _wall_pair(wall)
    spec, q = card_spec(_meta(row["cand"]), cat, Processed.from_result(row["result"]), base,
                        value if value is not None and value < ch.THIN else None, probe)
    obj_text, mat_text = cw.card_texts(spec, base)
    for kind, suffix, text in (("objects", "", obj_text), ("materials", cw.EFF_SUFFIX, mat_text)):
        p = root / kind / LIBRARY / f"{spec.name}{suffix}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    local = root / mf.MESH_DIR / spec.name
    local.mkdir(parents=True, exist_ok=True)
    for f in (PHYSICS, HULLS):
        shutil.copyfile(mesh_dir / f, local / f)
    return _entry(row, spec, wall), q


def require_centred(picked: Sequence[Mapping[str, Any]]) -> None:
    """Raise unless every picked mesh has its origin at its bounding-box centre (``recentre`` ran)."""
    off = [row["result"].get("name", row["cand"]["uid"]) for row in picked if not is_centred(row["result"])]
    if off:
        raise ImporterConfigError(f"{len(off)} meshes are not centred on their origin (first: {off[0]}); run "
                                  "`python -m alloy.importers.objaverse recentre --meshes <dir>` before assemble")


def write_root(root: Path, picked: Sequence[Mapping[str, Any]], mesh_dirs: Mapping[str, Path],
               walls: Mapping[str, Wall], lib: CardLibrary | None = None) -> dict[str, Any]:
    """Write every picked object plus the shared files; returns a JSON summary."""
    require_centred(picked)
    lib = lib or CardLibrary()
    foods = {m.name: m for m in food_materials()}
    man, rows, rejected = mf.load(root / "manifest.json", LIBRARY), [], []
    for row in picked:
        name = card_name(row["category"], row["cand"]["uid"])
        material = load_categories()[row["category"]].material
        try:
            entry, q = write_object(root, row, mesh_dirs[name], foods.get(material) or lib.material(material),
                                    walls.get(name))
        except ImportRejected as e:
            rejected.append(e.as_row())
            continue
        thumb = man["objects"].get(name, {}).get("files", {}).get("thumb.png")
        if thumb is not None:                         # thumbnails are re-rendered and re-hashed by the thumbnail owner
            entry["files"]["thumb.png"] = thumb
        man["objects"][name] = entry
        rows.append(q)
    _write_shared(root, foods, man)
    return {"written": len(picked) - len(rejected), "rejected": rejected, "queue": qu.add(root / "queue.jsonl", rows),
            "objects": len(man["objects"])}


def _write_shared(root: Path, foods: Mapping[str, ct.MaterialCard], man: dict[str, Any]) -> None:
    for name, mat in foods.items():
        p = root / "materials" / LIBRARY / f"{name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dump_card(mat, header="Objaverse import: rigid stand-in for food (ESTIMATE values)"))
    mf.save(root / "manifest.json", man)
    write_attribution(root, man)
    (root / ".gitignore").write_text("meshes/\n", encoding="utf-8")


def write_attribution(root: Path, man: Mapping[str, Any]) -> None:
    """``ATTRIBUTION.md``: dataset credit, licences, changes, and one table row per object."""
    lines = ["# Attribution: imports/objaverse", "", f"Dataset: {DATASET_CREDIT}.", "",
             "Each object keeps its author's licence: CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/) or "
             "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).", f"Changes made by Alloy: {CHANGES}.", "",
             "| Card | Title | Author | Licence | Source |", "|---|---|---|---|---|"]
    cell = lambda x: str(x).replace("|", "/").replace("\n", " ")  # noqa: E731 - table-safe text
    lines += [f"| {n} | {cell(e['title'])} | {cell(e['author'])} | {e['licence']} | {e['source']} |"
              for n, e in sorted(man["objects"].items())]
    (root / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
