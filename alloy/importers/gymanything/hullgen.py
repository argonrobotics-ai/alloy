# SPDX-License-Identifier: Apache-2.0
"""Convex-hull sidecars (``physics.hulls.obj``, the shared importer format of ``alloy.importers.common``) for every
rigid imported mesh, and the collision mode that follows from them:

- convex meshes get a one-hull sidecar (the drop test and other tools read hulls) and keep ``collision: convex``;
- non-convex meshes get CoACD hulls; ``collision: decompose`` when the hulls hold at most ``MAX_FILL`` x the mesh
  volume, else the hulls fill a cavity (a bowl, a rack) and the card keeps the exact mesh (``collision: sdf``).

Needs the importer extra (``trimesh``, ``coacd``). A cache folder of earlier sidecars (``<card>.hulls.obj``) is used
when its source sha256 matches the mesh, so a rerun does not redo CoACD.

Example::

    >>> stats = hull_all(Path("gar"), ["gar_wine_pouring_glass_bordeaux"], convex=set(), workers=2)   # doctest: +SKIP
    >>> stats["gar_wine_pouring_glass_bordeaux"]["n_hulls"]                                            # doctest: +SKIP
    32
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Any

from alloy.cards import geometry as geo
from alloy.cards.loader import dump_card, load_object

log = logging.getLogger("alloy.importers.gymanything")

PHYSICS, HULLS = "physics.obj", "physics.hulls.obj"
MAX_FILL = 3.0               # sum(hull volume) / mesh volume above this: the hulls fill a cavity -> keep sdf
SHA_TAG = "# source_sha256:"


def mesh_dir(root: Path, card: str) -> Path:
    return root / "meshes" / card


def _cached(cache: Path | None, card: str, sha: str, dest: Path) -> bool:
    if cache is None or not (cache / f"{card}.hulls.obj").is_file():
        return False
    src = cache / f"{card}.hulls.obj"
    with open(src, encoding="utf-8") as fh:
        if fh.readline().strip() != f"{SHA_TAG} {sha}":
            return False
    shutil.copyfile(src, dest)
    return True


def _hull_stats(path: Path, volume: float) -> dict[str, Any]:
    from alloy.importers.common.objio import read_hulls_obj  # noqa: PLC0415

    _, hulls = read_hulls_obj(path)
    total = sum(geo.signed_volume(geo.MeshData(v, f)) for v, f in hulls)
    return {"n_hulls": len(hulls), "volume_ratio": float(total / volume), "max_verts": max(len(v) for v, _ in hulls)}


def hull_one(args: tuple[str, str, bool, str | None]) -> tuple[str, dict[str, Any]]:
    """Write ``meshes/<card>/physics.hulls.obj`` (one hull if ``convex``, else CoACD); picklable for a pool."""
    from alloy.importers.common import hulls as hl  # noqa: PLC0415 - needs the importer extra
    from alloy.importers.common import objio  # noqa: PLC0415

    root, card, convex, cache = Path(args[0]), args[1], args[2], (Path(args[3]) if args[3] else None)
    src, dest = mesh_dir(root, card) / PHYSICS, mesh_dir(root, card) / HULLS
    try:
        m = geo.read_mesh(src)
        sha, vol = geo.sha256_file(src), geo.signed_volume(m)
        cached = _cached(cache, card, sha, dest)
        if not cached:
            hulls = [hl.convex_hull(m.vertices)] if convex else hl.decompose(m.vertices, m.faces, vol)[0]
            text = objio.hulls_obj_text(hulls, sha, header=f"{card}: {len(hulls)} convex hulls ({'hull' if convex else 'CoACD'})")
            dest.write_text(text, encoding="utf-8")
        return card, {"ok": True, "cached": cached, **_hull_stats(dest, vol)}
    except Exception as exc:  # noqa: BLE001 - one object's failure is reported, the batch goes on
        return card, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def hull_all(root: Path, cards: list[str], convex: set[str], workers: int = 2,
             cache: Path | None = None) -> dict[str, dict[str, Any]]:
    """``hull_one`` over ``cards`` (``convex`` names get one hull) in a spawn pool; card -> stats or error."""
    jobs = [(str(root), c, c in convex, str(cache) if cache else None) for c in cards]
    if workers <= 1:
        return dict(map(hull_one, jobs))
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        return dict(pool.imap_unordered(hull_one, jobs))


def collision_for(convex_ratio: float, hull: dict[str, Any] | None) -> str:
    """``convex`` / ``decompose`` / ``sdf`` from the mesh's convexity and its hull fill (see the module docstring)."""
    from alloy.importers.gymanything.meshes import CONVEX_RATIO  # noqa: PLC0415

    if convex_ratio <= CONVEX_RATIO:
        return "convex"
    if hull and hull.get("ok") and hull["volume_ratio"] <= MAX_FILL:
        return "decompose"
    return "sdf"


def set_collision(card_path: Path, collision: str, n_hulls: int | None, header: str) -> None:
    """Rewrite an object card's ``shape.collision`` (and ``max_parts`` for decompose) in canonical form."""
    card = load_object(card_path)
    shape = dataclasses.replace(card.shape, collision=collision)
    if collision == "decompose" and n_hulls:
        shape = dataclasses.replace(shape, max_parts=max(1, min(n_hulls, card.budget.collision_parts)))
    card_path.write_text(dump_card(dataclasses.replace(card, shape=shape), header=header), encoding="utf-8")
