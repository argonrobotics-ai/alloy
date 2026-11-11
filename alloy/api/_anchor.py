# SPDX-License-Identifier: Apache-2.0
"""Where ``Scene.add_object`` puts a library object: a point of its own geometry (its anchor) goes to ``pos``.

A card's body frame is wherever its source put it: YCB scans have origins 19-85 mm off the mesh, so an object placed
by its origin landed inside walls and tables. The anchor comes from the card's geometry (no card field, so no card
hash changes):

- ``"auto"`` (default): ``"base"`` for an object from an imported library (YCB, Google Scanned Objects, RoboCasa,
  Objaverse, the Hub: their file origins are wherever the scan put them), so ``pos`` is where it stands, as for
  Alloy's own objects; ``"origin"`` for Alloy's own library and your own cards (their origins are authored: a cup's
  base, a bed's floor, which existing scenes rely on);
- ``"center"``: the centre of the object's axis-aligned box in its own frame;
- ``"base"``: that box's centre in x and y, its bottom in z (``pos`` z = the table top it stands on);
- ``"origin"``: the card's frame itself (the behaviour before anchors; scenes saved before them keep their poses).

Geometry: a mesh, an SDF or tet source from a mesh, and an MJCF object's point cloud (``CardLibrary.mesh_for``);
primitives (centred on their frame, cylinders and capsules along z); curves (their points plus the radius) and
particle regions.

What you set is what you read: for an object placed by its anchor, ``"<name>.pos"`` in ``sim.state()``,
``sim.run(record=...)`` and trajectories is that anchor point in the world (it turns with the body), and
``sim.set_pose(name, pos=...)`` puts the anchor there; ``"<name>.body_pos"`` is the body frame's origin. Everything
else (primitives, Alloy's own and your own cards, anchor="origin") reads its body origin as ``.pos``, as before.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

log = logging.getLogger("alloy.api")
MODES = ("auto", "center", "base", "origin")


def resolve(scene: Any) -> Any:
    """``scene`` with every anchored library object's pose, per-world rows and randomized x / y / z ranges turned
    into its body origin's (a copy; ``scene`` itself when nothing is anchored). The build and ``Scene.save`` use it.

    Example: ``materialize(resolve(scene), root)``."""
    shifts = {}
    for n, it in scene.items.items():
        if it["kind"] == "card" and it.get("raw_placement") is None and it.get("anchor", "origin") != "origin":
            c = offset(it["card_name"], scene.roots, it["anchor"])
            if np.any(c != 0.0):
                shifts[n] = (c, _rotate(np.asarray(it["pose"][3:7], np.float64), c))
    if not shifts:
        return scene                                          # nothing moves: the scene as given, not a copy
    out = type(scene).from_dict(scene.to_dict())
    for name, (c, d) in shifts.items():
        it = out.items[name]
        it["pose"] = [*(np.asarray(it["pose"][:3]) - d).tolist(), *it["pose"][3:7]]
        if it.get("per_world_pos"):
            it["per_world_pos"] = (np.asarray(it["per_world_pos"], np.float64) - d).tolist()
        for r in it.get("randomize", []):
            if str(r["field"]).startswith("pose."):
                k = int(r["field"].split(".")[1])
                r["lo"], r["hi"] = float(r["lo"] - d[k]), float(r["hi"] - d[k])
        it["anchor"], it["anchor_shift"], it["anchor_local"] = "origin", d.tolist(), c.tolist()
        log.info("%s: %s anchor, body origin placed %s m from pos", name, scene.items[name]["anchor"],
                 np.round(-d, 4).tolist())
    return out


def shift(card: str, roots: Sequence[str], mode: str, quat_xyzw: Sequence[float]) -> np.ndarray:
    """World offset from the anchor to the body origin, R(q)·c: the body origin goes to ``pos - shift``.

    Example: ``shift("ycb_005_tomato_soup_can", [], "center", (0, 0, 0, 1))``."""
    return _rotate(np.asarray(quat_xyzw, np.float64), offset(card, roots, mode))


def offset(card: str, roots: Sequence[str], mode: str) -> np.ndarray:
    """The anchor point c in the card's body frame for ``mode`` ("auto" decided; zeros for "origin").

    Example: ``offset("ycb_005_tomato_soup_can", [], "base")``."""
    mode = effective(card, roots, mode)
    return np.zeros(3) if mode == "origin" else anchor(card, tuple(roots), mode)


def anchors(resolved: Any) -> dict[str, list[float]]:
    """Body-frame anchor points of the objects ``resolve`` moved: ``{name: c}``. ``sim.state()`` reports
    ``"<name>.pos"`` at that point (``body_q`` applied to c) and the frame itself as ``"<name>.body_pos"``.

    Example: ``anchors(resolve(scene))`` -> ``{"can": [0.0, 0.0, -0.05]}``."""
    return {n: list(it["anchor_local"]) for n, it in resolved.items.items() if it.get("anchor_local")}


def world_points(body_q: np.ndarray, c: Any) -> np.ndarray:
    """(N, 3) world points of body-frame point ``c`` for N body poses ``body_q`` (N, 7: position, quat xyzw), in
    ``body_q``'s dtype. Example: ``world_points(np.array([[0, 0, 1, 0, 0, 0, 1.0]]), (0, 0, -0.05))`` -> z 0.95."""
    q = np.asarray(body_q, np.float64)
    n = np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    u, w = q[:, 3:6] / n, q[:, 6:7] / n
    v = np.broadcast_to(np.asarray(c, np.float64), u.shape)
    t = 2.0 * np.cross(u, v)
    return (q[:, :3] + v + w * t + np.cross(u, t)).astype(np.asarray(body_q).dtype)


def body_origins(points: np.ndarray, quat_xyzw: np.ndarray, c: Any) -> np.ndarray:
    """(N, 3) body origins that put body-frame point ``c`` at world ``points`` with orientations ``quat_xyzw``
    (N, 4): the inverse of ``world_points``. Example: ``body_origins(np.zeros((1, 3)), np.array([[0, 0, 0, 1.0]]),
    (0, 0, -0.05))`` -> z 0.05."""
    q = np.asarray(quat_xyzw, np.float64)
    zero = np.concatenate([np.zeros((len(q), 3)), q], axis=1)
    return np.asarray(points, np.float64) - world_points(zero, c).astype(np.float64)


def effective(card: str, roots: Sequence[str], mode: str) -> str:
    """``mode`` with "auto" decided: "base" for a card from an imported library or the Hub, else "origin".

    Example: ``effective("ycb_005_tomato_soup_can", [], "auto")`` -> ``"base"``."""
    if mode != "auto":
        return mode
    from alloy.api import catalog  # noqa: PLC0415

    path = catalog.card_path("objects", card, roots)
    if path is None:                                          # only in the Hub's index: an imported library
        return "base"
    imported = [*catalog.import_roots(), *catalog.fetched_roots()]
    return "base" if any(path.resolve().is_relative_to(r.resolve()) for r in imported) else "origin"


def anchor(card: str, roots: tuple[str, ...], mode: str) -> np.ndarray:
    """The anchor in the card's body frame (zeros when its geometry is unknown).

    Example: ``anchor("ycb_005_tomato_soup_can", (), "base")``."""
    box = local_box(card, roots)
    if box is None:
        log.info("%s: geometry unknown, placed by its origin", card)
        return np.zeros(3)
    lo, hi = box
    c = 0.5 * (lo + hi)
    if mode == "base":
        c[2] = lo[2]
    return c


def local_box(card: str, roots: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    """(lo, hi) of the card's geometry in its body frame, or None. Example: ``local_box("peg_d12", ())``."""
    from alloy.api import catalog  # noqa: PLC0415
    from alloy.api._build import fetch_imported  # noqa: PLC0415
    from alloy.cards import types as ct  # noqa: PLC0415

    lib = _library(roots, os.environ.get("ALLOY_ASSETS"), os.environ.get("ALLOY_CACHE"),
                   tuple(str(r) for r in catalog.fetched_roots()))
    fetch_imported(lib, [card])                               # its mesh, from the store or the Hub, as a build does
    obj = lib.peek_object(card)
    s = obj.shape
    if isinstance(s, ct.Primitive):
        return _primitive_box(s)
    if isinstance(s, ct.CurveSource):
        pts = np.asarray(s.points, np.float64)
        return pts.min(axis=0) - s.radius, pts.max(axis=0) + s.radius
    if isinstance(s, ct.ParticleSource):
        lo, hi = np.asarray(s.region[0], np.float64), np.asarray(s.region[1], np.float64)
        return np.minimum(lo, hi), np.maximum(lo, hi)
    mesh = lib.mesh_for(obj)
    if mesh is None or not len(mesh.vertices):
        return None
    v = np.asarray(mesh.vertices, np.float64)
    return v.min(axis=0), v.max(axis=0)


def _primitive_box(p: Any) -> tuple[np.ndarray, np.ndarray] | None:
    s = list(p.size)
    half = {"box": lambda: s[:3], "ellipsoid": lambda: s[:3], "sphere": lambda: [s[0]] * 3,
            "capsule": lambda: [s[0], s[0], s[1] + s[0]], "cylinder": lambda: [s[0], s[0], s[1]],
            "cone": lambda: [s[0], s[0], s[1]]}.get(p.kind)
    if half is None:                                          # a plane has no box
        return None
    h = np.asarray(half(), np.float64)
    return -h, h


@functools.lru_cache(maxsize=4)
def _library(roots: tuple[str, ...], assets: str | None, cache: str | None, fetched: tuple[str, ...]) -> Any:
    """One card library per (roots, $ALLOY_ASSETS, $ALLOY_CACHE, fetch caches on disk): indexing every root takes
    about 0.25 s."""
    from alloy.api import catalog  # noqa: PLC0415
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.cards.roots import assets_root  # noqa: PLC0415

    return CardLibrary(roots=catalog.card_roots(list(roots)), assets=assets_root())


def _rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw / np.linalg.norm(q_xyzw)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r @ v
