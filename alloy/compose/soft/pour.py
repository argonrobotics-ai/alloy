# SPDX-License-Identifier: Apache-2.0
"""A pour: a stream of liquid (an emitter) falling into a placed container, from a vessel held tilted over it when
the sentence names one ("pour water from a bottle into a cup").

The vessel is a still copy of its card (motion static) with its mouth just above the stream's start; its own
liquid is the stream. The stream runs at ``POUR_RATE`` until the asked volume has left (default 150 ml), then the
scene settles for ``POUR_TAIL_S``. A pour moves by design, so the checks judge where the liquid ends up.

Example::

    >>> plan_pour(planner, layout, 2, item)        # doctest: +SKIP
    >>> planner.soft.emitters[0].make(layout.placed).rate        # doctest: +SKIP
    5e-05
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alloy.cards import types as ct
from alloy.compose import geom
from alloy.compose.body import Body
from alloy.compose.errors import ComposeError
from alloy.compose.layout import Layout, Placed
from alloy.compose.soft import cards as K
from alloy.compose.soft import fill as F
from alloy.compose.soft.interior import measure
from alloy.compose.soft.probe import Watch
from alloy.compose.soft.scene import Extra, LazyEmitter
from alloy.compose.soft.tilt import plan_tilt
from alloy.compose.soft.vessel import interior_of
from alloy.compose.spec import ItemSpec

YAWS = 8
STILL_NOTE = ("the scene's objects are held still (static copies): beside free rigid bodies Alloy's liquid does not "
              "yet feel a moving (kinematic) vessel")


def plan_pour(planner: Any, layout: Layout, idx: int, item: ItemSpec) -> None:
    """Stream (and held vessel) into the first placed object of ``item.target``."""
    t = planner._template(idx, item)                                                  # noqa: SLF001
    targets = planner.targets(layout, item.target)
    if not targets:
        raise ComposeError(f"{item.query}: the container it pours into ({item.target!r}) was not placed",
                           "see the missing list for why")
    cup = targets[0]
    it = interior_of(planner, cup)
    mat = planner.library.material(t.material)
    volume, how = F.amount_m3(item.soft, None, float(mat.density.value))
    F.check_capacity(it, volume, item.query, cup.card)
    if item.soft.tilt_deg_s or item.soft.tilt_to_deg:
        row = plan_tilt(planner, layout, item, cup, it, volume, t.material, float(mat.density.value))
        planner._row(f"{item.query}_tilt", item, t, form="pour", **{"into": cup.key},            # noqa: SLF001
                     volume_ml=round(volume * 1e6, 2), amount=how, **row)
        planner.soft.settle_s = max(planner.soft.settle_s, 0.3)
        return
    pour = F.pour_over(cup, it, volume)
    key = cup.key
    k = sum(e.material == t.material for e in planner.soft.emitters)
    liquid = f"{t.material}_pour_{k}"                         # the name Scene.load gives this card emitter
    planner.soft.emitters.append(LazyEmitter((key,), lambda placed, v=volume: _emitter(placed[key], it, v,
                                                                                          t.material), t.material))
    row: dict[str, Any] = {"into": key, "liquid": liquid, "volume_ml": round(volume * 1e6, 2), "amount": how,
                           "rate_ml_s": round(F.POUR_RATE * 1e6, 1), "seconds": round(pour.seconds, 3)}
    if item.soft.source is not None:
        row["from"] = _vessel(planner, layout, item, cup, it)
    planner._row(f"{item.query}_stream", item, t, form="pour", **row)                   # noqa: SLF001
    planner.soft.estimate(f"{item.query} pour: a {F.POUR_RATE * 1e6:.0f} ml/s stream starting "
                          f"{F.POUR_DROP * 100:.0f} cm above {cup.key} (rate and start height estimated); "
                          f"alloy.load(path) or alloy.build(result.scene(i)) pours it as {liquid}; alloy.check does not "
                          f"run it")
    planner.soft.watches.append(Watch("pour", None, key, it, cup.body.rest,
                                      {"volume": volume, "density": float(mat.density.value)}))
    planner.soft.expect_rest = False
    planner.soft.settle_s = max(planner.soft.settle_s, pour.seconds + F.POUR_TAIL_S)


def _emitter(cup: Placed, it: Any, volume: float, material: str) -> ct.EmitterSpec:
    p = F.pour_over(cup, it, volume)
    return ct.EmitterSpec("emitter", material, (*p.start, 0.0, 0.0, 0.0, 1.0), (F.POUR_MOUTH, F.POUR_MOUTH, 1e-3),
                          rate=p.rate, velocity=(0.0, 0.0, -F.POUR_SPEED), start_s=0.0, stop_s=p.seconds)


def _vessel(planner: Any, layout: Layout, item: ItemSpec, cup: Placed, it: Any) -> str:
    """Add the held vessel (a static copy of the source item's card); returns its card name."""
    source = next(i for i in planner.spec.items if i.query == item.soft.source)
    name, card, body = vessel_body(planner.catalog, source)
    mouth = _mouth(body)
    vname = planner._name(source.query)                                                # noqa: SLF001
    sc = K.static_copy(vname, card)
    planner.soft.add_card(sc, planner.work)
    yaw = _free_yaw(layout, cup, body, mouth, it)
    planner.soft.extras.append(Extra(sc, (cup.key,), lambda placed: _held_pose(placed[cup.key], it, body, mouth, yaw),
                                     {}))
    planner.soft.rows.append({"name": vname, "card": vname, "item": source.query, "kind": "vessel", "template": name,
                              "held": "static, tilted over the stream"})
    planner.soft.estimate(f"{source.query}: held still and tilted {math.degrees(F.TILT):.0f} deg over "
                          f"{cup.key}; the stream stands for its liquid")
    return vname


def vessel_body(catalog: Any, item: ItemSpec) -> tuple[str, ct.ObjectCard, Body]:
    """The first catalog match for a held vessel with a mesh (any route: it is held still)."""
    from alloy.compose.catalog import _mesh  # noqa: PLC0415 - the catalog's own mesh reader

    for m in catalog.search(item.query, item.words, k=25):
        catalog.info(m.name)                                         # fetches an imported object's files
        try:
            card = catalog.library.object(m.name)
        except Exception:  # noqa: BLE001 - an unreadable card is skipped
            continue
        mesh = _mesh(catalog.library, card, card.shape) if isinstance(
            card.shape, (ct.Primitive, ct.MeshSource, ct.MJCFSource)) else "no mesh"
        if isinstance(mesh, str):
            continue
        body = Body.from_mesh(m.name, mesh[0], mesh[1], np.asarray(mesh[0]).mean(axis=0))
        if body.reoriented:
            continue
        return m.name, card, body
    raise ComposeError(f"no {item.query} with a mesh to hold over the pour", "name another vessel ('from a jar', "
                       "'from a mug'), or leave 'from ...' out for a stream from above", field="items.soft.source")


def _mouth(body: Body) -> np.ndarray:
    it = measure(body.vertices, body.faces)
    top = float(body.vertices[:, 2].max())
    c = it.centre if it is not None else body.vertices[:, :2].mean(axis=0)
    return np.array([c[0], c[1], top])


def _held_pose(cup: Placed, it: Any, body: Body, mouth: np.ndarray, yaw: float) -> tuple[float, ...]:
    start = np.asarray(F.pour_over(cup, it, 1e-6).start)
    pose = F.hold_vessel(body, mouth, start, yaw)
    rot = geom.matrix_from_quat(pose[3:]) @ geom.matrix_from_quat(body.rest)
    return tuple(pose[:3]) + geom.quat_from_matrix(rot)


def _free_yaw(layout: Layout, cup: Placed, body: Body, mouth: np.ndarray, it: Any) -> float:
    """The tilt direction whose held vessel stays farthest from the other objects."""
    best, best_d = 0.0, -math.inf
    others = [p for k, p in layout.placed.items() if k != cup.key]
    for k in range(YAWS):
        yaw = 2 * math.pi * k / YAWS
        pose = _held_pose(cup, it, body, mouth, yaw)
        rot = geom.matrix_from_quat(pose[3:]) @ geom.matrix_from_quat(body.rest).T
        xy = (body.vertices @ rot.T + np.asarray(pose[:3]))[:, :2]
        d = min((float(np.min(np.linalg.norm(xy[:, None, :] - o.outline[None, :, :], axis=2))) for o in others),
                default=1.0)
        if d > best_d:
            best, best_d = yaw, d
    return best


__all__ = ["plan_pour", "vessel_body"]
