# SPDX-License-Identifier: Apache-2.0
"""A tilting pour: the named vessel, filled and held still in the air beside a container with its lip over it, and
the tilt that pours it (pivot on the lip, axis, rate, duration) as data for ``sim.drive``, with the container's
opening (centre at rim height, inscribed radius) to judge where the liquid went.

"pour 150 ml of water from a jar tilted at 30 deg/s into a mug" places the mug, stands 150 ml at rest in a
kinematic copy of the jar whose inner lip sits over the mug's opening (its bottom clear of the mug's rim), and adds a
row with the tilt. The scene's other objects are held still (static): beside free rigid bodies Alloy's liquid does
not yet feel a moving vessel. The scene itself is at rest (the full check runs it so); the pour happens when you
drive it:

    sim = alloy.build(result.scene(0))
    sim.run(plan["start_s"])
    alloy.compose.tilt(sim, plan)            # sim.drive(vessel, angular_velocity, pivot, duration)
    sim.run(plan["duration_s"] + 1.0)

The tilt turns the vessel's opening towards the container (its far wall rises).

Example::

    >>> import numpy as np
    >>> axis_towards(np.array([-1.0, 0.0]))       # vessel on the -x side: opening tips towards +x about +y
    (0.0, 1.0, 0.0)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alloy.compose import geom
from alloy.compose.errors import ComposeError
from alloy.compose.layout import Layout, Placed
from alloy.compose.soft import cards as K
from alloy.compose.soft import fill as F
from alloy.compose.soft.interior import Interior, measure
from alloy.compose.soft.probe import Watch
from alloy.compose.soft.scene import Extra
from alloy.compose.spec import ItemSpec

DEFAULT_RATE_DEG_S = 40.0
DEFAULT_TOP_DEG = 120.0
START_S = 0.5               # s of rest before the tilt starts
CLEAR = 0.01                # m between the vessel's bottom and the container's rim
OVER = 0.35                 # share of the container's opening radius the lip sits inside its edge
DIRECTIONS = 8


def axis_towards(u: np.ndarray) -> tuple[float, float, float]:
    """The horizontal tilt axis that turns a vessel standing on side ``u`` (unit xy, from the container) so its
    opening tips towards the container: z -> -u under a positive rotation."""
    return (float(u[1]), float(-u[0]), 0.0)


def plan_tilt(planner: Any, layout: Layout, item: ItemSpec, cup: Placed, cup_it: Interior, volume: float,
              material: str, density: float) -> dict[str, Any]:
    """Fill and hold the source vessel over ``cup``; returns the result row with the tilt."""
    from alloy.compose.soft.pour import STILL_NOTE, vessel_body  # noqa: PLC0415 - pour imports this module

    source = next(i for i in planner.spec.items if i.query == item.soft.source)
    tname, tcard, body = vessel_body(planner.catalog, source)
    it = measure(body.vertices, body.faces)
    if it is None:
        raise ComposeError(f"the {source.query} ({tname}) has no cavity to hold the {item.query}",
                           "name a cup, jar, mug, bin or bottle to pour from", field="items.soft.source")
    F.check_capacity(it, volume, item.query, tname)
    col = F.column(it, volume, margin=F.sph_margin(2 * it.border), box=False)
    if col is None:
        raise ComposeError(f"{volume * 1e6:.0f} ml of {item.query} does not stand in the {source.query}",
                           "ask for less, or a larger vessel", field="items.soft.volume_ml")
    lip = np.array([it.centre[0] + _opening_radius(it), it.centre[1], it.rim])
    u = _free_side(layout, cup, body, lip, cup_it)
    vname, fname = planner._name(source.query), planner._name(item.query)            # noqa: SLF001
    vcard = K.static_copy(vname, tcard, motion="kinematic")
    planner.soft.add_card(vcard, planner.work)
    sc = K.liquid_column(fname, material, radius=col.radius, height=col.height, passage=2 * it.border, where=tname)
    planner.soft.add_card(sc, planner.work)
    key = cup.key
    hold = lambda placed: _hold(placed[key], cup_it, body, lip, u)                      # noqa: E731
    planner.soft.extras.append(Extra(vcard, (key,), lambda placed: _pose(*hold(placed), body), {}))
    planner.soft.extras.append(Extra(sc, (key,), lambda placed: _column_world(*hold(placed), col), {}))
    planner.soft.watches.append(Watch("fill", fname, vname, it, body.rest, {"volume": volume, "density": density}))
    planner.soft.rows += [{"name": vname, "card": vname, "item": source.query, "kind": "vessel", "template": tname,
                           "held": "kinematic: still until the tilt drives it"},
                          {"name": fname, "card": fname, "item": item.query, "kind": "liquid", "form": "fill",
                           "in": vname, "material": material, "volume_ml": round(volume * 1e6, 2),
                           "start": col.to_dict()}]
    rate = math.radians(item.soft.tilt_deg_s or DEFAULT_RATE_DEG_S)
    top = math.radians(item.soft.tilt_to_deg or DEFAULT_TOP_DEG)
    axis = np.asarray(axis_towards(u))
    planner.soft.estimate(f"{source.query}: held still (kinematic) with its lip over {key}; it pours when driven "
                          f"by the tilt in its row (alloy.compose.tilt)")
    planner.soft.hold_still = True
    planner.soft.estimate(STILL_NOTE)
    return {"vessel": vname, "liquid": fname, "start_s": START_S, "duration_s": round(top / rate, 4),
            "rate_deg_s": round(math.degrees(rate), 3), "to_deg": round(math.degrees(top), 3),
            "angular_velocity": [round(float(v) * rate, 6) for v in axis],
            "pivot": lambda placed: [round(float(v), 6) for v in hold(placed)[2]],
            "opening_centre": lambda placed: [round(float(v), 6) for v in _opening(placed[key], cup_it)[0]],
            "opening_radius_m": round(_opening_radius(cup_it), 5)}


def tilt(sim: Any, plan: dict[str, Any]) -> Any:
    """Start a tilting pour's drive on ``sim`` (a row of ``result.scenes[i].soft["items"]`` with "vessel").

    Example: ``alloy.compose.tilt(sim, next(r for r in scene.soft["items"] if "vessel" in r))``."""
    return sim.drive(plan["vessel"], angular_velocity=tuple(plan["angular_velocity"]), pivot=tuple(plan["pivot"]),
                     duration=float(plan["duration_s"]))


def _hold(cup: Placed, cup_it: Interior, body: Any, lip: np.ndarray, u: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vessel yaw rotation, vessel origin, lip world point) with the lip over the cup's opening."""
    top, r_open = _opening(cup, cup_it)
    z = top[2] + CLEAR + (lip[2] - float(body.vertices[:, 2].min()))
    point = np.array([top[0] + u[0] * OVER * r_open, top[1] + u[1] * OVER * r_open, z])
    rot = geom.yaw_matrix(math.atan2(-u[1], -u[0]))
    return rot, point - rot @ lip, point


def _opening_radius(cup_it: Interior) -> float:
    return cup_it.radius_at(cup_it.rim - 2e-3, cup_it.rim - 1e-3)


def _opening(cup: Placed, cup_it: Interior) -> tuple[np.ndarray, float]:
    """(world point at the centre of the cup's opening, at its rim height; the opening's inscribed radius)."""
    centre = geom.yaw_matrix(cup.yaw) @ np.array([*cup_it.centre, 0.0]) + cup.pos
    return np.array([centre[0], centre[1], float(cup.triangles()[..., 2].max())]), _opening_radius(cup_it)


def _pose(rot: np.ndarray, origin: np.ndarray, _lip: np.ndarray, body: Any) -> tuple[float, ...]:
    full = rot @ geom.matrix_from_quat(body.rest)
    return tuple(float(v) for v in origin) + geom.quat_from_matrix(full)


def _column_world(rot: np.ndarray, origin: np.ndarray, _lip: np.ndarray, col: F.Column) -> tuple[float, ...]:
    p = rot @ np.array([col.x, col.y, col.z0 + 0.5 * col.height]) + origin
    return tuple(float(v) for v in p) + geom.quat_from_matrix(rot)


def _free_side(layout: Layout, cup: Placed, body: Any, lip: np.ndarray, cup_it: Interior) -> np.ndarray:
    """The side of the cup (unit xy) where the held vessel stays farthest from other objects, inside the table."""
    best, best_d = np.array([-1.0, 0.0]), -math.inf
    others = [p.outline for k, p in layout.placed.items() if k != cup.key] + list(layout.obstacles)
    table = layout.support.outline
    for k in range(DIRECTIONS):
        a = 2 * math.pi * k / DIRECTIONS
        u = np.array([math.cos(a), math.sin(a)])
        rot, origin, _ = _hold(cup, cup_it, body, lip, u)
        xy = (body.vertices @ rot.T + origin)[:, :2]
        if geom.depths(table, xy).min() < 0:
            continue
        d = min((float(np.min(np.linalg.norm(xy[:, None, :] - o[None, :, :], axis=2))) for o in others), default=1.0)
        if d > best_d:
            best, best_d = u, d
    return best


__all__ = ["DEFAULT_RATE_DEG_S", "DEFAULT_TOP_DEG", "axis_towards", "plan_tilt", "tilt"]
