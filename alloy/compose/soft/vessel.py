# SPDX-License-Identifier: Apache-2.0
"""A liquid or grains filled into one placed container: measure its cavity, size the amount, make the resting
column's card, route it, and hang it off the container so it follows the container's pose.

Example::

    >>> fill_container(planner, layout, cup, item, template)        # doctest: +SKIP
    >>> planner.soft.rows[-1]["volume_ml"]                           # doctest: +SKIP
    200.0
"""

from __future__ import annotations

import re
from typing import Any

from alloy.cards import types as ct
from alloy.compose.errors import ComposeError
from alloy.compose.layout import Layout, Placed
from alloy.compose.soft import cards as K
from alloy.compose.soft import fill as F
from alloy.compose.soft.interior import Interior, measure
from alloy.compose.soft.materials import Template
from alloy.compose.soft.probe import Watch
from alloy.compose.soft.scene import Extra
from alloy.compose.spec import ItemSpec

SOLID_COLLISION = ("convex",)
RIGID_BUDGET = re.compile(r"\((\d+) grains > (\d+)")
CAP_MARGIN = 0.95
SETTLE_FILL_S = 0.3


def interior_of(planner: Any, p: Placed) -> Interior:
    """The cavity of a placed container (cached per card); ``ComposeError`` when it has none."""
    cache = planner.__dict__.setdefault("_interiors", {})
    if p.card not in cache:
        cache[p.card] = measure(p.body.vertices, p.body.faces)
    it = cache[p.card]
    shape = planner.catalog.library.peek_object(p.card).shape
    if it is None or getattr(shape, "collision", "") in SOLID_COLLISION:
        why = "it has no cavity" if it is None else "its collision shape is its convex hull (solid inside)"
        raise ComposeError(f"{p.card} cannot hold anything: {why}", "put it in a cup, mug, jar, bowl, bin or tub "
                           "(a card with collision 'sdf' or 'mesh')", field="items.target")
    return it


def fill_container(planner: Any, layout: Layout, p: Placed, item: ItemSpec, t: Template) -> None:
    """Fill placed container ``p`` with ``item`` (a liquid or grain fill) made from template ``t``."""
    it = interior_of(planner, p)
    mat = planner.library.material(t.material)
    rho = F.bulk_density(mat, item.soft.kind)                          # grains: packed, not solid
    volume, how = F.amount_m3(item.soft, it, rho)
    level = F.check_capacity(it, volume, item.query, p.card)
    passage = 2.0 * it.radius_at(it.floor + 1e-3, it.floor + 2e-3)
    name = planner._name(item.query)                                            # noqa: SLF001
    sc, col, route, volume = _card(planner, name, item, t, it, volume, passage, p.card)
    region = item.soft.kind == "grain" and isinstance(sc.obj.shape, ct.ParticleSource)
    planner.soft.extras.append(Extra(sc, (p.key,), lambda placed, key=p.key, c=col, b=region:
                                     F.column_pose(placed[key], c, bottom=b), {}))
    planner.soft.watches.append(Watch("fill", name, p.key, it, p.body.rest,
                                      {"volume": volume, "density": rho}))
    planner._row(name, item, t, form="fill", **{"in": p.key},                    # noqa: SLF001
                 volume_ml=round(volume * 1e6, 2), amount=how,
                 mass_g=round(volume * rho * 1e3, 2),
                 level_mm=round((it.level_for(volume) - it.floor) * 1e3, 1),
                 capacity_ml=round(it.capacity_ml(F.MAX_FILL), 1), route=route,
                 start=col.to_dict())
    if level - it.floor > 0:
        planner.soft.settle_s = max(planner.soft.settle_s, SETTLE_FILL_S)


def _card(planner: Any, name: str, item: ItemSpec, t: Template, it: Interior, volume: float, passage: float,
          where: str) -> tuple[K.SoftCard, F.Column, str, float]:
    """The column card that fits, routed; grains first shrink to the router's rigid-grain budget if it asks."""
    grain = item.soft.kind == "grain"
    tmpl = planner.library.object(t.card) if (grain and t.card) else None
    rigid = grain and getattr(getattr(tmpl, "shape", None), "grain_diameter", None) is not None
    if rigid:                                     # grains with a size may route as rigid grains: mind the budget
        volume = cap_grains(planner, name, lambda v: _box_card(name, t, tmpl, v, passage, where), volume)
    margin = F.GRAIN_MARGIN if grain else F.sph_margin(passage)
    col = F.column(it, volume, margin=margin, box=rigid)
    if col is None:
        raise ComposeError(f"{volume * 1e6:.0f} ml of {item.query} does not stand as one resting column in "
                           f"{where}", "ask for less, or a wider container", field="items.soft.volume_ml")
    half = (col.hx, col.hy) if col.box else None
    if rigid:
        sc = K.grain_region(name, t.material, tmpl, half=half, height=col.height, passage=passage, where=where)
    else:
        make = K.grain_column if grain else K.liquid_column
        sc = make(name, t.material, radius=col.radius, height=col.height, passage=passage, where=where, half=half)
    planner.soft.add_card(sc, planner.work)
    planner.library.rescan()
    route, _nums, err = K.route_of(planner.library, name)
    if err:
        raise ComposeError(f"{name}: the engine cannot route it: {err[:300]}", "ask for less, or another material")
    return sc, col, route, volume


def _box_card(name: str, t: Template, tmpl: Any, volume: float, passage: float | None, where: str) -> K.SoftCard:
    a = volume ** (1.0 / 3.0)
    return K.grain_region(name, t.material, tmpl, half=(a / 2, a / 2), height=a, passage=passage, where=where)


def cap_grains(planner: Any, name: str, make: Any, volume: float) -> float:
    """``volume``, or less when the router finds more rigid grains than its budget (then it says so in the
    estimates). ``make(volume) -> SoftCard`` builds the trial card; the caller writes the final card."""
    planner.soft.add_card(make(volume), planner.work)
    planner.library.rescan()
    _route, _nums, err = K.route_of(planner.library, name)
    m = RIGID_BUDGET.search(err)
    if not err or m is None:
        return volume
    capped = volume * int(m.group(2)) / int(m.group(1)) * CAP_MARGIN
    planner.soft.estimate(f"{name}: capped at {capped * 1e6:.0f} ml (asked {volume * 1e6:.0f} ml): rigid grains are "
                          f"limited to {m.group(2)} per world by the engine's router")
    return capped


__all__ = ["cap_grains", "fill_container", "interior_of"]
