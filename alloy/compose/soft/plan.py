# SPDX-License-Identifier: Apache-2.0
"""Soft items of one scene, start to end: the cloth and piles that take room on the table go into the layout with
the rigid objects; once everything is placed, liquids and grains go into their containers, pours get their stream
(and held vessel), and cables are laid and plugged in. Every generated card is routed by the engine's router
before it is used, and a request it cannot build is reported with the router's reason and what to change.

Example::

    >>> planner = SoftPlanner(spec, catalog, library, work, rng, "scene")        # doctest: +SKIP
    >>> layout_items = planner.layout_items()                                     # doctest: +SKIP
    >>> planner.finish(layout)                                                     # doctest: +SKIP
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alloy.cards import types as ct
from alloy.compose.body import Body
from alloy.compose.errors import ComposeError
from alloy.compose.layout import Layout, Placed
from alloy.compose.soft import cable as C
from alloy.compose.soft import cards as K
from alloy.compose.soft import sheet as SH
from alloy.compose.soft.materials import Template, resolve
from alloy.compose.soft.pour import plan_pour
from alloy.compose.soft.probe import Watch
from alloy.compose.soft.scene import Extra, SoftScene, Weld
from alloy.compose.soft.vessel import cap_grains, fill_container
from alloy.compose.spec import ItemSpec, SceneSpec

log = logging.getLogger("alloy.compose")

SIZE_SCALE = {"small": 0.6, "large": 1.5}
SETTLE_CABLE_S = 0.3
CABLE_TURNS = (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0)   # rad: where a cable leaves its objects, in turn
PORT_TRIES = 3                                   # ports drawn for a cable plugged in at one end


@dataclass(frozen=True)
class LayoutItem:
    """A soft item that takes room on the table: placed like an object (``mode`` "layout"), over its target
    ("drape") or over the table edge ("hang")."""

    item: int
    query: str
    card: str
    body: Body
    target: str | None
    mode: str
    thickness: float = 0.0


def _box_size(item: ItemSpec, card: Any) -> tuple[float, float, float]:
    """(length, width, thickness) of a thin-box sheet template at the size the item asks (or a size word's scale)."""
    hx, hy, hz = card.shape.size
    thick = float(card.features.thickness.value) if card.features.thickness is not None else 2 * hz
    length, width = max(2 * hx, 2 * hy), min(2 * hx, 2 * hy)
    scale = next((SIZE_SCALE[w] for w in item.words if w in SIZE_SCALE), 1.0)
    length, width = item.soft.length_m or length * scale, item.soft.width_m or (
        width * (item.soft.length_m / length) if item.soft.length_m else width * scale)
    return length, width, thick


class SoftPlanner:
    """Makes and places the soft items of one scene (``name``) into ``soft`` (a ``SoftScene``)."""

    def __init__(self, spec: SceneSpec, catalog: Any, library: Any, work: Path, rng: np.random.Generator,
                 name: str) -> None:
        self.spec, self.catalog, self.library, self.work, self.rng, self.name = spec, catalog, library, work, rng, name
        self.soft = SoftScene()
        self._templates: dict[int, Template] = {}
        self._n: dict[str, int] = {}
        self._draped: set[str] = set()                 # layout keys of drapes (a cable may run under one)
        self._laid: list[np.ndarray] = []              # keep-out polygons of the cables laid so far (world xy)

    # --- items placed with the layout ---------------------------------------------------------------------------

    def layout_items(self) -> list[LayoutItem]:
        """Cloth sheets and grain piles, with their cards written; items that cannot be made are reported."""
        out: list[LayoutItem] = []
        for idx, item in enumerate(self.spec.items):
            s = item.soft
            if s is None or not (s.kind == "cloth" or (s.kind == "grain" and s.form == "pile")):
                continue
            try:
                out += self._cloth(idx, item) if s.kind == "cloth" else self._pile(idx, item)
            except ComposeError as exc:
                self._miss(item, exc.message, exc.fix)
        self.library.rescan()
        return out

    def place(self, layout: Layout, li: LayoutItem, key: str, target_key: str | None) -> Placed | None:
        """Put a drape or a hanging sheet down (layout items go through ``layout.place``)."""
        if li.mode == "drape":
            self._draped.add(key)
            return SH.drape(layout, li.body, li.card, key, target_key, li.item, li.thickness) if target_key else None
        return SH.hang(layout, li.body, li.card, key, li.item, li.thickness)

    def _cloth(self, idx: int, item: ItemSpec) -> list[LayoutItem]:
        t = self._template(idx, item)
        card = self.library.object(t.card)
        mode = {"flat": "layout", "folded": "layout", "draped": "drape", "hanging": "hang"}[item.soft.form]
        if mode == "drape" and item.target is None:
            raise ComposeError(f"{item.query}: draped over nothing", "say what it covers: 'a towel draped over a "
                               "bowl'", field="items.target")
        length, width, thick = self._mesh_size(item, t.card, card) if isinstance(card.shape, ct.MeshSource) else \
            _box_size(item, card)
        out = []
        for _ in range(item.count):
            sc, body = self._sheet(item, card, length, width, thick)
            self.soft.add_card(sc, self.work)
            self._row(sc.name, item, t, form=item.soft.form, size_m=[round(length, 4), round(width, 4)],
                      thickness_mm=round(thick * 1e3, 3), **({"layers": 2} if item.soft.form == "folded" else {}))
            self.soft.watches.append(Watch("cloth", sc.name))
            out.append(LayoutItem(idx, item.query, sc.name, body, item.target if mode != "hang" else None, mode,
                                  thick))
        if mode != "layout":
            self.soft.settle_s = max(self.soft.settle_s, SH.SETTLE_DRAPE_S)
        return out

    def _mesh_size(self, item: ItemSpec, name: str, card: Any) -> tuple[float, float, float]:
        """(length, width, thickness) of a sheet whose shape is its own mesh (the scored kraft sheet: its crease is
        in the mesh), at its own size; asking for another size, or for it folded in half, raises with what to ask."""
        sx, sy, _sz = self.catalog.info(name).size
        length, width = max(sx, sy), min(sx, sy)
        if item.soft.form == "folded":
            raise ComposeError(f"{item.query}: {name} has its crease in its mesh, so the builder does not fold it in "
                               "half", f"ask for it flat ('{item.query} on a table'): it folds on its crease when "
                               "pushed; or ask for kraft paper folded in half", field="items.soft.form")
        if item.soft.length_m or item.soft.width_m or any(w in SIZE_SCALE for w in item.words):
            raise ComposeError(f"{item.query}: {name} comes at its own size ({length:.2f} x {width:.2f} m)",
                               "leave the size out, or ask for kraft paper at that size", field="items.soft")
        return length, width, float(card.features.thickness.value)

    def _sheet(self, item: ItemSpec, card: Any, length: float, width: float, thick: float) -> tuple[Any, Any]:
        """(card, layout body) of one sheet: the template at this size, a mesh sheet as it is, or folded in half as a
        two-layer mesh."""
        name = self._name(item.query)
        if isinstance(card.shape, ct.MeshSource):
            return K.mesh_sheet(name, card), SH.sheet_body(name, length, width, thick)
        if item.soft.form != "folded":
            return K.cloth_sheet(name, card, length, width), SH.sheet_body(name, length, width, thick)
        from alloy.scene.emitters.deformable import CLOTH_VERTS_TARGET  # noqa: PLC0415 - the grid sheets' density

        spacing = math.sqrt(length * width / CLOTH_VERTS_TARGET)
        v, f = SH.folded_mesh(length, width, thick, spacing)
        sc = K.cloth_mesh_card(name, card, v, f, thick, f"a {length:.2f} x {width:.2f} m sheet folded in half")
        span = v.max(axis=0) - v.min(axis=0)
        self.soft.estimate(f"{item.query}: folded in half as two layers {SH.FOLD_GAP:.0f} thicknesses apart, "
                           f"joined at the fold; the fold is its rest shape")
        return sc, SH.sheet_body(name, float(span[0]), float(span[1]), float(span[2]) + thick)

    def _pile(self, idx: int, item: ItemSpec) -> list[LayoutItem]:
        from alloy.compose.soft.fill import amount_m3, bulk_density  # noqa: PLC0415

        t = self._template(idx, item)
        mat = self.library.material(t.material)
        rho = bulk_density(mat, "grain")                    # a pile packs; the card holds the grain density
        volume, how = amount_m3(item.soft, None, rho)
        angle = mat.plastic.friction_angle
        tmpl = self.library.object(t.card) if t.card else None
        out = []
        for _ in range(item.count):
            name = self._name(item.query)
            volume = self._grain_card(name, t, tmpl, volume, angle.value)
            r, h = SH.pile_size(volume, float(angle.value))
            self.soft.watches.append(Watch("pile", name, expect={"angle": float(angle.value), "volume": volume,
                                                                 "density": rho}))
            self._row(name, item, t, form="pile", volume_ml=round(volume * 1e6, 2), amount=how,
                      angle_of_repose_deg=round(math.degrees(float(angle.value)), 2),
                      angle_source=f"{t.material} plastic.friction_angle ({angle.source}: {angle.note})"[:200],
                      cone_mm={"radius": round(r * 1e3, 1), "height": round(h * 1e3, 1)})
            out.append(LayoutItem(idx, item.query, name, SH.pile_body(name, r, h), item.target, "layout"))
        self.soft.settle_s = max(self.soft.settle_s, SH.SETTLE_PILE_S)
        return out

    def _grain_card(self, name: str, t: Template, tmpl: Any, volume: float, angle: float) -> float:
        """Write the pile's start box (shrunk to the router's rigid-grain budget if it asks); returns the volume."""
        def make(v: float) -> K.SoftCard:
            half, height = SH.pile_start(v, SH.pile_size(v, angle)[0])
            return K.grain_region(name, t.material, tmpl, half=(half, half), height=height)

        volume = cap_grains(self, name, make, volume)
        self.soft.add_card(make(volume), self.work)
        self.library.rescan()
        _route, _nums, err = K.route_of(self.library, name)
        if err:
            raise ComposeError(f"{name}: the engine cannot route it: {err[:300]}", "ask for a smaller amount or "
                               "another grain material")
        return volume

    # --- after the layout ------------------------------------------------------------------------------------------

    def finish(self, layout: Layout) -> SoftScene:
        """Fills, pours and cables, once every object is placed; returns the scene's soft parts."""
        for idx, item in enumerate(self.spec.items):
            s = item.soft
            if s is None:
                continue
            try:
                if s.kind in ("liquid", "grain") and s.form == "fill":
                    self._fills(layout, idx, item)
                elif s.kind == "liquid" and s.form == "pour":
                    plan_pour(self, layout, idx, item)
                elif s.kind == "cable":
                    self._cables(layout, idx, item)
            except ComposeError as exc:
                self._miss(item, exc.message, exc.fix)
        self.library.rescan()
        return self.soft

    def _fills(self, layout: Layout, idx: int, item: ItemSpec) -> None:
        t = self._template(idx, item)
        targets = self.targets(layout, item.target)
        if not targets:
            raise ComposeError(f"{item.query}: its container {item.target!r} was not placed", "see the missing list "
                               "for why, or ask for another container")
        for p in targets:
            fill_container(self, layout, p, item, t)

    def _cables(self, layout: Layout, idx: int, item: ItemSpec) -> None:
        t = self._template(idx, item)
        card = self.library.object(t.card)
        radius = item.soft.diameter_mm * 5e-4 if item.soft.diameter_mm else float(card.shape.radius)
        base = float(np.linalg.norm(np.subtract(card.shape.points[-1], card.shape.points[0])))
        length = item.soft.length_m or base
        ends = self._ends(layout, item)
        for _ in range(item.count):
            pts, keys = self._lay(layout, ends, length, radius, item.soft.form == "straight")
            name = self._name(item.query)
            sc = K.cable_curve(name, card, pts - pts[0], radius, length if len(ends) < 2 else None)
            pts = pts[0] + np.asarray(sc.obj.shape.points)             # as laid: a free end is cut to the length
            self.soft.add_card(sc, self.work)
            self.library.rescan()
            route, nums, err = K.route_of(self.library, name)
            if err:
                raise ComposeError(f"{name}: the engine cannot route a {length:.2f} m cable: {err[:300]}",
                                   "ask for a shorter or thicker cable")
            links = int(nums.get("links", 0))
            self._laid += C.footprint(pts, radius)                     # the next cables keep clear of this one
            start = tuple(float(v) for v in pts[0])
            self.soft.extras.append(Extra(sc, tuple(keys), lambda placed, s=start: s + (0.0, 0.0, 0.0, 1.0), {}))
            self.soft.watches.append(Watch("cable", name, expect={"length": C.path_length(pts)}))
            for k, key in enumerate(keys):
                self.soft.welds.append(Weld(name, 0 if k == 0 else max(links - 1, 0), key))
            if radius != float(card.shape.radius):
                self.soft.estimate(f"{name}: gauge {2e3 * radius:.2g} mm; bending and twist stiffness scaled from "
                                   f"{t.card} by (r/r0)^4")
            self.soft.estimate(f"{name}: the cable's rest shape is the curve it was laid in (it keeps that set)")
            self._row(name, item, t, form=item.soft.form, length_m=round(C.path_length(pts), 4),
                      diameter_mm=round(2e3 * radius, 3), route=route, links=links, attached=list(keys))
            self.soft.settle_s = max(self.soft.settle_s, SETTLE_CABLE_S)

    def _ends(self, layout: Layout, item: ItemSpec) -> list[Placed]:
        """The objects a cable's ends are fixed to, one per named end, never one object for both ("from a mug to a
        mug" takes two mugs)."""
        ends: list[Placed] = []
        for q in item.soft.attach:
            found = self.targets(layout, q)
            if not found:
                raise ComposeError(f"{item.query}: an object it is attached to was not placed", "see the missing list")
            free = [p for p in found if all(p.key != e.key for e in ends)]
            if not free:
                raise ComposeError(f"{item.query}: both its ends are on the one {q} in the scene",
                                   f"ask for a second {q}, or end the {item.query} on another object")
            ends.append(free[0])
        return ends

    def _lay(self, layout: Layout, ends: list[Placed], length: float, radius: float, straight: bool
             ) -> tuple[np.ndarray, list[str]]:
        """A free path for one cable, clear of the cables laid before it: between its two ends (leaving them further
        round at each try, ``CABLE_TURNS``), from its one end, or anywhere on the table."""
        keys = [e.key for e in ends]
        under = tuple(keys) + tuple(k for k in self._draped if k in layout.placed and layout.placed[k].target in keys)
        centre = np.mean(layout.area, axis=0)
        pts = None
        if len(ends) == 2:
            for turn in CABLE_TURNS:                   # the ends turn opposite ways: the cable shifts to one side
                a, _ = C.port(ends[0], ends[1].outline.mean(axis=0), radius, turn)
                b, _ = C.port(ends[1], ends[0].outline.mean(axis=0), radius, -turn)
                pts = C.between(layout, a, b, max(length, 1.15 * float(np.linalg.norm(b[:2] - a[:2]))), radius,
                                under, self._laid)
                if pts is not None:
                    break
        elif len(ends) == 1:
            for _ in range(PORT_TRIES):
                toward = centre + layout.rng.normal(0.0, 0.1, 2)
                a, heading = C.port(ends[0], toward, radius)
                pts = C.loose_path(layout, length, radius, straight=straight, start=a, heading=heading, skip=under,
                                   avoid=self._laid)
                if pts is not None:
                    break
        else:
            pts = C.loose_path(layout, length, radius, straight=straight, avoid=self._laid)
        if pts is None:
            raise ComposeError(f"no free path for a {length:.2f} m cable on the table", "ask for a shorter cable, a "
                               "tidier table or fewer objects")
        return pts, keys

    # --- helpers -----------------------------------------------------------------------------------------------------

    def targets(self, layout: Layout, query: str | None) -> list[Placed]:
        """Placed objects of the item whose query is ``query``."""
        idx = next((i for i, it in enumerate(self.spec.items) if it.query == query), None)
        return [p for p in layout.placed.values() if p.item == idx]

    def _template(self, idx: int, item: ItemSpec) -> Template:
        if idx not in self._templates:
            t = resolve(item, self.catalog)
            for note in t.estimated:
                self.soft.estimate(note)
            self._templates[idx] = t
        return self._templates[idx]

    def _name(self, query: str) -> str:
        k = self._n[query] = self._n.get(query, 0) + 1
        return f"{self.name}_{re.sub(r'[^a-z0-9]+', '_', query.lower())}_{k}"

    def _row(self, name: str, item: ItemSpec, t: Template, **extra: Any) -> None:
        self.soft.rows.append({"name": name, "card": name, "item": item.query, "kind": item.soft.kind,
                               "material": t.material, "template": t.card, **extra})

    def _miss(self, item: ItemSpec, message: str, fix: str = "") -> None:
        self.soft.missing.append(f"{item.query}: {message}" + (f" ({fix})" if fix else ""))
        log.info("compose soft: %s not built: %s", item.query, message)


__all__ = ["LayoutItem", "SoftPlanner"]
