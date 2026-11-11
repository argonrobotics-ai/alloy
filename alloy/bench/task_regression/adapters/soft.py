# SPDX-License-Identifier: Apache-2.0
"""Soft-asset pack checks through `alloy.check` (cards, overlap, rest drift, health, penetration, energy), on CPU, no
wall budget.

- SOFT-1: each film and cloth object dropped 0.5 mm onto the library table top (alloy.testing.soft_film), at the
  check's default size (2 worlds x 0.5 s).
- SOFT-2: each liquid / paste / grain fill resting in the open box liquid_box_56 on the table top, 1 world x 0.3 s
  (the scene card of [internal note], rebuilt here so the suite runs repo code only).
- SOFT-3: each soft food / tissue object through ``alloy.check.check_object`` (its drop test on a generated floor) at
  the check's CPU size (2 worlds x 0.5 s), run in full (no wall budget)."""

from __future__ import annotations

from collections.abc import Sequence

#: SOFT-2 fill card -> the route the router gives it (task-regression-sources.md, SOFT-2 row 1)
FILLS: dict[str, str] = {
    "water_box_200ml": "liquid_sph", "glycerol_80_box_100ml": "liquid_sph", "oil_olive_box_150ml": "liquid_sph",
    "honey_box_150ml": "paste_mpm", "sauce_ketchup_box_150ml": "paste_mpm",
    "sand_box_200ml": "grain_mpm", "rice_box_200ml": "grain_mpm",
}
#: SOFT-2's viscoplastic food pastes (main 442896119; sources SOFT-2 row 1, "Added 2026-11-06"): fill -> route
PASTES: dict[str, str] = {name: "paste_mpm" for name in (
    "sauce_mayonnaise_box_150ml", "batter_bread_box_150ml", "batter_cake_box_150ml", "puree_fruit_box_150ml",
    "tomato_concentrate_box_150ml", "yogurt_box_150ml", "peanut_butter_box_150ml")}
#: SOFT-3 library object -> route (task-regression-sources.md, SOFT-3 row; SOFT-3 messages WP6 when it adds one)
FOODS: dict[str, str] = {
    "cucumber_slicing": "soft_vbd", "fruit_tomato": "soft_vbd", "fruit_peach": "soft_vbd",
    "tissue_pad_skin": "soft_vbd", "tissue_pad_fat": "soft_vbd", "gel_block_ballistic": "soft_vbd",
    "foam_pad_pu": "soft_vbd", "sponge_cellulose": "soft_vbd", "dough_ball": "paste_mpm",
}
BOX_SCENE = """schema: alloy.scene/1
robots: []
objects:
  - {{object: table_top_plywood, pose: [0, 0, -0.009, 0, 0, 0, 1]}}
  - {{object: liquid_box_56, pose: [0, 0, 0, 0, 0, 0, 1]}}
  - {{object: {fill}, pose: [0, 0, 0, 0, 0, 0, 1]}}
worlds: 1
"""


def film_drop(seeds: Sequence[int], device: str) -> dict[str, float]:
    """Every alloy.testing.soft_film object on route cloth_film (plastic film sheets, the grocery bag)."""
    return _drops("cloth_film", device)


def cloth_drop(seeds: Sequence[int], device: str) -> dict[str, float]:
    """Every alloy.testing.soft_film object on route cloth_vbd (towels, cloth sheets)."""
    return _drops("cloth_vbd", device)


def _drops(route: str, device: str) -> dict[str, float]:
    from alloy.bench.task_regression.adapters import keep_json
    from alloy.testing import soft_film as SF

    rows = [SF.check_row(name, device=device, budget_s=None) for name, r in SF.OBJECTS.items() if r == route]
    keep_json(f"soft_{route}_rows.json", rows)
    return _tally(rows)


def sph_fills(seeds: Sequence[int], device: str) -> dict[str, float]:
    """Water, glycerol 80 % and olive oil in the box (route liquid_sph)."""
    return _fills("liquid_sph", device)


def paste_fills(seeds: Sequence[int], device: str) -> dict[str, float]:
    """Honey and ketchup in the box (route paste_mpm)."""
    return _fills("paste_mpm", device)


def grain_fills(seeds: Sequence[int], device: str) -> dict[str, float]:
    """Sand and rice in the box (route grain_mpm)."""
    return _fills("grain_mpm", device)


def viscoplastic_fills(seeds: Sequence[int], device: str) -> dict[str, float]:
    """Mayonnaise and six shear-thinning food pastes in the box (route paste_mpm; five on the Herschel-Bulkley
    solver), PASTES."""
    return _fills("paste_mpm", device, table=PASTES, tag="viscoplastic")


def _fills(route: str, device: str, table: dict[str, str] = FILLS, tag: str = "") -> dict[str, float]:
    """Each fill of ``table`` on ``route`` through alloy.check; a fill passes when the check is ok and the router
    still gives it ``route`` (the pass rule's "routes as listed")."""
    import time

    import alloy.check as AC
    from alloy.bench.task_regression.adapters import keep_json

    rows = []
    for fill in (f for f, r in table.items() if r == route):
        t0 = time.perf_counter()
        rep = AC.check(BOX_SCENE.format(fill=fill), worlds=1, seconds=0.3, device=device, budget_s=None)
        row = _row(fill, rep, time.perf_counter() - t0)
        row["route"] = _routed(fill, device)
        if row["route"] != route:
            row["ok"] = False
            row["errors"].append(f"routed {row['route']}, expected {route}")
        rows.append(row)
    keep_json(f"soft_{tag or route}_rows.json", rows)
    return _tally(rows)


def _routed(fill: str, device: str) -> str:
    """The route the router gives ``fill`` in the box scene (one more assembly, a few seconds: alloy.check does not
    report routes, and the catalog leaves a box fill's route empty)."""
    from alloy.check.source import load_source
    from alloy.scene.assembler import SceneAssembler

    src = load_source(BOX_SCENE.format(fill=fill), None)
    decision = SceneAssembler(src.lib).assemble(src.card, device, world_count=1).report.decisions.get(fill)
    return str(decision.route) if decision is not None else "missing"


def soft_vbd_drops(seeds: Sequence[int], device: str) -> dict[str, float]:
    """SOFT-3's food and tissue objects on route soft_vbd (FOODS)."""
    return _object_drops("soft_vbd", device)


def dough_drops(seeds: Sequence[int], device: str) -> dict[str, float]:
    """SOFT-3's dough on route paste_mpm (FOODS)."""
    return _object_drops("paste_mpm", device)


def _object_drops(route: str, device: str) -> dict[str, float]:
    import time

    import alloy.check as AC
    from alloy.bench.task_regression.adapters import keep_json

    rows = []
    for name in (n for n, r in FOODS.items() if r == route):
        t0 = time.perf_counter()
        with _full_run():
            rep = AC.check_object(name, device=device)
        rows.append(_row(name, rep, time.perf_counter() - t0))
    keep_json(f"soft3_{route}_rows.json", rows)
    return _tally(rows)


def _full_run():
    """check_object runs alloy.check.api.check at its 25 s wall budget (no keyword for it), which cuts a busy host's
    run short; this lifts the budget for the call (the drop module reads ``check`` from alloy.check.api at call
    time)."""
    import contextlib
    import functools

    import alloy.check.api as API

    @contextlib.contextmanager
    def lifted():
        orig = API.check
        API.check = functools.partial(orig, budget_s=None)
        try:
            yield
        finally:
            API.check = orig

    return lifted()


def _row(name: str, rep, wall_s: float) -> dict:
    return {"object": name, "ok": rep.ok, "checks": rep.to_dict().get("checks", {}),
            "errors": [f.message for f in rep.errors()], "warnings": [f.message for f in rep.warnings()],
            "wall_s": round(wall_s, 1)}


def _tally(rows: list[dict]) -> dict[str, float]:
    return {"objects": float(len(rows)), "failed": float(sum(not r["ok"] for r in rows)),
            "errors": float(sum(len(r["errors"]) for r in rows)),
            "warnings": float(sum(len(r["warnings"]) for r in rows))}
