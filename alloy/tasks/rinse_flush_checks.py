# SPDX-License-Identifier: Apache-2.0
"""flush-floor's gates, settle and score (:mod:`alloy.tasks.rinse_flush_floor`), ported from Gym-Anything's
surface-rinsing task set (``gym-anything-robotics/benchmarks/roboworld/environments/surface_rinsing``, read-only in
[internal note]; file:line):

- Gates, ``task.py:755-779``:
  - floor_flushed: the worst 1 cm gauge of adhered + suspended soil at most 10 µg/cm².
  - valve_parked: the valve body's origin within 0.15 m of its docked centre (``task.py:602-606``; ``scene.py:281,
    397``: the holster floor + 0.105 m).
  - contained: at most 1 g of water on the dunnage shelf. That is every packet striking a ``shelf_*`` or
    ``shelf_post`` collider, on any face (``water/system.py:309-310``). Here: the boards' top charts plus what the
    boards and posts caught (DC5's ``caught_by``).
  - water_off_and_hands_free: no outlet flowing, and no finger pad touching a movable body. Here only the outlet is
    judged: there is no hand, so the gate can never be True (audit AB-gym-3); it is False if the outlet runs.
  - no_simulation_fault: every measured value finite; water, soil species and the soil's chain closed to 1e-9 or 1 ng;
    no contact or constraint dropped by MuJoCo Warp.
- Settle, ``task.py:817-912``: hands-off for at least 2 s, then 1 s at a time until quiet, at most 30 s. Quiet means no
  outlet flowing, no scrap faster than 2 mm/s, and under 1 g/s change in the film and in what left it (drained or off
  an open edge). Each world is judged when it is quiet. Gym-Anything also waits for its water packets in the air; DC5's
  jets land within the step, so there are none.
- Score, ``scoring.py:77-87``: (a + a·k)/2 - excess(shelf, 1 g), 0.9 x progress on a fail.
  - a: the mean log reduction over the cells dirty at the start, read before the settle.
  - k: parked x water off, after it.
  - The shelf term is read before the settle (``scoring.py:47``).

Alloy's gates inside ``success`` (``EXTRA_GATES``, not counted in ``judged``): floor_water (at most 50 mL off
the bay) and scraps_on_drain (every scrap's centre within the drain; the scraps are Alloy's). ``success_ga`` leaves
them out: every judged Gym-Anything gate passes. ``judged`` says how many are judged.

Not checked:
- intact_and_not_dropped: vacuous in Gym-Anything, where the can is exempt (``task.py:393,691``): in the verdict as
  None, "vacuous in Gym-Anything too".
- hands-free: there is no robot; the scripted valve docks. Not judged (the gate judges the outlet part only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.tasks import rinse_score as rs
from alloy.tasks.paint_supply import supply_drift
from alloy.tasks.rinse_sink import GPM

RHO = 998.0                     # kg/m³, water
FLUSHED = 1.0e-4                # kg/m²: 10 µg/cm², the worst 1 cm gauge (one 10 mm cell) Gym-Anything allows
PARK_M = 0.15                   # m, valve body to its docked centre (Gym-Anything's park_m)
SHELF_G = 1.0                   # g of water on the dunnage shelf (Gym-Anything's)
SETTLE_MIN = 2.0                # s: Gym-Anything's settle_min_s
SETTLE_STEP = 1.0               # s between quiet checks
QUIET_V = 2.0e-3                # m/s: an item moving
QUIET_KG = 1.0e-3               # kg per second: water still running
EXTRA_GATES = ("floor_water", "scraps_on_drain")      # Alloy's own gates, not Gym-Anything criteria
GA_GATES = ("floor_flushed", "valve_parked", "contained", "water_off_and_hands_free", "intact_and_not_dropped",
            "no_simulation_fault")


@dataclass(frozen=True)
class Columns:
    """Where the scene's columns sit in ``SurfaceFlow.read()``: the valve's edge in ``supply_discharge`` and the
    shelf's occluder slots in ``supply_caught_by``: one tuple per world (:func:`shelf_slots`), or one tuple of slots
    for every world."""

    outlet: int = 1                     # Hose.network(): edges ("hose", "valve")
    shelf: tuple = ()

    def shelf_of(self, world: int) -> list[int]:
        """The shelf's slots in ``world``'s row of ``supply_caught_by``."""
        if self.shelf and isinstance(self.shelf[0], tuple):
            return list(self.shelf[world])
        return list(self.shelf)


@dataclass
class End:
    """One world's state for its verdict: the valve body's origin (m), the scraps' centres [n, 2] (m), the floor's
    gauge before the settle [nx, ny] (kg/m², the score's reduction), the water set on the floor at the start (m³, the
    wash film), the shelf's water before the settle (g, the score's term) and the overflow bits seen."""

    body: np.ndarray
    scraps: np.ndarray
    pre_gauge: np.ndarray
    set_volume: float
    pre_shelf_g: float = 0.0
    overflow: int = 0                   # 1: MuJoCo Warp dropped contacts or rows in some substep (0: none)
    settle_s: float = 0.0


def shelf_slots(model: Any, occluder_shapes: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Per world, the occluder slots (its row of ``Supply.occluder_shapes`` [W, S]) of the shelf's boards and posts."""
    labels = list(getattr(model, "shape_label", None) or [])
    return tuple(tuple(j for j, s in enumerate(row) if 0 <= s < len(labels)
                       and labels[s].split("/")[-1].startswith("shelf_")) for row in np.asarray(occluder_shapes))


def shelf_grams(read: dict, n_floor: int, cols: Columns, world: int) -> float:
    """Water on the shelf [g]: landed on the boards' top charts (the cells past the floor's) plus caught by them or
    by the posts."""
    on_top = float(np.asarray(read["supplied"])[world, n_floor:].sum())
    slots = cols.shelf_of(world)
    caught = float(np.asarray(read["supply_caught_by"])[world, slots].sum()) if slots else 0.0
    return (on_top + caught) * RHO * 1e3


def _floor(bay: Any, read: dict, key: str, world: int) -> np.ndarray:
    nx, ny = bay.cells
    v = np.asarray(read[key])[world, : nx * ny]
    return (v[:, 0] if v.ndim == 2 else v).reshape(nx, ny)


def gauge(bay: Any, read: dict, world: int) -> np.ndarray:
    """Gym-Anything's 1 cm gauge on the 10 mm floor: adhered plus suspended soil per cell [kg/m²]."""
    return _floor(bay, read, "soil", world) + _floor(bay, read, "species", world)


def quiet_books(read: dict) -> np.ndarray:
    """Per world [W, 2] kg: the film's water and what has left it (drained, off an open edge)."""
    film = np.asarray(read["film_volume"], dtype=float)
    away = np.asarray(read["drained_total_volume"], dtype=float) + np.asarray(read["spilled_volume"], dtype=float)
    return np.stack([film, away], 1) * RHO


def quiet(prev: np.ndarray, now: np.ndarray, q_off_gpm: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Per world: quiet over the last ``SETTLE_STEP`` (``prev``, ``now``: :func:`quiet_books`; ``speed``: the fastest
    scrap's, m/s)."""
    return ((np.abs(q_off_gpm) <= rs.OUTLET_OFF_GPM) & (speed <= QUIET_V)
            & (np.abs(now - prev) <= QUIET_KG * SETTLE_STEP).all(axis=1))


LANDED = "drops_species_landed"   # [W, K] kg the film's drops put back on landing (WP9's species drops), if any
DROP_OUT = ("species_pending", "species_to_sph", "species_in_flight", "species_landed", "species_floored",
            "species_to_tanks")     # where the drops' shed species are (WP9's drops.py identity)


def drops_soil_gap(read: dict, world: int) -> float:
    """The drop stage's soil identity's gap [kg], read as ``drops_<key>`` (WP9: shed + the lumps' dropped = pending +
    to SPH + in flight + landed + floored + to tanks); 0 without species drops."""
    if "drops_species_shed" not in read:
        return 0.0

    def get(key: str) -> float:
        arr = read.get(f"drops_{key}")
        return float(np.asarray(arr)[world, 0]) if arr is not None else 0.0

    return abs(get("species_shed") + get("lump_species_dropped") - sum(get(k) for k in DROP_OUT))


def _books(read: dict, end: End, world: int) -> tuple[bool, dict]:
    """no_simulation_fault's books: water (film ledger and DC5's cross ledger), the soil's species and its chain. Soil
    the drops carried off and put back on landing comes in again as ``species_supplied``; it is not new soil off the
    floor, so the chain takes it out (``LANDED``, 0 without species drops), and the drop stage's own soil identity
    must close too (:func:`drops_soil_gap`: soil lost in flight would otherwise leave the chain closed)."""
    put_in = float(read["supplied_volume"][world]) + float(read["pending_volume"][world]) + end.set_volume
    water_rel = max(abs(float(read["ledger"][world]) - end.set_volume) / max(put_in, 1e-30), supply_drift(read, world))
    start, left, removed = (float(np.asarray(read[k]).reshape(-1)[world])
                            for k in ("soil_start_mass", "soil_mass", "soil_removed_mass"))
    landed = read.get(LANDED)
    back = float(np.asarray(landed)[world, 0]) if landed is not None else 0.0
    into_film = float(np.asarray(read["species_supplied"])[world, :, 0].sum()) - back
    soil_kg = max(abs(start - left - removed), abs(removed - into_film),       # the state's and the chain's books,
                  drops_soil_gap(read, world))                                 # and the drops'
    sp_kg = float(np.asarray(read["species_ledger"])[world, 0])
    ok = rs.ledger_closed(water_rel, abs(sp_kg) / max(start, 1e-30), sp_kg) and \
        rs.ledger_closed(0.0, soil_kg / max(start, 1e-30), soil_kg)
    return ok, {"water_rel": water_rel, "soil_kg": soil_kg, "species_kg": sp_kg, "overflow": end.overflow}


def verdict(bay: Any, soil0: np.ndarray, read: dict, end: End, world: int, cols: Columns | None = None) -> dict:
    """One world's gates, progress and score (module docstring), read after its settle."""
    cols = cols or Columns()
    n0, g = soil0.size, gauge(bay, read, world)
    shelf_g = shelf_grams(read, n0, cols, world)
    floor_ml = float(np.asarray(read["spilled"])[world, :n0].sum()) * 1e6
    q_off = abs(float(np.asarray(read["supply_discharge"])[world, cols.outlet])) / GPM
    park = float(np.linalg.norm(end.body - np.asarray(bay.holster)))
    books, detail = _books(read, end, world)
    scrap_r = float(np.linalg.norm(end.scraps - np.asarray(bay.drain), axis=1).max()) if len(end.scraps) else 0.0
    measured = [g.tolist(), shelf_g, floor_ml, q_off, park, scrap_r, list(detail.values())]
    gates = {"floor_flushed": (float(g.max()) <= FLUSHED, round(float(g.max()) * 1e5, 4)),
             "valve_parked": (park <= PARK_M, round(park, 4)),
             "contained": (shelf_g <= SHELF_G, round(shelf_g, 4)),
             "water_off_and_hands_free": rs.hands_gate(q_off <= rs.OUTLET_OFF_GPM, None, outlet_gpm=q_off),
             "intact_and_not_dropped": (None, "vacuous in Gym-Anything too: the can is a floor item, exempt from the "
                                              "impact check (task.py:393,691)"),
             "floor_water": (floor_ml <= rs.FLOOR_WATER_ML, round(floor_ml, 3)),
             "scraps_on_drain": (scrap_r <= bay.drain_r, round(scrap_r, 4)),
             "no_simulation_fault": (books and rs.finite(measured) and end.overflow == 0, detail)}
    v = rs.Verdict(criteria=gates, extra=EXTRA_GATES)
    dirty = soil0 > FLUSHED                                     # cells: equal areas
    a = float(np.mean([rs.reduction(s0, gp, FLUSHED) for s0, gp in zip(soil0[dirty], end.pre_gauge[dirty],
                                                                          strict=True)])) if dirty.any() else 0.0
    k = float(gates["valve_parked"][0] and q_off <= rs.OUTLET_OFF_GPM)
    v.terms = {"a": a, "k": k, "shelf": rs.excess(end.pre_shelf_g, SHELF_G)}
    rs.finish(v, (a + a * k) / 2.0 - v.terms["shelf"], gates["no_simulation_fault"][0])
    start, left = (float(np.asarray(read[key]).reshape(-1)[world]) for key in ("soil_start_mass", "soil_mass"))
    ga = [gates[n][0] for n in GA_GATES if gates[n][0] is not None]      # judged Gym-Anything gates only
    return {"success": v.success, "success_ga": bool(ga) and all(ga), "failed": v.failed(),
            "judged": list(v.judged), "not_judged": v.not_judged(), "summary": v.summary(),
            "score": v.score, "progress": v.progress, "terms": v.terms, "settle_s": end.settle_s,
            "gates": {name: {"ok": ok, "value": val} for name, (ok, val) in gates.items()},
            "soil_mg": [round(start * 1e6, 3), round(left * 1e6, 3),
                        round(float(np.asarray(read["species_mass"])[world, 0]) * 1e6, 3)],
            "drained_L": round(float(read["drained_total_volume"][world]) * 1e3, 3),
            "sent_L": round(float(read["supply_sent"][world]) * 1e3, 3), "dirty_cells": dirty_cells(bay, g)}


def dirty_cells(bay: Any, g: np.ndarray, most: int = 10) -> list[list[float]]:
    """The worst cells over the limit: [x, y (m, bay frame), gauge (µg/cm²), off the can's side (m)], worst first."""
    over = np.argwhere(g > FLUSHED)
    if not len(over):
        return []
    order = np.argsort(-g[over[:, 0], over[:, 1]])[:most]
    c = bay.centres()
    out = []
    for i, j in over[order]:
        x, y = (float(v) for v in c[i, j])
        off = math.hypot(x - bay.can[0], y - bay.can[1]) - bay.can[2] if bay.can is not None else float("nan")
        out.append([round(x, 3), round(y, 3), round(float(g[i, j]) * 1e5, 2), round(off, 3)])
    return out


__all__ = ["Columns", "DROP_OUT", "End", "EXTRA_GATES", "FLUSHED", "GA_GATES", "LANDED", "PARK_M", "QUIET_KG",
           "QUIET_V", "RHO", "SETTLE_MIN", "SETTLE_STEP", "SHELF_G", "dirty_cells", "drops_soil_gap", "gauge", "quiet",
           "quiet_books", "shelf_grams", "shelf_slots", "verdict"]
