# SPDX-License-Identifier: Apache-2.0
"""Verdict of rinse-garbage-can (Gym-Anything's can-wash bay): the gates and the score on measured state.

"The 32 gal garbage can in the can-wash bay has been scrubbed with detergent. Rinse its inside and outside free of
soap and loosened debris with the hose, empty the rinse water into the floor drain, and leave the can upright"
(``task_set/catalogue.py:42-45``; spec: [internal note] notes wp8-results/surface/rinse_spec_more.md §2).

Sources (Gym-Anything's RoboWorld ``surface_rinsing`` environment; file:line):
- gates ``soap_free``, ``visibly_clean``, ``emptied_upright``: ``task.py:755-767``; ``valve_parked``, ``contained``:
  ``task.py:778-779``; the common three (``rinse_score``): ``task.py:677-692``;
- soap_free: the can's mean detergent (film + adsorbed + cavity pool over its charted area) <= 10 µg per 0.085 m² of
  ware (1.18e-7 kg/m²), its worst 1 cm detergent gauge <= 1 mg/m², foam < 1e-3 mL (``params.py:251-254`` CRITERIA);
  no foam model in v1 (WP15's G7 ruling): the foam part is not judged (below);
- visibly_clean: worst 1 cm adhered-soil gauge, inside and out, <= 1 µg/cm² (``params.py:158`` VRL);
- emptied_upright: free water <= 100 mL (``drained_can``) and the can's axis < 20° from vertical (``place_tilt_deg``);
  free water = cavity pool + the inner film above what clings, ``free_film`` (``task.py:507-523``);
- valve_parked: the hose valve <= 0.15 m from its holster point; contained: dunnage shelf water <= 1 g;
- score (``task_set/scoring.py:112-120``): r = reduction(detergent, limit = area x 1.18e-7 kg/m²,
  ``task_set/evaluator.py:84-85``), d = reduction(soil gauge, 1 µg/cm²), c = (r + d)/2, e = emptied_upright,
  k = valve_parked x released; progress = clip((c + c e + c e k)/3 - excess(shelf g, 1)); 1 if every gate passes,
  else 0.9 x progress (``rinse_score.finish``). GA reads r, d and the shelf live, before the settle, and the gates
  after it (``task_set/evaluator.py:110-118,130-136``): ``verdict``'s ``live``.

Judged here: 5 of GA's 8 criteria (audit AB-gym-3; ``rinse_score``: a gate is True or False only when read from the
simulation's state, None when this port does not judge it; success and the score count the judged parts only, and
each row says how many that is). Not judged, each with its judged parts still able to fail it:
- ``soap_free``: the mean and the gauge are judged; foam is not (no foam model);
- ``water_off_and_hands_free``: the hose's flow is judged. GA's "nothing held" is any finger pad on a movable body
  (``task.py:227,609``); this port's hand is privileged and its valve kinematic, so there are no pads to read
  (``rinse_score.hands_gate``). The script's own state can fail the gate, never pass it: a hand still gripping, or
  the robot stage's valve, fixed in the FR3's hand and never released;
- ``intact_and_not_dropped``: vacuous in GA too (the can is a floor item and nothing here is breakable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks.rinse_score import (OUTLET_OFF_GPM, Verdict, excess, finish, finite, hands_gate, ledger_closed,
                                     partly, reduction)

SOAP_PER_AREA = 10.0e-9 / 0.085       # kg/m²: 10 µg per 0.085 m² item (surf_per_item / 0.085)
SOAP_GAUGE = 1.0e-6                   # kg/m², 1 mg/m² (surf_gauge)
VRL = 1.0e-5                          # kg/m², 1 µg/cm²
DRAINED_ML = 100.0                    # drained_can 100e-6 m³
TILT_DEG = 20.0                       # place_tilt_deg
VALVE_PARK_M = 0.15
SHELF_G = 1.0
H_RET_MIN = 50.0e-6                   # m: free_film never lets less than 50 µm cling
SIN_MIN = 0.04                        # a level floor clings h_break / sqrt(0.04) = 5 h_break
NO_FOAM = "not judged: no foam model (Gym-Anything: foam < 1e-3 mL)"
VACUOUS = "vacuous in Gym-Anything too: the can is a floor item and nothing here is breakable"


@dataclass
class CanReadout:
    """Measured state of every world, ``[W]`` each, at one instant (start, live or settled)."""

    soap_kg: np.ndarray               # detergent on and in the can: film + adsorbed + cavity pool
    soap_gauge: np.ndarray            # worst 1 cm detergent gauge on the can [kg/m²] (film + adsorbed)
    soil_gauge: np.ndarray            # worst 1 cm adhered-soil gauge, inside and out [kg/m²]
    free_ml: np.ndarray               # cavity pool + free_film [mL]
    tilt_deg: np.ndarray              # the can's axis from vertical
    valve_m: np.ndarray               # hose valve to its holster point
    shelf_g: np.ndarray               # water caught by the dunnage shelf
    hose_gpm: np.ndarray              # hose outlet flow
    held: np.ndarray                  # the script's own hold on a body (a hand's grip; a tool fixed in a robot's
                                      # hand), not a state read: it can fail hands-free, never pass it
    water_rel: np.ndarray             # water ledger residual, relative
    species_rel: np.ndarray           # detergent and soil ledger residual, relative
    species_kg: np.ndarray            # the same, absolute
    finite: np.ndarray                # every field finite and no engine fault
    area_m2: float                    # the can's charted area (the mean's denominator)


def free_film(depth: np.ndarray, normal_z: np.ndarray, area: np.ndarray, h_break: np.ndarray) -> float:
    """Water on the can's inner cells that would still run if tipped [m³] (``task.py:507-523``): per cell the film
    above h_ret = max(h_break / sqrt(max(sin slope, 0.04)), 50 µm), slope from the cell's world normal z."""
    sin_a = np.sqrt(np.clip(1.0 - normal_z * normal_z, 0.0, 1.0))
    h_ret = np.maximum(h_break / np.sqrt(np.maximum(sin_a, SIN_MIN)), H_RET_MIN)
    return float((np.maximum(depth - h_ret, 0.0) * area).sum())


def counted(ok: bool | None) -> float:
    """A gate as a factor of the score: 0 if it fails, 1 if it passes or is not judged (judged parts only)."""
    return 0.0 if ok is False else 1.0


def row(v: Verdict, settle_s: float) -> dict:
    """A world's verdict as a run's row: success over the judged gates, how many of GA's criteria that is
    (``judged``: [judged, all]), the ones not judged, and each gate (``ok`` None: not judged)."""
    return {"success": v.success, "failed": v.failed(), "judged": list(v.judged), "not_judged": v.not_judged(),
            "summary": v.summary(), "score": v.score, "progress": v.progress, "terms": v.terms, "settle_s": settle_s,
            "gates": {g: {"ok": ok, "value": val} for g, (ok, val) in v.criteria.items()}}


def gates(r: CanReadout, w: int) -> dict[str, tuple[bool | None, object]]:
    """World ``w``'s eight gates (module doc); None: not judged."""
    mean = float(r.soap_kg[w]) / r.area_m2
    closed = bool(r.finite[w]) and ledger_closed(float(r.water_rel[w]), float(r.species_rel[w]),
                                                 float(r.species_kg[w]))
    return {
        "soap_free": (partly(bool(mean <= SOAP_PER_AREA), bool(r.soap_gauge[w] <= SOAP_GAUGE), None),
                      {"mean_mg_m2": mean * 1e6, "gauge_mg_m2": float(r.soap_gauge[w]) * 1e6, "foam": NO_FOAM}),
        "visibly_clean": (bool(r.soil_gauge[w] <= VRL), {"gauge_ug_cm2": float(r.soil_gauge[w]) * 1e5}),
        "emptied_upright": (bool(r.free_ml[w] <= DRAINED_ML and r.tilt_deg[w] < TILT_DEG),
                            {"free_ml": float(r.free_ml[w]), "tilt_deg": float(r.tilt_deg[w])}),
        "valve_parked": (bool(r.valve_m[w] <= VALVE_PARK_M), float(r.valve_m[w])),
        "contained": (bool(r.shelf_g[w] <= SHELF_G), float(r.shelf_g[w])),
        "water_off_and_hands_free": hands_gate(bool(r.hose_gpm[w] <= OUTLET_OFF_GPM), bool(r.held[w]),
                                               hose_gpm=float(r.hose_gpm[w])),
        "intact_and_not_dropped": (None, VACUOUS),
        "no_simulation_fault": (closed, {"water_rel": float(r.water_rel[w]), "species_rel": float(r.species_rel[w]),
                                         "species_kg": float(r.species_kg[w]), "finite": bool(r.finite[w])}),
    }


def verdict(start: CanReadout, r: CanReadout, w: int, live: CanReadout | None = None) -> Verdict:
    """World ``w``'s gates on the settled state ``r`` and its score's detergent, debris and shelf terms on ``live``
    (the state before the settle; None: ``r``), against the reset state ``start`` (module doc). GA's "x released"
    counts the gate's judged parts (``counted``)."""
    v = Verdict(criteria=gates(r, w))
    lv = live if live is not None else r
    limit = r.area_m2 * SOAP_PER_AREA
    rinsed = reduction(float(start.soap_kg[w]), float(lv.soap_kg[w]), limit)
    debris = reduction(float(start.soil_gauge[w]), float(lv.soil_gauge[w]), VRL)
    c = (rinsed + debris) / 2.0
    e = float(v.criteria["emptied_upright"][0])
    k = float(v.criteria["valve_parked"][0]) * counted(v.criteria["water_off_and_hands_free"][0])
    v.terms = {"rinsed": rinsed, "debris_removed": debris, "clean": c, "emptied_upright": e, "docked_released": k}
    ok = bool(r.finite[w]) and bool(v.criteria["no_simulation_fault"][0]) and finite(v.terms)
    return finish(v, (c + c * e + c * e * k) / 3.0 - excess(float(lv.shelf_g[w]), SHELF_G), ok)


__all__ = ["CanReadout", "counted", "free_film", "gates", "row", "verdict"]
