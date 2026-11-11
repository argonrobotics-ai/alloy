# SPDX-License-Identifier: Apache-2.0
"""Verdict of task rinse_glassware (Gym-Anything's rinse-glassware): its gates and score, per glass and per world.

Gates (``task.py:694-711`` and the common ones ``task.py:677-692``), every glass must pass:
- ``soap_free``: surfactant on the glass (faces + adsorbed + cavity) <= 10 µg, its worst 1 cm surfactant gauge
  <= 1 mg/m², foam < 1e-3 mL (no foam model in v1, WP15's ruling G7: the foam part is not judged);
- ``sanitized``: the tracker's clock >= 30 s and the soap carried into the sanitizer <= 10 µg;
- ``drained``: free water (cavity + inner film above what clings) <= 2 mL;
- ``placed``: in a rack compartment, mouth down: centre within 25 mm of a compartment's centre and inside the rack,
  the foot ``H`` above the rack's floor (-15 mm..+20 mm) (``task.py:525-538``), tilt > 160°;
- ``contained``: floor water <= 50 mL (the supply's ``lost`` ledger); the drainboard gauge is not judged (the board
  is not charted);
- ``intact_and_not_dropped``: no released glass faster than 1 m/s (body centre), none on the floor (below 0.25 m);
  contacts while held are not judged;
- ``water_off_and_hands_free``: the faucet closed (judged); the hands part is not judged; ``no_simulation_fault``.
A criterion with a part this port cannot read is None unless a judged part fails (``rinse_score``).
Score (``task_set/scoring.py:88-102``): per glass r = reduction(start µg, end µg, 10), z = clip(s/30),
t = placed and free water <= 2 mL and gauge <= 1 mg/m² and no foam, times released; credit (r + r z + r z t)/3, 0 if
damaged; progress = clip(mean credit - (damaged + board soil + floor excess)/glasses); 1 if every gate passes, else
0.9 x progress.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from alloy.tasks.rinse_score import Verdict, clip, excess, finish, hands_gate, partly, reduction

SOAP_UG = 10.0                        # surf_per_item (params.py:251)
SOAP_GAUGE = 1.0e-6                   # kg/m²
DRAINED_ML = 2.0                      # drained_glass 2e-6 m³
FLOOR_ML = 50.0
IMPACT = 1.0                          # m/s (params.py:223)
DROP_Z = 0.25
TILT_MIN = 160.0


@dataclass
class GlassEnd:
    """One glass at the verdict (µg, kg/m², mL, m, deg, s)."""

    soap_ug0: float                   # at the start
    soap_ug: float
    soap_gauge: float
    sanitized_s: float
    into_ug: float
    free_water_ml: float
    rack_xyz: np.ndarray              # origin in the rack frame (the rack's floor centre at its base)
    foot_up: float                    # the foot's height above the rack's base [m] (an inverted glass: origin + base)
    height: float                     # the glass's H
    tilt_deg: float
    peak_impact: float                # fastest body-centre speed once released [m/s] (theirs: contact normal speed)
    z: float
    foam_ml: float = 0.0


def in_cell(g: GlassEnd, cells: np.ndarray, floor: float) -> bool:
    """Centred in a compartment, inside the rack, foot H above the floor, mouth down (``task.py:525-538``)."""
    x, y = float(g.rack_xyz[0]), float(g.rack_xyz[1])
    near = float(np.min(np.hypot(cells[:, 0] - x, cells[:, 1] - y)))
    return (near < 0.025 and abs(x) < 0.25 and abs(y) < 0.25 and -0.015 < g.foot_up - (floor + g.height) < 0.02
            and g.tilt_deg > TILT_MIN)


def reported(g: GlassEnd) -> GlassEnd:
    """Their report's rounding, which their gates read (``task.py:484-504``): seconds and degrees to 2 places, µg and
    mL to 3 (1,500 samples of 0.02 s sum to 29.99999999999945 s, a 30.00 s soak in their report)."""
    return replace(g, soap_ug=round(g.soap_ug, 3), into_ug=round(g.into_ug, 3), sanitized_s=round(g.sanitized_s, 2),
                   free_water_ml=round(g.free_water_ml, 3), tilt_deg=round(g.tilt_deg, 2))


def verdict(glasses: list[GlassEnd], cells: np.ndarray, floor: float, floor_ml: float, outlet_off: bool,
            gripping: bool, books: bool) -> Verdict:
    """One world: every glass's gates (on their report's rounding), then the score. ``cells`` [25, 2] compartment
    centres in the rack frame; ``outlet_off``: the faucet is closed; ``gripping``: the script still grips."""
    glasses = [reported(g) for g in glasses]
    released = outlet_off and not gripping
    placed = [in_cell(g, cells, floor) for g in glasses]
    damaged = [g.peak_impact > IMPACT or g.z < DROP_Z for g in glasses]
    soap_ok = [g.soap_ug <= SOAP_UG and g.soap_gauge <= SOAP_GAUGE for g in glasses]
    v = Verdict(criteria={
        "soap_free": (partly(all(soap_ok), None),
                      {"soap_ug": [round(g.soap_ug, 3) for g in glasses],
                       "foam": "not judged: no foam model (Gym-Anything: foam < 1e-3 mL)"}),
        "sanitized": (all(g.sanitized_s >= 30.0 and g.into_ug <= SOAP_UG for g in glasses),
                      [round(g.sanitized_s, 2) for g in glasses]),
        "drained": (all(g.free_water_ml <= DRAINED_ML for g in glasses), [round(g.free_water_ml, 3) for g in glasses]),
        "placed": (all(placed), {"in_cell": placed, "rack": "static here (Gym-Anything: a free 1.6 kg body)"}),
        "contained": (partly(floor_ml <= FLOOR_ML, None),
                      {"floor_mL": floor_ml, "drainboard_gauge": "not judged: the board is not charted"}),
        "intact_and_not_dropped": (partly(not any(damaged), None),
                                   {"peak_speed": [round(g.peak_impact, 3) for g in glasses],
                                    "contacts_while_held": "not judged: only the released glass's body-centre "
                                                           "speed and the drop height are read (Gym-Anything: "
                                                           "every contact's normal speed, held or not)"}),
        "water_off_and_hands_free": hands_gate(outlet_off, gripping),
        "no_simulation_fault": (books, "film water, film + wall species, supply cross ledger (rinse_books, 1e-9); "
                                       "no contact overflow"),
    })
    credits = []
    for g, p, d in zip(glasses, placed, damaged, strict=True):
        r, z = reduction(g.soap_ug0, g.soap_ug, SOAP_UG), clip(g.sanitized_s / 30.0)
        t = float(p and g.free_water_ml <= DRAINED_ML and g.soap_gauge <= SOAP_GAUGE and g.foam_ml < 1e-3) * released
        credits.append(0.0 if d else (r + r * z + r * z * t) / 3.0)
    units = (sum(damaged) + excess(floor_ml, FLOOR_ML)) / len(glasses)
    v.terms = {"credit": float(np.mean(credits)), "units": units}
    return finish(v, float(np.mean(credits)) - units, books)


__all__ = ["GlassEnd", "in_cell", "reported", "verdict"]
