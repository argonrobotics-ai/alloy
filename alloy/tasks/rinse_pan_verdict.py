# SPDX-License-Identifier: Apache-2.0
"""Verdict of task rinse_pan (Gym-Anything's scrub-rinse-pan): its nine gates and the progress-v1 score.

Gates (``task.py:714-733`` and the common ones ``task.py:677-692``):
- ``visibly_clean``: the worst 1 cm adhered-soil gauge over the pan <= 1 µg/cm²;
- ``soap_free``: surfactant on the pan (faces + adsorbed + cavity) per pan area <= 10 µg/0.085 m², the worst 1 cm
  surfactant gauge <= 1 mg/m², foam < 1e-3 mL (no foam model in v1, WP15's ruling G7: the foam part is not judged);
- ``sanitized``: the tracker's clock >= 30 s, and the soap carried into the sanitizer per area under the same limit;
- ``drained``: free water (cavity + inner film above what clings) <= 5 mL;
- ``placed``: the pan's origin on the shelf over the clean drainboard (x over the board, y within 0.32 m of the back
  wall, 1.50 < z < 1.65) and mouth down (tilt > 160°) (``task.py:570-576``);
- ``sponge_on_ledge``: the sponge's scour face (their origin, ``rinse_pan_scene.Station.scour_face``) on the deck
  behind the three bowls (``task.py:578-587``);
- ``water_off_and_hands_free`` (no outlet, the faucet or the spray valve, over 1e-4 gpm in that world, and no hand
  holding; the hands part is only the script's grip flag, so it can fail the gate and not pass it),
  ``intact_and_not_dropped`` (the pan and sponge above 0.25 m; the pan is not breakable),
  ``no_simulation_fault`` (``rinse_books``: the film's water, each species with the walls' load and the supply's cross
  ledger; the soil's own book; no contact overflow in any substep).
The gates read their report's rounding (``reported``, ``task.py:484-504``). Score: ``rinse_tracker.pan_score``
(progress-v1, ``task.py:369-382, 782-802``; its Z reads the rounded clock).

Coordinates: theirs have the bowls' front at y 4.676 and the back wall at 5.4; here the apron is y = 0, so the back
wall is at 0.724 and x is theirs (the wash bowl starts at 2.6096).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from alloy.tasks.rinse_tracker import VRL_KG_M2, pan_score
from alloy.tasks.rinse_score import Verdict, hands_gate, partly
from alloy.tasks.rinse_pan_scene import Y_BACK
from alloy.tasks.rinse_sink import BOWL, PotSink

INCH = 0.0254
SOAP_PER_AREA = 10.0e-9 / 0.085       # kg/m², surf_per_item over their reference area (params.py:251; task.py:719)
SOAP_GAUGE = 1.0e-6                   # kg/m², surf_gauge (params.py:251)
DRAINED_ML = 5.0                      # drained_bowl 5e-6 m³ (params.py:251)
TILT_MIN = 160.0                      # deg, 180 - place_tilt_deg (params.py:254)
SHELF_Z = (1.50, 1.65)
DROP_Z = 0.25


def clean_board_x(sink: PotSink) -> tuple[float, float]:
    """The clean drainboard: 24 in after the sanitizer bowl (layout.py:34-38)."""
    x0 = sink.x0 + 3 * BOWL[0]
    return x0, x0 + 24 * INCH


@dataclass
class PanEnd:
    """One world at the verdict (SI units; masses kg, areas m², volumes mL)."""

    gauge: float                      # worst adhered-soil gauge
    soap: float                       # surfactant on the pan: faces + adsorbed + cavity
    soap_gauge: float                 # worst 1 cm surfactant gauge
    area: float                       # the pan chart's area
    sanitized_s: float
    into_sanitizer: float             # soap carried into the sanitizer by the entry the clock depends on
    free_water_ml: float
    pan_pos: np.ndarray
    tilt_deg: float                   # the pan's axis from up (180: mouth down)
    sponge_pos: np.ndarray
    outlet_running: bool
    held: bool
    books: bool
    foam_ml: float = 0.0


def on_shelf(sink: PotSink, p: np.ndarray) -> bool:
    x0, x1 = clean_board_x(sink)
    return bool(x0 < p[0] < x1 and Y_BACK - 0.32 < p[1] < Y_BACK and SHELF_Z[0] < p[2] < SHELF_Z[1])


def on_ledge(sink: PotSink, p: np.ndarray) -> bool:
    """On the deck behind the three bowls; its height ceiling admits the sponge on any face (task.py:578-587)."""
    return bool(sink.x0 < p[0] < sink.x0 + 3 * BOWL[0] and BOWL[1] < p[1] < Y_BACK
                and sink.rim_z - 0.005 < p[2] < sink.rim_z + 0.055)


def reported(end: PanEnd) -> PanEnd:
    """Their report's rounding, which their gates read (``task.py:484-504``): surfactant and carry-in to 3 places in
    µg, the surfactant gauge to 4 in mg/m², the soil gauge to 4 in µg/cm², free water to 3 in mL, tilt and the clock
    to 2."""
    return replace(end, soap=round(end.soap * 1e9, 3) * 1e-9, soap_gauge=round(end.soap_gauge * 1e6, 4) * 1e-6,
                   gauge=round(end.gauge * 1e5, 4) * 1e-5, into_sanitizer=round(end.into_sanitizer * 1e9, 3) * 1e-9,
                   free_water_ml=round(end.free_water_ml, 3), tilt_deg=round(end.tilt_deg, 2),
                   sanitized_s=round(end.sanitized_s, 2))


def verdict(sink: PotSink, end: PanEnd, clean: float, rinsed: float) -> Verdict:
    """One world's gates (on their report's rounding) and score; ``clean`` and ``rinsed`` are progress-v1's C and R
    (``rinse_tracker``)."""
    end = reported(end)
    limit = SOAP_PER_AREA * end.area
    v = Verdict(criteria={
        "visibly_clean": (end.gauge <= VRL_KG_M2, round(end.gauge * 1e5, 4)),
        "soap_free": (partly(end.soap <= limit and end.soap_gauge <= SOAP_GAUGE, None),
                      {"foam": "not judged: no foam model (Gym-Anything: foam < 1e-3 mL)",
                       "soap_ug": round(end.soap * 1e9, 3), "limit_ug": round(limit * 1e9, 3),
                       "gauge_mg_m2": round(end.soap_gauge * 1e6, 4)}),
        "sanitized": (end.sanitized_s >= 30.0 and end.into_sanitizer <= limit,
                      {"sanitized_s": round(end.sanitized_s, 2), "into_ug": round(end.into_sanitizer * 1e9, 3)}),
        "drained": (end.free_water_ml <= DRAINED_ML, round(end.free_water_ml, 3)),
        "placed": (on_shelf(sink, end.pan_pos) and end.tilt_deg > TILT_MIN,
                   {"position": np.round(end.pan_pos, 4).tolist(), "tilt_deg": round(end.tilt_deg, 2)}),
        "sponge_on_ledge": (on_ledge(sink, end.sponge_pos), np.round(end.sponge_pos, 4).tolist()),
        "water_off_and_hands_free": hands_gate(not end.outlet_running, end.held, outlet=end.outlet_running),
        "intact_and_not_dropped": (min(end.pan_pos[2], end.sponge_pos[2]) >= DROP_Z, None),
        "no_simulation_fault": (end.books, "film water, species (walls included), supply cross ledger, soil; no "
                                           "contact overflow in any substep"),
    })
    return pan_score(v, clean, rinsed, end.sanitized_s, end.books)


__all__ = ["DRAINED_ML", "DROP_Z", "PanEnd", "SHELF_Z", "SOAP_GAUGE", "SOAP_PER_AREA", "Y_BACK", "clean_board_x",
           "on_ledge", "on_shelf", "reported", "verdict"]
