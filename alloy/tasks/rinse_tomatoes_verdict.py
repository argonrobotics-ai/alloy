# SPDX-License-Identifier: Apache-2.0
"""Verdict of task rinse_tomatoes (Gym-Anything's rinse-tomatoes): its eight gates and score, per world.

Gates (``task.py:743-754`` and the common ones ``task.py:677-692``):
- ``visibly_clean``: every tomato's gauge <= 1 µg/cm². Theirs is the worst 1 cm bin of its 2 mm chart. Here a tomato
  is one lump (WP9's lumped films) with no pattern, so the gauge is conservative: its remaining dust over 1 cm²
  (no 1 cm² bin can hold more than all of it). A pass here is a pass there; a tomato could fail here and pass there.
  Its mean load would be lenient: their run A3 kept a 21.3 µg/cm² bin on a tomato whose mean was about 1;
- ``none_lost``: every tomato's centre inside the colander's cavity frustum: z within (base - 1 cm, rim + 2 cm), radial
  within r(z) + 1 cm, r interpolated from the cavity's base radius to its mouth's (``task.py:553-562``);
- ``placed``: the colander's origin (their: the foot ring's underside) inside the clean bowl's frustum, the same test;
- ``drained``: the colander's free water (cavity + inner film above what clings) <= 10 mL (``params.py:251``);
- ``contained``: the cutting board's and the clean bowl's worst 1 cm gauges <= 1 µg/cm², floor water <= 50 mL
  (``task.py:618-642``). Their board reads soil + suspended + tracer, and their dust feeds suspended and tracer alike
  (``water/system.py:410-411``), so the board reads twice the dust its water carries; the bowl reads soil + tracer,
  once (``rinse_tomatoes.soil_gauge``);
- ``water_off_and_hands_free`` (no outlet over 1e-4 gpm, nothing held), ``intact_and_not_dropped`` (no tomato or the
  colander finally below 0.25 m), ``no_simulation_fault``.
Score (``task_set/scoring.py:67-76``): credit_n = reduction(start gauge, end gauge, 1) for a kept tomato, else 0, the
start gauge the worst 1 cm bin of its start dust (about 2.3x its mean, as theirs; ``rinse_tomatoes.start_dust``);
c = mean credit, p = placed x drained x released (released: ``water_off_and_hands_free`` passed, their
``scoring.py:42, 73``); units = (lost + excess(board, 1) + excess(bowl, 1) +
excess(floor mL, 50)) / 15; progress = clip((c + c p)/2 - units); 1 if every gate passes, else 0.9 x progress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks.rinse_score import FLOOR_WATER_ML, Verdict, excess, finish, hands_gate, reduction

GAUGE_LIMIT = 1.0e-5                  # kg/m² (1 µg/cm²)
DRAINED_ML = 10.0                     # drained_colander 10e-6 m³
DROP_Z = 0.25
UG_CM2 = 1.0e5                        # µg/cm² per kg/m²


@dataclass(frozen=True)
class Frustum:
    """A cavity in its body frame: z from ``base`` to ``rim``, radius ``r_base`` to ``r_rim`` (linear)."""

    base: float
    rim: float
    r_base: float
    r_rim: float

    def holds(self, p: np.ndarray) -> np.ndarray:
        """``[..., 3]`` points in the body frame -> inside (``task.py:553-562``: 1 cm below, 2 cm above, 1 cm out)."""
        p = np.asarray(p, np.float64)
        s = np.clip((p[..., 2] - self.base) / (self.rim - self.base), 0.0, 1.0)
        r = self.r_base + (self.r_rim - self.r_base) * s
        return ((p[..., 2] > self.base - 0.01) & (p[..., 2] < self.rim + 0.02)
                & (np.hypot(p[..., 0], p[..., 1]) < r + 0.01))


@dataclass
class TomatoesEnd:
    """One world at the verdict (gauges in kg/m², water in mL, heights in m)."""

    gauge0: np.ndarray                # [15] the worst 1 cm bin at the start
    gauge: np.ndarray                 # [15] remaining dust over 1 cm² (module docstring)
    kept: np.ndarray                  # [15] bool: in the colander
    placed: bool
    free_water_ml: float
    board_gauge: float
    bowl_gauge: float
    floor_ml: float
    lowest_z: float                   # the lowest tomato or the colander at the end
    faucet_open: bool
    held: bool
    books: bool


def verdict(e: TomatoesEnd) -> Verdict:
    """The gates and the score of one world."""
    n, released = len(e.gauge), not e.held and not e.faucet_open
    v = Verdict(criteria={
        "visibly_clean": (bool(np.all(e.gauge <= GAUGE_LIMIT)),
                          {"gauge_ug_cm2": [round(float(g) * UG_CM2, 3) for g in e.gauge],
                           "scale": "judged, stricter than Gym-Anything: each tomato's total remaining dust over "
                                    "1 cm2 (about 20x its mean on a ~20 cm2 tomato), not the worst 1 cm bin"}),
        "none_lost": (bool(np.all(e.kept)), int(n - np.count_nonzero(e.kept))),
        "placed": (e.placed, None),
        "drained": (e.free_water_ml <= DRAINED_ML, round(e.free_water_ml, 3)),
        "contained": (e.board_gauge <= GAUGE_LIMIT and e.bowl_gauge <= GAUGE_LIMIT and e.floor_ml <= FLOOR_WATER_ML,
                      {"board_ug_cm2": e.board_gauge * UG_CM2, "bowl_ug_cm2": e.bowl_gauge * UG_CM2,
                       "floor_mL": e.floor_ml,
                       "board": "judged, stricter: read as 2x the film's dust (soil + suspended + tracer)",
                       "floor": "judged, defined differently: the supply's lost water + water the ground caught + "
                                "drops that fell past the floor; Gym-Anything: floor sinks and floor-chart films"}),
        "water_off_and_hands_free": hands_gate(not e.faucet_open, e.held),
        "intact_and_not_dropped": (e.lowest_z >= DROP_Z, round(e.lowest_z, 4)),
        "no_simulation_fault": (e.books, "film water and soil, supply cross ledger, lumps' water and soil (1e-9); "
                                         "no contact overflow"),
    })
    credit = [reduction(float(g0) * UG_CM2, float(g) * UG_CM2, 1.0) if k else 0.0
              for g0, g, k in zip(e.gauge0, e.gauge, e.kept, strict=True)]
    c = float(np.mean(credit))
    p = float(e.placed and e.free_water_ml <= DRAINED_ML and released)
    units = (n - np.count_nonzero(e.kept) + excess(e.board_gauge, GAUGE_LIMIT) + excess(e.bowl_gauge, GAUGE_LIMIT)
             + excess(e.floor_ml, FLOOR_WATER_ML)) / n
    v.terms = {"credit": c, "p": p, "units": float(units)}
    return finish(v, (c + c * p) / 2.0 - units, e.books)


__all__ = ["DRAINED_ML", "Frustum", "GAUGE_LIMIT", "TomatoesEnd", "UG_CM2", "verdict"]
