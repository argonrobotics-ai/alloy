# SPDX-License-Identifier: Apache-2.0
"""Verdict of rinse-sponge (Gym-Anything's pot sink): the gates and the score on measured state.

"The scrub sponge on the back ledge is full of wash solution. Rinse it under the faucet until it releases no more
suds, wring it out and put it back on the ledge" (``task_set/catalogue.py:31-33``; spec: [internal note] notes
wp8-results/surface/rinse_spec_more.md §3).

Sources (Gym-Anything's RoboWorld ``surface_rinsing`` environment; file:line):
- gates (v1 contract, ``task.py:731-733``, ``task_set/evaluator.py:97-98``; six with the common three of
  ``rinse_score``): ``no_suds`` = the pores' dissolved detergent <= 0.02 g/L (0 when the pores hold under 1 µL;
  foam and adsorbed detergent not counted, ``task.py:595``), ``wrung_out`` = pore water / the uncompressed pore
  capacity (79.1 mL) <= 0.60 (``params.py:253``: sponge_suds_g_L, sponge_wrung), ``sponge_on_ledge`` = the sponge's
  origin (its scour face's centre) on the deck behind the three bowls (``task.py:578-587``, compartments=True:
  x between the wash bowl's left wall and the sanitizer's right wall, y between the bowls' back edge and the wall,
  z from 5 mm under the rim to 55 mm over it);
- score (``task_set/scoring.py:49-54``): s = reduction(c, 0.02 g/L), w = clip((1 - fill)/0.40),
  p = sponge_on_ledge x released; progress = clip((s + s w + s w p)/3); 1 if every gate passes, else 0.9 x progress.
  GA reads s and w live, before the settle, and the gates after it (``task_set/evaluator.py:110-118``): ``verdict``'s
  ``live``.

Judged here: 5 of GA's 6 criteria (audit AB-gym-3; ``rinse_score``: a gate is True or False only when read from the
simulation's state, None when this port does not judge it; success and the score count the judged parts only, and
each row says how many that is). ``water_off_and_hands_free`` is not judged: the faucet's flow is, but GA's "nothing
held" is any finger pad on a movable body (``task.py:227,609``) and this port's privileged hand has no pads
(``rinse_score.hands_gate``). The script's own grip can fail the gate, never pass it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks.can_wash_score import counted
from alloy.tasks.rinse_score import (OUTLET_OFF_GPM, Verdict, clip, finish, finite, hands_gate, ledger_closed,
                                     reduction)
from alloy.tasks.rinse_sink import BOWL, BOWLS, PotSink

SUDS_KG_M3 = 0.02                     # g/L = kg/m³ (sponge_suds_g_L)
WRUNG = 0.60                          # pore fill (sponge_wrung)
V_FULL = 0.9 * 0.111 * 0.066 * 0.012  # m³: phi A H_c, 79.1 mL uncompressed (sponge.py:12-17,90)
EMPTY_M3 = 1.0e-9                     # pores under 1 µL read c = 0 (task.py:595)
LEDGE_BELOW, LEDGE_ABOVE = 0.005, 0.055   # m around the rim (compartments=True)


@dataclass(frozen=True)
class Ledge:
    """The deck behind the three bowls, as an open box (lo, hi) in the scene's frame."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    @staticmethod
    def behind(sink: PotSink, bowl_back_y: float, wall_y: float) -> Ledge:
        """The deck of ``sink`` between its bowls' back edge and the wall (the scene gives both)."""
        x1 = sink.x0 + len(BOWLS) * BOWL[0]
        return Ledge((sink.x0, bowl_back_y, sink.rim_z - LEDGE_BELOW), (x1, wall_y, sink.rim_z + LEDGE_ABOVE))

    def holds(self, p: np.ndarray) -> bool:
        return bool(np.all((np.asarray(p) > self.lo) & (np.asarray(p) < self.hi)))


@dataclass
class SpongeReadout:
    """Measured state of every world at one instant (``[W]``, or ``[W, 3]`` for ``pos``)."""

    pore_soap_kg: np.ndarray          # detergent dissolved in the pore water
    pore_m3: np.ndarray               # pore water
    pos: np.ndarray                   # the sponge's origin (scour face's centre), scene frame
    faucet_gpm: np.ndarray            # every outlet's flow, the largest
    held: np.ndarray                  # the script's own grip on the sponge, not a state read: it can fail
                                      # hands-free, never pass it
    intact: np.ndarray
    water_rel: np.ndarray
    species_rel: np.ndarray
    species_kg: np.ndarray
    finite: np.ndarray


def pore_conc(r: SpongeReadout, w: int) -> float:
    """Dissolved detergent in the pores [kg/m³ = g/L]; 0 under 1 µL of pore water: their contract
    (``task.py:595``, ``p.sp[0]/p.vol if p.vol > 1e-9 else 0``) kept as is, not a rule of this port."""
    v = float(r.pore_m3[w])
    return float(r.pore_soap_kg[w]) / v if v > EMPTY_M3 else 0.0


def gates(r: SpongeReadout, w: int, ledge: Ledge) -> dict[str, tuple[bool | None, object]]:
    """World ``w``'s six gates (module doc); None: not judged."""
    c, fill = pore_conc(r, w), float(r.pore_m3[w]) / V_FULL
    closed = bool(r.finite[w]) and ledger_closed(float(r.water_rel[w]), float(r.species_rel[w]),
                                                 float(r.species_kg[w]))
    return {
        "no_suds": (c <= SUDS_KG_M3, {"pore_g_l": c}),
        "wrung_out": (fill <= WRUNG, {"fill": fill, "pore_ml": float(r.pore_m3[w]) * 1e6}),
        "sponge_on_ledge": (ledge.holds(r.pos[w]), [float(x) for x in r.pos[w]]),
        "water_off_and_hands_free": hands_gate(bool(r.faucet_gpm[w] <= OUTLET_OFF_GPM), bool(r.held[w]),
                                               faucet_gpm=float(r.faucet_gpm[w])),
        "intact_and_not_dropped": (bool(r.intact[w]), bool(r.intact[w])),
        "no_simulation_fault": (closed, {"water_rel": float(r.water_rel[w]), "species_rel": float(r.species_rel[w]),
                                         "species_kg": float(r.species_kg[w]), "finite": bool(r.finite[w])}),
    }


def verdict(start: SpongeReadout, r: SpongeReadout, w: int, ledge: Ledge, live: SpongeReadout | None = None) -> Verdict:
    """World ``w``'s gates on the settled state ``r`` and its score's s and w on ``live`` (the state before the
    settle; None: ``r``), against the reset state ``start`` (module doc). GA's "x released" counts the gate's judged
    parts (``can_wash_score.counted``)."""
    v = Verdict(criteria=gates(r, w, ledge))
    lv = live if live is not None else r
    s = reduction(pore_conc(start, w), pore_conc(lv, w), SUDS_KG_M3)
    wrung = clip((1.0 - float(lv.pore_m3[w]) / V_FULL) / (1.0 - WRUNG))
    p = float(v.criteria["sponge_on_ledge"][0]) * counted(v.criteria["water_off_and_hands_free"][0])
    v.terms = {"rinsed": s, "wrung": wrung, "returned": p}
    ok = bool(r.finite[w]) and bool(v.criteria["no_simulation_fault"][0]) and finite(v.terms)
    return finish(v, (s + s * wrung + s * wrung * p) / 3.0, ok)


__all__ = ["Ledge", "SpongeReadout", "V_FULL", "gates", "pore_conc", "verdict"]
