# SPDX-License-Identifier: Apache-2.0
"""Adhesive cards for the ``Bonds`` plug-in: an effective fracture energy G(theta) per substrate, plus formation and a
lap-shear cap.

**G(theta) is a calibration, not a prediction.** A Kendall peel with one G gives F180/F90 = 0.5. The 3M 7818 sheet
gives 1.24 on stainless steel and 1.11-1.69 over its six substrates. So each card stores two effective energies:
- ``g90 = F90/b``;
- ``g180 = 2·F180/b``.

Between 90 and 180 deg, G is linear in c = 1 - cos(theta). Below 90 deg it is held at ``g90``, because there is no
data there. A peel at 90 or 180 deg therefore reproduces the sheet's row by construction. The laws check that the
engine's front geometry (angle, width, advance) keeps it so.

Sources: [internal note] notes/wp13-bonds-sources.md. The 3M 7818 sheet, May 2017, ASTM D3330, 10 min dwell at room
temperature, gives these values (N/cm, x100 = N/m):

| Substrate | 90 deg | 180 deg |
|---|---|---|
| stainless steel | 3.8 | 4.7 |
| polycarbonate | 4.0 | 5.1 |
| polypropylene | 1.8 | 2.0 |
| glass | 3.7 | 5.7 |
| HDPE | 1.8 | 2.6 |
| LDPE | 1.3 | 2.2 |

The 7818 face stock is 0.08 mm PET on 0.02 mm of 310 acrylic.

ESTIMATES:
- **``p_form`` = 100 kPa.** 3M VHB's "15 psi" firm application pressure, applied to 7818, which states no rubdown
  pressure.
- **``dwell`` = 0.05 s.** Contact time under a roller. The data's 10 min to 72 h wet-out growth is not modelled: v1
  formation is binary.
- **``tau`` = 0.5 MPa.** The lap-shear cap, a generic acrylic PSA value. It only bounds near-0 deg pulls.

**3M 371 on kraft liner** (box-sealing tape on a corrugated carton, SOFT-1's open-carton): the 371 sheet (Oct 2006,
doc 70-0710-0274-8) gives "Adhesion to Steel: 45 oz./in. width (44 N/100 mm width) D-3330" (180 deg: g180 = 880 J/m²
on steel), BOPP backing, hot-melt rubber-resin adhesive, "Tape Thickness: 1.9 mil (0.048 mm)". On corrugated board
the bond is stronger than the liner, which tears in its plane: the peel energy is the liner's internal bond, Scott
bond 200-350 J/m² for kraftliners (TAPPI T569; Billerud Pure Performance 350 J/m²). ESTIMATE: g90 = g180 = 200 J/m²
(one G, Kendall: F/b = 200 N/m at 90 deg, 100 N/m at 180 deg), the reference task set's kraft-liner value, at the
range's low end. ``p_form``, ``dwell`` and ``tau`` as for 7818 (ESTIMATES).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SHEET_7818 = "3M 7818 datasheet (May 2017), ASTM D3330, 10 min RT; [internal note] notes/wp13-bonds-sources.md section 0"
P_FORM_EST = 1.0e5        # Pa, ESTIMATE: 3M VHB "15 psi" firm application pressure (VHB 4920 sheet)
DWELL_EST = 0.05          # s, ESTIMATE: contact time under a roller; binary formation in v1
TAU_EST = 5.0e5           # Pa, ESTIMATE: lap-shear cap, generic acrylic PSA


@dataclass(frozen=True)
class Adhesive:
    """One adhesive on one substrate: effective fracture energies at 90 and 180 deg [J/m²], plus formation and the
    lap-shear cap."""

    name: str
    g90: float
    g180: float
    p_form: float = P_FORM_EST
    dwell: float = DWELL_EST
    tau: float = TAU_EST
    source: str = SHEET_7818

    def __post_init__(self) -> None:
        if not (self.g90 > 0.0 and self.g180 > 0.0):
            raise ValueError(f"adhesive {self.name!r}: g90 and g180 must be > 0 (got {self.g90}, {self.g180})")
        if self.p_form < 0.0 or self.dwell < 0.0 or self.tau <= 0.0:
            raise ValueError(f"adhesive {self.name!r}: p_form, dwell >= 0 and tau > 0 required")

    def g(self, c: float) -> float:
        """Effective G at c = 1 - cos(theta) (module docstring)."""
        return g_of_c(self.g90, self.g180, c)

    def peel_force(self, theta_deg: float, width: float) -> float:
        """The steady peel force [N] at ``theta_deg`` for a front ``width`` wide: G(theta)·b/(1 - cos theta)."""
        c = 1.0 - math.cos(math.radians(theta_deg))
        return self.g(c) * width / max(c, 1e-9)


def g_of_c(g90: float, g180: float, c: float) -> float:
    """G at c = 1 - cos(theta): ``g90`` for c <= 1, then linear in c up to ``g180`` at c = 2."""
    return g90 if c <= 1.0 else g90 + (g180 - g90) * (min(c, 2.0) - 1.0)


def _card(name: str, n90: float, n180: float) -> Adhesive:
    return Adhesive(name, g90=100.0 * n90, g180=2.0 * 100.0 * n180)


TAPE_7818 = {           # substrate -> card, from the N/cm rows above
    "stainless_steel": _card("3m_7818/stainless_steel", 3.8, 4.7),
    "polycarbonate": _card("3m_7818/polycarbonate", 4.0, 5.1),
    "polypropylene": _card("3m_7818/polypropylene", 1.8, 2.0),
    "glass": _card("3m_7818/glass", 3.7, 5.7),
    "hdpe": _card("3m_7818/hdpe", 1.8, 2.6),
    "ldpe": _card("3m_7818/ldpe", 1.3, 2.2),
}
ROWS_7818 = {k: (v.g90, v.g180 / 2.0) for k, v in TAPE_7818.items()}      # F/b [N/m] at 90 and 180 deg (laws)

SHEET_371 = ("3M Scotch Box Sealing Tape 371 TDS (Oct 2006), ASTM D3330 on steel 44 N/100 mm; on kraft liner the "
             "liner's Scott bond, TAPPI T569 (module docstring)")
TAPE_371 = {           # substrate -> card
    "kraft_liner": Adhesive("3m_371/kraft_liner", g90=200.0, g180=200.0, source=SHEET_371),
}
CARDS = {"3m_7818": TAPE_7818, "3m_371": TAPE_371}


def adhesive(name: str) -> Adhesive:
    """A card by name: ``"<tape>/<substrate>"``, tapes and substrates the keys of ``CARDS`` and its tables."""
    head, _, sub = name.partition("/")
    table = CARDS.get(head, {})
    if sub in table:
        return table[sub]
    raise KeyError(f"no adhesive card {name!r}; cards: {sorted(t + '/' + k for t, tab in CARDS.items() for k in tab)}")


__all__ = ["Adhesive", "CARDS", "ROWS_7818", "TAPE_371", "TAPE_7818", "adhesive", "g_of_c"]
