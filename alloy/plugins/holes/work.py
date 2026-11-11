# SPDX-License-Identifier: Apache-2.0
"""Work materials for drilling, one source per value (a sourced table in the plug-in, not a material-card group): the
work-material factor Kd of Shaw and Oxford's thrust and torque for twist drilling (``WORK_MATERIALS``), and the blow
energy per removed volume for rotary-hammer drilling of concrete (``PERCUSSIVE``).

Kd enters Machinery's Handbook form of Shaw and Oxford's model, T = 2 Kd f^0.8 d^0.8 B W + Kd d^2 J W [lbf] and
M = Kd f^0.8 d^1.8 A W [in lbf] (``alloy.plugins.drilling``). The table gives one Kd for thrust and torque, except for
the alloys where the source prints two.

Source of every row: Machinery's Handbook, "Work Material Factor, Kd, for Drilling with a Sharp Drill", as reproduced
in Dynomax, "Book of Spindles" part 2, p. 56, Table 14 (read in the PDF on 2026-11-08,
https://dynospindles.com/vault/technical/Book-of-Spindles-Part-2.pdf). Rows quoted as printed, e.g. "Steel, 200 Bhn
24,000" and "Most Aluminum Alloys 7,000"; the stainless and titanium pairs carry the footnote "Values based upon a
limited number of tests". The same page gives the chisel-edge factors the drill's defaults use ("For drills of
standard design, use c/d = 0.18": A 1.085, B 1.355, J 0.030).

Except ``pine_southern``: Kd fitted to the one chip thickness McMillin & Woodson (1972) measured (see the row). The
source gives no feed law, so the row assumes Shaw and Oxford's exponents and the Handbook's factors; a softwood
stands in for the Douglas-fir and plywood the tasks drill (no drilling source for those was found).

``PERCUSSIVE`` rows give the energy per removed volume e_v at a few bit diameters (linear in between, held past the
ends); a rotary hammer of blow energy E at blow rate f deepens a hole of diameter d at E f / (e_v pi d^2 / 4).

Example::

    >>> work_material("steel_200bhn").kd_thrust
    24000.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.plugins.errors import PluginError

TABLE_SOURCE = ("Machinery's Handbook Kd table, reproduced in Dynomax 'Book of Spindles' part 2 p. 56 Table 14 "
                "(https://dynospindles.com/vault/technical/Book-of-Spindles-Part-2.pdf)")


@dataclass(frozen=True)
class WorkMaterial:
    """One row: Kd [lbf/in^1.6] for thrust and for torque, and the row as the source prints it."""

    name: str
    kd_thrust: float
    kd_torque: float
    quote: str
    source: str = TABLE_SOURCE


def _row(name: str, kd: float, quote: str, torque: float | None = None) -> WorkMaterial:
    return WorkMaterial(name, float(kd), float(kd if torque is None else torque), quote)


MCMILLIN = ("McMillin & Woodson 1972, 'Moisture content of southern pine as related to thrust, torque, and chip "
            "formation in boring', Forest Products Journal 22(11):55-59 "
            "(https://research.fs.usda.gov/download/treesearch/8240.pdf), abstract")
PINE = WorkMaterial(
    "pine_southern", 449.0, 307.0,
    "1-inch spur machine bit, southern pine of specific gravity 0.53, 2,400 rpm, chips 0.020 inch thick: thrust "
    "across the grain average 138 pounds, torque across the grain average 33 inch-pounds. Kd fitted: f = 0.040 "
    "in/rev (two cutters), 138 = Kd (2 f^0.8 B W + J W), 33 = Kd f^0.8 A W with B 1.355, J 0.030, A 1.085, W 1.30 "
    "(Shaw-Oxford exponents assumed: one chip thickness measured)", MCMILLIN)


WORK_MATERIALS: dict[str, WorkMaterial] = {m.name: m for m in (
    _row("steel_1117", 12000, "AISI 1117 (Resulfurized free machining mild steel) 12,000"),
    _row("steel_200bhn", 24000, "Steel, 200 Bhn 24,000"),
    _row("steel_300bhn", 31000, "Steel, 300 Bhn 31,000"),
    _row("steel_400bhn", 34000, "Steel, 400 Bhn 34,000"),
    _row("cast_iron_150bhn", 14000, "cast Iron, 150 Bhn 14,000"),
    _row("aluminium", 7000, "Most Aluminum Alloys 7,000"),
    _row("magnesium", 4000, "Most Magnesium Alloys 4,000 (printed 'Magnedum')"),
    _row("brass", 14000, "Most Brasses 14,000"),
    _row("brass_leaded", 7000, "Leaded Brass 7,000"),
    _row("stainless_316", 35000, "Austenitic Stainless Steel (Type 316) 35,000 for Thrust, 24,000 for Torque "
         "(values based upon a limited number of tests)", torque=24000),
    _row("titanium_t16a", 29000, "Titanium Alloy T16A 29,000 for Thrust, 18,000 for Torque (values based upon a "
         "limited number of tests)", torque=18000),
    PINE,
)}


@dataclass(frozen=True)
class PercussiveMaterial:
    """One row: blow energy per removed volume e_v [J/m^3] at bit diameters [m], and the source as printed."""

    name: str
    diameters: tuple[float, ...]
    energy: tuple[float, ...]
    quote: str
    source: str


HILTI = ("Hilti TE 6-S operating instructions, '2. Technical data', p. 7 "
         "(https://staff.washington.edu/nelsonr/manuals/Hilti%20TE%206-S.pdf)")
ZIMPRICH = ("Zimprich, 'Unabhaengige Einstellung von Vorschubkraft, Schlagfrequenz und Einzelschlagenergie beim "
            "Hammerbohren durch zyklische Motoransteuerungen', dissertation, KIT (https://publikationen.bibliothek."
            "kit.edu/1000191911): Table 6.1, Table 6.2 and p. 111")
PERCUSSIVE: dict[str, PercussiveMaterial] = {m.name: m for m in (
    PercussiveMaterial(                     # 1.8 J x 5100/min = 153 W over the rated volume rates
        "concrete_medium_hard", (0.008, 0.012, 0.016), (153.0 / (28e-6 / 60.0), 153.0 / (45e-6 / 60.0),
                                                        153.0 / (49e-6 / 60.0)),
        "Single impact energy 1.8 J; hammering speed 0-5100 1/min; drilling performance in medium-hard concrete "
        "8 mm dia. = 28 cm3/min, 12 mm dia. = 45 cm3/min, 16 mm dia. = 49 cm3/min (0.328, 0.204, 0.187 J/mm^3)", HILTI),
    PercussiveMaterial(
        "concrete_c50_60", (0.016,), (0.59 * 24.60 / (0.21e-3 * math.pi * 0.008**2),),
        "16 mm four-cutter bit (DeWalt DT9680) in C50/60 concrete, constant motor control, 200 N feed force: mean "
        "single-blow energy 0.59 J at 24.60 Hz (Table 6.2, 25 Hz, 'Konstant'), mean ROP 0.21 mm/s (SD 0.06): "
        "0.344 J/mm^3 on the nominal bit area", ZIMPRICH),
)}


def work_material(name: str) -> WorkMaterial | PercussiveMaterial:
    """The table row ``name`` (twist drilling or rotary hammer); an unknown name raises with the valid ones."""
    row = WORK_MATERIALS.get(name) or PERCUSSIVE.get(name)
    if row is None:
        raise PluginError(f"no drilling work material {name!r}; known: {sorted(WORK_MATERIALS)} and, for a rotary "
                          f"hammer, {sorted(PERCUSSIVE)}")
    return row


def energy_per_volume(row: PercussiveMaterial, d: float) -> float:
    """e_v [J/m^3] at bit diameter ``d`` (host side; the kernels interpolate the same way)."""
    return float(np.interp(d, row.diameters, row.energy))


__all__ = ["PERCUSSIVE", "TABLE_SOURCE", "WORK_MATERIALS", "PercussiveMaterial", "WorkMaterial", "energy_per_volume",
           "work_material"]
