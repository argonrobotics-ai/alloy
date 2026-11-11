# SPDX-License-Identifier: Apache-2.0
"""ESTIMATE masses for Google Scanned Objects (no masses are published) and YCB parts without a measured mass.

Order of preference for a GSO model:

1. **Net contents in the product name or description** (``net_contents``): a retail label quantity such as
   ``12_oz``, ``1.5 lb``, ``355 ml`` or ``144 fl oz`` (count x unit for "3 count, 8 fl oz each"). The object mass is
   contents + max(``PACKAGING`` x contents, ``MIN_PACKAGING``), with an interval from the contents alone to contents
   + twice the packaging; liquids count at water density. Glass jars are heavier than this rule says. Used only if
   the implied effective density is plausible (``PLAUSIBLE``).
2. **Category rule** (``CATEGORY``, keyed by the first GSO category): an effective density, or for objects whose
   closed scan volume says little about their mass (bags, headphones, hats) a typical mass.

Every value is labelled ESTIMATE in its card note and gets a sourcing-queue row with a search query and the GTIN.
Category densities derived from YCB use the fitted effective densities of measured YCB objects (measured mass over
this pipeline's scan volume), so they come with the same scan-volume bias as the GSO scans they are applied to.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

PACKAGING = 0.10                # packaging as a fraction of net contents (retail cartons, bottles, blister packs)
MIN_PACKAGING = 0.03            # kg: a container weighs at least this (small jars and tubes of a few grams of contents)
PLAUSIBLE = (30.0, 3000.0)      # kg/m^3: effective densities outside this make a name-based mass suspect
WATER = 1000.0                  # kg/m^3

# YCB cards whose fitted effective densities stand in for GSO categories (fills ycb_density at build time)
YCB_ANALOGUES: dict[str, tuple[str, ...]] = {
    "carton": ("ycb_003_cracker_box", "ycb_004_sugar_box", "ycb_008_pudding_box", "ycb_009_gelatin_box"),
    "bottle_can": ("ycb_002_master_chef_can", "ycb_005_tomato_soup_can", "ycb_006_mustard_bottle",
                   "ycb_007_tuna_fish_can", "ycb_010_potted_meat_can", "ycb_021_bleach_cleanser",
                   "ycb_022_windex_bottle"),
    "toy": ("ycb_011_banana", "ycb_013_apple", "ycb_017_orange", "ycb_054_softball", "ycb_072_a_toy_airplane",
            "ycb_070_b_colored_wood_blocks"),
}


@dataclass(frozen=True)
class CategoryRule:
    """How to estimate one GSO category: an effective density (kg/m^3) or a typical mass (kg), with an interval."""

    material: str               # library material for surface and elastic values
    density: float | None = None
    mass: float | None = None
    unc: tuple[float, float] = (0.0, 0.0)
    analogue: str = ""          # YCB_ANALOGUES key: density and interval come from those fits instead
    note: str = ""


CATEGORY: dict[str, CategoryRule] = {
    "Shoe": CategoryRule("nitrile_70a", density=300.0, unc=(150.0, 500.0),
                         note="single shoe, rubber sole; engineering estimate 0.25-0.45 kg over a 0.8-1.2 L scan"),
    "Consumer Goods": CategoryRule("cardboard_corrugated", analogue="carton",
                                   note="packaged retail goods; YCB food cartons as analogues"),
    "Bottles and Cans and Cups": CategoryRule("hdpe", analogue="bottle_can",
                                              note="filled bottles and cans; YCB bottles and cans as analogues"),
    "Toys": CategoryRule("abs", analogue="toy", note="moulded toys; YCB toys as analogues"),
    "Action Figures": CategoryRule("vinyl_soft", density=800.0, unc=(400.0, 1250.0),
                                   note="mostly solid PVC/ABS figure, 0.3-1.0 of vinyl_soft's density"),
    "Legos": CategoryRule("abs", analogue="toy", note="LEGO sets (boxed or built); YCB toys as analogues"),
    "Board Games": CategoryRule("cardboard_corrugated", density=250.0, unc=(150.0, 400.0),
                                note="boxed game (board, cards, pieces); engineering estimate"),
    "Media Cases": CategoryRule("pp_homopolymer", density=300.0, unc=(200.0, 450.0),
                                note="game or disc keep-case with disc and insert; engineering estimate"),
    "Bag": CategoryRule("cotton_woven", mass=0.3, unc=(0.1, 0.8),
                        note="empty fabric bag or case; typical mass, scan volume not used"),
    "Keyboard": CategoryRule("abs", density=600.0, unc=(350.0, 900.0), note="computer keyboard; engineering estimate"),
    "Mouse": CategoryRule("abs", density=600.0, unc=(350.0, 900.0), note="computer mouse; engineering estimate"),
    "Headphones": CategoryRule("abs", mass=0.3, unc=(0.2, 0.45),
                               note="over-ear headset; typical mass, scan volume not used"),
    "Stuffed Toys": CategoryRule("cotton_woven", density=50.0, unc=(20.0, 120.0),
                                 note="plush toy (fibre fill); engineering estimate"),
    "Hat": CategoryRule("cotton_woven", mass=0.1, unc=(0.05, 0.2), note="hat; typical mass, scan volume not used"),
    "Car Seat": CategoryRule("abs", density=300.0, unc=(100.0, 800.0), note="engineering estimate"),
    "Camera": CategoryRule("abs", density=1500.0, unc=(1000.0, 2000.0),
                           note="compact camera; engineering estimate"),
}
DEFAULT_RULE = CategoryRule("abs", density=500.0, unc=(100.0, 1500.0),
                            note="uncategorised object; generic engineering estimate")

_FL_OZ = 29.5735295625e-6      # m^3 per US fluid ounce
_UNITS = {"oz": 0.028349523125, "ounce": 0.028349523125, "ounces": 0.028349523125, "lb": 0.45359237,
          "lbs": 0.45359237, "g": 1e-3, "gram": 1e-3, "grams": 1e-3, "kg": 1.0,
          "fl_oz": _FL_OZ * WATER, "ml": 1e-6 * WATER, "l": 1e-3 * WATER, "lt": 1e-3 * WATER,
          "liter": 1e-3 * WATER, "liters": 1e-3 * WATER, "litre": 1e-3 * WATER, "litres": 1e-3 * WATER}
_QTY = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(fl\.?\s?oz|ounces?|oz|lbs?|grams?|g|kg|ml|liters?|litres?|lt|l)"
                  r"(?![a-z])")


def _kg(number: str, unit: str) -> float:
    """Mass in kg of a label quantity (volumes at water density)."""
    u = re.sub(r"fl\.?\s?oz", "fl_oz", unit)
    return float(number) * _UNITS[u]


_UNIT = r"(?:fl\.?\s?oz|ounces?|oz|lbs?|grams?|g|kg|ml|liters?|litres?|lt|l)(?![a-z])"
_COUNT = re.compile(r"(?<![\w.])(\d+)\s*(?:count|ct|pack|pk)\b"                  # "3 count", "6 pack"
                    r"|(?<![\w.])(\d+)\s*-\s*(?=\d+(?:\.\d+)?\s*" + _UNIT + ")")   # "12 - 12 fl oz", not "250-300 miles"


def _count(text: str) -> tuple[int, int] | None:
    """``(count, end position)`` of a multi-pack count ("3 count", "6 pack", "12 - 12 fl oz"), if any."""
    m = _COUNT.search(text)
    if m is None:
        return None
    n = int(m.group(1) or m.group(2))
    return (n, m.end()) if n > 1 else None


def net_contents(text: str) -> tuple[float, str] | None:
    """Net contents of a product description in kg (liquids at water density), with how it was read.

    The largest stated quantity is taken, unless the text gives a pack count and a per-unit quantity after it
    ("3 count, 8 fl oz bottles each", "6 pack, 0.5 oz packs") whose product is larger: then count x unit. A
    quantity followed by "total" is a total. Use the description, not the model name: names drop decimal points
    ("462_oz" for 4.62 oz).

    Example::

        net_contents("Pepsi Max, 12 - 12 fl oz (355 ml) cans [144 fl oz (4.26 lt)]")   # (4.26, "4.26 lt")
        net_contents("Fingerpaint - 3 count, 8 fl oz bottles each")                       # (0.71, "3 x 8 fl oz")
    """
    t = text.lower()
    found = [(m, _kg(m.group(1), m.group(2))) for m in _QTY.finditer(t)]
    found = [(m, kg) for m, kg in found if kg > 0]
    if not found:
        return None
    m_big, kg_big = max(found, key=lambda x: x[1])
    best = (kg_big, m_big.group(0))
    cnt = _count(t)
    if cnt is None:
        return best
    after = [(m, kg) for m, kg in found if m.start() >= cnt[1]]
    if not after or t[after[0][0].end():after[0][0].end() + 7].strip().startswith("total"):
        return best
    m_unit, kg_unit = after[0]
    if kg_big >= 0.9 * cnt[0] * kg_unit:
        return best
    return cnt[0] * kg_unit, f"{cnt[0]} x {m_unit.group(0)}"


def ycb_density(fits: dict[str, float], key: str) -> tuple[float, tuple[float, float], str]:
    """Median and min-max of the fitted effective densities of a YCB analogue group, with a note.

    Example::

        ycb_density({"ycb_003_cracker_box": 189.1, ...}, "carton")   # (554.0, (189.1, 806.0), "...")
    """
    names = [n for n in YCB_ANALOGUES[key] if n in fits]
    vals = sorted(fits[n] for n in names)
    note = (f"median effective density of {len(vals)} measured YCB objects ({', '.join(names)}: measured mass over "
            f"scan volume), interval their min-max")
    return statistics.median(vals), (vals[0], vals[-1]), note
