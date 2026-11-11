# SPDX-License-Identifier: Apache-2.0
"""Words for soft items: which nouns are liquids, grains, cloth or cables, their units, fill words and form words.

Plain data plus small helpers; the library cards behind each noun are found at build time
(``alloy.compose.soft.materials``), so new library materials (``glycerol``, ``sugar``) need no change here beyond
their noun.

Example::

    >>> kind_of("water"), kind_of("towel"), kind_of("mug")
    ('liquid', 'cloth', None)
    >>> to_ml(0.5, "l"), to_g(1, "kg"), to_m(50, "cm")
    (500.0, 1000.0, 0.5)
"""

from __future__ import annotations

LIQUIDS = frozenset({"water", "oil", "honey", "glycerol", "glycerin", "milk", "juice", "coffee", "tea", "wine", "beer",
                     "soda", "syrup", "vinegar", "sauce", "soup", "broth", "liquid", "fluid", "lemonade", "ketchup"})
GRAINS = frozenset({"sand", "rice", "grain", "sugar", "salt", "flour", "lentil", "gravel", "soil", "dirt", "seed",
                    "couscous", "quinoa", "bead"})
CLOTH = frozenset({"towel", "cloth", "napkin", "rag", "dishcloth", "tablecloth", "fabric", "sheet", "handkerchief",
                   "washcloth", "paper", "film", "blanket", "tarp", "wrap", "paperboard", "chipboard"})
CABLES = frozenset({"cable", "cord", "wire", "rope", "string", "lace", "shoelace", "twine", "thread", "hose"})
KIND_OF: dict[str, str] = {**{w: "liquid" for w in LIQUIDS}, **{w: "grain" for w in GRAINS},
                           **{w: "cloth" for w in CLOTH}, **{w: "cable" for w in CABLES}}

#: unit words -> (quantity, factor to ml / g / m)
UNITS: dict[str, tuple[str, float]] = {
    "ml": ("volume", 1.0), "millilitre": ("volume", 1.0), "milliliter": ("volume", 1.0), "cc": ("volume", 1.0),
    "cl": ("volume", 10.0), "centilitre": ("volume", 10.0), "centiliter": ("volume", 10.0),
    "dl": ("volume", 100.0), "l": ("volume", 1000.0), "litre": ("volume", 1000.0), "liter": ("volume", 1000.0),
    "g": ("mass", 1.0), "gram": ("mass", 1.0), "gramme": ("mass", 1.0), "kg": ("mass", 1000.0),
    "kilogram": ("mass", 1000.0), "kilo": ("mass", 1000.0),
    "mm": ("length", 1e-3), "millimetre": ("length", 1e-3), "millimeter": ("length", 1e-3),
    "cm": ("length", 1e-2), "centimetre": ("length", 1e-2), "centimeter": ("length", 1e-2),
    "m": ("length", 1.0), "metre": ("length", 1.0), "meter": ("length", 1.0),
}

#: fill words -> share of the container's depth
FILL_WORDS: dict[str, float] = {"full": 0.85, "filled": 0.85, "brimming": 0.88, "nearly full": 0.8,
                                "almost full": 0.8, "three quarters full": 0.75, "half full": 0.5, "half filled": 0.5,
                                "half empty": 0.5, "a quarter full": 0.25, "quarter full": 0.25,
                                "a third full": 0.33, "a little": 0.15, "a splash": 0.1, "a drop": 0.05}
DEFAULT_FILL = 0.6                  # a liquid or grain fill with no amount
DEFAULT_PILE_ML = 100.0             # a pile with no amount

#: form words per kind (the word -> the form)
CLOTH_FORMS: dict[str, str] = {"flat": "flat", "spread": "flat", "laid": "flat", "unfolded": "flat",
                               "smooth": "flat", "draped": "draped", "covering": "draped", "covered": "draped",
                               "hanging": "hanging", "hung": "hanging", "dangling": "hanging",
                               "folded": "folded", "doubled": "folded"}
CABLE_FORMS: dict[str, str] = {"straight": "straight", "stretched": "straight", "taut": "straight",
                               "loose": "loose", "curled": "loose", "curved": "loose", "lying": "loose"}
PILE_WORDS = frozenset({"pile", "heap", "mound", "cone", "piled", "heaped", "spilled", "poured"})
POUR_WORDS = frozenset({"pour", "pours", "pouring", "poured"})
TILT_WORDS = frozenset({"tilt", "tilted", "tilting", "tipped", "tipping", "tip"})
DEGREES = frozenset({"deg", "degree", "degrees"})
DEFAULT_TILT_DEG_S = 40.0
ATTACH_WORDS = frozenset({"plugged", "connected", "attached", "tied", "fixed", "clipped", "hooked", "fastened",
                          "wired", "running", "leading", "going"})
ROBOT_WORDS = frozenset({"robot", "robotic", "arm", "arms", "panda", "franka", "manipulator"})
#: the library robot a sentence's "robot arm" stands for (Franka Panda, MuJoCo Menagerie, Apache-2.0)
DEFAULT_ROBOT = "robots/emb_e2_panda/scene.xml"

#: every word the soft parser consumes (kept out of object nouns)
FORM_WORDS = frozenset({*CLOTH_FORMS, *CABLE_FORMS, *PILE_WORDS, *POUR_WORDS, *ATTACH_WORDS, *TILT_WORDS, "out",
                        "edge", "off", "half", "empty", "quarter", "quarters", "third", "nearly", "almost", "brimming",
                        "containing", "holding", "splash", "drop", "full", "filled", "fill"})


DEFAULT_FORMS = {"cloth": "flat", "cable": "loose", "grain": "pile"}


def kind_of(noun: str) -> str | None:
    """The soft kind of a singular noun ("water" -> "liquid"), or None for an ordinary object.

    Example: ``kind_of("rope") == "cable"``."""
    return KIND_OF.get(noun)


def _unit(unit: str) -> tuple[str, float]:
    u = unit.lower()
    u = u if u in UNITS else u[:-1] if u.endswith("s") else u
    if u not in UNITS:
        raise KeyError(unit)
    return UNITS[u]


def unit_kind(unit: str) -> str | None:
    """"volume", "mass", "length" or None for a unit word ("ml", "kilograms", "cm").

    Example: ``unit_kind("litres") == "volume"``."""
    try:
        return _unit(unit)[0]
    except KeyError:
        return None


def to_ml(value: float, unit: str) -> float:
    """A volume in ml. Example: ``to_ml(2, "dl") == 200.0``."""
    return float(value) * _unit(unit)[1]


def to_g(value: float, unit: str) -> float:
    """A mass in g. Example: ``to_g(0.3, "kg") == 300.0``."""
    return float(value) * _unit(unit)[1]


def to_m(value: float, unit: str) -> float:
    """A length in m. Example: ``to_m(3, "mm") == 0.003``."""
    return float(value) * _unit(unit)[1]


__all__ = ["ATTACH_WORDS", "CABLES", "CABLE_FORMS", "CLOTH", "CLOTH_FORMS", "DEFAULT_FILL", "DEFAULT_FORMS",
           "DEFAULT_PILE_ML", "DEFAULT_ROBOT", "DEFAULT_TILT_DEG_S", "DEGREES", "FILL_WORDS", "FORM_WORDS", "GRAINS",
           "KIND_OF", "LIQUIDS", "PILE_WORDS", "POUR_WORDS", "ROBOT_WORDS", "TILT_WORDS", "UNITS", "kind_of", "to_g",
           "to_m", "to_ml", "unit_kind"]
