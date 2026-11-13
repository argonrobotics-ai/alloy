# SPDX-License-Identifier: Apache-2.0
"""Vocabulary for the sentence parser and the library search: numbers, plurals, synonyms, material and size words.

Everything here is plain data plus two small helpers, so a new library only needs new synonyms when its object
names use words nobody would type (most imported names already contain the everyday noun, e.g. ``..._mug_...``).

Example::

    >>> singular("knives"), singular("boxes"), singular("glass")
    ('knife', 'box', 'glass')
    >>> sorted(expand("block"))
    ['block', 'brick', 'cube']
"""

from __future__ import annotations

import re

NUMBERS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "pair": 2, "couple": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12,
    "few": 3, "several": 4, "some": 3, "many": 6, "lots": 8,
}
MAX_COUNT = 40                                   # per item phrase; more than this is almost surely a typo
PLURAL_COUNT = 3                                 # "cubes", "towels" with no number

CLUTTER: dict[str, str] = {
    "cluttered": "cluttered", "messy": "cluttered", "crowded": "cluttered", "packed": "cluttered",
    "busy": "cluttered", "dense": "cluttered", "jumbled": "cluttered", "untidy": "cluttered",
    "tidy": "tidy", "neat": "tidy", "sparse": "tidy", "spaced": "tidy", "uncluttered": "tidy", "clean": "tidy",
    "organised": "tidy", "organized": "tidy", "minimal": "tidy",
}
ARRANGEMENT: dict[str, str] = {"row": "row", "line": "row", "lined": "row", "grid": "grid", "rows": "grid"}

#: support words -> the support kind (``fixtures.SUPPORTS``)
SUPPORTS: dict[str, str] = {
    "table": "table", "desk": "table", "bench": "table", "workbench": "table", "counter": "table",
    "countertop": "table", "worktop": "table", "tabletop": "table", "floor": "floor", "ground": "floor",
}

#: relation words between an item and a container or another item
RELATIONS: dict[str, str] = {"in": "in", "inside": "in", "into": "in", "within": "in", "on": "on", "onto": "on",
                             "atop": "on", "upon": "on", "over": "on"}

#: query word -> words that also count as that object (both directions are listed where it makes sense)
SYNONYMS: dict[str, tuple[str, ...]] = {
    "cube": ("block", "brick"), "block": ("cube", "brick"), "brick": ("block", "cube"),
    "ball": ("sphere", "marble"), "sphere": ("ball",), "marble": ("ball",),
    "cup": ("mug", "tumbler", "glass"), "mug": ("cup",), "tumbler": ("cup", "glass"),
    "jar": ("pot", "container"), "bin": ("box", "tub", "crate"), "box": ("bin", "crate", "case"),
    "tray": ("pan", "plate"), "plate": ("dish",), "dish": ("plate", "bowl"), "bowl": ("dish",),
    "screw": ("bolt",), "bolt": ("screw",), "spanner": ("wrench",), "wrench": ("spanner",),
    "knife": ("blade",), "spatula": ("turner",), "turner": ("spatula",), "pen": ("pencil", "marker"),
    "pencil": ("pen",), "bottle": ("flask",), "can": ("tin",), "tin": ("can",), "rod": ("shaft", "peg"),
    "shaft": ("rod",), "peg": ("pin", "dowel"), "board": ("plank", "plate"), "toy": ("car", "figure"),
    "fruit": ("apple", "banana", "orange", "pear", "lemon", "peach", "plum"), "tool": ("screwdriver", "wrench",
    "hammer", "pliers"), "utensil": ("spoon", "fork", "knife", "spatula", "ladle"), "cutlery": ("spoon", "fork",
    "knife"), "container": ("box", "bin", "jar", "tub"),
}

#: adjectives that point at a material family: the word -> substrings of material names
MATERIAL_WORDS: dict[str, tuple[str, ...]] = {
    "wooden": ("wood", "beech", "pine", "oak", "alder", "plywood", "bamboo", "cedar", "maple", "birch"),
    "wood": ("wood", "beech", "pine", "oak", "alder", "plywood", "bamboo", "cedar", "maple", "birch"),
    "steel": ("steel", "stainless", "crv"), "metal": ("steel", "stainless", "alumin", "copper", "brass", "iron",
    "tinplate", "zinc"), "metallic": ("steel", "stainless", "alumin", "copper", "brass", "iron"),
    "aluminium": ("alumin",), "aluminum": ("alumin",), "copper": ("copper",), "brass": ("brass",),
    "iron": ("iron", "steel"), "glass": ("glass",), "ceramic": ("stoneware", "porcelain", "ceramic"),
    "porcelain": ("porcelain", "stoneware", "ceramic"), "stoneware": ("stoneware",),
    "plastic": ("pp", "pe", "hdpe", "ldpe", "lldpe", "abs", "pet", "pom", "pa6", "pa66", "nylon", "polycarbonate",
                "polystyrene", "pmma", "ptfe", "pvc", "plastic", "pbt"),
    "rubber": ("rubber", "silicone", "nitrile"), "foam": ("foam",), "paper": ("paper", "cardboard", "kraft"),
    "cardboard": ("cardboard",), "kraft": ("kraft",), "cotton": ("cotton",), "stone": ("stone", "granite", "marble"),
}
#: library cards whose names leave out everyday words they answer to (searched as if in the name); a flat shell sheet
#: (``Info.sheet``) is placed as an object only when one of these words names it ("a bag insert")
CARD_WORDS: dict[str, tuple[str, ...]] = {
    "gift_bag_kraft": ("shopping", "carrier"),
    "gift_bag_board": ("bottom", "insert"),
}
SIZE_WORDS: dict[str, str] = {"small": "small", "tiny": "small", "little": "small", "mini": "small",
                              "large": "large", "big": "large", "huge": "large", "long": "large", "tall": "large"}

#: words the parser drops without a note
STOP: frozenset[str] = frozenset({
    "the", "with", "and", "of", "some", "its", "it", "them", "there", "is", "are", "scene", "set", "to",
    "for", "that", "which", "has", "have", "plus", "also", "lying", "sitting", "standing", "placed", "put",
    "random", "randomly", "scattered", "around", "about", "near", "next", "beside", "together", "by", "at",
    "top", "each", "other", "kitchen", "lab", "workshop", "office", "robot", "make", "build", "create", "me",
    "please", "simple", "realistic", "physically", "checked", "valid", "variety", "different", "various",
    "assorted", "mixed", "kinds", "kind", "type", "types", "objects", "object", "things", "items", "stuff",
})

_IRREGULAR = {"knives": "knife", "leaves": "leaf", "shelves": "shelf", "halves": "half", "mice": "mouse",
              "glasses": "glass", "boxes": "box", "dishes": "dish", "brushes": "brush", "matches": "match",
              "watches": "watch", "batteries": "battery", "berries": "berry", "cherries": "cherry",
              "tomatoes": "tomato", "potatoes": "potato", "pieces": "piece", "wrenches": "wrench",
              "benches": "bench", "lunches": "lunch", "sandwiches": "sandwich", "peaches": "peach",
              "tubes": "tube", "plates": "plate", "pipes": "pipe", "dice": "die", "teeth": "tooth"}
_KEEP = frozenset({"glass", "brass", "grass", "dress", "press", "class", "boss", "moss", "lens", "series",
                   "scissors", "pliers", "tongs", "chess", "bus", "gas", "plus", "status", "cactus"})
TOKEN = re.compile(r"[a-z]+")


def singular(word: str) -> str:
    """The singular of an English noun, good enough for object names.

    Example: ``singular("cubes") == "cube"``."""
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w in _KEEP or len(w) <= 3 or w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "zes")):
        return w[:-2]
    return w[:-1] if w.endswith("s") else w


def tokens(text: str) -> list[str]:
    """Lower-case alphabetic tokens, singular, of a name, title or phrase (digits and punctuation split words).

    Example: ``tokens("YCB 025 Mug, blue") == ["ycb", "mug", "blue"]``."""
    out = []
    for part in re.split(r"(?<=[a-z])(?=[A-Z])", text):        # camelCase -> camel Case
        out.extend(singular(t) for t in TOKEN.findall(part.lower()))
    return out


def expand(word: str) -> set[str]:
    """``word`` and its synonyms (singular).

    Example: ``"mug" in expand("cup")``."""
    w = singular(word)
    return {w, *SYNONYMS.get(w, ())}


__all__ = ["ARRANGEMENT", "CARD_WORDS", "CLUTTER", "MATERIAL_WORDS", "MAX_COUNT", "NUMBERS", "PLURAL_COUNT",
           "RELATIONS", "SIZE_WORDS", "STOP", "SUPPORTS", "SYNONYMS", "expand", "singular", "tokens"]
