# SPDX-License-Identifier: Apache-2.0
"""Which library cards a soft item is made from: the liquid or grain material, the cloth or cable template card.

Liquids are the library materials with ``fluid`` values, grains the ones with a ``plastic.friction_angle``; cloth
templates are sheet object cards (a thin box or a mesh with ``features.thickness``) and cable templates are curve
cards. New library materials and cards are found by name with no change here. Everyday words with no card of their
own use a close stand-in, and the stand-in is listed in the result as an estimate (milk as water, a rag as the
cotton towel, a cord as the USB cable); a word with no card and no stand-in is an error that lists what exists.

Example::

    >>> t = resolve(ItemSpec("milk", soft=SoftSpec("liquid")), Catalog())          # doctest: +SKIP
    >>> t.material, t.estimated                                                   # doctest: +SKIP
    ('water', ('milk: simulated as water (density and viscosity within ~5 % of milk)',))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from alloy.compose import words as W
from alloy.compose.errors import ComposeError
from alloy.compose.spec import ItemSpec

log = logging.getLogger("alloy.compose")

ALIASES = {"oil": "olive_oil", "glycerin": "glycerol"}
WATERY = frozenset({"milk", "juice", "coffee", "tea", "wine", "beer", "soda", "lemonade", "broth", "liquid", "fluid",
                    "vinegar"})
CLOTH_DEFAULT = {"napkin": "napkin_cotton_aniso", "paper": "paper_a4", "film": "film_ldpe_sheet",
                 "wrap": "film_ldpe_sheet", "tarp": "film_hdpe_sheet", "paperboard": "paper_board_sheet",
                 "chipboard": "paper_board_sheet"}
#: words before "paper" (or "wrap") that name a library paper sheet; plain "paper" is an A4 sheet
PAPER_WORDS = {"kraft": "paper_kraft_sheet", "brown": "paper_kraft_sheet", "wrapping": "paper_wrap_sheet",
               "gift": "paper_wrap_sheet", "packing": "paper_wrap_sheet", "scored": "paper_kraft_scored_sheet",
               "creased": "paper_kraft_scored_sheet"}
#: everyday cloth words with no card of their own -> the closest library sheet (listed as an estimate)
CLOTH_STANDIN = {"rag": "towel_cotton", "dishcloth": "towel_cotton", "washcloth": "towel_cotton",
                 "fabric": "cloth_cotton_sheet", "tablecloth": "cloth_cotton_sheet", "blanket": "cloth_cotton_sheet",
                 "handkerchief": "cloth_cotton_sheet"}
PLASTIC = ("plastic", "polythene", "polyethylene", "pe", "ldpe", "hdpe", "lldpe")
CLOTH_FALLBACK = "towel_cotton"
ROPES = frozenset({"rope", "string", "lace", "shoelace", "twine", "thread"})
ROPE_FALLBACK = "shoelace_cotton"
CABLE_FALLBACK = "usb_cable_1m"


@dataclass(frozen=True)
class Template:
    """What a soft item is made from: ``material`` (a material card), ``card`` (an object card whose shape and
    features it copies: the cloth, the cable, the grain fill that gives grain size and spacing; None for liquids),
    and ``estimated`` (stand-ins used, for the result's estimate list)."""

    kind: str
    material: str
    card: str | None = None
    estimated: tuple[str, ...] = ()


def resolve(item: ItemSpec, catalog: Any) -> Template:
    """The template of a soft item (see the module docstring); ``ComposeError`` with the fix when none fits.

    Example: ``resolve(ItemSpec("towel", soft=SoftSpec("cloth")), catalog).card == "towel_cotton"``."""
    kind = item.soft.kind
    if kind == "liquid":
        return _liquid(item, catalog.library)
    if kind == "grain":
        return _grain(item, catalog)
    return _cloth(item, catalog) if kind == "cloth" else _cable(item, catalog)


def materials_of(library: Any, kind: str) -> list[str]:
    """Library material names that are liquids (``kind="liquid"``) or grains (``"grain"``).

    Example: ``"water" in materials_of(CardLibrary(), "liquid")``."""
    out = []
    for name in library.names("materials"):
        try:
            mat = library.material(name)
        except Exception as exc:  # noqa: BLE001 - an invalid material is skipped and logged
            log.debug("compose soft: material %s unreadable: %s", name, exc)
            continue
        grain = mat.plastic is not None and mat.plastic.friction_angle is not None
        if (kind == "grain" and grain) or (kind == "liquid" and mat.fluid is not None and not grain):
            out.append(name)
    return sorted(out)


def _pick_material(item: ItemSpec, names: list[str]) -> str | None:
    noun = ALIASES.get(item.query, item.query)
    if item.object is not None:
        return item.object if item.object in names else None
    if noun in names:
        return noun
    hits = [n for n in names if noun in n.split("_") or item.query in n.split("_")]
    hits.sort(key=lambda n: (-sum(w in n.split("_") for w in item.words), len(n), n))
    return hits[0] if hits else None


def _liquid(item: ItemSpec, library: Any) -> Template:
    names = materials_of(library, "liquid")
    name = _pick_material(item, names)
    if name is not None:
        return Template("liquid", name)
    if item.query in WATERY and "water" in names and item.object is None:
        return Template("liquid", "water", estimated=(f"{item.query}: simulated as water (density and viscosity "
                                                      f"within ~5 % of {item.query})",))
    raise ComposeError(f"no liquid material for {item.object or item.query!r}",
                       f"use one of {names} (alloy.compose.soft.materials.materials_of lists them), or add a material "
                       "card with fluid values", field="items.query")


def _grain(item: ItemSpec, catalog: Any) -> Template:
    names = materials_of(catalog.library, "grain")
    name = _pick_material(item, names)
    if name is None:
        raise ComposeError(f"no grain material for {item.object or item.query!r}",
                           f"use one of {names} (sand, rice), or add a material card with plastic.friction_angle",
                           field="items.query")
    return Template("grain", name, _grain_card(catalog, name))


def _grain_card(catalog: Any, material: str) -> str | None:
    """A library particle card of this grain material (it carries grain size and lattice spacing), if any; the soft
    pack's ``<grain>_box_<n>ml`` fills first (SOFT-2), the older task fills after."""
    from alloy.cards import types as ct  # noqa: PLC0415

    for name in sorted(catalog.entries, key=lambda n: ("_box_" not in n, n)):     # the soft pack's fills first
        if catalog.entries[name].source not in ("library", "assets"):
            continue
        try:
            card = catalog.library.peek_object(name)
        except Exception:  # noqa: BLE001 - unreadable cards are not templates
            continue
        if card.material == material and isinstance(card.shape, ct.ParticleSource):
            return name
    return None


def _shape_kind(catalog: Any, name: str) -> str | None:
    """"cloth" for sheet cards (a thin box, or a flat film mesh with a thickness), "cable" for curve cards, else
    None (read without validation)."""
    from alloy.cards import types as ct  # noqa: PLC0415

    try:
        card = catalog.library.peek_object(name)
    except Exception:  # noqa: BLE001 - unreadable cards are not templates
        return None
    if isinstance(card.shape, ct.CurveSource):
        return "cable"
    if card.motion != "dynamic":
        return None
    if isinstance(card.shape, ct.Primitive):
        size = card.shape.size if card.shape.kind == "box" else None
    elif isinstance(card.shape, ct.MeshSource) and card.features.thickness is not None and card.route is not None \
            and card.route.route == "cloth_film":
        size = catalog.info(name).size                 # a flat film mesh (a scored sheet), not a bag or a shell
    else:
        size = None
    return "cloth" if size is not None and min(size) < 0.05 * sorted(size)[1] else None


def _search(item: ItemSpec, catalog: Any, kind: str) -> str | None:
    if item.object is not None:
        if _shape_kind(catalog, item.object) != kind:
            raise ComposeError(f"{item.object} is not a {kind} card", f"pin a {kind} card, or leave 'object' out",
                               field="items.object")
        return item.object
    for m in catalog.search(item.query, item.words, k=30):
        if m.entry.source in ("library", "assets") and W.singular(item.query) in m.entry.words and \
                _shape_kind(catalog, m.name) == kind:
            return m.name
    return None


def _cloth(item: ItemSpec, catalog: Any) -> Template:
    """A sheet card: "plastic sheet" / "film" / "wrap" -> a film, else a card named by the noun (towel, cloth,
    napkin, paper, polyester sheet), else the closest stand-in sheet (listed as an estimate)."""
    if any(w in PLASTIC for w in item.words) and item.object is None:
        item = ItemSpec("film", item.count, tuple(w for w in item.words if w != "plastic"), item.relation,
                        item.target, None, item.soft)
    name = _paper(item, catalog) or _search(item, catalog, "cloth") or _have(catalog, CLOTH_DEFAULT.get(item.query))
    if name is not None:
        return Template("cloth", catalog.library.peek_object(name).material, name)
    fall = _have(catalog, CLOTH_STANDIN.get(item.query)) or CLOTH_FALLBACK
    what = catalog.library.peek_object(fall).material
    return Template("cloth", what, fall, (f"{item.query}: made as the library sheet {fall} ({what}: material and "
                                          "thickness)",))


def _paper(item: ItemSpec, catalog: Any) -> str | None:
    """The library paper sheet a paper phrase names ("kraft paper", "gift wrap", "a sheet of paper": A4), else None."""
    if item.query not in ("paper", "wrap"):
        return None
    named = next((PAPER_WORDS[w] for w in item.words if w in PAPER_WORDS), None)
    return _have(catalog, named or (CLOTH_DEFAULT["paper"] if item.query == "paper" else None))


def _have(catalog: Any, name: str | None) -> str | None:
    return name if name is not None and name in catalog.entries else None


def _cable(item: ItemSpec, catalog: Any) -> Template:
    name = _search(item, catalog, "cable")
    if name is None and "usb" in item.words:
        name = CABLE_FALLBACK
    if name is not None:
        return Template("cable", catalog.library.peek_object(name).material, name)
    fall = ROPE_FALLBACK if item.query in ROPES else CABLE_FALLBACK
    what = "cotton shoelace cord" if fall == ROPE_FALLBACK else "USB cable (composite bending and twist stiffness)"
    return Template("cable", catalog.library.peek_object(fall).material, fall,
                    (f"{item.query}: made as the library {what}, {fall}",))


__all__ = ["Template", "materials_of", "resolve"]
