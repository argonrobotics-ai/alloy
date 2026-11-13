# SPDX-License-Identifier: Apache-2.0
"""Physical values for a new object: what it is (category), its base material, its real size and its mass, each with
where it comes from. Nothing is refused for lack of a citation; estimates carry ``tag: ESTIMATE`` and a range.

Sources, best first:

- mass: your ``mass`` (``USER``, with ``mass_source``); the sourcing cache's category values
  (``alloy.sourcing``, cited evidence, ``ESTIMATE`` for one object); the category table's typical mass
  (``ESTIMATE``); the base material's density times the mesh volume (``ESTIMATE``, +-30 %);
- material: your ``material`` (a library material name or a word such as "wood"); the category's; "abs" plastic;
- size: see ``alloy.create.scale``; the category's typical size is the estimate for generated meshes.

Example::

    >>> cat = find_category("a ceramic coffee mug", None)              # doctest: +SKIP
    >>> cat.name, cat.size                                             # doctest: +SKIP
    ('mug', (0.08, 0.11, 0.14))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alloy.compose import words as W
from alloy.create.request import CreateError

log = logging.getLogger("alloy.create")

DEFAULT_MATERIAL = "abs"
#: a material word -> the library material it means first (else the first library name containing a hint)
PREFERRED = {"wooden": "wood_beech", "wood": "wood_beech", "steel": "steel_carbon", "metal": "steel_carbon",
             "metallic": "steel_carbon", "iron": "steel_carbon", "aluminium": "aluminium_6061_t6",
             "aluminum": "aluminium_6061_t6", "glass": "glass_soda_lime", "ceramic": "stoneware_glazed",
             "porcelain": "stoneware_glazed", "stoneware": "stoneware_glazed", "plastic": "abs",
             "rubber": "silicone_40a", "paper": "paper_80gsm", "cardboard": "cardboard_corrugated",
             "brass": "brass_c360", "copper": "copper_c110"}
DENSITY_REL = 0.3                # +- share on a density-times-volume estimate
#: Young's modulus below which a material makes a soft body (rubber, foam, fabric); a word never picks one, because
#: create makes rigid objects (name a soft library material yourself to try a soft body)
SOFT_YOUNGS_PA = 1e8


@dataclass(frozen=True)
class Category:
    """What the object is, with typical values: ``size`` (lo, typical, hi) longest side in m, ``mass`` (lo, typical,
    hi) in kg, a suggested base ``material``, and where each came from."""

    name: str
    size: tuple[float, float, float] | None = None
    mass: tuple[float, float, float] | None = None
    material: str | None = None
    mass_note: str = ""
    mass_tag: str = "ESTIMATE"
    evidence: tuple[str, ...] = ()
    size_note: str = ""


@dataclass
class Value:
    """One physical value of the new card and its provenance (a row of the result's ``values``)."""

    field: str
    value: float
    unit: str
    tag: str                       # USER, CITED, LIBRARY or ESTIMATE
    source: str
    lo: float | None = None
    hi: float | None = None
    verify: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in ("field", "value", "unit", "tag", "source", "lo", "hi", "verify",
                                           "evidence")}
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items() if v not in (None, "", [])}


def _candidates(*texts: str | None) -> list[str]:
    """Words and two-word runs to look a category up by, most specific (the last noun) first."""
    out: list[str] = []
    for text in texts:
        if not text:
            continue
        toks = [t for t in W.tokens(text.split(" with ")[0]) if t not in W.STOP and t not in W.NUMBERS]
        out += [f"{a}_{b}" for a, b in zip(toks[::-1][1:], toks[::-1])] + toks[::-1]
    return list(dict.fromkeys(out))


def find_category(text: str | None, category: str | None, name: str | None = None) -> Category | None:
    """The best category for the object: the sourcing cache first (cited values), the category table filling what
    the cache lacks (typically the size), else the table alone."""
    keys = _candidates(category, text, (name or "").replace("_", " "))
    for key in keys:
        entry = _cache_entry(key)
        if entry is not None:
            names = [*keys, entry.key, *(a.replace(" ", "_") for a in entry.aliases)]
            table = next((t for t in map(_from_table, names) if t is not None), None)
            return _merge(entry, table)
        hit = _from_table(key)
        if hit is not None:
            return hit
    return None


def _cache_entry(key: str):
    try:
        from alloy.sourcing.cache import Cache  # noqa: PLC0415

        return Cache().lookup(key, "category")
    except Exception as exc:  # noqa: BLE001 - no alloy-assets sources in this install: the table still answers
        log.debug("create: sourcing cache unavailable (%s)", exc)
        return None


def _merge(entry, table: Category | None) -> Category:
    """A sourcing-cache entry (cited mass) with the table's size and material where the entry has none."""
    m = entry.values.get("mass")
    if m is not None:
        mass, note = (m.lo, m.value, m.hi), m.reason
        evidence = tuple(e.url for e in m.evidence if getattr(e, "url", ""))[:3]
    else:
        mass, note, evidence = (table.mass, table.mass_note, ()) if table else (None, "", ())
    return Category(entry.key, table.size if table else None, mass, entry.material or (table.material if table
                    else None), note, "ESTIMATE", evidence, table.size_note if table else "")


def _from_table(key: str) -> Category | None:
    try:
        from alloy.importers.objaverse.categories import load_categories, lvis_index  # noqa: PLC0415

        cats = load_categories()
        cat = cats.get(key) or lvis_index(cats).get(key)
    except Exception as exc:  # noqa: BLE001 - the table is optional data; no table means no category estimate
        log.debug("create: category table unavailable (%s)", exc)
        return None
    if cat is None:
        return None
    return Category(cat.name, (cat.size.lo, cat.size.value, cat.size.hi), (cat.mass.lo, cat.mass.value, cat.mass.hi),
                    cat.material, f"typical {cat.spec or cat.name} (category table, not measured)", "ESTIMATE", (),
                    f"typical {cat.spec or cat.name} (category table, not measured)")


def choose_material(lib, wanted: str | None, cat: Category | None, text: str | None) -> tuple[str, Value]:
    """A library material name for the base surface and density, with how it was chosen. ``wanted`` (your
    ``material=``) is a library material name or words ("Steel", "stainless steel"); one that is neither, or that
    names a liquid or a granular material, raises ``CreateError``. Without it, a material word in ``text`` is used."""
    if wanted:
        name = _named(lib, wanted)
        if name is not None:
            return name, Value("material", 0.0, "", "USER", f"library material {name} (you chose it)")
    words = W.tokens(wanted or text or "")
    hint = next((w for w in words if w in W.MATERIAL_WORDS), None)
    if wanted and hint is None:
        raise CreateError(f"material {wanted!r} is neither a library material nor a material word",
                          f"pass a library material name (alloy list materials) or one of {sorted(W.MATERIAL_WORDS)}",
                          field="material")
    soft = ""
    if hint is not None:
        name = _for_word(lib, hint, [w for w in words if w != hint] if wanted else [])
        if name is not None:
            return name, Value("material", 0.0, "", "ESTIMATE", f"library material {name} for '{wanted or hint}'",
                               verify=f"check that {name} matches the real material")
        soft = f"'{hint}' only matches soft materials and create makes rigid objects; "
    if cat is not None and cat.material and lib.has_material(cat.material) and _rigid_ok(lib, cat.material):
        return cat.material, Value("material", 0.0, "", "ESTIMATE", f"{soft}library material {cat.material}, "
                                   f"typical for a {cat.name}", verify=f"check that a {cat.name} of {cat.material} "
                                   "is right")
    return DEFAULT_MATERIAL, Value("material", 0.0, "", "ESTIMATE", f"{soft}library material {DEFAULT_MATERIAL} "
                                   "(no rigid material given or known)",
                                   verify="pass material= (e.g. 'wood', 'steel', 'plastic')")


def _named(lib, wanted: str) -> str | None:
    """The library material ``wanted`` names (as written, or lower-case with ``_`` for spaces and dashes), or None.
    A liquid or a granular material raises: create makes a solid object and would drop what makes it flow."""
    for name in dict.fromkeys((wanted, "_".join(wanted.lower().replace("-", " ").split()))):
        if not lib.has_material(name):
            continue
        m = lib.material(name)
        flows = "a liquid" if m.fluid is not None else \
            "a granular material" if getattr(m.plastic, "friction_angle", None) is not None else None
        if flows:
            raise CreateError(f"material {name!r} is {flows}; create makes rigid objects and would keep only its "
                              "density", "pass a solid material (e.g. 'plastic', 'glass', 'steel'); for a container "
                              "of it, create the container and fill it in the scene (Scene.add_liquid, alloy.compose)",
                              field="material")
        return name
    return None


def _for_word(lib, hint: str, others: list[str]) -> str | None:
    """The first rigid library material for the material word ``hint``: the preferred one, then names holding one
    of the word's substrings, Alloy's own library before imported ones; names that also hold one of ``others`` (the
    rest of your words: "stainless" in "stainless steel") come first."""
    from alloy.cards.library import default_root  # noqa: PLC0415

    home = default_root().resolve()
    found = [n for sub in W.MATERIAL_WORDS[hint] for n in lib.names("materials") if sub in n]
    found.sort(key=lambda n: home not in Path(lib.path_of("materials", n)).resolve().parents)
    first = PREFERRED.get(hint)
    names = list(dict.fromkeys([*([first] if first and lib.has_material(first) else []), *found]))
    names.sort(key=lambda n: -sum(o in n for o in others))
    return next((n for n in names if _rigid_ok(lib, n)), None)


def _rigid_ok(lib, name: str) -> bool:
    """``name`` can be the base of a rigid object: no liquid, stiff, and no flow law. A ``j2`` metal yields but
    stays a solid, and a sheet material's crease law (paper, with a ``film`` block) does not make it flow: the
    object keeps only its density, surface and stiffness."""
    try:
        m = lib.material(name)
    except Exception:  # noqa: BLE001 - an unreadable material is simply not chosen
        return False
    youngs = getattr(getattr(getattr(m, "elastic", None), "youngs", None), "value", None)
    plastic_ok = m.plastic is None or m.plastic.model == "j2" or getattr(m, "film", None) is not None
    return m.fluid is None and plastic_ok and (youngs is None or youngs >= SOFT_YOUNGS_PA)


def mass_value(req, cat: Category | None, density: float, volume: float, material: str | None = None) -> Value:
    """The object's mass in kg with its source (see the module docstring). ``material``: the chosen base material; a
    category's typical mass is not used when you named a material other than the category's (a plastic "cup" is not a
    ceramic mug's weight): the material's density times the mesh volume is."""
    if req.mass is not None:
        src = req.mass_source or "given by you (no source stated)"
        tag = "CITED" if req.mass_source else "USER"
        return Value("mass", float(req.mass), "kg", tag, src, verify="" if req.mass_source else
                     "add mass_source= (a datasheet, a scale reading)")
    if cat is not None and cat.mass is not None and not _other_material(req, cat, material):
        lo, typ, hi = cat.mass
        return Value("mass", typ, "kg", "ESTIMATE", f"typical {cat.name}: {cat.mass_note}".strip(), lo, hi,
                     verify=f"weigh it (typical {cat.name} {lo * 1e3:.0f}-{hi * 1e3:.0f} g)",
                     evidence=list(cat.evidence))
    m = density * volume
    why = f" (not the typical {cat.name}: {material} is not its usual {cat.material})" if cat and cat.mass else ""
    return Value("mass", m, "kg", "ESTIMATE", f"{density:.0f} kg/m^3 {material or 'base material'} x "
                 f"{volume * 1e6:.4g} cm^3 mesh volume (solid){why}", m * (1 - DENSITY_REL), m * (1 + DENSITY_REL),
                 verify="weigh it; a hollow object weighs less than this")


def _other_material(req, cat: Category, material: str | None) -> bool:
    """True when you (``material=`` or a material word in the text) chose a material of another family than the
    category's (plastic for a ceramic mug; not beech for a hardwood board)."""
    if material is None or cat.material in (None, material):
        return False
    words = [req.material] if req.material else [w for w in W.tokens(req.text or "") if w in W.MATERIAL_WORDS]
    return bool(words) and not _same_family(material, cat.material, words)


def _same_family(chosen: str, usual: str, words: list[str]) -> bool:
    parts = usual.split("_")
    if chosen.split("_")[0] == parts[0]:
        return True
    subs = [sub for w in words for sub in W.MATERIAL_WORDS.get(w, ())]
    return any(part.startswith(sub) for part in parts for sub in subs)


__all__ = ["Category", "Value", "choose_material", "find_category", "mass_value"]
