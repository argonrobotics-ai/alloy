# SPDX-License-Identifier: Apache-2.0
"""Per-object mass specs, card headers and sourcing-queue rows for YCB and GSO imports.

YCB: a measured mass (``ycb.OBJECTS``) becomes a ``fit`` density; parts without one follow their ``estimate`` rule.
GSO: a catalogue weight in a shoe description, else net contents from a packaged product's description, else the
category rule (``estimates``); all are ESTIMATE and queued for the sourcing tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alloy.cards.validate import DENSITY_RANGE
from alloy.importers.common.cards import MassSpec
from alloy.importers.common.queue import QueueRow
from alloy.importers.ycb_gso import estimates as est
from alloy.importers.ycb_gso import gso, ycb
from alloy.importers.ycb_gso.relabels import RELABELS, Relabel

MASS_REL_UNC = 0.01            # digital-scale reading: +-1 % or +-1 g, whichever is larger (ICAR15 Sec. II)
NAMED_RANGE = (100.0, 1500.0)  # kg/m^3 a packaged product's name-based mass must imply
CAPACITY_WORDS = re.compile(r"\b(cup|mug|bowl|ramekin|basket|holder|tumbler|pitcher|container|planter|pot|vase|"
                            r"bin|canister set|storage)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ObjectSpec:
    """Everything library-specific about one object: mass, surface material, source, licence, queue rows."""

    name: str
    library: str                 # "ycb" | "gso"
    mass: MassSpec
    material: str
    material_note: str
    source_url: str
    attribution: str
    title: str
    queue: tuple[QueueRow, ...] = ()


def _measured(grams: float, ref: str) -> MassSpec:
    kg = grams / 1e3
    d = max(1e-3, MASS_REL_UNC * kg)
    return MassSpec("mass", kg, (max(kg - d, 0.5 * kg), kg + d), f"{grams:g} g measured, {ref}")


def ycb_fits(results: dict[str, dict]) -> dict[str, float]:
    """Fitted effective density (kg/m^3) of every YCB card with a measured mass and a result.

    Example::

        ycb_fits({"ycb_025_mug": {"volume": 1.378e-4}})   # {"ycb_025_mug": 856.3}
    """
    return {o.card_name: (o.mass_g / 1e3) / results[o.card_name]["volume"]
            for o in ycb.OBJECTS if o.mass_g is not None and o.card_name in results}


def _ycb_estimate(o: ycb.YcbObject, base_density: float, fits: dict[str, float]) -> tuple[float, tuple, str]:
    rule = o.estimate or "material"
    if rule.startswith("packed:"):
        f = float(rule.split(":")[1])
        return base_density * f, (base_density * f * 0.6, base_density * min(1.0, f * 1.4)), (
            f"{o.material} density {base_density:g} kg/m^3 x packing fraction {f:g} ({o.material_note})")
    if rule.startswith("like:"):
        other = rule.split(":")[1]
        rho = fits[other]
        return rho, (0.5 * rho, 2.0 * rho), f"effective density of {other} (same make, measured mass), {o.material_note}"
    return base_density, (0.8 * base_density, 1.1 * base_density), f"solid {o.material} ({o.material_note})"


def ycb_spec(o: ycb.YcbObject, base_density: float, fits: dict[str, float]) -> ObjectSpec:
    """The spec of one YCB object.

    Example::

        ycb_spec(ycb.by_card_name()["ycb_025_mug"], 905.0, fits).mass.kind   # "mass"
    """
    title = o.dataset_id
    common = dict(name=o.card_name, library="ycb", material=o.material, material_note=o.material_note,
                  source_url=o.url, attribution=ycb.ATTRIBUTION, title=title)
    if o.mass_g is not None:
        return ObjectSpec(mass=_measured(o.mass_g, o.ref), **common)
    rho, unc, note = _ycb_estimate(o, base_density, fits)
    row = QueueRow(o.card_name, "material.density", rho, unc, f"ESTIMATE: {note}", "YCB part without a mass",
                   f"YCB {o.dataset_id.split('_', 1)[1].replace('_', ' ')} mass grams", {"dataset_id": o.dataset_id})
    return ObjectSpec(mass=MassSpec("density", rho, unc, note), queue=(row,), **common)


def _named_mass(meta: gso.GsoMeta, volume: float) -> tuple[MassSpec, str] | None:
    """Mass from a shoe's catalogue weight or a packaged product's net contents, if plausible."""
    cat = meta.categories[0] if meta.categories else ""
    found = est.net_contents(meta.description)
    if found is None:
        return None
    kg, text = found
    if cat == "Shoe":
        return MassSpec("mass", kg, (0.9 * kg, 1.1 * kg), f"catalogue weight {text!r} in the product description "
                                                         "(per shoe, size not stated)"), "shoe weight"
    if CAPACITY_WORDS.search(meta.description) or cat not in ("Consumer Goods", "Bottles and Cans and Cups", ""):
        return None
    pack = max(est.PACKAGING * kg, est.MIN_PACKAGING)
    m = kg + pack
    if not NAMED_RANGE[0] <= m / volume <= NAMED_RANGE[1]:
        return None
    note = (f"net contents {text!r} in the product description (liquids at water density) + {pack * 1e3:.0f} g "
            f"packaging (max of {est.PACKAGING:.0%} and {est.MIN_PACKAGING * 1e3:.0f} g)")
    return MassSpec("mass", m, (kg, kg + 2 * pack), note), "net contents"


def _category_mass(meta: gso.GsoMeta, volume: float, fits: dict[str, float]) -> tuple[MassSpec, str, str]:
    cat = meta.categories[0] if meta.categories else ""
    return _rule_mass(est.CATEGORY.get(cat, est.DEFAULT_RULE), volume, fits)


def _rule_mass(rule: est.CategoryRule, volume: float, fits: dict[str, float]) -> tuple[MassSpec, str, str]:
    if rule.analogue:
        rho, unc, src = est.ycb_density(fits, rule.analogue)
        return MassSpec("density", rho, unc, f"{rule.note}: {src}"), rule.material, rule.note
    if rule.mass is not None:
        lo, hi = rule.unc
        rho = max(rule.mass / volume, DENSITY_RANGE[0])
        floor = "" if rho == rule.mass / volume else f", raised to the {DENSITY_RANGE[0]:g} kg/m^3 card floor"
        return (MassSpec("density", rho, (min(lo / volume, rho), max(hi / volume, rho)),
                         f"{rule.note}: {rule.mass:g} kg ({lo:g}-{hi:g}) over the scan volume{floor}"),
                rule.material, rule.note)
    return MassSpec("density", rule.density, rule.unc, rule.note), rule.material, rule.note


def gso_spec(meta: gso.GsoMeta, volume: float, fits: dict[str, float]) -> ObjectSpec:
    """The spec of one GSO model (always ESTIMATE, always queued).

    Example::

        gso_spec(gso.read_metadata(Path("metadata.pbtxt")), 9.1e-4, fits).queue[0].field   # "material.density"
    """
    rel = RELABELS.get(meta.name)
    if rel is not None:
        return _relabelled_spec(meta, rel, volume, fits)
    name = gso.card_name(meta.name)
    cat = meta.categories[0] if meta.categories else ""
    mass, material, mat_note = _category_mass(meta, volume, fits)
    named = _named_mass(meta, volume)
    if named is not None:
        mass = MassSpec("density", named[0].value / volume, (named[0].unc[0] / volume, named[0].unc[1] / volume),
                        f"{named[0].note}: {named[0].value * 1e3:.4g} g over the scan volume")
    rho_unc = mass.unc
    row = QueueRow(name, "material.density", mass.value, rho_unc, f"ESTIMATE: {mass.note}", cat or "uncategorised",
                   f"{meta.description.splitlines()[0] if meta.description else meta.name} weight",
                   {"gtin": meta.gtin, "brand": meta.brand, "gso_name": meta.name, "volume_m3": volume,
                    "mass_kg": mass.value * volume})
    return ObjectSpec(name=name, library="gso", mass=mass, material=material,
                      material_note=f"{cat or 'uncategorised'}: {mat_note}", source_url=gso.archive_url(meta.name),
                      attribution=gso.ATTRIBUTION, title=meta.name, queue=(row,))


def _relabelled_spec(meta: gso.GsoMeta, rel: Relabel, volume: float, fits: dict[str, float]) -> ObjectSpec:
    """A GSO model whose upstream title names another object: named and estimated for what the mesh shows."""
    mass, material, mat_note = _rule_mass(rel.rule, volume, fits)
    row = QueueRow(rel.card, "material.density", mass.value, mass.unc, f"ESTIMATE: {mass.note}", rel.what,
                   f"{rel.what} weight", {"gso_name": meta.name, "mislabel": True, "volume_m3": volume,
                                          "mass_kg": mass.value * volume})
    title = f"{meta.name} (upstream title is a mislabel: the mesh is a {rel.what})"
    return ObjectSpec(name=rel.card, library="gso", mass=mass, material=material, material_note=f"{rel.what}: {mat_note}",
                      source_url=gso.archive_url(meta.name), attribution=gso.ATTRIBUTION, title=title, queue=(row,))
