# SPDX-License-Identifier: Apache-2.0
"""Materials for RoboCasa imports: the category table, new material cards and part densities.

``data/categories.yaml`` maps each category to the material whose surface the collision shapes get and to a density
material per visual-part role. ``data/materials.yaml`` defines the new ``robocasa_*`` materials: ``surface`` ones
become Alloy material cards, ``fill`` ones are densities used only for part masses. Everything else is an Alloy
library material (``library/materials``).

Example::

    >>> from alloy.importers.robocasa.materials import MaterialTable
    >>> table = MaterialTable.load()
    >>> table.object_spec("glass_cup").surface
    'glass_soda_lime'
    >>> table.density("robocasa_ice").value
    917.0
    >>> part_role("GlassCup003_Clear"), part_role("Saucepan002"), part_role("Jar010_Lid")
    ('Clear', 'body', 'Lid')
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from alloy.cards import CardLibrary
from alloy.cards import types as ct
from alloy.core.quantity import Q

DATA = Path(__file__).with_name("data")
ROLES = ("Clear", "Prop", "prop", "Prpo", "Porp", "Lid", "Paper", "Handle", "Tong", "CreamCheese")


def part_role(stem: str) -> str:
    """The role of a visual part from its mesh name: the trailing Lightwheel role word, else ``body``."""
    for word in reversed(re.split(r"[_\W]+", stem)):
        clean = re.sub(r"\d+$", "", word)
        if clean in ROLES:
            return "body" if clean == "CreamCheese" else clean
    return "body"


@dataclass(frozen=True)
class ObjectSpec:
    """How one object category maps to materials, and its plausible mass and size (sanity bounds)."""

    category: str
    surface: str
    parts: dict[str, str]
    mass_kg: tuple[float, float]
    size_m: tuple[float, float]
    shell_mm: float = 2.0

    def material_for(self, role: str) -> str:
        """Density material of a part role (unlisted roles use ``body``)."""
        return self.parts.get(role, self.parts["body"])


@dataclass(frozen=True)
class FixtureSpec:
    """An appliance kind: surface material and a typical whole-unit mass with its note."""

    kind: str
    surface: str
    mass_kg: float
    note: str


def _q(row: list, i: int = 0) -> Q:
    value, lo, hi, source, note = row[i], row[i + 1], row[i + 2], row[i + 3], row[i + 4]
    return Q(float(value), source, (float(lo), float(hi)), note)


def _surface_card(name: str, row: dict) -> ct.MaterialCard:
    mu_s, mu_d, lo, hi, source, note = row["mu"]
    surface = ct.Surface(mu_static=Q(float(mu_s), source, (float(lo), float(hi)), note),
                         mu_dynamic=Q(float(mu_d), source, (float(lo), float(min(hi, mu_s))), note),
                         restitution=_q(row["restitution"]))
    return ct.MaterialCard(name=name, density=_q(row["density"]), surface=surface)


@dataclass
class MaterialTable:
    """Category specs, new material cards and fill densities; library materials come from ``library``."""

    objects: dict[str, ObjectSpec]
    fixtures: dict[str, FixtureSpec]
    cards: dict[str, ct.MaterialCard]
    fills: dict[str, Q]
    library: CardLibrary = field(default_factory=CardLibrary)

    @classmethod
    def load(cls, data: Path = DATA, library: CardLibrary | None = None) -> MaterialTable:
        """Read ``categories.yaml`` and ``materials.yaml`` from ``data``."""
        cats = yaml.safe_load((data / "categories.yaml").read_text())
        mats = yaml.safe_load((data / "materials.yaml").read_text())
        objects = {k: ObjectSpec(k, v["surface"], dict(v["parts"]), tuple(v["mass_kg"]), tuple(v["size_m"]),
                                 float(v.get("shell_mm", 2.0))) for k, v in cats["objects"].items()}
        fixtures = {k: FixtureSpec(k, v["surface"], float(v["mass_kg"]), v["note"]) for k, v in cats["fixtures"].items()}
        cards = {k: _surface_card(k, v) for k, v in mats["surface"].items()}
        fills = {k: _q(v) for k, v in mats["fill"].items()}
        return cls(objects, fixtures, cards, fills, library or CardLibrary())

    def object_spec(self, category: str) -> ObjectSpec:
        """The spec of an object category; ``KeyError`` names the table to extend."""
        try:
            return self.objects[category]
        except KeyError:
            raise KeyError(f"no RoboCasa object category {category!r} in {DATA / 'categories.yaml'}; add a row "
                           "(surface, parts, mass_kg, size_m)") from None

    def fixture_spec(self, kind: str) -> FixtureSpec:
        """The spec of a fixture kind; ``KeyError`` names the table to extend."""
        try:
            return self.fixtures[kind]
        except KeyError:
            raise KeyError(f"no RoboCasa fixture kind {kind!r} in {DATA / 'categories.yaml'}; add a row "
                           "(surface, mass_kg, note)") from None

    def material(self, name: str) -> ct.MaterialCard:
        """A new ``robocasa_*`` surface card or an Alloy library material card."""
        return self.cards[name] if name in self.cards else self.library.material(name)

    def density(self, name: str) -> Q:
        """Density of a fill, a new card or a library material (kg/m^3)."""
        if name in self.fills:
            return self.fills[name]
        return self.material(name).density

    def missing(self) -> list[str]:
        """Material names the tables use that resolve nowhere (should be empty)."""
        names = {s.surface for s in self.objects.values()} | {s.surface for s in self.fixtures.values()}
        names |= {m for s in self.objects.values() for m in s.parts.values()}
        return sorted(n for n in names if n not in self.fills and n not in self.cards
                      and not self.library.has_material(n))


@cache
def default_table() -> MaterialTable:
    """The packaged tables over the default card library (cached)."""
    return MaterialTable.load()


__all__ = ["DATA", "FixtureSpec", "MaterialTable", "ObjectSpec", "default_table", "part_role"]
