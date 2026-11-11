# SPDX-License-Identifier: Apache-2.0
"""Real-world scale and mass for an Objaverse object (Objaverse meshes carry neither).

Scale: the longest side of the oriented, Z-up mesh is set to its category's size prior.
Mass: the base material's density x the repaired mesh volume is kept when it falls inside the category's mass range
(a scan with modelled walls); otherwise (a closed outer shell of a hollow item, or a filled package) the object gets
an effective density so its mass equals the category's typical mass. Both are labelled ESTIMATE.

Example:
    >>> from alloy.importers.objaverse.categories import load_categories
    >>> from alloy.importers.objaverse.physical import choose_mass, scale_to_category
    >>> mug = load_categories()["mug"]
    >>> round(scale_to_category(55.0, mug), 6)    # a 55-unit-tall upload
    0.002
    >>> m = choose_mass(mug, density=2300.0, volume=1.4e-4)    # walls modelled: 0.32 kg
    >>> m.effective, round(m.mass, 3)
    (False, 0.322)
    >>> choose_mass(mug, density=2300.0, volume=6.0e-4).effective    # closed shell: 1.38 kg, outside the range
    True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from alloy.importers.objaverse.categories import Category
from alloy.importers.objaverse.errors import ImportRejected

MASS_SLACK = 1.5            # density x volume within [lo / 1.5, hi x 1.5] of the category range keeps the material
DENSITY_RANGE = (10.0, 25_000.0)   # kg/m^3, the card validator's range (SPEC 3.2)


@dataclass(frozen=True)
class MassChoice:
    """How an object gets its mass: the base material, or an effective density (``effective``) with its note."""

    effective: bool
    density: float          # kg/m^3 the card's material carries
    mass: float             # kg = density x volume
    note: str

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def scale_to_category(longest_raw: float, cat: Category) -> float:
    """Uniform scale that makes the longest side equal the category's typical size.

    Raises :class:`ImportRejected` (stage ``scale``) for a degenerate mesh.
    """
    if not longest_raw > 0.0:
        raise ImportRejected("?", "scale", f"longest side {longest_raw!r} is not positive (empty or flat mesh)")
    return cat.size.value / longest_raw


def choose_mass(cat: Category, density: float, volume: float, uid: str = "?") -> MassChoice:
    """Keep ``density`` if density x volume is plausible for the category, else an effective density.

    ``volume`` [m^3] is the repaired (watertight) mesh volume at real scale. Raises :class:`ImportRejected` (stage
    ``mass``) when the volume is not positive or the effective density leaves the card validator's range.
    """
    if not volume > 0.0:
        raise ImportRejected(uid, "mass", f"mesh volume {volume:.3g} m^3 is not positive (repair failed)")
    solid = density * volume
    if cat.mass.contains(solid, slack=MASS_SLACK):
        note = (f"base material density x mesh volume = {solid:.4g} kg, inside the {cat.name} range "
                f"[{cat.mass.lo:g}, {cat.mass.hi:g}] kg")
        return MassChoice(False, density, solid, note)
    eff = cat.mass.value / volume
    lo, hi = DENSITY_RANGE
    if not lo <= eff <= hi:
        raise ImportRejected(uid, "mass", f"effective density {eff:.4g} kg/m^3 outside [{lo:g}, {hi:g}] "
                                          f"(volume {volume:.3g} m^3 implausible for a {cat.name})")
    note = (f"ESTIMATE effective density: typical {cat.name} mass {cat.mass.value:g} kg / mesh volume {volume:.4g} m^3 "
            f"(base material x volume would give {solid:.4g} kg, outside [{cat.mass.lo:g}, {cat.mass.hi:g}] kg)")
    return MassChoice(True, eff, cat.mass.value, note)
