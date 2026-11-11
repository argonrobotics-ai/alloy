# SPDX-License-Identifier: Apache-2.0
"""Film specs: the per-object numbers the film emitter writes (sources: alloy-assets sources/clips/sc8/*.md).

``FilmSpec`` holds the material card values (E, ν, ρ, h, Rayleigh β, yield stress; for a ``plastic.model: j2`` card
its σ_y0 and H, which give the hinges a fibre section, sheet_j2.py) and the film-only ones the card schema has no
field for (dead-fold yield angle, membrane scale, contact radius, air). ``FILM_DEFAULTS`` maps material card names to
their film-only values; a card not listed gets ``FilmExtras()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FilmExtras:
    """Film-only values (design note §3-§6)."""

    yield_angle: float = 0.0            # rad, hinge elastic range (0 = elastic); the dead-fold return angle
    membrane_scale: float = 1.0e-4      # VBD membrane ke multiplier (the strain cap carries in-plane stiffness)
    compress_cap: float = 1.0           # 1 - s_min limit (1 = off)
    contact_radius: float = 5.0e-4      # m, collision radius of film particles (render: mid-surface, no offset)
    air: float = 1.0                    # air force scale
    bend_damping: float = 0.003         # s, hinge-only Rayleigh β: damps crease spring-back (sweep in sc8-film-laws.md)
    # fold lines (creases: scored or factory-folded hinges, the mesh's crease edges, emit.py): their own stiffness
    # and elastic range. A crease is as stiff as crease_ratio thicknesses of plate bending, κ = D/(crease_ratio·h)
    # per metre of crease (VBD's ½·ke·L·(θ-θ̄)², so ke = κ); Lechenault et al. 2014 measured ≈ 200 (Mylar)
    crease_ratio: float = 0.0           # L*/h; 0 = a crease hinge keeps the sheet's own ke
    crease_yield: float = 0.0           # rad; 0 = a crease hinge keeps the sheet's yield_angle


@dataclass(frozen=True)
class FilmSpec:
    youngs: float                       # Pa
    poisson: float
    density: float                      # kg/m^3
    thickness: float                    # m
    yield_stress: float                 # Pa; strain cap = yield_stress / youngs
    rayleigh_beta: float = 0.0          # s
    extras: FilmExtras = FilmExtras()
    j2_yield: float = 0.0               # Pa, a J2 card's σ_y0: the hinges get a fibre section (0 = none)
    j2_hardening: float = 0.0           # Pa, its H

    @property
    def strain_cap(self) -> float:
        return self.yield_stress / self.youngs

    @property
    def areal_density(self) -> float:
        return self.density * self.thickness

    @property
    def plate_rigidity(self) -> float:
        nu = self.poisson
        return self.youngs * self.thickness**3 / (12.0 * (1.0 - nu * nu))

    @property
    def bending_length(self) -> float:
        """D1388 bending length c = (D / w)^(1/3), w = ρ·h·g [N/m²]."""
        return (self.plate_rigidity / (self.areal_density * 9.81)) ** (1.0 / 3.0)

    def with_extras(self, **kw) -> FilmSpec:
        return replace(self, extras=replace(self.extras, **kw))


# HDPE T-shirt bag film. E: ExxonMobil HD 5201 1 % secant MD 1000 MPa (ASTM D882, 12.7 um film); yield 37 MPa MD;
# density 952 kg/m^3; 15 um (12.5-15 um retail T-sacks). ν 0.4 (PE, generic). Dead fold: 76 um HDPE coex returns
# 61 / 84.8 deg MD/CD (US10562684 Table 1): θ_y = 73 deg.
HDPE_EXTRAS = FilmExtras(yield_angle=math.radians(73.0))
HDPE_15UM = FilmSpec(youngs=1.0e9, poisson=0.4, density=952.0, thickness=15.0e-6, yield_stress=37.0e6,
                     extras=HDPE_EXTRAS)

# LLDPE bag film. E: ExxonMobil C4LL 1018 1 % secant MD 190 MPa, yield 9.6 MPa (D882, 25.4 um), 918 kg/m^3.
# Dead fold: 85 um blown PE returns 153 / 152 deg (US10562684): θ_y = 152 deg; "polyethylene ... dead-fold of 0%"
# (US7241832).
LLDPE_EXTRAS = FilmExtras(yield_angle=math.radians(152.0))
LLDPE_30UM = FilmSpec(youngs=1.9e8, poisson=0.44, density=918.0, thickness=30.0e-6, yield_stress=9.6e6,
                      extras=LLDPE_EXTRAS)

# Coffee-filter paper for FILM-4 (the falling filter): paper_80gsm card E 4 GPa (copy paper MD 5-7, CD 2-3 GPa),
# ν 0.3, 0.1 mm; mass set by the law from the measured filter (0.86 g). Yield 40 MPa only sets a 1 % cap (stiff
# shape). THIN_PAPER: the same filter at 10 um, i.e. a film-regime bending stiffness (design note §9, FILM-4).
PAPER_100UM = FilmSpec(youngs=4.0e9, poisson=0.3, density=800.0, thickness=1.0e-4, yield_stress=4.0e7)
THIN_PAPER = FilmSpec(youngs=4.0e9, poisson=0.3, density=800.0, thickness=1.0e-5, yield_stress=4.0e7)

FILM_DEFAULTS: dict[str, FilmExtras] = {
    "hdpe_film_sc8": HDPE_EXTRAS,
    "lldpe_film_sc8": LLDPE_EXTRAS,
    # library film materials (SCR-SOFT-1-1, until the yield angle is a card field): HDPE keeps a fold (73 deg); LDPE
    # and LLDPE take the blown-PE value (152 deg, nearly elastic)
    "film_hdpe": HDPE_EXTRAS,
    "film_ldpe": LLDPE_EXTRAS,
    "film_lldpe": LLDPE_EXTRAS,
}
