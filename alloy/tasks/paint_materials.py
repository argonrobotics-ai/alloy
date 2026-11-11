# SPDX-License-Identifier: Apache-2.0
"""Paints for the painting tasks: the library card's film values, plus the values a material card has no field for.

The library cards (``library/materials/paint_*.yaml``) hold density, the low-shear viscosity that sets sag and levelling
on a wall, the yield stress and the surface tension, each with its source. :func:`film_material` turns a card into the
surface-flow plug-in's :class:`~alloy.plugins.surface_flow.FilmMaterial`. :data:`PAINTS` adds the rest: volume solids,
the recommended wet film, the Kubelka-Munk scattering per µm of dry film, the wet-edge time, and the yield stress a coat
rebuilds at rest. A card ``film:`` block for these is a later card-schema share (design note §9).

Sources:
- latex eggshell, volume solids 0.42 (0.40-0.44) and 4 mil (102 µm) wet, 1.7 mil (43 µm) dry per coat: the
  Sherwin-Williams ProMar 200 Zero VOC Eg-Shel data sheet 101.83A, "Volume Solids: 42 ±2%" and "350-400 sq. ft. per
  gallon @ 4 mils wet 1.7 mils dry";
- latex white-base scattering 0.187 /µm (class range 0.08-0.30): an estimate, no measured S per µm was found. It hides
  a beige wall at 31 µm DFT (Gym-Anything's white base: about 33 µm);
- tint bases, scattering relative to white (pastel 0.58, medium 0.33, deep 0.020): the TiO2 share of the solids by
  base, white ~30 %, pastel 15-20 %, medium 8-12 %, deep 3-5 % or none (tint-base guides). The deep value is
  Gym-Anything's calibration on coats to hide;
- latex wet edge 180 s (90-240): "a wet edge time of at least about 1.5 minutes", example films 3-7 min open time
  (US 8444758 B2, Eastman);
- latex at rest: the card's yield stress (0.25 Pa) is the fresh coat's, just after the roller's or brush's shear. At
  rest the paint rebuilds it: "A yield stress τ y larger than 1 Pa" (AZoM, sag and levelling of paints), "greater
  than 10 dynes/cm2" before high shear (Hester & Squire 1997). 2 Pa (1-4) is an estimate inside that bound:
  Gym-Anything's τ_y,max and the surface-flow plug-in's latex preset. The 20 s recovery time is Gym-Anything's estimate.
  :func:`aged_yield` rebuilds it to first order: a coat older than about three recovery times holds its rest value;
- fluid acrylic volume solids 0.65 (0.60-0.75): "all acrylic paints shrink considerably (approximately 25-40%) upon
  drying" (Golden Artist Colors). Its scattering (0.06 /µm, semi-transparent artist pigments), wet film (100 µm) and
  wet edge (300 s) are estimates;
- aerosol enamel (Painter's Touch 2X class): the landed wet film is 0.25 solids by volume, after 60 % of the can's
  mass (propellant, acetone) flashes in flight (Gym-Anything's estimates; the safety data sheet lists propane 10-30,
  n-butane 5-10 and acetone 15-40 wt%). A light coat is 75 µm wet, 19 µm dry: two make the data sheet's "1.0 - 2.0
  mils (25 - 50µ)" per coat, as it asks ("Apply 2 or more light coats a few minutes apart", Rust-Oleum PT2X-01). Its
  scattering (0.20 /µm) is Gym-Anything's estimate. Its wet edge is the touch-dry time, "Dries to touch in 20
  minutes";
- colours, sRGB: Sherwin-Williams SW 6244 Naval (47, 61, 76), SW 7036 Accessible Beige (209, 199, 184), SW 7008
  Alabaster (237, 234, 224), from paint-colour databases (search snippets); cadmium red (196, 32, 38) is
  Gym-Anything's, as are gloss red (178, 28, 34) and sealed plywood (214, 182, 140).

Until the surface-flow process states (drying, solids) land, a film's dry thickness is ``h·φ0``: :func:`dft`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks.paint_optics import Optics, srgb8_to_linear

COLOURS = {
    "naval": (47, 61, 76),
    "accessible_beige": (209, 199, 184),
    "alabaster": (237, 234, 224),
    "primer_grey": (110, 112, 114),     # a P-shade primer grey (Gym-Anything's pshade_primer)
    "cadmium_red": (196, 32, 38),
    "gloss_red": (178, 28, 34),          # Gym-Anything's spray enamel
    "plywood": (214, 182, 140),          # Gym-Anything's sealed plywood
}
#: (least luminance Y of the colour, scattering relative to the white base), lightest base first
TINT_BASES = (("white", 0.70, 1.0), ("pastel", 0.40, 0.58), ("medium", 0.15, 0.33), ("deep", 0.0, 0.020))


@dataclass(frozen=True)
class PaintParams:
    card: str                   # library material card
    volume_solids: float        # φ0, dry volume / wet volume
    wet_film: float             # m, the recommended wet film per coat
    s_per_um: float             # Kubelka-Munk scattering per µm of dry film (a white base, for wall paint)
    wet_edge: float             # s
    tinted: bool = True         # wall paint: mixed in the tint base its colour needs
    rest_yield: float = 0.0     # Pa, the yield stress a coat rebuilds at rest (0: the card's, no rebuild)
    recovery: float = 0.0       # s, its first-order recovery time


PAINTS = {
    "latex_eggshell": PaintParams("paint_latex_eggshell", 0.42, 4 * 25.4e-6, 0.187, 180.0, tinted=True,
                                  rest_yield=2.0, recovery=20.0),
    "acrylic_fluid": PaintParams("paint_acrylic_fluid", 0.65, 100e-6, 0.06, 300.0, tinted=False),
    "enamel_aerosol": PaintParams("paint_enamel_aerosol", 0.25, 75e-6, 0.20, 1200.0, tinted=False),
}


def paint(name: str) -> PaintParams:
    if name not in PAINTS:
        raise KeyError(f"unknown paint {name!r}; known: {sorted(PAINTS)}")
    return PAINTS[name]


def film_material(name: str, library=None):
    """The surface-flow :class:`FilmMaterial` of paint ``name`` from its library card (no contact angle; drying is a
    later process state, so no evaporation)."""
    from alloy.cards.library import CardLibrary  # noqa: PLC0415 - card I/O only when asked
    from alloy.plugins.surface_flow.film import FilmMaterial  # noqa: PLC0415

    card = (library or CardLibrary(assets=None)).material(paint(name).card)
    if card.fluid is None:
        raise ValueError(f"paint card {card.name!r} has no fluid block")
    tau = card.plastic.yield_stress.value if card.plastic and card.plastic.yield_stress else 0.0
    sigma = card.fluid.surface_tension.value if card.fluid.surface_tension else 0.0
    return FilmMaterial(name=card.name, density=card.density.value, viscosity=card.fluid.viscosity.value,
                        yield_stress=tau, surface_tension=sigma)  # fmt: skip


def aged_yield(name: str, age: np.ndarray, library=None) -> np.ndarray:  # noqa: ANN001
    """The yield stress [Pa] of a coat ``age`` seconds after it was laid: from the card's (fresh, just sheared) to the
    paint's rest value, ``τ = τ_fresh + (τ_rest - τ_fresh)·(1 - exp(-age/t_r))``. A negative age means not laid yet
    (fresh)."""
    p = paint(name)
    fresh = film_material(name, library).yield_stress
    rest = max(p.rest_yield, fresh)
    if p.recovery <= 0.0 or rest == fresh:
        return np.full(np.shape(age), fresh)
    a = np.clip(np.asarray(age, dtype=np.float64), 0.0, None)
    return fresh + (rest - fresh) * (1.0 - np.exp(-a / p.recovery))


def dft(depth: np.ndarray, name: str) -> np.ndarray:
    """Dry film thickness [m] of a wet film ``depth`` [m] (all its solids, no drying yet)."""
    return np.asarray(depth) * paint(name).volume_solids


def tint_factor(colour: str) -> float:
    """Scattering of the tint base a wall colour is mixed in, relative to the white base (by its luminance Y)."""
    y = float(srgb8_to_linear(COLOURS[colour]) @ np.array([0.2126, 0.7152, 0.0722]))
    return next(f for _, y_min, f in TINT_BASES if y >= y_min)


def optics(name: str, colour: str) -> Optics:
    p = paint(name)
    return Optics(srgb8_to_linear(COLOURS[colour]), p.s_per_um * (tint_factor(colour) if p.tinted else 1.0))


__all__ = ["COLOURS", "PAINTS", "PaintParams", "TINT_BASES", "aged_yield", "dft", "film_material", "optics", "paint",
           "tint_factor"]
