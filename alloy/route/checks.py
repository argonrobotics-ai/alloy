# SPDX-License-Identifier: Apache-2.0
"""Router physics checks (SPEC §3.3 steps 2-6). Pure float64 functions on SI numbers; no card types here.

Formulas (object-model §4.6, SPEC §3.3 step 4):
- local squeeze (flat punch of radius a on a half-space): δ_loc = F(1−ν²)/(2aE), a = min(l3/2, 5 mm)
- sheet: clamped circular plate, central load: δ_b = F·R²/(16πD), R = l2/2, D = E·h³/(12(1−ν²));
  bending length c = (D/(ρ·h·g))^(1/3)
- rod: cantilever tip load δ_b = F·L³/(3EI); sag number γ = ρAgL³/(EI) (= 8·tip sag/L under self-weight);
  bending length c = (EI/(ρAg))^(1/3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

G = 9.81
Morphology = Literal["rod", "sheet", "volume"]
SQUEEZE_RADIUS_MAX = 5.0e-3  # m, flat-punch radius cap (SPEC §3.3 step 4)
GRAIN_PACKING = 0.6  # random close packing of monodisperse spheres
SPH_RADIUS_MAX_DEFAULT = 2.0e-3  # m


@dataclass(frozen=True)
class Threshold:
    """One routing comparison: `value` against `limit`; margin is |log10(value/limit)|."""

    name: str
    value: float
    limit: float

    @property
    def passed(self) -> bool:
        return self.value <= self.limit

    @property
    def margin(self) -> float:
        if self.value <= 0.0 or self.limit <= 0.0:
            return math.inf
        return abs(math.log10(self.value / self.limit))


@dataclass(frozen=True)
class RodSection:
    length: float  # m
    area: float  # m^2
    inertia: float  # m^4, weakest bending axis
    radius: float  # m, equivalent radius for capsule links (sqrt(area/π))


def morphology(extents: tuple[float, float, float], is_curve: bool, has_thickness: bool) -> tuple[Morphology,
                                                                                                    list[Threshold]]:
    """Rod / sheet / volume from OBB extents l1 ≥ l2 ≥ l3 (SPEC §3.3 step 3), with the ratios used."""
    l1, l2, l3 = extents
    r21 = l2 / l1 if l1 > 0 else 1.0
    r32 = l3 / l2 if l2 > 0 else 1.0
    ratios = [Threshold("l2/l1 rod", r21, 0.1), Threshold("l3/l2 sheet", r32, 0.05)]
    if is_curve or (r21 < 0.1 and r32 > 0.3):
        return "rod", ratios
    if r32 < 0.05 or has_thickness:
        return "sheet", ratios
    return "volume", ratios


def local_squeeze(F: float, E: float, nu: float, l3: float) -> float:
    """Flat-punch indentation δ_loc = F(1−ν²)/(2aE), a = min(l3/2, 5 mm) [m]."""
    a = min(0.5 * l3, SQUEEZE_RADIUS_MAX)
    return F * (1.0 - nu * nu) / (2.0 * a * E)


def plate_rigidity(E: float, nu: float, h: float) -> float:
    return E * h**3 / (12.0 * (1.0 - nu * nu))


def sheet_bend(F: float, R: float, D: float) -> float:
    """Clamped circular plate centre deflection under a central point load F·R²/(16πD) [m]."""
    return F * R * R / (16.0 * math.pi * D)


def sheet_bending_length(D: float, rho: float, h: float) -> float:
    """c = (D/(ρ·h·g))^(1/3) [m]."""
    return (D / (rho * h * G)) ** (1.0 / 3.0)


def rod_section(kind: str | None, size: tuple[float, ...], extents: tuple[float, float, float],
                curve_length: float | None = None, curve_radius: float | None = None) -> RodSection:
    """Rod length and section from the shape: circle for capsule/cylinder/curve, rectangle for box, else ellipse."""
    if curve_length is not None and curve_radius is not None:
        return _circle(curve_length, curve_radius)
    l1, l2, l3 = extents
    if kind in ("capsule", "cylinder") and len(size) >= 1:
        return _circle(l1, size[0])
    if kind == "box":
        b, h = l2, l3  # bending about the weak axis: I = b·h³/12
        return RodSection(l1, b * h, b * h**3 / 12.0, math.sqrt(b * h / math.pi))
    a, b = 0.5 * l2, 0.5 * l3  # elliptical section, weak axis
    return RodSection(l1, math.pi * a * b, math.pi * a * b**3 / 4.0, math.sqrt(a * b))


def _circle(length: float, r: float) -> RodSection:
    return RodSection(length, math.pi * r * r, math.pi * r**4 / 4.0, r)


def rod_bend(F: float, L: float, EI: float) -> float:
    """Cantilever tip deflection under a tip load F·L³/(3EI) [m]."""
    return F * L**3 / (3.0 * EI)


def rod_gamma(rho: float, area: float, L: float, EI: float) -> float:
    """Self-weight sag number γ = ρAgL³/(EI)."""
    return rho * area * G * L**3 / EI


def rod_bending_length(rho: float, area: float, EI: float) -> float:
    """c = (EI/(ρAg))^(1/3) [m]: the length over which self-weight bends the rod."""
    return (EI / (rho * area * G)) ** (1.0 / 3.0)


CHAIN_LINKS_MIN = 3


def chain_links(L: float, c: float, radius: float, override: int | None = None,
                minimum: int = CHAIN_LINKS_MIN) -> tuple[int, int]:
    """(links to emit, links required). Required: interior segments no longer than the bending length c,
    L = (N−1)·l with half-length ends. Emitted: the required count (SCR-DC3-7: no 30-link floor), or a card's
    `features.chain_links`; at least `minimum` (RouterConfig.chain_links_min), never segments shorter than one
    diameter. The caller sends required > its cap to ROD_VBD (D8)."""
    required = int(math.ceil(L / max(c, 1e-9))) + 1
    links = required if override is None else int(override)
    return min(max(minimum, CHAIN_LINKS_MIN, links), max_chain_links(L, radius, links)), required


def max_chain_links(L: float, radius: float, default: int = 10**6) -> int:
    """Most links with segments no shorter than one diameter (at least CHAIN_LINKS_MIN); `default` without a radius."""
    if radius <= 0:
        return max(default, CHAIN_LINKS_MIN)
    return max(int(math.floor(L / (2.0 * radius))) + 1, CHAIN_LINKS_MIN)


def mpm_voxel(d_feature: float, voxel_max: float) -> float:
    """MPM voxel = min(d_feature/4, voxel_max) (SPEC §3.3 step 2, object-model §3 :356)."""
    return min(0.25 * d_feature, voxel_max)


def grain_count(volume: float, d_grain: float, packing: float = GRAIN_PACKING) -> int:
    """Grains filling `volume` at random close packing."""
    return int(math.ceil(packing * volume / (math.pi * d_grain**3 / 6.0)))


def sph_radius(passage_width: float | None, r_max: float = SPH_RADIUS_MAX_DEFAULT, diameters: float = 8.0) -> float:
    """r = min(r_max, passage/(2·diameters)) so a passage spans ≥ `diameters` particle diameters (LIQ-6)."""
    if passage_width is None or passage_width <= 0.0:
        return r_max
    return min(r_max, passage_width / (2.0 * diameters))


def sph_particle_count(volume: float, r: float) -> int:
    """Particles on a cubic lattice of spacing 2r filling `volume`."""
    return int(math.ceil(volume / (2.0 * r) ** 3))


def min_margin(thresholds: list[Threshold]) -> float:
    """Routing margin: smallest log10 distance over the thresholds that were evaluated."""
    finite = [t.margin for t in thresholds if math.isfinite(t.margin)]
    return min(finite) if finite else math.inf
