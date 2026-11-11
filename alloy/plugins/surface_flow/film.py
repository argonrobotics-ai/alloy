# SPDX-License-Identifier: Apache-2.0
"""Film materials: the liquid or paste a ``SurfaceFlow`` instance carries (one material per instance).

Phase 1 rheology is Newtonian with an optional Bingham yield stress (design note §3.2). Shear thinning (Cross),
thixotropy and solids come with phase 2's process states. Values and sources: design note §7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from alloy.plugins.errors import PluginError


@dataclass(frozen=True)
class FilmMaterial:
    """Density [kg/m³], viscosity [Pa·s] (Newtonian, or Bingham plastic viscosity), yield stress [Pa], surface
    tension [N/m]. ``contact_angle`` [deg] switches the contact-line condition on (``kernels.potential_kernel``):
    where the film meets dry cells its surface slopes at θ, and the line advances only where the edge is steeper.
    None: no contact angle (a film spreads freely). ``evaporation`` [m/s] is a uniform sink."""

    name: str = "water"
    density: float = 998.2
    viscosity: float = 1.002e-3
    yield_stress: float = 0.0
    surface_tension: float = 0.0728
    contact_angle: float | None = None
    evaporation: float = 0.0

    def __post_init__(self) -> None:
        for name in ("density", "viscosity"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise PluginError(f"film material {self.name!r}: {name} must be finite and > 0, got {v}")
        for name in ("yield_stress", "surface_tension", "evaporation"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0.0):
                raise PluginError(f"film material {self.name!r}: {name} must be finite and >= 0, got {v}")
        if self.contact_angle is not None and not 0.0 < self.contact_angle < 180.0:
            raise PluginError(f"film material {self.name!r}: contact_angle must be in (0, 180) degrees")

    @property
    def nu(self) -> float:
        return self.viscosity / self.density


# Sources (design note §7): water, CRC Handbook at 20 °C; latex wall paint, high-shear viscosity 0.18 Pa·s
# (ASTM D4287 range 0.1-0.2), density 1295 kg/m³ (product sheet), yield stress <= 2 Pa and σ 30-35 mN/m are estimates;
# concrete surface paste: plastic viscosity 5 Pa·s (estimate), yield stress of a 100 mm slump mix 1570 Pa (NISTIR 6094
# eq. 19), density 2350 kg/m³; silicone oil (PDMS, 1000 cSt): 0.97 Pa·s, 970 kg/m³, 21 mN/m (manufacturer data).
MATERIALS = {
    "water": FilmMaterial(),
    "latex_paint": FilmMaterial("latex_paint", density=1295.0, viscosity=0.18, yield_stress=2.0,
                                surface_tension=0.032),
    "concrete_paste": FilmMaterial("concrete_paste", density=2350.0, viscosity=5.0, yield_stress=1570.0,
                                   surface_tension=0.0),
    "silicone_oil": FilmMaterial("silicone_oil", density=970.0, viscosity=0.97, surface_tension=0.021),
}


def material(spec: FilmMaterial | str) -> FilmMaterial:
    if isinstance(spec, FilmMaterial):
        return spec
    if spec not in MATERIALS:
        raise PluginError(f"unknown film material {spec!r}; known: {sorted(MATERIALS)} (or pass a FilmMaterial)")
    return MATERIALS[spec]


__all__ = ["FilmMaterial", "MATERIALS", "material"]
