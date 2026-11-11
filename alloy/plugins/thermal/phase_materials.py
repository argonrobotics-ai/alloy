# SPDX-License-Identifier: Apache-2.0
"""Heat v2 materials: thermal and phase-change properties per material name (docs/notes/heat-v2-design.md §3).

``MaterialCard`` carries no thermal block yet, so heat v2 keeps this table keyed by material name. Every value has a
source; uncited values say ESTIMATE in ``source``. Moving the table onto cards is a later SCR to WP1.

Flow (heat v2 step 3, design §6): a material with ``mu_liquid`` > 0 feeds its state back into the solver's per-particle
material arrays each exchange:
- viscosity μ(T, φ) = μ_s^(1−φ) · μ_l(T)^φ (log-linear in φ; μ = μ_l(T) when ``mu_solid`` = 0), with the liquid's
  Arrhenius law μ_l(T) = ``mu_liquid`` · exp(E_μ / R · (1/T − 1/T_l)), T clamped to ≥ T_s;
- a liquid that thickens as it cooks (a starch sauce gelatinising) sets ``mu_done``: the liquid's μ at the liquidus
  moves log-linearly with the doneness X from ``mu_liquid`` (X = 0) to ``mu_done`` (X = 1), with the same Arrhenius
  T factor: μ_l(T, X) = ``mu_liquid``^(1−X) · ``mu_done``^X · exp(E_μ / R · (1/T − 1/T_l));
- MPM: yield stress τ_y(φ) = ``yield_solid`` · (1 − φ)^n, internal friction μ_f(φ) = ``friction_solid`` · (1 − φ)^n
  (a melt is not granular; Newton's MPM default is 0.5) and Young's modulus E(φ) = E_l + (E_s − E_l) · (1 − φ)^n.

Which parts are sourced (WP13 step-3 review): the Arrhenius (Andrade) liquid law is standard. The rest is a MODELLING
CHOICE with no citation: log-linear mixing in φ is a Grunberg-Nissan-type rule (made for liquid mixtures), and the
(1 − φ)^n laws for yield, friction and stiffness are smooth stand-ins. Semi-solid and suspension rheology usually uses
solid-fraction laws with a packing threshold (Krieger and Dougherty 1959 for μ; a power of the solid fraction above a
percolation threshold for the yield stress); none is implemented. A material may set ``mu_liquid`` only with a source
or an ESTIMATE label for each flow value, in ``source``; the step-4 foods label theirs. Mushy-zone conductivity is
the parallel (linear) mix k = (1 − φ) k_s + φ k_l, also a modelling choice.

Enthalpy convention, per unit mass: H = 0 for the solid at its solidus T_s. Below it, H = c_s (T − T_s). The mushy
zone runs from H = 0 to H_l = L + c̄ (T_l − T_s), with c̄ the mean of c_s and c_l: φ = H / H_l and T rises linearly
from T_s to T_l. Above it, H = H_l + c_l (T − T_l). A pure substance has T_l = T_s (an isothermal plateau). A material
without phase change has L = 0, T_l = T_s and c_s = c_l. Boiling starts at H_b = H_l + c_l (T_b − T_l).

Example::

    >>> m = MATERIALS["water"]
    >>> round(m.temperature(m.enthalpy(353.15))[0], 3)
    353.15
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseMaterial:
    """Thermal properties of one material (SI)."""

    name: str
    density: float  # kg/m³, rest density for the conduction sums
    c_solid: float  # J/(kg·K)
    c_liquid: float  # J/(kg·K)
    k_solid: float  # W/(m·K)
    k_liquid: float  # W/(m·K)
    t_solidus: float  # K
    t_liquidus: float  # K (= t_solidus for a pure substance)
    latent: float  # J/kg, latent heat of fusion (0: no phase change)
    t_boil: float = math.inf  # K
    latent_vapor: float = 0.0  # J/kg
    doneness_rate: float = 0.0  # 1/s, Arrhenius pre-exponential A of the cooking progress (0: none)
    doneness_energy: float = 0.0  # J/mol, activation energy E_a
    mu_liquid: float = 0.0  # Pa·s at the liquidus (0: no flow feedback)
    mu_energy: float = 0.0  # J/mol, Arrhenius activation of the liquid's viscosity (0: constant)
    mu_done: float = 0.0  # Pa·s at the liquidus once fully cooked, doneness 1 (0: doneness leaves μ alone)
    mu_solid: float = 0.0  # Pa·s at φ = 0 (0: the liquid law at every φ)
    yield_solid: float = 0.0  # Pa, MPM yield stress at φ = 0
    yield_exponent: float = 1.0  # n in (1 − φ)^n for the yield stress, friction and Young's modulus
    friction_solid: float = 0.0  # MPM internal (Drucker-Prager) friction at φ = 0
    young_solid: float = 0.0  # Pa, MPM Young's modulus at φ = 0 (0: leave the solver's value)
    young_liquid: float = 0.0  # Pa, at φ = 1 (a finite floor keeps MPM on its compliant path)
    source: str = ""
    density_table: tuple[tuple[float, ...], tuple[float, ...]] = ()  # (T [K], ρ [kg/m³]) rows: thermal buoyancy

    def __post_init__(self) -> None:
        if min(self.density, self.c_solid, self.c_liquid, self.k_solid, self.k_liquid) <= 0.0:
            raise ValueError(f"{self.name}: density, specific heats and conductivities must be > 0")
        if self.t_liquidus < self.t_solidus or self.latent < 0.0:
            raise ValueError(f"{self.name}: needs t_liquidus >= t_solidus and latent >= 0")
        if self.t_boil < self.t_liquidus or (math.isfinite(self.t_boil) and self.latent_vapor <= 0.0):
            raise ValueError(f"{self.name}: t_boil must be >= t_liquidus with latent_vapor > 0")
        if min(self.mu_liquid, self.mu_energy, self.mu_solid, self.mu_done, self.yield_solid, self.friction_solid,
               self.young_solid, self.young_liquid) < 0.0 or self.yield_exponent <= 0.0:
            raise ValueError(f"{self.name}: flow parameters must be >= 0 and yield_exponent > 0")
        if self.mu_done > 0.0 and (self.mu_liquid <= 0.0 or self.doneness_rate <= 0.0):
            raise ValueError(f"{self.name}: mu_done needs mu_liquid > 0 and doneness kinetics")
        if self.young_solid > 0.0 and not 0.0 < self.young_liquid <= self.young_solid:
            raise ValueError(f"{self.name}: needs 0 < young_liquid <= young_solid")

    @property
    def h_liquidus(self) -> float:
        """H at the liquidus [J/kg]: the top of the mushy zone."""
        return self.latent + 0.5 * (self.c_solid + self.c_liquid) * (self.t_liquidus - self.t_solidus)

    @property
    def h_boil(self) -> float:
        """H at the boiling point [J/kg] (inf if the material does not boil)."""
        if not math.isfinite(self.t_boil):
            return math.inf
        return self.h_liquidus + self.c_liquid * (self.t_boil - self.t_liquidus)

    def enthalpy(self, t: float | np.ndarray) -> np.ndarray:
        """H(T) [J/kg]. At the solidus of a pure substance this is the solid side (φ = 0)."""
        t = np.asarray(t, dtype=np.float64)
        span = self.t_liquidus - self.t_solidus
        mushy = (t - self.t_solidus) / span * self.h_liquidus if span > 0.0 else np.zeros_like(t)
        return np.where(t < self.t_solidus, self.c_solid * (t - self.t_solidus),
                        np.where(t <= self.t_liquidus, mushy, self.h_liquidus + self.c_liquid * (t - self.t_liquidus)))

    def temperature(self, h: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T [K], liquid fraction φ) of H, before boiling (see ``particle_kernels.temperature`` for the device)."""
        h = np.asarray(h, dtype=np.float64)
        hl = self.h_liquidus
        phi = np.clip(h / hl, 0.0, 1.0) if hl > 0.0 else (h >= 0.0).astype(np.float64)
        t = np.where(h < 0.0, self.t_solidus + h / self.c_solid,
                     np.where(h <= hl, self.t_solidus + phi * (self.t_liquidus - self.t_solidus),
                              self.t_liquidus + (h - hl) / self.c_liquid))
        return t, phi

    def doneness_k(self, t: float | np.ndarray) -> np.ndarray:
        """First-order cooking rate k(T) = A exp(−E_a / (R T)) [1/s]; d(done)/dt = k (1 − done)."""
        return self.doneness_rate * np.exp(-self.doneness_energy / (8.314462618 * np.asarray(t, np.float64)))

    @property
    def flows(self) -> bool:
        """Does this material feed viscosity (and yield, stiffness) back into the solver?"""
        return self.mu_liquid > 0.0

    def viscosity(self, t: float | np.ndarray, phi: float | np.ndarray, done: float | np.ndarray = 0.0) -> np.ndarray:
        """μ(T, φ, X) [Pa·s] (module doc); the device copy is ``phase_kernels.viscosity``."""
        t = np.maximum(np.asarray(t, np.float64), self.t_solidus)
        mu_ref = self.mu_liquid
        if self.mu_done > 0.0:
            x = np.asarray(done, np.float64)
            mu_ref = np.exp((1.0 - x) * np.log(self.mu_liquid) + x * np.log(self.mu_done))
        mu_l = mu_ref * np.exp(self.mu_energy / 8.314462618 * (1.0 / t - 1.0 / self.t_liquidus))
        if self.mu_solid <= 0.0:
            return mu_l
        phi = np.asarray(phi, np.float64)
        return np.exp((1.0 - phi) * np.log(self.mu_solid) + phi * np.log(mu_l))

    def yield_stress(self, phi: float | np.ndarray) -> np.ndarray:
        """τ_y(φ) [Pa] = yield_solid · (1 − φ)^n."""
        return self.yield_solid * (1.0 - np.asarray(phi, np.float64)) ** self.yield_exponent

    def friction(self, phi: float | np.ndarray) -> np.ndarray:
        """μ_f(φ) = friction_solid · (1 − φ)^n (MPM internal friction)."""
        return self.friction_solid * (1.0 - np.asarray(phi, np.float64)) ** self.yield_exponent

    def young(self, phi: float | np.ndarray) -> np.ndarray:
        """E(φ) [Pa] = E_l + (E_s − E_l) · (1 − φ)^n."""
        w = (1.0 - np.asarray(phi, np.float64)) ** self.yield_exponent
        return self.young_liquid + (self.young_solid - self.young_liquid) * w

    def diffusivity(self, liquid: bool = False) -> float:
        """α = k / (ρ c) [m²/s] of the solid or the liquid."""
        k, c = (self.k_liquid, self.c_liquid) if liquid else (self.k_solid, self.c_solid)
        return k / (self.density * c)


# Liquid water's density at 0.101325 MPa, 0.01-95.01 C in 5 K steps: NIST Chemistry WebBook SRD 69, fluid properties,
# isobaric (https://webbook.nist.gov/cgi/fluid.cgi?P=0.101325&TLow=0&THigh=95&TInc=5&Digits=6&ID=C7732185&Action=Load
# &Type=IsoBar&TUnit=C&PUnit=MPa&DUnit=kg%2Fm3, fetched 2026-11-08). It carries the 4 C maximum. Thermal buoyancy
# (alloy.plugins.thermal.buoyancy) reads it.
WATER_DENSITY = (tuple(273.15 + 0.01 + 5.0 * k for k in range(20)),
                 (999.844, 999.966, 999.702, 999.101, 998.205, 997.045, 995.646, 994.030, 992.213, 990.209, 988.031,
                  985.688, 983.191, 980.545, 977.759, 974.837, 971.784, 968.605, 965.303, 961.881))

MATERIALS: dict[str, PhaseMaterial] = {
    "water": PhaseMaterial(
        "water", density=997.05, c_solid=2096.8, c_liquid=4180.0, k_solid=2.22, k_liquid=0.607,
        t_solidus=273.15, t_liquidus=273.15, latent=333.55e3, t_boil=373.15, latent_vapor=2256.4e3,
        source="IAPWS-95 / CRC Handbook: liquid c_p 4.18 kJ/(kg·K) and k 0.607 W/(m·K) at 25 °C; ice c_p 2.097 "
               "kJ/(kg·K) at 0 °C (IAPWS R10-06) and k 2.22 W/(m·K) near 0 °C; fusion 333.55 kJ/kg; vaporisation "
               "2256.4 kJ/kg at 100 °C; density = library/materials/water.yaml; density_table: NIST WebBook "
               "(WATER_DENSITY)",
        density_table=WATER_DENSITY),
    # Foods for the heat-v2 tasks (step 4). Published values disagree, often by 2-10x (search notes, 2026-11-06), so
    # every value not read from a table says ESTIMATE; the MPM stiffness and strength are numerical stand-ins.
    "dark_chocolate": PhaseMaterial(
        "dark_chocolate", density=1260.0, c_solid=2000.0, c_liquid=2000.0, k_solid=0.3, k_liquid=0.3,
        t_solidus=303.4, t_liquidus=309.3, latent=60.0e3, mu_liquid=1.5, mu_energy=30.0e3, mu_solid=50.0,
        yield_solid=2.0e4, yield_exponent=2.0, friction_solid=0.5, young_solid=1.0e6, young_liquid=1.0e4,
        source="ESTIMATE, CENTRAL values of the ranges quoted in snippets (full texts not read; WP13 review of step "
               "4): c 2000 J/(kg·K) (1670 liquid at 40-60 °C, ECCE6 2007; Bikos et al. 2022 2600 solid, 2000 liquid); "
               "k 0.3 W/(m·K) (Bikos et al. 2022 0.45 solid, 0.55 liquid; ECCE6 0.16-0.26 liquid); L 60 J/g (46.5-78 "
               "J/g across DSC studies, Calva-Estrada et al. 2020 46.5 for a 70 % dark); melting 30.25-36.15 °C "
               "(Calva-Estrada et al. 2020); density 1260 (liquid at 35 °C, ECCE6 2007, also used for the solid); "
               "μ 1.5 Pa·s at the liquidus (Casson values at 40 °C span 0.3-7.2 Pa·s), E_μ 30 kJ/mol (two-point "
               "Arrhenius of anhydrous milk fat, a fat-continuous analogue); solid μ 50 Pa·s, yield 20 kPa and E 1 MPa "
               "are numerical stand-ins (real solid chocolate is GPa-stiff), enough to hold a 1 cm block within 1 µm"),
    "beef_lean": PhaseMaterial(
        "beef_lean", density=1056.0, c_solid=2093.0, c_liquid=3517.0, k_solid=0.48, k_liquid=0.48,
        t_solidus=263.15, t_liquidus=271.43, latent=239.6e3, t_boil=373.15, latent_vapor=1620.0e3,
        source="ASHRAE Handbook—Refrigeration 2014, sirloin, lean (via engineersedge.com): initial freezing 28.9 °F, "
               "c 0.84 / 0.50 Btu/(lb·°F) above / below freezing, L 103 Btu/lb (water fraction 0.718 = L / 333.55 "
               "kJ/kg); k 0.48 W/(m·K) across the fibres at 7 °C (79 % water, engineeringtoolbox.com), also used "
               "frozen (not sourced; the tasks start thawed); ESTIMATE: solidus −10 °C (a linear mushy zone stands in "
               "for gradual freezing), density 1056 from Choi-Okos (1986) component densities at 72 % water, 22 % "
               "protein, 5 % fat, 1 % ash, boiling at 100 °C taking 0.718 x 2256.4 kJ/kg (the water's share)"),
    "starch_sauce": PhaseMaterial(
        "starch_sauce", density=1083.0, c_solid=1786.0, c_liquid=3549.0, k_solid=0.52, k_liquid=0.52,
        t_solidus=263.15, t_liquidus=272.15, latent=250.5e3, t_boil=373.15, latent_vapor=1694.6e3,
        doneness_rate=9.19e20, doneness_energy=150.0e3, mu_liquid=5.4e-3, mu_energy=18.4e3, mu_done=7.8,
        source="cold milk with corn starch and sugar, Gym-Anything's stir-sauce: ESTIMATE recipe 500 g whole milk "
               "(USDA SR 01077: water 88.13, protein 3.15, fat 3.25, carbohydrate 4.80, ash 0.67 %), 30 g corn "
               "starch (USDA SR 20027: water 8.32 %), 60 g sucrose; water fraction 0.751; c, k and density from "
               "Choi-Okos (1986) component models at 20 °C (Siebel below freezing), L = 333.55 x and boiling "
               "2256.4 x kJ/kg, freezing −1 °C to −10 °C as porridge_oat. Doneness = the gelatinised share of the "
               "starch, first order: E_a 150 kJ/mol and A 9.19e20 1/s, calibrated to the reference task's own targets "
               "(Gym-Anything ingredient-mixing report §6: a 5 K/min ramp is 18 % cooked at 60 °C and 95 % at 77 °C; "
               "law H-9c), inside maize starch DSC ranges (onset about 62-66 °C in excess water; Pineda-Gómez et al. "
               "2010, Ingeniería y Ciencia 6(11):129). Viscosity ESTIMATE, Newtonian (SPH): raw 3 mPa·s at 20 °C "
               "(whole milk about 2, raised by sugar and swollen granules), cooked 3.8 Pa·s at 25 °C (a 5 % corn "
               "starch paste's power law K 16.7 Pa·s^n, n 0.36, at 10 1/s, the spoon's shear), log-linear in the "
               "gelatinised share; E_μ 18.4 kJ/mol (the reference task's consistency law) for both; values at the "
               "liquidus 272.15 K"),
    "porridge_oat": PhaseMaterial(
        "porridge_oat", density=1050.0, c_solid=1893.0, c_liquid=3641.0, k_solid=0.56, k_liquid=0.56,
        t_solidus=263.15, t_liquidus=272.15, latent=278.8e3, t_boil=373.15, latent_vapor=1886.0e3,
        mu_liquid=15.75, mu_energy=14.3e3,
        source="water fraction 0.836 (USDA SR 08121, oats cooked with water: 195.6 g water per 234 g cup); ESTIMATE "
               "from water-fraction rules: c = 4.19 x + 0.84 (1 − x) above and 2.1 x + 0.84 (1 − x) below freezing "
               "(Siebel), k = 0.148 + 0.493 x (Sweat 1974), L = 333.55 x, boiling 2256.4 x kJ/kg, freezing −1 °C to "
               "−10 °C, density 1050 (Choi-Okos with 12 % carbohydrate, 2.5 % protein, 1.5 % fat); viscosity "
               "ESTIMATE 10 Pa·s at 20 °C falling to 4 Pa·s at 74 °C (E_μ 14.3 kJ/mol; no measured porridge "
               "viscosity found, oat systems are shear-thinning), 15.75 Pa·s at the liquidus by that law"),
    # Law fixtures: synthetic values chosen for the laws' time and length scales (not physical materials).
    "law_conductor": PhaseMaterial(
        "law_conductor", density=1000.0, c_solid=1000.0, c_liquid=1000.0, k_solid=50.0, k_liquid=50.0,
        t_solidus=300.0, t_liquidus=300.0, latent=0.0, source="law fixture: alpha = 5e-5 m^2/s, no phase change"),
    "law_pcm": PhaseMaterial(
        "law_pcm", density=1000.0, c_solid=1000.0, c_liquid=1000.0, k_solid=50.0, k_liquid=50.0,
        t_solidus=300.0, t_liquidus=300.0, latent=5.0e4, source="law fixture: pure substance melting at 300 K"),
    "law_stefan_pcm": PhaseMaterial(
        "law_stefan_pcm", density=1000.0, c_solid=1000.0, c_liquid=1000.0, k_solid=200.0, k_liquid=200.0,
        t_solidus=300.0, t_liquidus=300.0, latent=5.0e4,
        source="law fixture: law_pcm with alpha = 2e-4 m^2/s, so the Stefan front crosses 12-17 particles"),
    "law_reservoir": PhaseMaterial(
        "law_reservoir", density=1000.0, c_solid=1.0e9, c_liquid=1.0e9, k_solid=200.0, k_liquid=200.0,
        t_solidus=300.0, t_liquidus=300.0, latent=0.0,
        source="law fixture: heat capacity 1e6x law_stefan_pcm's, a fixed-temperature wall for the Stefan law"),
    "law_doneness": PhaseMaterial(
        "law_doneness", density=1000.0, c_solid=1000.0, c_liquid=1000.0, k_solid=1.0, k_liquid=1.0,
        t_solidus=200.0, t_liquidus=200.0, latent=0.0, doneness_rate=1.0e10, doneness_energy=7.0e4,
        source="law fixture: Arrhenius cooking progress, k(350 K) = 0.36/s"),
    "law_paste": PhaseMaterial(
        "law_paste", density=1000.0, c_solid=1000.0, c_liquid=1000.0, k_solid=1.0, k_liquid=1.0,
        t_solidus=300.0, t_liquidus=300.0, latent=5.0e4, mu_liquid=0.5, mu_energy=3.0e4, mu_solid=50.0,
        yield_solid=2.0e4, yield_exponent=2.0, friction_solid=0.5, young_solid=1.0e6, young_liquid=1.0e4,
        source="law fixture: a paste that holds its shape solid (yield 20 kPa) and flows melted (0.5 Pa·s)"),
    "law_paste_x10": PhaseMaterial(
        "law_paste_x10", density=1000.0, c_solid=1000.0, c_liquid=1000.0, k_solid=1.0, k_liquid=1.0,
        t_solidus=300.0, t_liquidus=300.0, latent=5.0e4, mu_liquid=5.0, mu_energy=3.0e4, mu_solid=50.0,
        yield_solid=2.0e4, yield_exponent=2.0, friction_solid=0.5, young_solid=1.0e6, young_liquid=1.0e4,
        source="law fixture: law_paste with a 10x liquid viscosity (H-8b's scaling)"),
}


def material(name: str) -> PhaseMaterial:
    """The table entry ``name``; raises with the known names."""
    if name not in MATERIALS:
        raise KeyError(f"no heat-v2 material {name!r}; known: {sorted(MATERIALS)} (add it to "
                       f"alloy/plugins/thermal/phase_materials.py with a source)")
    return MATERIALS[name]
