# SPDX-License-Identifier: Apache-2.0
"""Parameters of the process states (``process.py``; design §4.8, §5, §7). Dataclasses, or dicts from JSON and cards.
Values are sourced where a source exists and marked *estimate* where none does."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow import process_kernels as PK


def _check(owner: str, obj: Any, positive: tuple[str, ...] = (), non_negative: tuple[str, ...] = ()) -> None:
    for name in positive + non_negative:
        v = getattr(obj, name)
        ok = math.isfinite(v) and (v > 0.0 if name in positive else v >= 0.0)
        if not ok:
            raise PluginError(f"process.{owner}: {name} must be finite and {'> 0' if name in positive else '>= 0'}, "
                              f"got {v}")


def _law(struct: Any, obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        setattr(struct, name, float(getattr(obj, name)))
    return struct


@dataclass(frozen=True)
class Setting:
    """``τ(a) = tau_0 + a_thix·a/s_t + tau_is·(a/(t_is·s_t))^m`` [Pa]. The defaults are a 100 mm slump concrete at
    21 °C: τ_0 1566 Pa (NISTIR 6094 eq. 19), A_thix 0.25 Pa/s (Roussel 2006), τ_IS 383 kPa at t_IS 4 h and m 5.13 (the
    ASTM C403 set anchors through punch factor 9; C:424-431). ``initial_age`` [s] is the age of a forming bulk at the
    start; ``film_factor`` scales the film's yield (a finished surface layer is 15-30 % of the bulk, C:432)."""

    tau_0: float = 1566.0
    a_thix: float = 0.25
    tau_is: float = 383.0e3
    t_is: float = 4.0 * 3600.0
    m: float = 5.13
    s_t: float = 1.0
    film_factor: float = 1.0
    initial_age: float = 0.0

    def __post_init__(self) -> None:
        _check("setting", self, ("tau_0", "t_is", "m", "s_t", "film_factor"), ("a_thix", "tau_is", "initial_age"))

    def yield_stress(self, age: float | np.ndarray) -> float | np.ndarray:
        """Host form of the law [Pa] (float64), for tests and tasks."""
        a = np.maximum(age, 0.0)
        return self.tau_0 + self.a_thix * a / self.s_t + self.tau_is * (a / (self.t_is * self.s_t)) ** self.m

    def law(self) -> PK.SetLaw:
        return _law(PK.SetLaw(), self, ("tau_0", "a_thix", "tau_is", "t_is", "m", "s_t", "film_factor"))


@dataclass(frozen=True)
class Soil:
    """A soil on the charts (R:257-270). Strength ``σ_c = sigma_dry·(1 - hyd)² + sigma_wet`` falls as the soil soaks
    (``dhyd/dt = (1 - hyd)/t_soak(T)`` while wet, ``t_soak`` halving per ``soak_doubling`` °C above 20 °C). A wet
    cell loses soil [kg/m²/s] by:
    - jet impact ``k_i·frac·(p_stag/σ_c - 1)+``, ``frac`` the jets' share of the cell (the supply's
      ``impact_share``: their footprint over the cell's area) and ``p_stag`` their stagnation pressure (the supply's
      ``impact_p`` is ``frac·p_stag``, so the kernel applies ``k_i·(impact_p/σ_c - frac)+``);
    - shear ``k_e·(τ_w - σ_c)+`` (Partheniades 1965), ``τ_w`` the supply's jet-sheet shear (``jet_tau``) under a
      jet's sheet (``jet_cover > 0``), else the larger of ``jet_tau`` and the film's own ``3μ|q̄|/h²`` (+ τ_y while
      flowing); cells under a sheet count as wet (the supply sweeps their film);
    - Wilson's cleaning front (Wilson et al. 2014): next to a clean cell (< ``m_clean``) whose film flows into it the
      soil recedes at ``v = k_front·hyd·beta·ρ·u_n²·h``, and the cell loses its start load at ``m_0·v·L/A``;
    - a tool wiping it, ``k_wipe·cover·τ_wipe/σ_c`` (Archard form), ``τ_wipe = μ|v|/max(gap, h, h_wet)``.
    A cell under a ceiling tool also loses soil, wet or dry, to abrasion ``k_abr·cover·press·|v|·σ_ref/σ_c`` (Archard;
    WP8's G4: GA ``params.SPONGE`` k_abr 1.2e-5 kg/(m²·Pa·m), σ_ref 1 kPa, estimates; ``tf.press`` the pad pressure).
    Defaults: dried tomato paste 0.20 kg/m² (ASTM F2324-03 recipe), σ 200 kPa dry and 150 Pa wet, t_soak 5.9 s at
    20 °C doubling per 15 °C (R:532-535), and Gym-Anything's removal constants (``surface_rinsing/params.py:95,
    135-148``, as WP8's ``rinse_soil``): k_i 2.0 kg/(m²·s), k_e 0.05 kg/(m²·s·Pa), k' 2e-3 s·m/kg, clean below 1e-6
    kg/m², β 1.2, a 0.6 mm jet. ``k_wipe`` is an *estimate*. ``charts``: the soiled charts, by index or name (empty:
    all); ``set_soil`` writes any map and restarts the start load. Removed soil joins the film as ``species``."""

    initial: float = 0.20
    charts: tuple[int | str, ...] = ()
    sigma_dry: float = 200.0e3
    sigma_wet: float = 150.0
    t_soak: float = 5.9
    soak_doubling: float = 15.0
    temperature: float = 20.0
    k_e: float = 0.05
    k_i: float = 2.0
    k_front: float = 2.0e-3
    m_clean: float = 1.0e-6
    beta: float = 1.2
    jet_area: float | None = None                   # ignored: the supply's impact_share is the jets' share (kept so
    #                                                 callers written before it build; WP8's rinse_plates drops it)
    k_wipe: float = 1.0e-3
    k_abr: float = 1.2e-5
    sigma_ref: float = 1.0e3
    species: str = "soil"

    def __post_init__(self) -> None:
        object.__setattr__(self, "charts", tuple(self.charts))
        _check("soil", self, ("sigma_wet", "t_soak", "soak_doubling", "m_clean", "beta", "sigma_ref"),
               ("initial", "sigma_dry", "k_e", "k_i", "k_front", "k_wipe", "k_abr"))

    @property
    def t_soak_at(self) -> float:
        """Soak time [s] at ``temperature``."""
        return self.t_soak * 2.0 ** (-(self.temperature - 20.0) / self.soak_doubling)

    def strength(self, hyd: float | np.ndarray) -> float | np.ndarray:
        return self.sigma_dry * (1.0 - hyd) ** 2 + self.sigma_wet

    def law(self, rho: float) -> PK.SoilLaw:
        law = _law(PK.SoilLaw(), self, ("sigma_dry", "sigma_wet", "k_e", "k_i", "k_front", "m_clean", "beta",
                                         "k_wipe", "k_abr", "sigma_ref"))
        law.t_soak, law.rho = float(self.t_soak_at), float(rho)
        return law


@dataclass(frozen=True)
class Drying:
    """Paint drying (design §3.5, §5; P:704-728). Only the solvent leaves: ``E = k_m·c_sat·(1 - rh)/rho_solvent``
    [m/s] until the solids reach ``phi_m``. The solids ride in the film as the species ``species`` in volume per area
    [m] (``set_source(species={"solids": phi_0})``, ``set_species``), so ``phi_s = solids/h`` and the dry film
    thickness is ``solids``. ``hide``: covered once that reaches ``hide_dft``. ``laps``: passes that land new paint on
    film already past the wet edge ``phi_we`` that carries at least ``lap_visible`` solids (Gym-Anything's visible-lap
    threshold, 4 µm: ``film.py:179-185``, params ``FILM lap_visible``). Defaults: latex wall paint, 42 % volume solids (product sheet), random
    close packing 0.64, wet edge at 0.57 (10x viscosity), k_m 0.006 m/s (Eastman wet edge 2-4 min), water vapour 17.3
    g/m³ at 20 °C (CRC) at 50 % RH, hide at 33 µm DFT (P:963, an optical *estimate*)."""

    phi_0: float = 0.42
    phi_m: float = 0.64
    phi_we: float = 0.57
    lap_visible: float = 4.0e-6
    k_m: float = 0.006
    c_sat: float = 0.0173
    rh: float = 0.5
    rho_solvent: float = 998.2
    hide_dft: float = 3.3e-5
    species: str = "solids"

    def __post_init__(self) -> None:
        _check("drying", self, ("phi_0", "phi_m", "phi_we", "rho_solvent", "hide_dft"), ("k_m", "c_sat", "lap_visible"))
        if not (0.0 < self.phi_0 < self.phi_we < self.phi_m < 1.0) or not 0.0 <= self.rh < 1.0:
            raise PluginError(f"process.drying: need 0 < phi_0 < phi_we < phi_m < 1 and 0 <= rh < 1, got {self}")

    @property
    def rate(self) -> float:
        """Solvent evaporation [m/s] while the solids are below ``phi_m``."""
        return self.k_m * self.c_sat * (1.0 - self.rh) / self.rho_solvent

    def law(self) -> PK.DryLaw:
        d = PK.DryLaw()
        d.rate, d.phi_m, d.phi_we, d.lap_visible = self.rate, float(self.phi_m), float(self.phi_we), float(self.lap_visible)
        return d


@dataclass(frozen=True)
class Substrate:
    """A porous substrate under a chart (design §3.5): Lucas-Washburn uptake ``absorbed = S·sqrt(t)`` [m] while wet, up
    to ``capacity`` [m]. Watercolour paper: S ≈ 3.0e-6 m/√s from a Cobb60 of 23 g/m² (P:815-816); its capacity is an
    *estimate* (a 0.3 mm sheet at 50 % porosity holds about 150 µm)."""

    sorptivity: float = 3.0e-6
    capacity: float = 1.5e-4

    def __post_init__(self) -> None:
        _check("substrate", self, ("sorptivity", "capacity"))


@dataclass(frozen=True)
class Sanitizer:
    """The sanitizer clock (design §5): ``expo`` [s] counts unbroken contact while the film's ``species``
    concentration is at least ``c_min`` [kg/m³]; ``expo_best`` keeps the longest. 50 ppm chlorine is 0.05 kg/m³ (the
    FDA Food Code's low end); the task asks 30 s (FDA Food Code 4-703.11)."""

    c_min: float = 0.05
    species: str = "sanitizer"

    def __post_init__(self) -> None:
        _check("sanitizer", self, ("c_min",))


def _substrates(spec: Any) -> dict[Any, Substrate]:
    """Chart (index or name; JSON turns indices into digit strings) -> Substrate."""
    out = {}
    for k, v in dict(spec or {}).items():
        key = int(k) if isinstance(k, str) and k.isdigit() else k
        out[key] = v if isinstance(v, Substrate) else Substrate(**v)
    return out


PARTS = {"setting": Setting, "soil": Soil, "drying": Drying, "sanitizer": Sanitizer}


@dataclass(frozen=True)
class Process:
    """The process states of one ``SurfaceFlow`` (``process.py``). None: that part is off."""

    setting: Setting | None = None
    soil: Soil | None = None
    drying: Drying | None = None
    sanitizer: Sanitizer | None = None
    substrates: dict[Any, Substrate] | None = None      # chart (index or name) -> its porous substrate


def _part(cls: type, spec: Any) -> Any:
    if spec is None or isinstance(spec, cls):
        return spec
    if isinstance(spec, dict):
        return cls(**spec)
    raise PluginError(f"process: expected a {cls.__name__} or a dict, got {spec!r}")


def process_spec(spec: Process | dict | None) -> Process | None:
    """A ``Process`` from a ``Process``, a dict (JSON, cards, the harness's copies) or None."""
    if spec is None or isinstance(spec, Process):
        return spec
    if not isinstance(spec, dict):
        raise PluginError(f"process: expected a Process or a dict, got {spec!r}")
    unknown = set(spec) - {f.name for f in fields(Process)}
    if unknown:
        raise PluginError(f"process: unknown parts {sorted(unknown)}; parts: {sorted(PARTS) + ['substrates']}")
    parts = {name: _part(cls, spec.get(name)) for name, cls in PARTS.items()}
    return Process(substrates=_substrates(spec.get("substrates")) or None, **parts)


__all__ = ["PARTS", "Drying", "Process", "Sanitizer", "Setting", "Soil", "Substrate", "process_spec"]
