# SPDX-License-Identifier: Apache-2.0
"""Heat v3 materials: temperature-dependent c(T) and k(T) tables for ``GridThermal`` (docs/notes/heat-v3-design.md).

The enthalpy convention is heat v2's (``phase_materials`` module doc), per unit mass: H = 0 for the solid at its
solidus T_s; below it H = -∫_T^T_s c dT; the mushy zone runs linearly in T from H = 0 to H_l = L + ∫_T_s^T_l c dT,
with the liquid fraction φ = H / H_l (a pure substance, T_l = T_s: the plateau H ∈ [0, L] at T_s, φ = H / L); above
it H = H_l + ∫_T_l^T c dT. Mushy-zone conductivity is the parallel mix k = (1 − φ) k_s(T) + φ k_l, as in v2.

v2's ``PhaseMaterial`` holds constant c and k; welding needs steel's c(T) (the 735 °C transformation peak) and k(T),
so v3 tabulates them. ``table()`` gives nodes every ``step`` K from ``t_min`` to ``t_boil`` with T_s, T_l and the
formulas' breakpoints as extra nodes, H exact at every node (c integrated on 0.01 K), and k at every node. Between
nodes the device interpolates linearly in H, which is exact in the mushy zone (H is linear in T there): the latent
heat is never smeared over a table cell (GA, report_welding §4: a tabulated mushy range was 23 % off Stefan).

Every value has a source; uncited values say ESTIMATE. Boiling: at ``t_boil`` the grid caps H and books the excess as
evaporated energy (GA §4 does the same); no vapour is modelled.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

T_MIN = 200.0  # K, the tables' bottom (H below it extrapolates with c(T_MIN))
STEP = 5.0  # K, table spacing (GA: a 5 K table)
SUB = 0.01  # K, integration spacing for H at the nodes


@dataclass(frozen=True)
class GridMaterial:
    """A conducting solid that melts: c(T) and k(T) of the solid as functions [K -> SI], constants for the liquid."""

    name: str
    density: float  # kg/m³ (held constant: no thermal expansion or melt shrinkage in v3)
    c_solid: Callable[[np.ndarray], np.ndarray]  # J/(kg·K) of T [K], solid and mushy sensible part
    k_solid: Callable[[np.ndarray], np.ndarray]  # W/(m·K) of T [K]
    c_liquid: float  # J/(kg·K) above T_l
    k_liquid: float  # W/(m·K) above T_l (mushy: the φ mix)
    t_solidus: float  # K
    t_liquidus: float  # K (= t_solidus: a pure substance)
    latent: float  # J/kg (0 with t_liquidus = t_solidus: no phase change)
    t_boil: float  # K, the enthalpy cap
    breaks: tuple[float, ...] = ()  # K, where c(T) or k(T) change formula (added as table nodes)
    source: str = ""
    _nodes: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.density <= 0.0 or self.c_liquid <= 0.0 or self.k_liquid <= 0.0:
            raise ValueError(f"{self.name}: density, c_liquid and k_liquid must be > 0")
        if not T_MIN < self.t_solidus <= self.t_liquidus < self.t_boil or self.latent < 0.0:
            raise ValueError(f"{self.name}: needs {T_MIN} K < t_solidus <= t_liquidus < t_boil and latent >= 0")
        if self.t_liquidus == self.t_solidus and self.latent == 0.0 and self.c_liquid != float(self._c(self.t_solidus)):
            raise ValueError(f"{self.name}: without a phase change c_liquid must equal c_solid at the solidus")
        probe = np.linspace(T_MIN, self.t_liquidus, 64)
        if np.any(self._c(probe) <= 0.0) or np.any(self._k(probe) <= 0.0):
            raise ValueError(f"{self.name}: c_solid and k_solid must be > 0 from {T_MIN} K to the liquidus")

    def _c(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return np.where(t <= self.t_liquidus, np.asarray(self.c_solid(np.minimum(t, self.t_liquidus)), np.float64),
                        self.c_liquid)

    def _k(self, t: np.ndarray | float) -> np.ndarray:
        return np.asarray(self.k_solid(np.asarray(t, dtype=np.float64)), np.float64)

    def _integral(self, a: float, b: float) -> float:
        """∫_a^b c dT (sensible), trapezoid on SUB; a <= b."""
        if b <= a:
            return 0.0
        n = max(2, int(math.ceil((b - a) / SUB)) + 1)
        t = np.linspace(a, b, n)
        return float(np.trapezoid(self._c(t), t))

    @property
    def h_liquidus(self) -> float:
        """H at the liquidus [J/kg]: the top of the mushy zone (L for a pure substance)."""
        return self.latent + self._integral(self.t_solidus, self.t_liquidus)

    @property
    def h_boil(self) -> float:
        """H at ``t_boil`` [J/kg], the cap."""
        return self.h_liquidus + self.c_liquid * (self.t_boil - self.t_liquidus)

    def table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(T [K], H [J/kg], k [W/(m·K)]) at the nodes, float64; H strictly increasing. A pure substance has two
        nodes at T_s (H = 0 and H = L), so the plateau is one table segment."""
        if "t" not in self._nodes:
            ts, tl = self.t_solidus, self.t_liquidus
            grid = np.arange(T_MIN, self.t_boil, STEP)
            extra = [b for b in self.breaks if T_MIN < b < self.t_boil]
            t = np.unique(np.concatenate([grid, [ts, tl, self.t_boil], extra]))
            h = np.empty_like(t)
            below = t <= ts
            # H below the solidus: -∫_T^Ts c, accumulated node to node from Ts downwards
            idx = np.flatnonzero(below)[::-1]
            acc = 0.0
            for j, i in enumerate(idx):
                if j:
                    acc -= self._integral(t[i], t[idx[j - 1]])
                h[i] = acc
            hl = self.h_liquidus
            mushy = (t > ts) & (t <= tl)
            h[mushy] = (t[mushy] - ts) / (tl - ts) * hl if tl > ts else hl
            above = np.flatnonzero(t > tl)
            h[above] = hl + self.c_liquid * (t[above] - tl)
            phi = np.clip(h / hl, 0.0, 1.0) if hl > 0.0 else np.zeros_like(h)
            k = (1.0 - phi) * self._k(np.minimum(t, tl)) + phi * self.k_liquid
            if tl == ts and self.latent > 0.0:    # the plateau: a second node at Ts on the liquid side
                at = int(np.flatnonzero(t == ts)[0])
                t = np.insert(t, at + 1, ts)
                h = np.insert(h, at + 1, hl)
                k = np.insert(k, at + 1, self.k_liquid)
            if np.any(np.diff(h) <= 0.0):
                raise ValueError(f"{self.name}: H(T) is not strictly increasing on the table")
            self._nodes.update(t=t, h=h, k=k)
        return self._nodes["t"], self._nodes["h"], self._nodes["k"]

    def enthalpy(self, t: float | np.ndarray) -> np.ndarray:
        """H(T) [J/kg] (the solid side at T_s of a pure substance), linear between table nodes."""
        tt, hh, _ = self.table()
        t = np.asarray(t, dtype=np.float64)
        first = self.c_solid(np.asarray(T_MIN))
        lo = hh[0] + float(first) * (t - T_MIN)
        tt_u, first_idx = np.unique(tt, return_index=True)  # the solid-side node at a plateau
        return np.where(t < T_MIN, lo, np.interp(t, tt_u, hh[first_idx]))

    def temperature(self, h: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T [K], φ) of H [J/kg], the device's lookup on the host: linear in H between nodes, no division by
        (T_l − T_s); φ = clip(H / H_l) (H_l = L on a plateau)."""
        tt, hh, _ = self.table()
        h = np.asarray(h, dtype=np.float64)
        c0 = float(self.c_solid(np.asarray(T_MIN)))
        t = np.where(h < hh[0], T_MIN + (h - hh[0]) / c0, np.interp(np.minimum(h, hh[-1]), hh, tt))
        hl = self.h_liquidus
        phi = np.clip(h / hl, 0.0, 1.0) if hl > 0.0 else np.zeros_like(h)
        return t, phi

    def conductivity(self, h: float | np.ndarray) -> np.ndarray:
        """k(H) [W/(m·K)], linear in H between nodes (the device's lookup)."""
        _, hh, kk = self.table()
        return np.interp(np.asarray(h, dtype=np.float64), hh, kk)

    def bounds(self, gamma: float) -> tuple[float, float, float]:
        """(max k, min sensible c, max over T of min(k_max, γ k(T)) / c(T)) for the explicit step: a face's series
        conductance is at most A k_max / δ and at most 2 A k_i / d_i = γ A k_i / δ with γ = 2 δ / d_i
        (``grid.stable_dt``)."""
        tt, _, kk = self.table()
        cc = self._c(tt)
        kmax = float(kk.max())
        ratio = float(np.max(np.minimum(kmax, gamma * kk) / cc))
        return kmax, float(cc.min()), ratio


# --- steel: EN 1993-1-2 (Eurocode 3, fire part) §3.4.1, θ in °C, defined 20-1200 °C ---------------------------------

def _c_en1993(t: np.ndarray) -> np.ndarray:
    """EN 1993-1-2 eq. (3.2) specific heat of carbon steel [J/(kg·K)]; held at the 20 °C value below 20 °C and at
    650 above 1200 °C (ESTIMATE: the code stops at 1200 °C; GA keeps the EN values, report_welding §4)."""
    th = np.clip(np.asarray(t, dtype=np.float64) - 273.15, 20.0, None)
    return np.where(th < 600.0, 425.0 + 7.73e-1 * th - 1.69e-3 * th**2 + 2.22e-6 * th**3,
                    np.where(th < 735.0, 666.0 + 13002.0 / (738.0 - np.minimum(th, 735.0)),
                             np.where(th < 900.0, 545.0 + 17820.0 / (np.maximum(th, 735.0) - 731.0), 650.0)))


def _k_en1993(t: np.ndarray) -> np.ndarray:
    """EN 1993-1-2 eq. (3.3) thermal conductivity of carbon steel [W/(m·K)]: 54 − 3.33e-2 θ to 800 °C, then 27.3
    (held to the solidus, ESTIMATE above 1200 °C); the 20 °C value below 20 °C."""
    th = np.clip(np.asarray(t, dtype=np.float64) - 273.15, 20.0, None)
    return np.where(th < 800.0, 54.0 - 3.33e-2 * th, 27.3)


def _const(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda t: np.full(np.shape(t), value, np.float64)


_BREAKS_EN = tuple(273.15 + b for b in (20.0, 600.0, 735.0, 800.0, 900.0, 1200.0))

MATERIALS: dict[str, GridMaterial] = {m.name: m for m in (
    GridMaterial(
        "steel_a36", density=7850.0, c_solid=_c_en1993, k_solid=_k_en1993, c_liquid=650.0, k_liquid=420.0,
        t_solidus=1745.15, t_liquidus=1785.15, latent=2.47e5, t_boil=3133.15, breaks=_BREAKS_EN,
        source="ρ 7850: EN 1993-1-2 §3.2.2. c(T), k(T): EN 1993-1-2 eqs. (3.2), (3.3) (GA report_welding §4). "
               "Solidus 1472 °C, liquidus 1512 °C: Kim, Zhang & DebRoy 2003 via GA §4. k_liquid 420: the Penn State "
               "conduction-only fillet models' stand-in for pool convection (Kim, Zhang & DebRoy 2003 Table II, via "
               "GA §4). ESTIMATE: latent 2.47e5 J/kg (the value those fillet models use for steel; not read here), "
               "c above 1200 °C and of the liquid held at 650. Boiling cap 2860 °C (GA §4)."),
    GridMaterial(
        "law_iron", density=7850.0, c_solid=_const(600.0), k_solid=_const(30.0), c_liquid=600.0, k_liquid=30.0,
        t_solidus=1000.0, t_liquidus=1000.0, latent=0.0, t_boil=4000.0,
        source="law fixture: constant c 600, k 30 (α = 6.37e-6 m²/s), no phase change"),
    GridMaterial(
        "law_pure", density=7850.0, c_solid=_const(600.0), k_solid=_const(30.0), c_liquid=600.0, k_liquid=30.0,
        t_solidus=1000.0, t_liquidus=1000.0, latent=2.5e5, t_boil=4000.0,
        source="law fixture: law_iron melting at one temperature, 1000 K, L 2.5e5 J/kg (Neumann's problem)"),
    GridMaterial(
        "law_fast", density=7850.0, c_solid=_const(600.0), k_solid=_const(400.0), c_liquid=600.0, k_liquid=400.0,
        t_solidus=1000.0, t_liquidus=1000.0, latent=0.0, t_boil=4000.0,
        source="law fixture: k 400 (copper-like) so a 5 mm block is near-lumped (Biot 0.025 at h_c 2000)"),
)}


def material(name: str) -> GridMaterial:
    """The registered material ``name``; KeyError lists the names."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"grid material {name!r}: use one of {sorted(MATERIALS)}") from None


__all__ = ["MATERIALS", "GridMaterial", "material"]
