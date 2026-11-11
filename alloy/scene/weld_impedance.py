# SPDX-License-Identifier: Apache-2.0
"""Weld/connect equality impedance (SCR-DC3-3 as amended by SCR-WP2-9, rulings in [internal note] and WP2.md).

An unbreakable weld at a CHAIN's root becomes a tree joint (`alloy.scene.tree_weld`). Every other weld/connect stays
a MuJoCo equality row, sized here instead of taking MuJoCo's default 0.02 s / 0.9-0.95.

A weld row follows r'' + d·(b·r' + k·r) = (1 − d)·a₀. So it holds a steady violation r = (1 − d)/(d·k)·a₀, where
a₀ = F/m_eff is the unconstrained relative acceleration across the weld. On DC3's QSFP28 DAC, MuJoCo's default let the
0.5 g first cable link separate 1.1 m at the 90 N SFF-8661 retention load.

- solref is direct, (−ω², −2ζω) with ζ = 1 and ω·h = 0.6, where h is the design step: the largest step the scene's
  MuJoCo entry can take (`native_gains.rigid_dt_max`). Semi-implicit Euler is stable for ω·h < 2(√2 − 1) ≈ 0.83.
  MuJoCo's refsafe clamp (τ ≥ 2·h) applies only to a positive solref.
- solimp is flat (d, d, 0.001, 0.5, 2) and sized for the target. MuJoCo's K = −solref[0]/d², so the separation at the
  endpoints' design force F is F·(1 − d)·d/(m_eff·ω²). d is the softest value that keeps this at WELD_SEP_TOL,
  within [WELD_IMP_MIN, WELD_IMP_CAP]. The cap, 0.999, is there because a near-hard row (0.9999) does not condition in
  float32 on light bodies. With 0.9999, T09's fast profile had 19/64 invalid worlds and its standard profile ran at
  half speed (SCR-WP2-9). When the cap cannot meet the target, the route report warns with the predicted separation.
  Without a design force there is no target, and d is the cap.

The constraint acts in acceleration space, so its stability does not depend on mass. The effective mass m_eff (the
reduced mass of the two bodies; a static end is the world) sets the force-space stiffness K = m_eff·ω²/((1 − d)·d) and
so the separation. A card's `Attachment.timeconst` / `dampratio` override the solref: a timeconst gives MuJoCo's
standard (timeconst, dampratio), clamped by MuJoCo to τ ≥ 2·h, and a dampratio alone keeps the rule's ω. The solimp is
then sized for that stiffness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WELD_OMEGA_H = 0.6          # ω·h at the design step (module docstring)
WELD_IMP_CAP = 0.999        # largest solimp (SCR-WP2-9)
WELD_IMP_MIN = 0.95         # softest solimp: MuJoCo's default dmax
WELD_ZETA = 1.0
WELD_SEP_TOL = 1.0e-4       # m, separation target at the design force (DC3's 0.1 mm)
STABLE_OMEGA_H = 2.0 * (math.sqrt(2.0) - 1.0)    # semi-implicit Euler bound at ζ = 1


@dataclass(frozen=True)
class WeldImpedance:
    solref: tuple[float, float]
    solimp: tuple[float, float, float, float, float]
    dt: float                           # design step, s
    m_eff: float                        # kg; inf when both ends are massless (not a weld this rule sizes)
    k_force: float                      # N/m, realised force-space stiffness of the translational rows
    design_force: float | None          # N, the larger of the endpoints' card design forces
    source: str                         # "rule", "card timeconst", "card dampratio"

    @property
    def imp(self) -> float:
        return self.solimp[0]

    @property
    def separation(self) -> float | None:
        """Steady separation at the design force, m (None without a design force)."""
        if self.design_force is None or not math.isfinite(self.k_force) or self.k_force <= 0.0:
            return None
        return self.design_force / self.k_force

    @property
    def meets_target(self) -> bool:
        sep = self.separation
        return sep is None or sep <= WELD_SEP_TOL * (1.0 + 1e-9)


def reduced_mass(m_a: float, m_b: float) -> float:
    """Reduced mass across a weld; a massless or static end (≤ 0) is the world."""
    inv = sum(1.0 / m for m in (m_a, m_b) if m > 0.0)
    return 1.0 / inv if inv > 0.0 else math.inf


def sized_imp(m_eff: float, omega2: float, design_force: float | None) -> float:
    """The softest d in [WELD_IMP_MIN, WELD_IMP_CAP] with F·(1 − d)·d/(m_eff·ω²) ≤ WELD_SEP_TOL."""
    if design_force is None or design_force <= 0.0 or not math.isfinite(m_eff):
        return WELD_IMP_CAP
    c = WELD_SEP_TOL * m_eff * omega2 / design_force             # need (1 − d)·d ≤ c
    if c >= 0.25:
        return WELD_IMP_MIN
    d = 0.5 * (1.0 + math.sqrt(1.0 - 4.0 * c))
    return min(WELD_IMP_CAP, max(WELD_IMP_MIN, d))


def weld_impedance(m_eff: float, dt: float, design_force: float | None = None, timeconst: float | None = None,
                   dampratio: float | None = None) -> WeldImpedance:
    """The rule's (or the card's) solref, and the sized solimp, for one weld/connect at design step `dt`."""
    if dt <= 0.0:
        raise ValueError(f"weld impedance: design step must be > 0, got {dt}")
    zeta = WELD_ZETA if dampratio is None else float(dampratio)
    if timeconst is not None:
        tau = max(float(timeconst), 2.0 * dt)               # MuJoCo's refsafe clamp, as the solver applies it
        omega2, solref, source = 1.0 / (tau * tau * zeta * zeta), (float(timeconst), zeta), "card timeconst"
    else:
        omega = WELD_OMEGA_H / dt
        omega2, solref = omega * omega, (-omega * omega, -2.0 * zeta * omega)
        source = "rule" if dampratio is None else "card dampratio"
    imp = sized_imp(m_eff, omega2, design_force)
    k_force = m_eff * omega2 / ((1.0 - imp) * imp) if math.isfinite(m_eff) else math.inf
    return WeldImpedance(solref, (imp, imp, 0.001, 0.5, 2.0), dt, m_eff, k_force, design_force, source)


def report_line(kind: str, a: str, b: str, w: WeldImpedance, note: str = "") -> str:
    """Route-report line for one equality attachment's impedance."""
    sep = w.separation
    at = "" if sep is None else f", separation {sep * 1e3:.3g} mm at the design force {w.design_force:g} N"
    why = ("card Attachment.timeconst/dampratio" if w.source != "rule" else
           f"SCR-DC3-3/WP2-9 rule: direct solref at ω·h {WELD_OMEGA_H} for the {w.dt * 1e3:.4g} ms design step")
    imp = (f"solimp={w.imp:.6g} sized for {WELD_SEP_TOL * 1e3:g} mm" if w.design_force else
           f"solimp={w.imp:.6g} (no design force: the cap)")
    return (f"mujoco {kind} {a}–{b}: solref=({w.solref[0]:.6g}, {w.solref[1]:.6g}) {imp}; {why}; "
            f"m_eff {w.m_eff * 1e3:.4g} g, K {w.k_force:.4g} N/m{at}{note}")


def separation_warning(kind: str, a: str, b: str, w: WeldImpedance) -> str | None:
    if w.meets_target:
        return None
    return (f"{kind} {a}–{b}: predicted separation {w.separation * 1e3:.3g} mm at the design force "
            f"{w.design_force:g} N exceeds {WELD_SEP_TOL * 1e3:g} mm at the solimp cap {WELD_IMP_CAP} "
            f"(m_eff {w.m_eff * 1e3:.4g} g, {w.dt * 1e3:.4g} ms design step)")


__all__ = ["STABLE_OMEGA_H", "WELD_IMP_CAP", "WELD_IMP_MIN", "WELD_OMEGA_H", "WELD_SEP_TOL", "WeldImpedance",
           "reduced_mass", "report_line", "separation_warning", "sized_imp", "weld_impedance"]
