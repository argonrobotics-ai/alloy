# SPDX-License-Identifier: Apache-2.0
"""Precision-pair numbers: stiffness defaults (D22), realised MuJoCo stiffness, gap-ladder rungs and L2 step sizing.

All functions are pure float64 host code. Formulas:

- MuJoCo soft contact at rest (mujoco-warp 3.16.0 ``_src/constraint.py:95-145``, ``_efc_row`` 153-220): with
  ``k = 1/(dmax²·τ²·ζ²)``, ``R = (1-imp)/imp · invweight``, the steady state of ``(A+R)f = aref - a0`` with
  ``aref = -k·imp·pos`` gives ``K = k·imp²/((1-imp)·invweight) = m_eff/((1-imp)·τ²·ζ²)``, ``m_eff = 1/invweight``,
  and MJWarp clamps ``τ ≥ 2·dt`` (refsafe) for non-discrete integrators (D22).
- L2 impact sink of a critically damped contact: ``v·τ/e`` (e = Euler's number, PB CON-48); with ``τ = 2·dt``,
  ``dt ≤ e·max_pen/(2·v)``. Static sink: ``F_design/K ≤ max_pen/2``. Explicit solvers: ``ω·dt ≤ 0.5``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from alloy.contact.types import PrecisionPairSpec
from alloy.core.errors import ClockError, PrecisionSpecError
from alloy.core.log import get_logger
from alloy.core.timebase import TICK_EXP, snap_to_grid

log = get_logger("contact.precision")

Integrator = Literal["euler", "implicitfast", "discrete", "rk4", "implicit"]
Native = Literal["mujoco", "newton"]

RUNG_NAMES = ("L0", "L1", "L2", "L3", "L4")
K_REAL_MIN_RATIO = 0.8          # D22: PrecisionSpecError if K_real < 0.8 x declared


def pair_stiffness(spec: PrecisionPairSpec, design_force: float) -> float:
    """Declared force-space K [N/m]: ``spec.pair_stiffness`` or ``2·design_force/max_penetration`` (D22)."""
    if spec.pair_stiffness is not None:
        return float(spec.pair_stiffness)
    if design_force <= 0.0:
        raise PrecisionSpecError(f"design_force must be positive to size K, got {design_force}")
    return 2.0 * design_force / spec.resolved_max_penetration()


def k_real_mujoco(m_eff: float, imp: float, dt: float, tau_req: float, zeta: float = 1.0) -> float:
    """Realised MuJoCo contact stiffness ``m_eff/((1-imp)·τ²·ζ²)`` with ``τ = max(τ_req, 2·dt)`` [N/m] (D22)."""
    if m_eff <= 0.0 or not 0.0 < imp < 1.0 or dt <= 0.0 or zeta <= 0.0:
        raise ValueError(f"k_real_mujoco needs m_eff>0, 0<imp<1, dt>0, zeta>0; got {m_eff}, {imp}, {dt}, {zeta}")
    tau = max(float(tau_req), 2.0 * float(dt))
    return float(m_eff) / ((1.0 - imp) * tau * tau * zeta * zeta)


def tau_for_stiffness(k: float, m_eff: float, imp: float, zeta: float = 1.0) -> float:
    """The MuJoCo timeconst that realises force-space ``k`` before the ``2·dt`` clamp: ``sqrt(m/((1-imp)·k))/ζ``."""
    if k <= 0.0 or m_eff <= 0.0 or not 0.0 < imp < 1.0:
        raise ValueError(f"tau_for_stiffness needs k>0, m_eff>0, 0<imp<1; got {k}, {m_eff}, {imp}")
    return math.sqrt(m_eff / ((1.0 - imp) * k)) / zeta


def check_k_real(declared: float, m_eff: float, imp: float, dt: float, label: str = "") -> float:
    """``K_real`` for the declared K at ``dt``; raises ``PrecisionSpecError`` if it is below 0.8x declared (D22)."""
    k_real = k_real_mujoco(m_eff, imp, dt, tau_for_stiffness(declared, m_eff, imp))
    if k_real < K_REAL_MIN_RATIO * declared:
        raise PrecisionSpecError(
            f"pair {label}: K_real {k_real:.4g} N/m < {K_REAL_MIN_RATIO} x declared {declared:.4g} N/m at dt={dt:.4g} s "
            f"(m_eff={m_eff:.4g} kg, imp={imp}); halve dt to <= {max_dt_for_stiffness(declared, m_eff, imp):.4g} s")
    return k_real


def max_dt_for_stiffness(k: float, m_eff: float, imp: float) -> float:
    """Largest dt whose ``2·dt`` clamp still realises 0.8x ``k``: ``τ_req·sqrt(1/0.8)/2`` with τ_req for ``k``."""
    return 0.5 * tau_for_stiffness(K_REAL_MIN_RATIO * k, m_eff, imp)


# ------------------------------------------------------------------------------------------------ gap ladder ---

@dataclass(frozen=True)
class LadderRow:
    """What one SolverMuJoCo integrator allows (SPEC §3.5 table, D10 spike result)."""

    l0: bool
    l1: bool
    noslip: bool
    sleeping: bool


_LADDER: dict[str, LadderRow] = {
    "euler": LadderRow(True, True, True, True),
    "implicitfast": LadderRow(True, True, True, True),
    "discrete": LadderRow(True, False, False, True),
}


def integrator_row(integrator: str) -> LadderRow:
    """The ladder row; ``NotImplementedError`` for rk4/implicit (collision re-runs at stale contacts)."""
    key = integrator.lower()
    if key not in _LADDER:
        raise NotImplementedError(f"precision pairs are not supported with integrator {integrator!r} "
                                  f"(supported: {sorted(_LADDER)})")
    return _LADDER[key]


def ladder_rungs(spec: PrecisionPairSpec, integrator: str, native: Native = "mujoco",
                 append_l1: bool = True) -> tuple[str, ...]:
    """Rungs a pair actually has (``RouteReport.ladder``). L1 needs a speculative-capable integrator, ``ccd`` not
    ``"off"`` and, under ``native="mujoco"``, the passive-callback L1 append (``append_l1``, D10 spike). Hydroelastic
    pairs have no L0 (Newton's pipeline cannot expand their gap by velocity); their L1 rows cover the authored gap."""
    row = integrator_row(integrator)
    # L0 (velocity-expanded gap) needs ccd on, and Newton cannot expand hydroelastic gaps
    rungs = [] if spec.mode == "hydroelastic" or spec.ccd == "off" else ["L0"]
    l1_path = native == "newton" or append_l1
    if row.l1 and spec.ccd != "off" and l1_path:
        rungs.append("L1")
    rungs += ["L2", "L3", "L4"]
    return tuple(rungs)


# (mode, native) combinations no router may pick for a precision pair: the measured reason (docs/notes/wp3-results.md).
UNSUPPORTED_NATIVE: dict[tuple[str, str], str] = {
    ("hydroelastic", "newton"): (
        "hydroelastic precision pairs need native='mujoco': on native='newton' Newton's contact conversion leaves the "
        "pressure-field rows softer under edge loads (CT-1/SB-3: a 5 N push wedging the peg at the bore edge gives "
        "102.7-104.3 um max penetration against the 100 um cap, native='mujoco' 85.6-86.1 um); use native='mujoco', "
        "or mode='sdf' when the scene needs native='newton' (CT-1 sdf/newton 26.6 um)"),
}


def supported_natives(spec: PrecisionPairSpec) -> tuple[Native, ...]:
    """The ``native`` contact paths a router may pick for ``spec`` (UNSUPPORTED_NATIVE removes measured failures)."""
    return tuple(n for n in ("mujoco", "newton") if (spec.mode, n) not in UNSUPPORTED_NATIVE)


def l2_impact_dt(max_penetration: float, contact_speed: float) -> float:
    """SolverMuJoCo impact rule ``v·(2·dt)/e ≤ max_pen`` → ``dt ≤ e·max_pen/(2·v)`` [s]."""
    if max_penetration <= 0.0 or contact_speed <= 0.0:
        raise ValueError(f"need max_penetration>0 and contact_speed>0, got {max_penetration}, {contact_speed}")
    return math.e * max_penetration / (2.0 * contact_speed)


def l2_explicit_dt(k: float, m_eff: float) -> float:
    """Explicit-solver rule ``ω·dt ≤ 0.5``, ``ω = sqrt(K/m)`` [s]."""
    if k <= 0.0 or m_eff <= 0.0:
        raise ValueError(f"need k>0 and m_eff>0, got {k}, {m_eff}")
    return 0.5 / math.sqrt(k / m_eff)


def l2_static_ok(design_force: float, k_real: float, max_penetration: float) -> bool:
    """Static sink ``F_design/K_real ≤ max_pen/2`` (relative tolerance 1e-9: the D22 default K sits on the bound)."""
    return design_force / k_real <= 0.5 * max_penetration * (1.0 + 1e-9)


@dataclass(frozen=True)
class L2Sizing:
    """Result of sizing a precision pair's step on the D4 grid."""

    dt: float                 # grid step that meets every L2 condition [s]
    dt_impact: float          # raw impact bound [s]
    dt_stiffness: float       # raw bound for K_real >= max(0.8 K, 2 F/max_pen) [s]
    k_declared: float         # N/m
    k_real: float             # N/m at dt
    static_ok: bool


def size_l2(spec: PrecisionPairSpec, *, m_eff: float, design_force: float, contact_speed: float,
            control_dt: float, label: str = "") -> L2Sizing:
    """Largest grid dt meeting the impact rule and the K_real rule; ``PrecisionSpecError`` below the base tick or if
    the static rule fails at that dt (never silently softens, §3.5 L2)."""
    max_pen = spec.resolved_max_penetration()
    k = pair_stiffness(spec, design_force)
    dt_imp = l2_impact_dt(max_pen, contact_speed)
    k_static = 2.0 * design_force / max_pen                     # K_real needed by the static rule
    dt_k = min(max_dt_for_stiffness(k, m_eff, spec.mj_imp), 0.5 * tau_for_stiffness(k_static, m_eff, spec.mj_imp))
    raw = min(dt_imp, dt_k, control_dt)
    try:
        dt = snap_to_grid(raw, control_dt)
    except ClockError as exc:
        raise PrecisionSpecError(
            f"pair {label}: L2 needs dt <= {raw:.4g} s (impact {dt_imp:.4g}, stiffness {dt_k:.4g}) below the base tick "
            f"control_dt/2^{TICK_EXP} = {control_dt / 2**TICK_EXP:.4g} s") from exc
    k_real = k_real_mujoco(m_eff, spec.mj_imp, dt, tau_for_stiffness(k, m_eff, spec.mj_imp))
    ok = l2_static_ok(design_force, k_real, max_pen)
    if not ok:
        raise PrecisionSpecError(
            f"pair {label}: static sink {design_force / k_real:.3g} m > max_pen/2 = {0.5 * max_pen:.3g} m "
            f"with K_real={k_real:.4g} N/m at dt={dt:.4g} s; raise pair_stiffness")
    log.debug("pair %s: L2 dt=%.4g s (impact %.4g, stiffness %.4g), K=%.4g, K_real=%.4g", label, dt, dt_imp, dt_k,
              k, k_real)
    return L2Sizing(dt, dt_imp, dt_k, k, k_real, ok)
