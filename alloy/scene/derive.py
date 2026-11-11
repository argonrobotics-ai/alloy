# SPDX-License-Identifier: Apache-2.0
"""Card value -> Newton parameter conversions (SPEC §3.4 `alloy/scene/derive.py`). Pure functions, float64 host math.

Every formula names its source. Inputs are SI; outputs are SI in Newton's units (N/m, N·m, Pa, kg/s).
"""

from __future__ import annotations

import math
from typing import Literal

MuMixRule = Literal["max", "mean", "geometric", "harmonic", "none"]

# The smallest positive values written into Newton per-contact arrays, where 0 means "unset" (D13 note): a zero
# friction scale or damping would silently fall back to the solver's own material mix or critical damping.
TINY_SCALE = 1.0e-12
TINY_DAMPING = 1.0e-12


def lame(E: float, nu: float) -> tuple[float, float]:
    """(mu, lambda) Lamé parameters: mu = E/(2(1+ν)), λ = Eν/((1+ν)(1−2ν)) (newton@ac7f7830:newton/_src/usd/utils.py:805-809)."""
    _check_poisson(nu)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def plate_rigidity(E: float, nu: float, h: float) -> float:
    """Kirchhoff plate flexural rigidity D = E·h³/(12(1−ν²)) [N·m]."""
    _check_poisson(nu)
    return E * h**3 / (12.0 * (1.0 - nu * nu))


def sheet_bending(mat, E: float, nu: float, h: float) -> float:  # noqa: ANN001 - a MaterialCard
    """A sheet's bending rigidity D [N·m] at modulus ``E``: the material's ``elastic.bending`` (KES B where measured,
    else derived from E at the card's fitted thickness) scaled by E over the card's E (a drawn or worst-case E scales
    it, as randomize does the hinges), else the plate value E·h³/(12(1−ν²)) at the sheet thickness ``h``.

    A card fits E at its own model thickness, so the plate value at another thickness (a contact thickness) scales as
    h³: cotton_woven at 4 mm bends ~300x its fabric (R2-E1: a fresh agent's fold sprang open). The emitter, the
    readback, the router and the coverage plate check all use this."""
    el = getattr(mat, "elastic", None)
    if el is not None and el.bending is not None:
        return float(el.bending.value) * (E / float(el.youngs.value))
    return plate_rigidity(E, nu, h)


def cloth_stiffness(E: float, nu: float, h: float) -> tuple[float, float, float]:
    """(tri_ke, tri_ka, edge_ke) for VBD cloth: the plane-stress Lame pair of the membrane E·h, ν, and D.

    Newton's VBD membrane is neo-Hookean with tri_ke = mu and tri_ka = lambda, the 2D Lame parameters
    (newton/_src/solvers/vbd/particle_vbd_kernels.py: mu_NH = mu, lambda_NH = lambda + mu, Smith et al. 2018). Its
    2D modulus is 4 mu (mu + lambda) / (2 mu + lambda) and its Poisson ratio lambda / (2 mu + lambda), so
    tri_ke = mu = E·h / (2(1+ν)) and tri_ka = lambda = E·h·ν / (1−ν²) give back E·h and ν exactly.
    Before 2026-11-08 this wrote Newton's USD importer map, tri_ke = E·h/(1−ν²) and tri_ka = 0. In this membrane that is
    a 2D modulus of 2 E·h / (1−ν²) (2.2x at ν = 0.3) with Poisson 0 ([internal note] notes/wp2-drape/diff, coordinator
    ruling 08:40Z). edge_ke = D = E·h³/(12(1−ν²)) [N·m], the plate rigidity (the VBD route rewrites hinges from it).
    """
    _check_poisson(nu)
    Eh = E * h
    return Eh / (2.0 * (1.0 + nu)), Eh * nu / (1.0 - nu * nu), plate_rigidity(E, nu, h)


def plate_hinge_ke(D, edge_length, area_sum, c: float):
    """Plate-consistent VBD hinge stiffness edge_ke = c·D·L/(A1+A2) [N] (0 where A1+A2 = 0: a boundary edge).

    Newton's hinge energy is ½·edge_ke·L·θ² (newton@ac7f7830:newton/_src/solvers/vbd/particle_vbd_kernels.py,
    k = edge_ke·rest_length); with θ ≈ κ·d for triangle centres ~d = 2(A1+A2)/(3L) apart and the edge's area share
    ~(A1+A2)/3, matching ½·D·κ² per area gives edge_ke = ¾·D·L/(A1+A2), so c = 0.75. Measured on Alloy's grid
    triangulation: cylindrical bending averaged over orientations gives c = 0.741-0.744 (docs/notes/wp11-bending.md
    §1), and the CARD-1 coupons converge to 0.988-1.016 x Navier at c = 0.75 (opt-in accurate cloth profile,
    `alloy.solvers.vbd.cloth_profile`).
    Accepts numpy arrays (elementwise)."""
    import numpy as np

    area = np.asarray(area_sum, dtype=np.float64)
    safe = np.where(area > 0.0, area, 1.0)
    return np.where(area > 0.0, c * np.asarray(D, np.float64) * np.asarray(edge_length, np.float64) / safe, 0.0)


def rod_section(E: float, G: float, r: float, seg_len: float) -> tuple[float, float, float]:
    """(EA/l, EI/l, GJ/l) of a solid circular rod segment: A = πr², I = πr⁴/4, J = πr⁴/2 (solvers §3.2, D7)."""
    if r <= 0.0 or seg_len <= 0.0:
        raise ValueError(f"rod_section needs r > 0 and seg_len > 0, got r={r}, seg_len={seg_len}")
    area = math.pi * r * r
    inertia = math.pi * r**4 / 4.0
    return E * area / seg_len, E * inertia / seg_len, G * 2.0 * inertia / seg_len


def shear_modulus(E: float, nu: float) -> float:
    """G = E/(2(1+ν))."""
    _check_poisson(nu)
    return E / (2.0 * (1.0 + nu))


def effective_modulus(E1: float, nu1: float, E2: float | None = None, nu2: float | None = None) -> float:
    """Hertz contact modulus E* = 1/((1−ν1²)/E1 + (1−ν2²)/E2); one material alone (rigid partner): E1/(1−ν1²)."""
    inv = (1.0 - nu1 * nu1) / E1
    if E2 is not None and nu2 is not None:
        inv += (1.0 - nu2 * nu2) / E2
    return 1.0 / inv


def torsional_lever(mu: float, F: float, R: float, E_star: float) -> float:
    """Torsional friction lever (3π/16)·μ·a of a Hertz contact, a = (3FR/(4E*))^(1/3) (object-model §4.5, OM:299)."""
    if F <= 0.0 or R <= 0.0 or E_star <= 0.0:
        return 0.0
    a = (3.0 * F * R / (4.0 * E_star)) ** (1.0 / 3.0)
    return 3.0 * math.pi / 16.0 * mu * a


def pair_stiffness(ka: float, kb: float) -> float:
    """Two contact springs in series: ka·kb/(ka+kb) [N/m] (D22); an infinite (0-compliance) side passes the other."""
    if math.isinf(ka):
        return kb
    if math.isinf(kb):
        return ka
    if ka <= 0.0 or kb <= 0.0:
        return 0.0
    return ka * kb / (ka + kb)


def harmonic_mu(mu1: float, mu2: float) -> float:
    """Drake's unsourced-pair rule 2μ1μ2/(μ1+μ2), 0 if both are 0 (D13)."""
    if mu1 + mu2 <= 0.0:
        return 0.0
    return 2.0 * mu1 * mu2 / (mu1 + mu2)


def mu_mix(rule: MuMixRule, mu1: float, mu2: float) -> float:
    """A solver's own combination of two shape μ values (SolverCapability.friction_mix, D13)."""
    if rule == "max":
        return max(mu1, mu2)
    if rule == "mean":
        return 0.5 * (mu1 + mu2)
    if rule == "geometric":
        return math.sqrt(max(mu1, 0.0) * max(mu2, 0.0))
    if rule == "harmonic":
        return harmonic_mu(mu1, mu2)
    return 0.0


def friction_scale(mu_pair: float, mu_mixed: float) -> float:
    """The value written to `Contacts.rigid_contact_friction` so the solver realises μ_pair: μ_pair/μ_mix (D13).

    Newton treats 0 as "unset" (scale ignored), so a frictionless pair gets TINY_SCALE instead of 0.
    """
    if mu_mixed <= 0.0:
        return 0.0
    return max(mu_pair / mu_mixed, TINY_SCALE)


def zeta_from_restitution(e: float) -> float:
    """Damping ratio of a linear spring-damper impact with restitution e: ζ = −ln e/√(π² + ln²e); ζ(0) = 1."""
    if not 0.0 <= e <= 1.0:
        raise ValueError(f"restitution must be in [0, 1], got {e}")
    if e <= 0.0:
        return 1.0
    if e >= 1.0:
        return 0.0
    ln_e = math.log(e)
    return min(1.0, -ln_e / math.sqrt(math.pi**2 + ln_e * ln_e))


def contact_damping(e: float, K: float, m_eff: float, imp: float | None = None) -> float:
    """Force-space contact damping kd [N·s/m] that realises restitution e on a spring K with effective mass m_eff.

    Linear impact: kd = 2ζ(e)·√(K·m_eff). Under SolverMuJoCo (imp = the row's flat solimp) Newton keeps K and kd as
    the static values but the soft constraint integrates the body as if its mass were m_eff/(1−imp)
    (mujoco_warp@a9daa05f:mujoco_warp/_src/constraint.py:104-123; newton@ac7f7830:newton/_src/solvers/mujoco/
    kernels.py:663-685), so the realised ratio is ζ·√(1−imp) and kd is divided by √(1−imp). Never returns 0
    (0 means critical damping in Newton's conversion).
    """
    if K <= 0.0 or m_eff <= 0.0:
        return 0.0
    kd = 2.0 * zeta_from_restitution(e) * math.sqrt(K * m_eff)
    if imp is not None:
        if not 0.0 <= imp < 1.0:
            raise ValueError(f"imp must be in [0, 1), got {imp}")
        kd /= math.sqrt(1.0 - imp)
    return max(kd, TINY_DAMPING)


def k_real_mujoco(m_eff: float, imp: float, dt: float, tau_req: float, zeta: float = 1.0,
                  *, integrator: str = "implicitfast") -> float:
    """Realised static contact stiffness of a SolverMuJoCo row [N/m] (D22).

    Continuous integrators: K = m_eff/((1−imp)·τ²·ζ²), τ = max(τ_req, 2·dt) (MJW refsafe clamp, constraint.py:104).
    `discrete`: no clamp, but k = 1/(imp²τ²ζ²) is capped at 1/(dt²·imp) when dt²·k·imp > 1 (constraint.py:190-205),
    giving K = imp²·k·m_eff/(1−imp).
    """
    if m_eff <= 0.0 or not 0.0 < imp < 1.0 or tau_req <= 0.0 or zeta <= 0.0:
        raise ValueError(f"k_real_mujoco: bad inputs m_eff={m_eff}, imp={imp}, tau_req={tau_req}, zeta={zeta}")
    if integrator != "discrete":
        tau = max(tau_req, 2.0 * dt)
        return m_eff / ((1.0 - imp) * tau * tau * zeta * zeta)
    k = 1.0 / (imp * imp * tau_req * tau_req * zeta * zeta)
    if dt * dt * k * imp > 1.0:
        k = 1.0 / (dt * dt * imp)
    return imp * imp * k * m_eff / (1.0 - imp)


def mujoco_solref_from_force(K: float, kd: float, m_eff: float, imp: float) -> tuple[float, float]:
    """(timeconst, dampratio) that Newton writes for per-contact force-space (K, kd) under an elliptic cone.

    Mirrors newton@ac7f7830:newton/_src/solvers/mujoco/kernels.py:663-685 with factor = (1−imp)/m_eff
    (`_normal_contact_gain_factor`, kernels.py:208-222): ke' = K·factor; kd > 0 → τ = 2/(kd·factor),
    ζ = 1/(τ·√ke'); kd = 0 → τ = √(1/ke'), ζ = 1.
    """
    factor = (1.0 - imp) / m_eff
    ke = K * factor
    if kd > 0.0:
        tau = 2.0 / (kd * factor)
        return tau, 1.0 / (tau * math.sqrt(ke))
    return math.sqrt(1.0 / ke), 1.0


def _check_poisson(nu: float) -> None:
    if not -1.0 < nu < 0.5:
        raise ValueError(f"Poisson ratio must be in (-1, 0.5), got {nu}")
