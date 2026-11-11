# SPDX-License-Identifier: Apache-2.0
"""One elasto-plastic material model for Alloy's solids: volumes, sheets, rods and joint springs (SCR-WP7-15, agreed
with WP9's rods; WP9 reviews this module, WP7 reviews WP9's rod use of it).

**Model.** Rate-independent von Mises (J2) plasticity, associative flow, linear isotropic hardening
σ_y(ε̄p) = σ_y0 + H ε̄p on the equivalent (work-conjugate) plastic strain ε̄p. There is no kinematic hardening: a
reversed load yields at -σ_y(ε̄p), with no Bauschinger effect (docs/known_gaps/plasticity-metal.md). The 1-D monotonic
curve past yield is σ = σ_y0 + E_t (ε - σ_y0/E), with E_t = E H / (E + H).

**Card.** `plastic.model: j2`, `plastic.yield_stress` = σ_y0 [Pa, uniaxial initial yield], `plastic.hardening_modulus`
= H [Pa; absent means 0, perfectly plastic]. Without `model: j2` nothing here runs, and a yield_stress makes a paste.

**When.** Once per solver step, on the converged positions (explicit in time, as the film hinge). The plastic state is
solver state: snapshot, rollback and reset restore it. Model arrays derived from it (tet_poses, edge_rest_angle, joint
rest rotations, spring references) are rewritten from it.

**Callers** pass the E and G their solver's elastic stiffness uses (a rod: E = EI/I and G = GJ/J from its card, also
for a cable whose measured EI / GJ are composite); else the elastic range itself shifts the rest shape. Each call
writes the state: call once per step, at commit, and register the state with the entry snapshot so a coupler's restart
pass (Aitken, the contact gate) restores it with everything else (WP9's review, 2026-11-08). H >= 0 (validated on the
card; the functions clamp it).

**Functions** (`wp.func`, float32; the caller keeps the state arrays):
- `return_1d(eps, eps_p, ebar, E, sy0, H) -> (sigma, eps_p, ebar)`: a uniaxial fibre, layer or spring (torque /
  angle units work the same way).
- `return_fibre(eps, gam, eps_p, gam_p, ebar, E, G, sy0, H) -> (sigma, tau, eps_p, gam_p, ebar)`: a fibre in tension
  and shear. q = sqrt(σ² + 3τ²), dε_p = dλ σ/q, dγ_p = dλ 3τ/q, dε̄p = dλ. A bracketed scalar Newton solve finds dλ;
  it is exact radial return when τ = 0 (E) or σ = 0 (3G).
- `return_3d(F, fp_inv, ebar, mu, sy0, H) -> (fp_inv, ebar)`: radial return on the deviatoric Hencky strain of
  Fe = F Fp⁻¹, so Fp is isochoric (det Fp = 1). q = sqrt(3/2) |2μ dev e|, dλ = (q - σ_y)/(3μ + H), and
  Fp⁻¹ ← Fp⁻¹ V exp(e_new - e_trial) Vᵀ, rescaled to det 1 (in float32 it drifts 1.5e-4 in 120 steps without). An
  inverted or collapsed element (det Fe <= 1e-6) is left as it is: its log strain would read a spurious yield.
  A tet hook writes Dm⁻¹ Fp⁻¹ into Newton's `tet_poses`; VBD then evaluates the elastic energy of Fe over the unchanged
  rest volume.
- `sheet_section` / `rod_section`: the section's resultants by fibre quadrature, updating each fibre's state.

**Sections** (WP9 amendments (a) and (b); the closed forms are in `closed_forms.py`, and the laws pin these numbers):
- **Sheet**, per unit width, thickness t: `N_SHEET` = 21 Gauss-Lobatto layers through t, in plane strain across the
  width: E' = E/(1-ν²), σ_y' = 2σ_y0/√3, H' = 4H/3 (incompressible plastic flow). Layer strain ε + κ z.
  - M(κ) is within 0.4 % of the closed form from M_y = σ_y' t²/6 to 0.98 M_p (M_p = σ_y' t²/4); the sup over a dense
    κ grid is 0.37 % (19 layers: 0.45 %).
  - Surface layers make the onset exact.
- **Round wire**, radius R: `N_R` = 12 Gauss-Lobatto radii (11 rings with area) × `N_T` = 20 sectors, 220 fibres.
  - A fibre at (y1, y2) = r(cos θ, sin θ) has axial strain ε + κ1 y2 - κ2 y1 and shear γ = φ' r.
  - Resultants: N = ∫σ, M1 = ∫σ y2, M2 = -∫σ y1, T = ∫τ r.
  - Bending is within 0.35 % of the closed form from M_y = σ_y0 π R³/4 to 0.98 M_p (M_p = 4σ_y0 R³/3), in any bending
    direction; the 20 sectors turn M up to 0.6° off κ's direction near the onset.
  - Torsion is within 0.35 % up to 0.98 T_p (T_p = 2π R³ τ_y/3, τ_y = σ_y0/√3); 10 radii would read 0.44 %.
  - These are the sups over dense κ, φ' grids (`tests/solvers/plastic/test_j2.py`).
"""

from __future__ import annotations

import numpy as np
import warp as wp

N_SHEET = 21                        # Gauss-Lobatto layers through a sheet
N_R, N_T = 12, 20                   # a round section's Gauss-Lobatto radii (r = 0 carries no area) and sectors
NEWTON_ITERS = wp.constant(12)      # return_fibre's scalar solve (bracketed; converged to float32 well before)
SQRT_3_2 = wp.constant(1.2247448713915890)
TINY = wp.constant(1.0e-30)


@wp.struct
class Quad:
    """A section quadrature. Sheet: `x` are layer heights z/t on [-1/2, 1/2] and `w` their weights per t (Σw = 1).
    Rod: `x` are ring radii r/R and `w` the fibre area weights per R² (Σ over rings × n_t sectors = π)."""
    x: wp.array(dtype=float)
    w: wp.array(dtype=float)
    n_t: int


@wp.func
def flow_stress(sy0: float, hmod: float, ebar: float) -> float:
    return sy0 + hmod * ebar


@wp.func
def return_1d(eps: float, eps_p: float, ebar: float, E: float, sy0: float, hmod_in: float):
    """Uniaxial return: (σ, ε_p, ε̄p) after the step. Radial in 1-D: dλ = (|σ_tr| - σ_y) / (E + H)."""
    hmod = wp.max(hmod_in, 0.0)
    sig = E * (eps - eps_p)
    f = wp.abs(sig) - flow_stress(sy0, hmod, ebar)
    if f > 0.0:
        dl = f / (E + hmod)
        s = wp.sign(sig)
        eps_p = eps_p + s * dl
        ebar = ebar + dl
        sig = sig - s * E * dl
    return sig, eps_p, ebar


@wp.func
def _excess(st: float, tt: float, E: float, G: float, y0: float, hmod: float, x: float) -> wp.vec2:
    """(g, dg/dλ) at dλ = x: g = σ² + 3τ² - σ_y², σ = σ_tr/(1 + E x/Y), τ = τ_tr/(1 + 3G x/Y), Y = y0 + H x.
    dσ/dλ = -σ_tr E y0 / (Y + E x)², dτ/dλ = -τ_tr 3G y0 / (Y + 3G x)² (Y - H x = y0)."""
    y = y0 + hmod * x
    a = y + E * x
    b = y + 3.0 * G * x
    s = st * y / a
    t = tt * y / b
    ds = -st * E * y0 / (a * a)
    dt = -tt * 3.0 * G * y0 / (b * b)
    return wp.vec2(s * s + 3.0 * t * t - y * y, 2.0 * s * ds + 6.0 * t * dt - 2.0 * y * hmod)


@wp.func
def return_fibre(eps: float, gam: float, eps_p: float, gam_p: float, ebar: float, E: float, G: float, sy0: float,
                 hmod_in: float):
    """Tension + shear return: (σ, τ, ε_p, γ_p, ε̄p) after the step (module doc)."""
    hmod = wp.max(hmod_in, 0.0)                       # the bracket and "g falls with dλ" need H >= 0
    st = E * (eps - eps_p)
    tt = G * (gam - gam_p)
    y0 = flow_stress(sy0, hmod, ebar)
    q = wp.sqrt(st * st + 3.0 * tt * tt)
    if q <= y0:
        return st, tt, eps_p, gam_p, ebar
    lo = (q - y0) / (wp.max(E, 3.0 * G) + hmod)      # the root lies between the two radial limits
    hi = (q - y0) / (wp.min(E, 3.0 * G) + hmod)
    x = 0.5 * (lo + hi)
    for _ in range(NEWTON_ITERS):
        g = _excess(st, tt, E, G, y0, hmod, x)
        if g[0] > 0.0:
            lo = x
        else:
            hi = x
        xn = x - g[0] / wp.min(g[1], -TINY)           # g falls with dλ
        if xn < lo or xn > hi:                        # strict: a converged step lands on the bracket's end
            xn = 0.5 * (lo + hi)
        x = xn
    y = y0 + hmod * x
    s = st * y / (y + E * x)
    t = tt * y / (y + 3.0 * G * x)
    return s, t, eps_p + x * s / y, gam_p + x * 3.0 * t / y, ebar + x


@wp.func
def return_3d(F: wp.mat33, fp_inv: wp.mat33, ebar: float, mu: float, sy0: float, hmod_in: float):
    """Volume return: (Fp⁻¹, ε̄p) after the step (module doc)."""
    hmod = wp.max(hmod_in, 0.0)
    fe = F * fp_inv
    if wp.determinant(fe) <= 1.0e-6:                  # inverted / collapsed: no meaningful log strain
        return fp_inv, ebar
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(fe, U, S, V)
    e = wp.vec3(wp.log(wp.max(S[0], 1.0e-6)), wp.log(wp.max(S[1], 1.0e-6)), wp.log(wp.max(S[2], 1.0e-6)))
    m = (e[0] + e[1] + e[2]) / 3.0
    dev = e - wp.vec3(m, m, m)
    nd = wp.length(dev)
    q = 2.0 * SQRT_3_2 * mu * nd
    f = q - flow_stress(sy0, hmod, ebar)
    if f <= 0.0 or nd <= 0.0:
        return fp_inv, ebar
    dl = f / (3.0 * mu + hmod)
    d = -SQRT_3_2 * dl / nd * dev                     # e_new - e_trial (traceless)
    D = wp.mat33(wp.exp(d[0]), 0.0, 0.0, 0.0, wp.exp(d[1]), 0.0, 0.0, 0.0, wp.exp(d[2]))
    fp = fp_inv * (V * D * wp.transpose(V))
    return fp * wp.pow(wp.max(wp.determinant(fp), 1.0e-6), -1.0 / 3.0), ebar + dl


@wp.func
def sheet_section(e: int, eps: float, kappa: float, t: float, Ep: float, syp: float, hp: float, q: Quad,
                  eps_p: wp.array2d(dtype=float), ebar: wp.array2d(dtype=float)) -> wp.vec2:
    """(N, M) per unit width of sheet element e at membrane strain ε and curvature κ; updates its layers."""
    n = float(0.0)
    m = float(0.0)
    for k in range(q.x.shape[0]):
        z = q.x[k] * t
        sig, ep, eb = return_1d(eps + kappa * z, eps_p[e, k], ebar[e, k], Ep, syp, hp)
        eps_p[e, k] = ep
        ebar[e, k] = eb
        w = q.w[k] * t
        n += w * sig
        m += w * sig * z
    return wp.vec2(n, m)


@wp.func
def rod_section(e: int, eps: float, kappa1: float, kappa2: float, twist: float, R: float, E: float, G: float,
                sy0: float, hmod: float, q: Quad, eps_p: wp.array2d(dtype=float), gam_p: wp.array2d(dtype=float),
                ebar: wp.array2d(dtype=float)) -> wp.vec4:
    """(N, M1, M2, T) of round-section element e at axial strain ε, curvatures κ1, κ2 and twist rate φ'; updates its
    fibres (index ring × n_t + sector)."""
    n = float(0.0)
    m1 = float(0.0)
    m2 = float(0.0)
    tq = float(0.0)
    dth = 6.283185307179586 / float(q.n_t)
    for i in range(q.x.shape[0]):
        r = q.x[i] * R
        a = q.w[i] * R * R
        for j in range(q.n_t):
            th = (float(j) + 0.5) * dth
            y1 = r * wp.cos(th)
            y2 = r * wp.sin(th)
            f = i * q.n_t + j
            sig, tau, ep, gp, eb = return_fibre(eps + kappa1 * y2 - kappa2 * y1, twist * r, eps_p[e, f], gam_p[e, f],
                                                ebar[e, f], E, G, sy0, hmod)
            eps_p[e, f] = ep
            gam_p[e, f] = gp
            ebar[e, f] = eb
            n += a * sig
            m1 += a * sig * y2
            m2 -= a * sig * y1
            tq += a * tau * r
    return wp.vec4(n, m1, m2, tq)


def _lobatto(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Lobatto nodes and weights on [-1, 1] (n >= 2)."""
    from numpy.polynomial import legendre as L  # noqa: N812

    c = np.zeros(n)
    c[-1] = 1.0
    x = np.concatenate(([-1.0], np.sort(L.legroots(L.legder(c)).real), [1.0]))   # legroots is complex128 (zero imag)
    return x, 2.0 / (n * (n - 1) * L.legval(x, c) ** 2)


def sheet_quad(device: str, n: int = N_SHEET) -> Quad:
    x, w = _lobatto(n)
    q = Quad()
    q.x = wp.array(0.5 * x, dtype=float, device=device)
    q.w = wp.array(0.5 * w, dtype=float, device=device)
    q.n_t = 1
    return q


def rod_quad(device: str, n_r: int = N_R, n_t: int = N_T) -> Quad:
    x, w = _lobatto(n_r)
    r = 0.5 * (x + 1.0)
    a = 0.5 * w * r * (2.0 * np.pi / n_t)            # dA / R² per fibre of ring i
    keep = r > 0.0
    q = Quad()
    q.x = wp.array(r[keep], dtype=float, device=device)
    q.w = wp.array(a[keep], dtype=float, device=device)
    q.n_t = n_t
    return q


def fibres(q: Quad) -> int:
    """Fibres (state columns) per element of a section quadrature."""
    return int(q.x.shape[0]) * int(q.n_t)


def plane_strain(E: float, nu: float, sy0: float, hmod: float) -> tuple[float, float, float]:
    """(E', σ_y', H') for a wide sheet bent across its width (module doc, Sections)."""
    return E / (1.0 - nu * nu), 2.0 * sy0 / np.sqrt(3.0), 4.0 * hmod / 3.0
