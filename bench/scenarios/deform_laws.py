# SPDX-License-Identifier: Apache-2.0
"""Analytic and exact-discrete references for the `deform` family (numpy only, no scipy on the bench venv).

- Cantilever under self weight: discrete chain of rigid segments with rotational springs k = EI/l and a clamp
  hinge 2EI/l at x = 0 (ghost segment), solved by fixed-point iteration; the continuum Kirchhoff elastica is the
  same solve at N = 2000.
- Linear chain dynamics: generalized eigenproblem of the clamped chain (pluck) and the pinned hanging chain.
- Uniaxial compressible Neo-Hookean (Psi = mu/2 (I1-3) - mu ln J + lam/2 (ln J)^2), lateral faces traction-free.
- ASTM D1388 / Peirce cantilever: invert the heavy elastica for the bending rigidity.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

HANGING_CHAIN_W1 = 2.404825557695773 / 2.0  # first zero of J0 / 2: w1 = 1.2024 sqrt(g/L)
CANTILEVER_BETA1 = 1.8751040687119611  # first root of 1 + cos(b) cosh(b) = 0


# ---- cantilever statics -------------------------------------------------------------------------------------------
def cantilever_chain(nseg: int, L: float, w: float, EI: float, iters: int = 20000, relax: float = 0.3,
                     tol: float = 1e-12) -> tuple[float, float]:
    """Tip (drop, horizontal reach) of a clamped self-weight chain of N rigid segments (large deflection, exact).

    Hinges sit at nodes x_0 = 0 (the clamp) .. x_{N-1}; hinge j has k_j = EI/l and the clamp hinge k_0 = 2EI/l
    (half Voronoi length, the ghost-segment clamp of discrete elastic rods), so the linear tip drop is the
    trapezoid rule of the continuum integral and converges as 1/N^2. Moment balance at node j about the outboard
    weights: k_j (th_j - th_{j-1}) = sum_{i>=j} w l (x_ci - x_j), th_{-1} = 0. Angles downward from horizontal.
    """
    l = L / nseg
    k = np.full(nseg, EI / l)
    k[0] = 2.0 * EI / l
    W = w * l
    th = np.zeros(nseg)
    cnt = np.arange(nseg, 0, -1)
    for _ in range(iters):
        c = np.cos(th)
        x_node = np.concatenate([[0.0], np.cumsum(l * c)])  # nseg + 1 nodes
        x_c = x_node[:-1] + 0.5 * l * c
        tail = np.cumsum((W * x_c)[::-1])[::-1]
        M = tail - W * cnt * x_node[:-1]  # moment at hinge j = 0 .. nseg-1
        new = np.cumsum(M / k)
        if np.max(np.abs(new - th)) < tol:
            th = new
            break
        th = (1.0 - relax) * th + relax * new
    drop = float(np.sum(l * np.sin(th)))
    reach = float(np.sum(l * np.cos(th)))
    return drop, reach


@lru_cache(maxsize=64)
def elastica_tip(gamma: float, L: float = 1.0) -> tuple[float, float]:
    """Continuum self-weight elastica tip (drop, reach) for gamma = w L^3 / EI (unit w)."""
    return cantilever_chain(2000, L, 1.0, L**3 / gamma)


def linear_tip(L: float, w: float, EI: float) -> float:
    return w * L**4 / (8.0 * EI)


# ---- linear chain dynamics ----------------------------------------------------------------------------------------
def _link_kinematics(n: int, l: float) -> np.ndarray:
    """A with z_c = A @ theta: COM transverse offset of link i from absolute link angles (link 0 first)."""
    A = np.tril(np.full((n, n), l), k=-1)
    A[np.diag_indices(n)] = 0.5 * l
    return A


def _gen_eig(K: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Lc = np.linalg.cholesky(M)
    Li = np.linalg.inv(Lc)
    w2, V = np.linalg.eigh(Li @ K @ Li.T)
    return np.sqrt(np.clip(w2, 0.0, None)), Li.T @ V


def pluck_reference(nseg: int, L: float, m_lin: float, EI: float, radius: float, g: float,
                    t: np.ndarray) -> dict:
    """Clamped chain released from straight under gravity (linear modal superposition, undamped).

    Coordinates: absolute angles of links 0..N-1, clamp hinge k_0 = 2EI/l (as cantilever_chain). Returns the tip
    drop history, the first-mode period and the static linear tip drop of the discrete model, plus the continuum
    Euler-Bernoulli values.
    """
    n, l = nseg, L / nseg
    m = m_lin * l
    Ic = m * (3 * radius**2 + l**2) / 12.0
    A = _link_kinematics(n, l)
    M = m * A.T @ A + Ic * np.eye(n)
    D = np.eye(n) - np.eye(n, k=-1)
    kj = np.full(n, EI / l)
    kj[0] = 2.0 * EI / l
    K = D.T @ np.diag(kj) @ D
    F = A.T @ np.full(n, m * g)
    omega, V = _gen_eig(K, M)
    q_static = np.linalg.solve(K, F)
    tip_row = np.full(n, l)
    modal = V.T @ F / omega**2  # V is M-orthonormal
    tip_hist = (tip_row @ V) @ (modal[:, None] * (1.0 - np.cos(np.outer(omega, t))))
    f1_cont = CANTILEVER_BETA1**2 / (2 * math.pi) * math.sqrt(EI / (m_lin * L**4))
    return {"tip": tip_hist, "T1": 2 * math.pi / omega[0], "static_tip": float(tip_row @ q_static),
            "T1_continuum": 1.0 / f1_cont, "static_tip_continuum": linear_tip(L, m_lin * g, EI)}


def hanging_chain_modes(nseg: int, L: float, m_lin: float, radius: float, g: float) -> tuple[float, np.ndarray]:
    """First angular frequency and link-angle mode shape of a pinned chain of compound links (small swing)."""
    n, l = nseg, L / nseg
    m = m_lin * l
    Ic = m * (3 * radius**2 + l**2) / 12.0
    A = _link_kinematics(n, l)
    M = m * A.T @ A + Ic * np.eye(n)
    K = np.diag(m * g * l * (n - np.arange(n) - 0.5))
    omega, V = _gen_eig(K, M)
    shape = V[:, 0] / np.abs(np.sum(l * V[:, 0]))  # unit lateral displacement at the free end
    return float(omega[0]), shape * np.sign(np.sum(shape))


def hanging_chain_w1(L: float, g: float) -> float:
    return HANGING_CHAIN_W1 * math.sqrt(g / L)


def hanging_lumped_modes(nseg: int, L: float, m_lin: float, g: float) -> tuple[float, np.ndarray]:
    """Same for point masses at the nodes 1..N (particle chain, node 0 pinned); returns w1 and node offsets."""
    n, l = nseg, L / nseg
    mass = np.full(n, m_lin * l)
    mass[-1] *= 0.5
    tension = np.array([np.sum(mass[i:]) * g for i in range(n)])  # tension in link i (node i -> i+1)
    K = np.zeros((n, n))
    for i in range(n):  # lateral offsets x_1..x_N, link i between nodes i and i+1 (x_0 = 0)
        kk = tension[i] / l
        a, b = i - 1, i
        K[b, b] += kk
        if a >= 0:
            K[a, a] += kk
            K[a, b] -= kk
            K[b, a] -= kk
    omega, V = _gen_eig(K, np.diag(mass))
    shape = V[:, 0] / abs(V[-1, 0])
    return float(omega[0]), shape * np.sign(shape[-1])


# ---- capstan / Michell ----------------------------------------------------------------------------------------------
def capstan_ratio(mu: float, wrap: float) -> float:
    return math.exp(mu * wrap)


def michell_critical_twist(bend: float, twist: float) -> float:
    return 2.0 * math.pi * math.sqrt(3.0 * bend / twist)


# ---- Neo-Hookean uniaxial --------------------------------------------------------------------------------------------
def lame(E: float, nu: float) -> tuple[float, float]:
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def nh_uniaxial(stretch: float, mu: float, lam: float) -> tuple[float, float]:
    """Compressible NH under uniaxial stress: (nominal stress P [Pa], lateral stretch a). Bisection on a."""
    def lateral(a: float) -> float:
        return mu * (a * a - 1.0) + lam * math.log(a * a * stretch)

    lo, hi = 0.3, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lateral(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    a = 0.5 * (lo + hi)
    J = a * a * stretch
    return mu * stretch - mu / stretch + lam * math.log(J) / stretch, a


def nh_incompressible(stretch: float, mu: float) -> float:
    return mu * (stretch - stretch**-2)


def nh_stretch_for_stress(P: float, mu: float, lam: float) -> float:
    """Inverse of nh_uniaxial (compression, P < 0): the axial stretch carrying nominal stress P."""
    lo, hi = 0.2, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if nh_uniaxial(mid, mu, lam)[0] < P:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def timoshenko_tip(L: float, b: float, hgt: float, rho: float, g: float, E: float, nu: float) -> float:
    """Cantilever block under self weight: Euler-Bernoulli plus shear (kappa = 5/6) tip deflection."""
    A, I = b * hgt, b * hgt**3 / 12.0
    w = rho * g * A
    Gs = E / (2 * (1 + nu))
    return w * L**4 / (8 * E * I) + w * L**2 / (2 * (5.0 / 6.0) * Gs * A)


# ---- cloth ----------------------------------------------------------------------------------------------------------
def tip_chord_angle(gamma: float) -> float:
    """Chord angle [rad] below horizontal from the clamp to the tip of a heavy cantilever, gamma = W l^3 / B."""
    drop, reach = cantilever_chain(400, 1.0, 1.0, 1.0 / gamma, relax=0.2)
    return math.atan2(drop, reach)


def gamma_for_chord_angle(angle: float) -> float:
    """Inverse of tip_chord_angle by bisection on log(gamma)."""
    lo, hi = math.log(1e-3), math.log(60.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tip_chord_angle(math.exp(mid)) < angle:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


D1388_ANGLE = math.radians(41.5)
