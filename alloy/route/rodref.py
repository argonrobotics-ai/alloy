# SPDX-License-Identifier: Apache-2.0
"""Reference solutions for the rod self-test (CARD-7, ROD-1, ROD-2) and law A11. Pure numpy, float64.

Chain layout (D7): N links, half-length ends, L = (N−1)·l. Clamped tests weld link 0 (length l/2) to the wall, so the
N−1 junction hinges sit at s_j = l/2 + (j−1)·l with k_j = EI/l: the midpoint rule of θ' = M/EI, error O(1/N²).
The continuum elastica is the same chain at N = 4001 (relative error ~1e-8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81
CONTINUUM_LINKS = 4001
HANGING_CHAIN_W1 = 2.404825557695773 / 2.0  # first zero of J0 / 2: ω1 = 1.2024·√(g/L) (hanging chain)
CANTILEVER_BETA1 = 1.8751040687119611  # first root of cos·cosh = −1


@dataclass(frozen=True)
class ChainLayout:
    """Clamped half-length-end chain: free link lengths/masses/inertias and hinge stiffnesses (link 0 welded)."""

    lengths: np.ndarray  # [N−1] links 1..N−1 (the welded half link excluded)
    masses: np.ndarray  # [N−1] kg
    inertias: np.ndarray  # [N−1] kg·m², about the COM, bending axis
    k: np.ndarray  # [N−1] hinge j between link j−1 and j, N·m/rad
    root: float  # m, length of the welded link 0


def chain_layout(n_links: int, L: float, lin_mass: float, EI: float, radius: float = 0.0) -> ChainLayout:
    """The D7 chain of a rod of length L, mass per length `lin_mass` and bending stiffness EI."""
    if n_links < 3:
        raise ValueError(f"a half-length-end chain needs >= 3 links, got {n_links}")
    l = L / (n_links - 1)
    lengths = np.full(n_links - 1, l)
    lengths[-1] = 0.5 * l
    masses = lin_mass * lengths
    inertias = masses * (3.0 * radius**2 + lengths**2) / 12.0
    return ChainLayout(lengths, masses, inertias, np.full(n_links - 1, EI / l), 0.5 * l)


def _tridiag_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a symmetric positive definite tridiagonal system (lower[i] = A[i+1, i])."""
    n = len(diag)
    c, d = np.empty(n), np.empty(n)
    c[0], d[0] = upper[0] / diag[0] if n > 1 else 0.0, rhs[0] / diag[0]
    for i in range(1, n):
        den = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / den if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / den
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _chain_energy(layout: ChainLayout, th: np.ndarray, lever: np.ndarray) -> float:
    rel = np.diff(np.concatenate([[0.0], th]))
    return float(0.5 * np.sum(layout.k * rel**2) - np.sum(lever * np.sin(th)))


def _newton_step(layout: ChainLayout, th: np.ndarray, lever: np.ndarray) -> np.ndarray:
    """Newton direction on U(θ) = Σ ½k_j(θ_j − θ_{j−1})² − Σ lever_j·sin θ_j (tridiagonal Hessian)."""
    k = layout.k
    k_next = np.concatenate([k[1:], [0.0]])
    rel = np.diff(np.concatenate([[0.0], th]))
    rel_next = np.concatenate([rel[1:], [0.0]])
    grad = k * rel - k_next * rel_next - lever * np.cos(th)
    diag = k + k_next + lever * np.maximum(np.sin(th), 0.0)
    return _tridiag_solve(-k[1:], diag, -k[1:], -grad)


def static_tip_drop(layout: ChainLayout, g: float = G) -> float:
    """Exact large-deflection self-weight tip drop of a clamped horizontal chain [m], any γ (`static_angles`)."""
    return float(np.sum(layout.lengths * np.sin(static_angles(layout, g))))


def static_angles(layout: ChainLayout, g: float = G, iters: int = 500, tol: float = 1e-14) -> np.ndarray:
    """Equilibrium absolute link angles below horizontal [rad] of the clamped chain under self-weight.

    In absolute link angles θ_j (link 0 welded at θ = 0) the potential energy is
    U = Σ ½k_j(θ_j − θ_{j−1})² − Σ_j l_j·T_j·sin θ_j with T_j = g·(½m_j + Σ_{i>j} m_i), the weight hanging off link
    j's centre and beyond. ∇U = 0 is the hinge moment balance M_j = k_j(θ_j − θ_{j−1}). The Hessian is tridiagonal
    and positive definite for θ ∈ [0, π], so damped Newton converges from θ = 0 for any γ (up to the hanging limit).
    """
    lens, w = layout.lengths, layout.masses * g
    lever = lens * (np.cumsum(w[::-1])[::-1] - 0.5 * w)
    th = np.zeros_like(lens)
    u = _chain_energy(layout, th, lever)
    for _ in range(iters):
        step = _newton_step(layout, th, lever)
        t = 1.0
        while t > 1e-12:
            trial = np.clip(th + t * step, 0.0, 0.5 * math.pi)
            u_new = _chain_energy(layout, trial, lever)
            if u_new <= u:
                break
            t *= 0.5
        done = np.max(np.abs(trial - th)) < tol
        th, u = trial, u_new
        if done:
            break
    return th


def elastica_tip_drop(L: float, lin_mass: float, EI: float, g: float = G) -> float:
    """Continuum self-weight cantilever (Kirchhoff elastica) tip drop [m]."""
    return static_tip_drop(chain_layout(CONTINUUM_LINKS, L, lin_mass, EI), g)


def linear_tip_drop(L: float, lin_mass: float, EI: float, g: float = G) -> float:
    """Euler-Bernoulli small-deflection tip drop w·L⁴/(8EI) [m]."""
    return lin_mass * g * L**4 / (8.0 * EI)


def _modal_matrices(layout: ChainLayout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(M, K, A) of the linearised clamped chain in absolute link angles; z_c = A·θ (COM transverse offsets)."""
    lens = layout.lengths
    n = len(lens)
    A = np.tril(np.tile(lens, (n, 1)), k=-1)
    A[np.diag_indices(n)] = 0.5 * lens
    M = A.T @ np.diag(layout.masses) @ A + np.diag(layout.inertias)
    D = np.eye(n) - np.eye(n, k=-1)
    K = D.T @ np.diag(layout.k) @ D
    return M, K, A


@dataclass(frozen=True)
class ReleaseReference:
    period: float  # s, first-mode period of the discrete chain
    period_continuum: float  # s, Euler-Bernoulli cantilever 2π/ω1
    static_drop: float  # m, linear static tip drop of the discrete chain
    peak_drop: float  # m, first maximum of the tip drop after release from straight (undamped)
    t: np.ndarray  # s
    drop: np.ndarray  # m, tip drop history


def release_reference(layout: ChainLayout, L: float, lin_mass: float, EI: float, t: np.ndarray,
                      g: float = G) -> ReleaseReference:
    """Linear modal superposition of the clamped chain released from straight under gravity, undamped."""
    M, K, A = _modal_matrices(layout)
    Q = -(A.T @ (layout.masses * g))  # generalised gravity load (downward drop positive below)
    th_s = np.linalg.solve(K, Q)
    Lc = np.linalg.cholesky(M)
    Li = np.linalg.inv(Lc)
    w2, V = np.linalg.eigh(Li @ K @ Li.T)
    phi = Li.T @ V  # mass-normalised modes
    w = np.sqrt(np.clip(w2, 0.0, None))
    a = phi.T @ M @ th_s  # θ(t) = θ_s − Σ φ_i a_i cos(ω_i t)
    tip_row = layout.lengths  # small-angle tip offset z_tip = Σ l_i·θ_i
    tip_hist = -(tip_row @ (th_s[:, None] - phi @ (a[:, None] * np.cos(np.outer(w, t)))))
    period = 2.0 * math.pi / w[0]
    w1_cont = CANTILEVER_BETA1**2 * math.sqrt(EI / (lin_mass * L**4))
    first = t <= period
    peak = float(np.max(tip_hist[first])) if first.any() else float(np.max(tip_hist))
    return ReleaseReference(period, 2.0 * math.pi / w1_cont, float(-(tip_row @ th_s)), peak, t, tip_hist)


def hanging_chain_w1(L: float, g: float = G) -> float:
    """First pendulum mode of a uniform hanging chain ω1 = 1.2024·√(g/L) [rad/s] (bench §4 A11)."""
    return HANGING_CHAIN_W1 * math.sqrt(g / L)


def catenary_sag(span: float, length: float) -> float:
    """Mid-span sag of an inextensible catenary of `length` hung between level supports `span` apart [m].

    Solves length = 2a·sinh(span/(2a)) for a, then sag = a·(cosh(span/(2a)) − 1).
    """
    if length <= span:
        raise ValueError(f"catenary needs length > span, got {length} <= {span}")
    lo, hi = span / 1400.0, 1e9 * span  # a; 2a·sinh(span/2a) decreases in a; span/2a ≤ 700 avoids overflow
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if 2.0 * mid * math.sinh(span / (2.0 * mid)) > length:
            lo = mid
        else:
            hi = mid
    a = math.sqrt(lo * hi)
    return a * (math.cosh(span / (2.0 * a)) - 1.0)


def euler_buckling_load(EI: float, L: float, K_eff: float = 1.0) -> float:
    """Euler critical load π²EI/(K·L)² [N] (law A12); K = 1 pinned-pinned, 2 clamped-free, 0.5 clamped-clamped."""
    return math.pi**2 * EI / (K_eff * L) ** 2
