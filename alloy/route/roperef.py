# SPDX-License-Identifier: Apache-2.0
"""Reference solutions for rope twins (R9) with bending: a heavy rope hung between level pins, and the first mode of a
rope hanging from a pin with hinge damping. Pure numpy/scipy, float64, n uniform links (n = 400 is the continuum to
~1e-5; the CHAIN twin uses its own D7 layout).

- `supported_sag`: links of length l = L/n with absolute angles θ below horizontal, pinned (moment-free) at both ends,
  hinge stiffness EI/l. Stationary point of U = Σ½k(θ_j − θ_{j−1})² − Σ lever_j·sin θ_j under Σ l·cos θ = S and
  Σ l·sin θ = 0, by Newton on the KKT system (tridiagonal Hessian, two multipliers by Schur complement), started
  from the inextensible catenary.
- `hanging_modes`: small swings φ from vertical: M = AᵀmA + diag(I), K = K_gravity (diag l·g·(weight below the link
  centre)) + K_bend (tridiagonal EI/l between links; the top pin is free), C = the hinge dampers c'/l. ω₁ and its
  mode from the generalised eigenproblem; ζ₁ = φ₁ᵀCφ₁ / (2ω₁) for an M-normalised φ₁.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from alloy.route.rodref import G, _tridiag_solve, catenary_sag

CONTINUUM = 400


def _bend_matrix(n: int, k: float) -> np.ndarray:
    """Tridiagonal Σ ½k(θ_j − θ_{j−1})² over the n − 1 inner hinges (free ends)."""
    K = np.zeros((n, n))
    idx = np.arange(n - 1)
    K[idx, idx] += k
    K[idx + 1, idx + 1] += k
    K[idx, idx + 1] -= k
    K[idx + 1, idx] -= k
    return K


def _catenary_angles(n: int, L: float, span: float) -> np.ndarray:
    """Link angles below horizontal of the inextensible catenary (tan φ = (s − L/2)/a at the link centres)."""
    sag = catenary_sag(span, L)
    lo, hi = 1e-6, 1e6                                      # a from sag = a(cosh(S/2a) − 1)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if mid * (math.cosh(span / (2 * mid)) - 1.0) > sag else (lo, mid)
    a = math.sqrt(lo * hi)
    s = (np.arange(n) + 0.5) * (L / n)
    return -np.arctan((s - 0.5 * L) / a)


@dataclass(frozen=True)
class _Rope:
    n: int
    l: float
    k: float
    lever: np.ndarray        # g·m·(n − j − ½)·l per link

    @staticmethod
    def make(L: float, lin: float, EI: float, n: int, g: float) -> "_Rope":
        l = L / n
        return _Rope(n, l, EI / l, g * lin * l * l * (n - np.arange(n) - 0.5))


def _kkt_residual(r: _Rope, th: np.ndarray, lam: np.ndarray, span: float) -> tuple[np.ndarray, np.ndarray]:
    rel = np.diff(np.concatenate([[th[0]], th]))                          # free first hinge (pin)
    nxt = np.concatenate([rel[1:], [0.0]])
    grad = r.k * (rel - nxt) - r.lever * np.cos(th) + r.l * (-lam[0] * np.sin(th) + lam[1] * np.cos(th))
    cons = np.array([r.l * np.sum(np.cos(th)) - span, r.l * np.sum(np.sin(th))])
    return grad, cons


def _kkt_step(r: _Rope, th: np.ndarray, lam: np.ndarray, grad: np.ndarray, cons: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    kd = np.full(r.n, 2.0 * r.k)
    kd[0] = kd[-1] = r.k
    diag = kd + r.lever * np.sin(th) - r.l * (lam[0] * np.cos(th) + lam[1] * np.sin(th))
    off = np.full(r.n - 1, -r.k)
    J = np.stack([-r.l * np.sin(th), r.l * np.cos(th)])                   # [2, n]
    X = np.stack([_tridiag_solve(off, diag, off, J[i]) for i in range(2)], axis=1)   # H⁻¹Jᵀ
    y = _tridiag_solve(off, diag, off, grad)
    dlam = np.linalg.solve(J @ X, cons - J @ y)
    return -y - X @ dlam, dlam


def supported_sag(L: float, span: float, lin: float, EI: float, n: int = CONTINUUM, g: float = G,
                  iters: int = 200, tol: float = 1e-13) -> float:
    """Mid-span sag [m] of a rope of length L, mass per length `lin` and bending stiffness EI between level pins."""
    if n % 2:
        raise ValueError("supported_sag needs an even link count (a node at mid-span)")
    r = _Rope.make(L, lin, EI, n, g)
    th = _catenary_angles(n, L, span)
    lam = np.zeros(2)
    grad, cons = _kkt_residual(r, th, lam, span)
    A = r.l * np.stack([-np.sin(th), np.cos(th)], axis=1)                 # least-squares multipliers at the start
    lam = np.linalg.lstsq(A, -grad, rcond=None)[0]
    for _ in range(iters):
        grad, cons = _kkt_residual(r, th, lam, span)
        dth, dlam = _kkt_step(r, th, lam, grad, cons)
        t, base = 1.0, np.linalg.norm(grad) + np.linalg.norm(cons)
        while t > 1e-6:
            g2, c2 = _kkt_residual(r, th + t * dth, lam + t * dlam, span)
            if np.linalg.norm(g2) + np.linalg.norm(c2) < base or t < 2e-6:
                break
            t *= 0.5
        th, lam = th + t * dth, lam + t * dlam
        if np.max(np.abs(t * dth)) < tol:
            break
    return float(r.l * np.sum(np.sin(th[: n // 2])))


@dataclass(frozen=True)
class HangingMode:
    omega: float               # rad/s
    zeta: float                # modal damping ratio
    angles: np.ndarray         # mode shape: link angles from vertical (top to bottom), max |φ| = 1

    @property
    def log_decrement(self) -> float:
        return 2.0 * math.pi * self.zeta / math.sqrt(max(1.0 - self.zeta**2, 1e-12))

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega


def hanging_modes(L: float, lin: float, EI: float, damping: float = 0.0, radius: float = 0.0, n: int = CONTINUUM,
                  g: float = G, pivot_damping: float = 0.0, lengths: np.ndarray | None = None) -> HangingMode:
    """First lateral mode of a rope pinned at the top: `damping` = hinge damping per length c' [N·m²·s],
    `pivot_damping` [N·m·s/rad] on the top link's absolute angle. `lengths` (top to bottom) replaces the n uniform
    links, e.g. the D7 chain's half-length ends; hinges keep k = EI/l with l = L/(links − 1) as D7 sets them.

    ζ₁ is the modal projection, valid for light damping only: hinge dampers strong enough to lock the hinges turn
    the rope into a rigid pendulum instead (measured on the CHAIN: c' = 0.0079 N·m²·s gave δ 0.006, not the
    projected 0.049)."""
    ls = np.full(n, L / n) if lengths is None else np.asarray(lengths, np.float64)
    n = len(ls)
    l_hinge = L / n if lengths is None else L / (n - 1)
    m = lin * ls
    A = np.tril(np.tile(ls, (n, 1)), -1) + np.diag(0.5 * ls)
    M = A.T @ (m[:, None] * A) + np.diag(m * (3.0 * radius**2 + ls * ls) / 12.0)
    below = np.cumsum(m[::-1])[::-1] - 0.5 * m
    K = np.diag(ls * g * below) + _bend_matrix(n, EI / l_hinge)
    w2, V = eigh(K, M, subset_by_index=[0, 0])
    phi = V[:, 0]                                                          # M-normalised by eigh
    omega = math.sqrt(float(w2[0]))
    C = _bend_matrix(n, damping / l_hinge)
    C[0, 0] += pivot_damping
    zeta = float(phi @ C @ phi) / (2.0 * omega)
    return HangingMode(omega, zeta, phi / np.max(np.abs(phi)))


__all__ = ["supported_sag", "hanging_modes", "HangingMode"]
