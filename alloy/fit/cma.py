# SPDX-License-Identifier: Apache-2.0
"""CMA-ES in the unit box (ask/tell), numpy only.

(mu/mu_w, lambda)-CMA-ES with cumulative step-size adaptation, rank-one and rank-mu covariance updates and the
default strategy constants of Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772 (2016), Table 1
and Fig. 7. Box constraints: samples are mirrored into [0, 1] and the mirrored points are evaluated and used in the
update (they are the points whose fitness is known). One candidate is one batched world group (SPEC §3.12), so
lambda is set by the world budget, not by the 4 + 3 ln n default (which is the minimum here). Deterministic in
`seed` (numpy Generator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TOL_X = 1e-4          # default: stop when every axis step is below this (unit box; 1e-4 of each range)
TOL_FUN = 1e-12       # stop when the best-fitness range over the last generations is below this
FLAT_GENERATIONS = 10


def default_popsize(n: int) -> int:
    return 4 + int(3 * math.log(n))


def mirror(u: np.ndarray) -> np.ndarray:
    """Reflect values into [0, 1] (period-2 mirror, so any real maps inside)."""
    v = np.mod(np.asarray(u, np.float64), 2.0)
    return np.where(v > 1.0, 2.0 - v, v)


@dataclass(frozen=True)
class CMAConstants:
    weights: np.ndarray
    mueff: float
    cc: float
    cs: float
    c1: float
    cmu: float
    damps: float
    chi_n: float

    @staticmethod
    def make(n: int, lam: int) -> "CMAConstants":
        mu = lam // 2
        w = math.log((lam + 1) / 2) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(1.0 / np.sum(w * w))
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        return CMAConstants(w, mueff, cc, cs, c1, cmu, damps, chi_n)


class CMAES:
    """Ask/tell CMA-ES over [0, 1]^n."""

    def __init__(self, mean: np.ndarray, sigma: float, popsize: int, seed: int, tol_x: float = TOL_X) -> None:
        self.n = int(np.size(mean))
        self.lam = max(int(popsize), default_popsize(self.n), 2)
        self.k = CMAConstants.make(self.n, self.lam)
        self.mean = mirror(np.asarray(mean, np.float64).copy())
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)
        self.C = np.eye(self.n)
        self.generation = 0
        self.tol_x = float(tol_x)
        self._best_hist: list[float] = []

    def ask(self, inject: np.ndarray | None = None) -> np.ndarray:
        """lambda candidates in the unit box, shape [lambda, n]; `inject` replaces candidate 0 (e.g. the start
        point, so a fit is never worse than its closed-form estimate). The draw is the same with or without it."""
        z = self.rng.standard_normal((self.lam, self.n))
        y = z @ (self.B * self.D).T
        x = mirror(self.mean + self.sigma * y)
        if inject is not None:
            x[0] = mirror(np.asarray(inject, np.float64))
        return x

    def tell(self, x: np.ndarray, f: np.ndarray) -> None:
        """Update from candidates `x` [lambda, n] and their fitness `f` [lambda] (lower is better; NaN = worst)."""
        k, n = self.k, self.n
        f = np.where(np.isfinite(f), f, np.inf)
        order = np.argsort(f, kind="stable")
        mu = k.weights.size
        sel = np.asarray(x, np.float64)[order[:mu]]
        old = self.mean
        self.mean = k.weights @ sel
        step = (self.mean - old) / self.sigma
        inv_sqrt_c = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - k.cs) * self.ps + math.sqrt(k.cs * (2 - k.cs) * k.mueff) * (inv_sqrt_c @ step)
        norm_ps = float(np.linalg.norm(self.ps))
        g1 = self.generation + 1
        hsig = norm_ps / math.sqrt(1 - (1 - k.cs) ** (2 * g1)) / k.chi_n < 1.4 + 2 / (n + 1)
        self.pc = (1 - k.cc) * self.pc + hsig * math.sqrt(k.cc * (2 - k.cc) * k.mueff) * step
        art = (sel - old) / self.sigma
        rank_mu = art.T @ (k.weights[:, None] * art)
        delta = (1 - hsig) * k.cc * (2 - k.cc)
        self.C = ((1 - k.c1 - k.cmu) * self.C + k.c1 * (np.outer(self.pc, self.pc) + delta * self.C)
                  + k.cmu * rank_mu)
        self.sigma *= math.exp((k.cs / k.damps) * (norm_ps / k.chi_n - 1))
        self.sigma = min(self.sigma, 1.0)    # never wider than the box
        self._decompose()
        self.generation += 1
        self._best_hist.append(float(f[order[0]]))

    def _decompose(self) -> None:
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        d2, b = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-30))
        self.B = b

    def axis_steps(self) -> np.ndarray:
        """Current 1-sigma step per coordinate in the unit box: sigma * sqrt(C_ii)."""
        return self.sigma * np.sqrt(np.maximum(np.diag(self.C), 0.0))

    def stop_reason(self) -> str | None:
        """'tolx' when every coordinate step < tol_x; 'tolfun' when the best fitness is flat; else None."""
        if float(np.max(self.axis_steps())) < self.tol_x:
            return "tolx"
        hist = self._best_hist[-FLAT_GENERATIONS:]
        if len(hist) == FLAT_GENERATIONS and np.all(np.isfinite(hist)) and max(hist) - min(hist) < TOL_FUN:
            return "tolfun"
        return None


__all__ = ["CMAES", "CMAConstants", "default_popsize", "mirror", "TOL_X"]
